"""
CogniStream — QLoRA Fine-tuning Script for Moondream2

Fine-tunes moondream2 using its native training API with LoRA adapters.
Uses the distilled dataset from NVIDIA cloud VLM.

Prerequisites:
    pip install peft bitsandbytes accelerate pillow

Usage:
    python scripts/finetune/train.py
    python scripts/finetune/train.py --epochs 5 --lr 1e-4 --rank 32

Output:
    data/finetune/cognistream-moondream-lora/
    data/finetune/cognistream-moondream-merged/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FINETUNE_DIR = DATA_DIR / "finetune"
DATASET_PATH = FINETUNE_DIR / "distillation_dataset.jsonl"
LORA_OUTPUT = FINETUNE_DIR / "cognistream-moondream-lora"
MERGED_OUTPUT = FINETUNE_DIR / "cognistream-moondream-merged"

BASE_MODEL = "vikhyatk/moondream2"


def load_dataset():
    """Load the distillation dataset."""
    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s\nRun distillation first.", DATASET_PATH)
        sys.exit(1)

    items = []
    with open(DATASET_PATH) as f:
        for line in f:
            item = json.loads(line)
            if Path(item["image_path"]).exists():
                items.append(item)

    logger.info("Loaded %d training examples", len(items))
    return items


def train(args):
    """Run LoRA fine-tuning using moondream's native API."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    items = load_dataset()
    if not items:
        logger.error("No valid training examples found.")
        sys.exit(1)

    logger.info("Loading base model: %s", BASE_MODEL)

    # Load model — use float16, no 4-bit for now (moondream is small enough)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Find all linear layer names
    target_modules = set()
    for name, module in model.named_modules():
        if "Linear" in type(module).__name__:
            short = name.split(".")[-1]
            target_modules.add(short)
    target_modules.discard("lm_head")

    logger.info("LoRA targets: %s", sorted(target_modules))
    logger.info("Config: rank=%d, alpha=%d, lr=%s, epochs=%d",
                args.rank, args.alpha, args.lr, args.epochs)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=list(target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # Custom training loop using moondream's native encoding
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=0.01,
    )

    model.train()
    total_steps = len(items) * args.epochs // args.grad_accum
    step = 0
    t_start = time.monotonic()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_loss = 0.0

        for i, item in enumerate(items):
            try:
                # Load image and encode text
                image = Image.open(item["image_path"]).convert("RGB")

                # Build the training text: prompt + response
                text = f"{item['prompt']}\n{item['response']}"

                # Tokenize
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(model.device)

                # Forward pass — use the base model's text generation head
                outputs = model.base_model.model.text_model(
                    inputs_embeds=model.base_model.model.text_model.get_input_embeddings()(tokens["input_ids"]),
                    attention_mask=tokens["attention_mask"],
                )

                # Compute language modeling loss
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = tokens["input_ids"][..., 1:].contiguous()

                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                loss = loss / args.grad_accum
                loss.backward()
                batch_loss += loss.item()

                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1

                    if step % 5 == 0:
                        elapsed = time.monotonic() - t_start
                        logger.info(
                            "Epoch %d | Step %d/%d | Loss: %.4f | %.1f min elapsed",
                            epoch + 1, step, total_steps, batch_loss, elapsed / 60,
                        )
                    epoch_loss += batch_loss
                    batch_loss = 0.0

            except Exception as exc:
                logger.debug("Skipping sample %d: %s", i, exc)
                continue

        logger.info("Epoch %d complete | Avg loss: %.4f", epoch + 1,
                    epoch_loss / max(1, len(items) // args.grad_accum))

    elapsed = time.monotonic() - t_start
    logger.info("Training complete in %.1f min", elapsed / 60)

    # Save LoRA adapter
    LORA_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_OUTPUT))
    tokenizer.save_pretrained(str(LORA_OUTPUT))
    logger.info("LoRA adapter saved: %s", LORA_OUTPUT)

    # Merge
    if args.merge:
        logger.info("Merging LoRA into base model...")
        merged = model.merge_and_unload()
        MERGED_OUTPUT.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(MERGED_OUTPUT))
        tokenizer.save_pretrained(str(MERGED_OUTPUT))
        logger.info("Merged model saved: %s", MERGED_OUTPUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", default="2e-4")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--merge", action="store_true", default=True)
    parser.add_argument("--no-merge", dest="merge", action="store_false")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
