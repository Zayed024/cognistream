"""
CogniStream — Moondream2 Fine-tuning Script

Fine-tunes moondream2's text model on NVIDIA-distilled captions.
Vision encoder is frozen (official recommendation).

Uses moondream's internal text encoding + LM head for proper loss
computation with the model's custom architecture.

Prerequisites:
    pip install transformers torch pillow einops

Usage:
    python scripts/finetune/train.py
    python scripts/finetune/train.py --epochs 3 --lr 3e-5

Output:
    data/finetune/cognistream-moondream/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
OUTPUT_DIR = FINETUNE_DIR / "cognistream-moondream"

BASE_MODEL = "vikhyatk/moondream2"
MD_REVISION = "2024-08-26"


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        sys.exit(1)

    items = []
    with open(DATASET_PATH) as f:
        for line in f:
            item = json.loads(line)
            if Path(item["image_path"]).exists():
                items.append(item)

    random.shuffle(items)
    logger.info("Loaded %d training examples", len(items))
    return items


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_dataset()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    logger.info("Device: %s | Model: %s (rev=%s)", DEVICE, BASE_MODEL, args.revision)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=args.revision, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=args.revision, trust_remote_code=True,
        dtype=DTYPE, device_map={"": DEVICE},
    )

    # Freeze vision encoder
    vision_frozen = 0
    text_trainable = 0
    for name, param in model.named_parameters():
        if "vision" in name.lower() or "visual" in name.lower() or "enc" in name.lower():
            param.requires_grad = False
            vision_frozen += param.numel()
        else:
            param.requires_grad = True
            text_trainable += param.numel()

    logger.info("Frozen: %dM params | Trainable: %dM params",
                vision_frozen // 1_000_000, text_trainable // 1_000_000)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr), weight_decay=0.01,
    )

    # Get the internal text model and embedding layer
    inner = model.model if hasattr(model, 'model') else model
    wte = inner.text.wte if hasattr(inner, 'text') else None

    if wte is None:
        logger.error("Cannot find text embedding layer (model.model.text.wte)")
        sys.exit(1)

    # Get tokenizer from the inner model (moondream uses its own tokenizer)
    md_tokenizer = inner.tokenizer if hasattr(inner, 'tokenizer') else None

    model.train()
    total_steps = (len(items) * args.epochs) // args.grad_accum
    step = 0
    t_start = time.monotonic()

    for epoch in range(args.epochs):
        random.shuffle(items)
        epoch_loss = 0.0
        batch_loss = 0.0
        valid = 0

        for i, item in enumerate(items):
            try:
                # Tokenize the answer text
                answer = item["response"]

                if md_tokenizer:
                    token_ids = md_tokenizer.encode(answer).ids
                else:
                    token_ids = tokenizer.encode(answer, add_special_tokens=False)

                if len(token_ids) < 5:
                    continue

                # Truncate to max length
                token_ids = token_ids[:512]
                input_ids = torch.tensor([token_ids], device=DEVICE)

                # Get embeddings
                embeds = F.embedding(input_ids, wte)

                # Forward through text model blocks
                hidden = embeds
                text_model = inner.text

                # Simple forward: embedding → transformer blocks → layer norm
                for block in text_model.blocks:
                    # Layer norm
                    ln_weight = block.ln.weight
                    ln_bias = block.ln.bias if hasattr(block.ln, 'bias') else None
                    normed = F.layer_norm(hidden, (hidden.size(-1),), ln_weight, ln_bias)

                    # Self-attention (simplified — no KV cache for training)
                    qkv = block.attn.qkv(normed)
                    d = hidden.size(-1)
                    q, k, v = qkv.split([d, d // 4 * 2, d // 4 * 2], dim=-1) if qkv.size(-1) != d * 3 else qkv.chunk(3, dim=-1)

                    # Scaled dot-product attention
                    bsz, seq_len = hidden.shape[:2]
                    n_heads = d // 64  # head_dim = 64
                    head_dim = 64

                    # Reshape for multi-head attention
                    q_heads = q[..., :d].view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
                    k_heads = k.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
                    v_heads = v.view(bsz, seq_len, -1, head_dim).transpose(1, 2)

                    # Causal mask
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1).bool()
                    attn_out = F.scaled_dot_product_attention(
                        q_heads, k_heads, v_heads, attn_mask=~causal_mask.unsqueeze(0).unsqueeze(0)
                    )
                    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, d)
                    attn_out = block.attn.proj(attn_out)

                    # MLP
                    mlp_out = block.mlp.fc2(F.gelu(block.mlp.fc1(normed)))

                    hidden = hidden + attn_out + mlp_out

                # LM head
                hidden = F.layer_norm(hidden, (hidden.size(-1),),
                                      text_model.post_ln.weight,
                                      text_model.post_ln.bias if hasattr(text_model.post_ln, 'bias') else None)
                logits = text_model.lm_head(hidden)

                # Cross-entropy loss (next token prediction)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                loss = loss / args.grad_accum
                loss.backward()
                batch_loss += loss.item()
                valid += 1

                if valid % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1

                    if step % 5 == 0:
                        elapsed = time.monotonic() - t_start
                        logger.info(
                            "Epoch %d | Step %d/%d | Loss: %.4f | %.1f min",
                            epoch + 1, step, total_steps, batch_loss, elapsed / 60,
                        )
                    epoch_loss += batch_loss
                    batch_loss = 0.0

            except Exception as exc:
                if i < 5:
                    logger.warning("Sample %d: %s", i, str(exc)[:100])
                continue

        n_steps = max(1, valid // args.grad_accum)
        logger.info("Epoch %d | Avg loss: %.4f | Valid: %d/%d",
                    epoch + 1, epoch_loss / n_steps, valid, len(items))

    elapsed = time.monotonic() - t_start
    logger.info("Training complete: %d steps in %.1f min", step, elapsed / 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("Saved: %s", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", default="3e-5")
    parser.add_argument("--revision", default=MD_REVISION)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
