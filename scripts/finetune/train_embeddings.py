"""
CogniStream — Embedding Model Fine-tuning

Fine-tunes all-MiniLM-L6-v2 on query→segment pairs generated from
the distilled captions. Uses MultipleNegativesRankingLoss (contrastive).

This improves retrieval precision for CogniStream-specific queries
by teaching the embedding model which captions match which queries.

Prerequisites:
    pip install sentence-transformers datasets

Usage:
    # Generate training pairs + train (one command)
    python scripts/finetune/train_embeddings.py

    # Custom settings
    python scripts/finetune/train_embeddings.py --epochs 5 --batch-size 32

Output:
    data/finetune/cognistream-embedder/  (fine-tuned model)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR, EMBEDDING_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FINETUNE_DIR = DATA_DIR / "finetune"
DATASET_PATH = FINETUNE_DIR / "distillation_dataset.jsonl"
PAIRS_PATH = FINETUNE_DIR / "embedding_pairs.jsonl"
OUTPUT_DIR = FINETUNE_DIR / "cognistream-embedder"


def generate_training_pairs():
    """Generate query→passage pairs from the distilled captions.

    For each caption, generates synthetic queries that a user might search for.
    Uses the caption text to derive natural search queries.
    """
    if not DATASET_PATH.exists():
        logger.error("Distillation dataset not found: %s", DATASET_PATH)
        logger.error("Run: python scripts/finetune/distill.py")
        sys.exit(1)

    items = []
    with open(DATASET_PATH) as f:
        for line in f:
            items.append(json.loads(line))

    logger.info("Generating training pairs from %d captions...", len(items))
    pairs = []

    for item in items:
        response = item["response"]

        # Extract structured fields
        scene = ""
        objects = ""
        activity = ""

        for line in response.split("\n"):
            line_lower = line.strip().lower()
            if line_lower.startswith("scene:") or line_lower.startswith("**scene:**"):
                scene = line.split(":", 1)[1].strip().strip("*")
            elif line_lower.startswith("objects:") or line_lower.startswith("**objects:**"):
                objects = line.split(":", 1)[1].strip().strip("*")
            elif line_lower.startswith("activity:") or line_lower.startswith("**activity:**"):
                activity = line.split(":", 1)[1].strip().strip("*")

        # Generate query variants from the caption content
        passage = response

        # Query from activity
        if activity and len(activity) > 10 and "no activity" not in activity.lower():
            pairs.append({"query": activity, "passage": passage})
            # Rephrase as question
            pairs.append({"query": f"when does {activity.lower().rstrip('.')}", "passage": passage})

        # Query from objects
        if objects and objects.lower() != "none" and len(objects) > 5:
            obj_list = [o.strip() for o in objects.split(",")]
            if len(obj_list) >= 2:
                # "show me X and Y"
                sample = random.sample(obj_list, min(2, len(obj_list)))
                pairs.append({"query": f"show me {' and '.join(sample)}", "passage": passage})
                # "find X near Y"
                pairs.append({"query": f"find {sample[0]}", "passage": passage})

        # Query from scene
        if scene and len(scene) > 20 and "black background" not in scene.lower():
            # First sentence as a search
            first_sent = scene.split(".")[0].strip()
            if len(first_sent) > 15:
                pairs.append({"query": first_sent, "passage": passage})

        # Full caption as self-match (identity pair)
        if len(passage) > 50:
            pairs.append({"query": passage[:100], "passage": passage})

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = p["query"][:50]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    random.shuffle(unique_pairs)
    logger.info("Generated %d unique training pairs", len(unique_pairs))

    # Save
    with open(PAIRS_PATH, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p) + "\n")

    return unique_pairs


def train(args):
    """Fine-tune the embedding model on query→passage pairs."""
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    # Generate or load pairs
    if PAIRS_PATH.exists() and not args.regenerate:
        pairs = []
        with open(PAIRS_PATH) as f:
            for line in f:
                pairs.append(json.loads(line))
        logger.info("Loaded %d existing training pairs", len(pairs))
    else:
        pairs = generate_training_pairs()

    if len(pairs) < 10:
        logger.error("Too few training pairs (%d). Need at least 10.", len(pairs))
        sys.exit(1)

    # Create training examples
    train_examples = [
        InputExample(texts=[p["query"], p["passage"]])
        for p in pairs
    ]

    logger.info("Training %s on %d pairs, %d epochs, batch=%d",
                EMBEDDING_MODEL, len(train_examples), args.epochs, args.batch_size)

    # Load model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    t_start = time.monotonic()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=min(100, len(train_dataloader)),
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
    )

    elapsed = time.monotonic() - t_start
    logger.info("Training complete in %.1f min", elapsed / 60)
    logger.info("Fine-tuned model saved: %s", OUTPUT_DIR)
    logger.info("")
    logger.info("To use it, set in .env:")
    logger.info("  EMBEDDING_MODEL=%s", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate training pairs")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
