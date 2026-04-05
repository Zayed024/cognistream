"""
CogniStream — SigLIP Visual Embedder

Embeds keyframe images directly into a text-searchable vector space
using SigLIP 2 (Sigmoid Loss for Language Image Pre-Training).

This provides fast visual similarity search WITHOUT needing VLM captioning.
A text query "red car at intersection" can match directly against frame
embeddings, skipping the slow VLM caption generation step entirely.

Used as a complement to text embeddings — not a replacement.  Text
embeddings from VLM captions capture semantic detail ("the car is turning
left").  SigLIP embeddings capture visual similarity ("this frame looks
like a car scene").  Together they provide better retrieval than either alone.

Usage:
    embedder = SigLIPEmbedder()
    vec = embedder.embed_image("path/to/frame.jpg")    # image → 768-dim
    vec = embedder.embed_text("red car at intersection") # text → 768-dim
    vecs = embedder.embed_images(["a.jpg", "b.jpg"])    # batch
    embedder.unload()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from backend.config import SIGLIP_DIM, SIGLIP_MODEL

logger = logging.getLogger(__name__)


class SigLIPEmbedder:
    """Embed images and text into a shared vector space via SigLIP 2."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or SIGLIP_MODEL
        self._model = None
        self._processor = None

    @property
    def enabled(self) -> bool:
        """True if SigLIP is configured (model name not empty)."""
        return bool(self._model_name)

    @property
    def dim(self) -> int:
        return SIGLIP_DIM

    def embed_image(self, image_path: str) -> list[float] | None:
        """Embed a single image into the SigLIP vector space."""
        if not self.enabled:
            return None

        results = self.embed_images([image_path])
        return results[0] if results else None

    def embed_images(self, image_paths: list[str]) -> list[list[float]] | None:
        """Batch embed multiple images.

        Returns list of vectors (same length as input), or None on failure.
        """
        if not self.enabled or not image_paths:
            return None

        try:
            model, processor = self._get_model()
            import torch
            from PIL import Image

            images = []
            valid_indices = []
            for i, p in enumerate(image_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_indices.append(i)
                except Exception:
                    logger.debug("SigLIP: could not open image %s", p)

            if not images:
                return None

            inputs = processor(images=images, return_tensors="pt", padding=True)
            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            vectors = image_features.cpu().numpy().tolist()

            # Map back to original indices (fill None for failed images)
            result: list[list[float] | None] = [None] * len(image_paths)
            for idx, vec in zip(valid_indices, vectors):
                result[idx] = vec

            # Filter out Nones — return only successful embeddings
            return [v for v in result if v is not None]

        except ImportError:
            logger.warning(
                "SigLIP requires: pip install transformers torch pillow. "
                "Disabling visual embeddings."
            )
            self._model_name = ""
            return None
        except Exception as exc:
            logger.error("SigLIP embed_images failed: %s", exc)
            return None

    def embed_text(self, text: str) -> list[float] | None:
        """Embed text into the same SigLIP vector space as images.

        This enables text-to-image search: embed the query text,
        then find nearest image embeddings in ChromaDB.
        """
        results = self.embed_texts([text])
        return results[0] if results else None

    def embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Batch embed multiple text queries."""
        if not self.enabled or not texts:
            return None

        try:
            model, processor = self._get_model()
            import torch

            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            return text_features.cpu().numpy().tolist()

        except ImportError:
            logger.warning("SigLIP requires: pip install transformers torch pillow")
            self._model_name = ""
            return None
        except Exception as exc:
            logger.error("SigLIP embed_texts failed: %s", exc)
            return None

    def _get_model(self):
        """Lazy-load the SigLIP model and processor."""
        if self._model is not None:
            return self._model, self._processor

        logger.info("Loading SigLIP model: %s", self._model_name)
        t0 = time.monotonic()

        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()

        logger.info("SigLIP loaded in %.1fs", time.monotonic() - t0)
        return self._model, self._processor

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            logger.info("SigLIP model unloaded.")
