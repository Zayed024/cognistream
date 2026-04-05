"""
CogniStream — VLM Runner

I/O layer for the Visual Narrative Engine.  Communicates with a local
Ollama instance to run Vision Language Model inference on keyframes.

Architecture
────────────
OllamaClient   Low-level HTTP client.  One method: send an image + prompt,
                get back text.  Handles base64 encoding, timeouts, retries.

VLMRunner       High-level orchestrator.  Iterates over keyframes, runs four
                analysis passes per frame (scene / objects / activity /
                anomaly), delegates parsing to CaptionProcessor, and
                returns a list of VisualCaption objects.

Why four passes instead of one?
    Small VLMs (Moondream2, LLaVA-7B) produce significantly better output
    when given a single focused instruction per call.  A combined mega-prompt
    leads to missed fields and hallucinated structure.  Each pass is also
    independently retryable.

Usage:
    client = OllamaClient()
    runner = VLMRunner(client)
    captions = runner.analyse_keyframes(keyframes)
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import httpx

from backend.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    PIPELINE_MODE,
    VLM_WORKERS,
    _SMALL_VLMS,
)
from backend.db.models import Keyframe, VisualCaption
from backend.visual.caption_processor import CaptionProcessor, PromptLibrary

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Low-level Ollama HTTP client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OllamaError(Exception):
    """Raised when communication with Ollama fails."""


class OllamaClient:
    """Thin HTTP wrapper around Ollama's ``/api/generate`` endpoint.

    Designed for **synchronous, sequential** inference — appropriate for
    edge hardware where the VLM can only serve one request at a time.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_MODEL
        self.timeout = timeout or OLLAMA_TIMEOUT
        self._generate_url = f"{self.base_url}/api/generate"

        # Persistent HTTP client — reuses TCP connections across all
        # generate() calls instead of opening 400 connections per video.
        self._http = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

    def close(self) -> None:
        """Close the persistent HTTP client."""
        self._http.close()

    def unload(self) -> None:
        """Unload the model from GPU VRAM to free memory for other models."""
        try:
            self._http.post(
                self._generate_url,
                json={"model": self.model, "keep_alive": 0},
            )
            logger.info("Ollama model '%s' unloaded from VRAM.", self.model)
        except Exception as exc:
            logger.debug("Ollama unload failed: %s", exc)

    # ── health ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model is loaded."""
        try:
            resp = self._http.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama tags may include `:latest` suffix
            return any(
                m == self.model or m.startswith(f"{self.model}:")
                for m in models
            )
        except (httpx.HTTPError, KeyError):
            return False

    # ── inference ───────────────────────────────────────────────

    def generate(self, prompt: str, image_path: str) -> str:
        """Send a single image + prompt to the VLM and return the response text.

        Args:
            prompt:     The text instruction for the model.
            image_path: Path to a JPEG/PNG keyframe on disk.

        Returns:
            The model's generated text (stripped of leading/trailing whitespace).

        Raises:
            OllamaError: On HTTP failures, timeouts, or missing image.
        """
        image_b64 = self._encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.2,     # low temp for factual descriptions
                "num_predict": 300,     # cap output length
            },
        }

        try:
            resp = self._http.post(
                self._generate_url,
                json=payload,
            )
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Ollama timed out after {self.timeout}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama HTTP {exc.response.status_code}: "
                f"{exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama connection error: {exc}") from exc

        body = resp.json()
        return body.get("response", "").strip()

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read an image file and return its base64 encoding."""
        path = Path(image_path)
        if not path.is_file():
            raise OllamaError(f"Image file not found: {image_path}")
        return base64.b64encode(path.read_bytes()).decode("ascii")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# High-level orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VLMRunner:
    """Run multi-pass VLM analysis on a batch of keyframes.

    For each keyframe, four inference passes are executed:

        1. **Scene**   — free-text description of the visual scene.
        2. **Objects** — list of prominent objects/entities.
        3. **Activity** — what action or event is taking place.
        4. **Anomaly** — anything unusual, suspicious, or out of place.

    Results are parsed by :class:`CaptionProcessor` and returned as
    :class:`VisualCaption` dataclass instances.
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        processor: CaptionProcessor | None = None,
        max_retries: int = 2,
        fast_mode: bool | None = None,
    ):
        self.client = client or OllamaClient()
        self.processor = processor or CaptionProcessor()
        self.max_retries = max_retries
        # Auto-detect mode based on model name
        if fast_mode is not None:
            self.fast_mode = fast_mode
        elif PIPELINE_MODE == "auto":
            model_name = (self.client.model or "").lower().split(":")[0]
            self.fast_mode = model_name not in _SMALL_VLMS
        else:
            self.fast_mode = PIPELINE_MODE == "fast"

    def analyse_keyframes(
        self,
        keyframes: list[Keyframe],
    ) -> list[VisualCaption]:
        """Analyse every keyframe and return structured captions.

        Args:
            keyframes: Output from the frame sampler.

        Returns:
            One :class:`VisualCaption` per keyframe, in the same order.
        """
        total = len(keyframes)
        if total == 0:
            logger.warning("No keyframes to analyse.")
            return []

        logger.info(
            "Starting VLM analysis: %d keyframes, model=%s",
            total,
            self.client.model,
        )

        if not self.client.is_available():
            raise OllamaError(
                f"Ollama is not reachable at {self.client.base_url} "
                f"or model '{self.client.model}' is not pulled."
            )

        t_start = time.monotonic()

        # Choose analysis function: NVIDIA cloud > local fast > local quality
        from backend.providers.nvidia import nvidia
        if nvidia.available:
            analyse_fn = self._analyse_single_nvidia
            mode_label = "nvidia (cloud)"
        elif self.fast_mode:
            analyse_fn = self._analyse_single_fast
            mode_label = "fast (single-pass)"
        else:
            analyse_fn = self._analyse_single
            mode_label = "quality (4-pass)"

        # Determine worker count:
        #   0 (auto) = 1 for local Ollama, 4 for NVIDIA cloud
        #   1 = sequential (original behavior, best for local Ollama)
        #   2+ = concurrent (best for NVIDIA cloud or fast local GPU)
        workers = VLM_WORKERS
        if workers <= 0:
            workers = 4 if nvidia.available else 1

        logger.info("VLM mode: %s, workers: %d", mode_label, workers)

        if workers == 1:
            # Sequential — no thread overhead, best for single-model Ollama
            captions = self._run_sequential(keyframes, analyse_fn, t_start, total)
        else:
            # Concurrent — multiple frames at once via thread pool
            captions = self._run_concurrent(keyframes, analyse_fn, workers, t_start, total)

        logger.info(
            "VLM analysis complete: %d captions in %.1fs",
            len(captions),
            time.monotonic() - t_start,
        )
        return captions

    def _run_sequential(
        self,
        keyframes: list[Keyframe],
        analyse_fn,
        t_start: float,
        total: int,
    ) -> list[VisualCaption]:
        """Process frames one at a time (original behavior)."""
        captions: list[VisualCaption] = []
        for idx, kf in enumerate(keyframes, 1):
            caption = analyse_fn(kf)
            captions.append(caption)
            if idx % 10 == 0 or idx == total:
                elapsed = time.monotonic() - t_start
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info(
                    "VLM progress: %d/%d keyframes (%.1f frames/min)",
                    idx, total, rate * 60,
                )
        return captions

    def _run_concurrent(
        self,
        keyframes: list[Keyframe],
        analyse_fn,
        workers: int,
        t_start: float,
        total: int,
    ) -> list[VisualCaption]:
        """Process multiple frames concurrently via thread pool.

        Results are returned in the same order as the input keyframes.
        Failed frames get a fallback caption instead of crashing the batch.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Map future → original index to preserve ordering
        captions: list[VisualCaption | None] = [None] * total
        completed = 0

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm") as pool:
            future_to_idx = {
                pool.submit(analyse_fn, kf): i
                for i, kf in enumerate(keyframes)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    captions[idx] = future.result()
                except Exception as exc:
                    logger.error(
                        "VLM worker failed on frame %d: %s",
                        keyframes[idx].frame_number, exc,
                    )
                    # Fallback: empty caption so the pipeline continues
                    captions[idx] = VisualCaption(
                        keyframe=keyframes[idx],
                        scene_description="Analysis failed.",
                        objects=[],
                        activity="unknown",
                        anomaly=None,
                    )

                completed += 1
                if completed % 10 == 0 or completed == total:
                    elapsed = time.monotonic() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        "VLM progress: %d/%d keyframes (%.1f frames/min, %d workers)",
                        completed, total, rate * 60, workers,
                    )

        return [c for c in captions if c is not None]

    # ── single-frame analysis ───────────────────────────────────

    def _analyse_single_nvidia(self, keyframe: Keyframe) -> VisualCaption:
        """Run analysis using NVIDIA cloud VLM (highest quality, requires API key)."""
        from backend.providers.nvidia import nvidia

        combined_raw = nvidia.caption_image(
            keyframe.file_path,
            PromptLibrary.combined_prompt(),
        ) or ""

        caption = self.processor.build_caption_from_combined(
            keyframe=keyframe,
            combined_raw=combined_raw,
        )

        logger.debug(
            "Frame %d (nvidia) → scene=%d chars, objects=%d",
            keyframe.frame_number,
            len(caption.scene_description),
            len(caption.objects),
        )
        return caption

    def _analyse_single_fast(self, keyframe: Keyframe) -> VisualCaption:
        """Run a single combined pass on one keyframe (fast mode).

        One VLM call instead of four — ~75% faster per frame.
        """
        image_path = keyframe.file_path

        combined_raw = self._call_with_retry(
            PromptLibrary.combined_prompt(), image_path, "combined"
        )

        caption = self.processor.build_caption_from_combined(
            keyframe=keyframe,
            combined_raw=combined_raw,
        )

        logger.debug(
            "Frame %d (fast) → scene=%d chars, objects=%d",
            keyframe.frame_number,
            len(caption.scene_description),
            len(caption.objects),
        )
        return caption

    def _analyse_single(self, keyframe: Keyframe) -> VisualCaption:
        """Run all four analysis passes on one keyframe."""
        image_path = keyframe.file_path

        scene_raw = self._call_with_retry(
            PromptLibrary.scene_prompt(), image_path, "scene"
        )
        objects_raw = self._call_with_retry(
            PromptLibrary.objects_prompt(), image_path, "objects"
        )
        activity_raw = self._call_with_retry(
            PromptLibrary.activity_prompt(), image_path, "activity"
        )
        anomaly_raw = self._call_with_retry(
            PromptLibrary.anomaly_prompt(), image_path, "anomaly"
        )

        caption = self.processor.build_caption(
            keyframe=keyframe,
            scene_raw=scene_raw,
            objects_raw=objects_raw,
            activity_raw=activity_raw,
            anomaly_raw=anomaly_raw,
        )

        logger.debug(
            "Frame %d → scene=%d chars, objects=%d, activity='%s', anomaly=%s",
            keyframe.frame_number,
            len(caption.scene_description),
            len(caption.objects),
            caption.activity[:40] if caption.activity else "",
            "yes" if caption.anomaly else "no",
        )
        return caption

    def _call_with_retry(
        self, prompt: str, image_path: str, pass_name: str
    ) -> str:
        """Call the VLM with retries on transient failures.

        Returns the raw response text, or an empty string if all retries
        are exhausted (so one failed pass doesn't block the whole frame).
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.generate(prompt, image_path)
            except OllamaError as exc:
                logger.warning(
                    "VLM %s pass failed (attempt %d/%d): %s",
                    pass_name,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(1)

        logger.error(
            "VLM %s pass exhausted retries for %s — returning empty.",
            pass_name,
            image_path,
        )
        return ""
