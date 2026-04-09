"""
CogniStream — LLM-Powered Video Report Generation

Uses an LLM (NVIDIA cloud or local Ollama) to generate structured
summary reports from processed video segments.

Inspired by NVIDIA Metropolis VSS 3 automatic report generation.

Templates:
    - executive: high-level summary for non-technical audience
    - incident: security incident report with timeline
    - timeline: chronological list of key moments
    - activity: detected actions and entities

Each template produces a structured JSON report that can be exported
to HTML/PDF or used directly via the API.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)
from backend.providers.nvidia import nvidia

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


REPORT_TEMPLATES = {
    "executive": {
        "name": "Executive Summary",
        "description": "High-level overview for non-technical audiences",
        "system": (
            "You are a video analyst writing a brief executive summary. "
            "Be concise, factual, and avoid jargon. Output 3-5 short paragraphs."
        ),
        "instruction": (
            "Write an executive summary of this video covering: "
            "(1) what the video shows, (2) key entities and activities, "
            "(3) any notable events or anomalies, (4) overall context. "
            "Keep it under 300 words."
        ),
    },
    "incident": {
        "name": "Incident Report",
        "description": "Security incident report with timeline",
        "system": (
            "You are a security analyst writing an incident report. "
            "Be factual and chronological. Highlight any anomalies or suspicious activity."
        ),
        "instruction": (
            "Write a security incident report covering: "
            "(1) Timeline of events with timestamps, "
            "(2) People and objects observed, "
            "(3) Suspicious activities or anomalies, "
            "(4) Recommended actions. "
            "Use bullet points and timestamps."
        ),
    },
    "timeline": {
        "name": "Chronological Timeline",
        "description": "Step-by-step narrative of the video",
        "system": "You are a video analyst writing a chronological narrative.",
        "instruction": (
            "Write a chronological timeline of this video. "
            "For each significant moment, provide: timestamp, what happened, who/what was involved. "
            "Format as a numbered list."
        ),
    },
    "activity": {
        "name": "Activity Report",
        "description": "Detected actions, entities, and interactions",
        "system": "You are an activity recognition analyst.",
        "instruction": (
            "Analyze the activities in this video. List: "
            "(1) All detected actions/activities, "
            "(2) People and objects involved, "
            "(3) Temporal patterns or repeated activities, "
            "(4) Unique or unusual moments."
        ),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ReportGenerator:
    """Generate LLM-summarized reports from processed video segments."""

    def __init__(self, model: str | None = None):
        # Use NVIDIA cloud if available, else Ollama
        self._use_nvidia = nvidia.available
        self._model = model or ("meta/llama-3.2-11b-vision-instruct" if self._use_nvidia else OLLAMA_MODEL)

    def generate(
        self,
        video_meta: dict,
        segments: list[dict],
        events: list[dict],
        annotations: list[dict] | None = None,
        template: str = "executive",
        scenario: str | None = None,
        events_to_track: list[str] | None = None,
        objects_of_interest: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a report for a processed video.

        Args:
            video_meta: Video metadata (filename, duration, etc.)
            segments: List of stored segments from ChromaDB
            events: List of detected events from SQLite
            annotations: Optional list of user annotations
            template: One of REPORT_TEMPLATES keys
            scenario: Optional scenario description (e.g., "warehouse aisle monitoring")
                      Borrowed from VSS 3 — gives the LLM domain context.
            events_to_track: Optional list of specific events the LLM should look for
                             (e.g., ["forklift speeding", "unattended pallet"])
            objects_of_interest: Optional list of specific objects to highlight
                                 (e.g., ["forklift", "worker", "pallet"])

        Returns:
            Report dict with summary text, structured fields, and metadata.
        """
        if template not in REPORT_TEMPLATES:
            template = "executive"
        tmpl = REPORT_TEMPLATES[template]

        # Build a compact context from segments + events
        context = self._build_context(video_meta, segments, events, annotations or [])

        # 3-parameter contract from VSS 3 — augment the instruction with user-provided context
        focus_lines = []
        if scenario:
            focus_lines.append(f"SCENARIO: {scenario}")
        if events_to_track:
            focus_lines.append(f"EVENTS TO TRACK: {', '.join(events_to_track)}")
        if objects_of_interest:
            focus_lines.append(f"OBJECTS OF INTEREST: {', '.join(objects_of_interest)}")

        focus_block = "\n".join(focus_lines)
        instruction = tmpl["instruction"]
        if focus_block:
            instruction = (
                f"{focus_block}\n\n"
                f"Pay particular attention to the scenario, events, and objects above. "
                f"{instruction}"
            )

        prompt = (
            f"{instruction}\n\n"
            f"VIDEO DATA:\n{context}\n\n"
            f"REPORT:"
        )

        logger.info("Generating %s report for %s using %s",
                    template, video_meta.get("filename"),
                    "NVIDIA cloud" if self._use_nvidia else "local Ollama")

        summary_text = self._call_llm(prompt, tmpl["system"])

        return {
            "video_id": video_meta.get("video_id"),
            "filename": video_meta.get("filename"),
            "duration_sec": video_meta.get("duration_sec"),
            "template": template,
            "template_name": tmpl["name"],
            "scenario": scenario,
            "events_to_track": events_to_track or [],
            "objects_of_interest": objects_of_interest or [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": self._model,
            "summary": summary_text,
            "stats": {
                "segments_analyzed": len(segments),
                "events_detected": len(events),
                "annotations": len(annotations or []),
            },
            "key_moments": self._extract_key_moments(segments, events),
        }

    def _build_context(
        self,
        video_meta: dict,
        segments: list[dict],
        events: list[dict],
        annotations: list[dict],
    ) -> str:
        """Build a compact text representation of the video for the LLM."""
        lines = []

        # Header
        duration = video_meta.get("duration_sec", 0)
        lines.append(f"Filename: {video_meta.get('filename', 'unknown')}")
        lines.append(f"Duration: {duration:.0f} seconds ({duration/60:.1f} min)")
        lines.append("")

        # Segments — sample evenly across the video
        # Cap at ~50 segments to fit in context window
        if segments:
            lines.append("CAPTIONS (sorted by time):")
            sorted_segs = sorted(segments, key=lambda s: s.get("start_time", 0))
            sample_size = min(50, len(sorted_segs))
            step = max(1, len(sorted_segs) // sample_size)
            for s in sorted_segs[::step][:sample_size]:
                t = s.get("start_time", 0)
                text = (s.get("text") or "").strip()[:200]
                if text:
                    lines.append(f"  [{t:.1f}s] {text}")
            lines.append("")

        # Events
        if events:
            lines.append(f"DETECTED EVENTS ({len(events)}):")
            for e in events[:20]:
                t = e.get("start_time", 0)
                lines.append(
                    f"  [{t:.1f}s] {e.get('event_type', 'event')}: "
                    f"{e.get('description', '')[:150]}"
                )
            lines.append("")

        # Annotations
        if annotations:
            lines.append(f"USER ANNOTATIONS ({len(annotations)}):")
            for a in annotations[:10]:
                lines.append(f"  [{a.get('start_time', 0):.1f}s] {a.get('label', '')}")

        return "\n".join(lines)

    def _extract_key_moments(self, segments: list[dict], events: list[dict]) -> list[dict]:
        """Extract a list of key moments for the report's timeline section."""
        moments = []

        # All events are key moments
        for e in events[:20]:
            moments.append({
                "time_sec": e.get("start_time", 0),
                "type": "event",
                "label": e.get("event_type", "event"),
                "description": (e.get("description", "") or "")[:200],
            })

        # Segments with anomalies are key moments
        for s in segments:
            text = (s.get("text") or "").lower()
            if "anomaly:" in text and "none" not in text.split("anomaly:")[-1][:50].lower():
                moments.append({
                    "time_sec": s.get("start_time", 0),
                    "type": "anomaly",
                    "label": "anomaly",
                    "description": (s.get("text") or "")[:200],
                })

        moments.sort(key=lambda m: m["time_sec"])
        return moments[:30]

    def _call_llm(self, prompt: str, system: str) -> str:
        """Call NVIDIA cloud or local Ollama for text generation."""
        if self._use_nvidia:
            return self._call_nvidia(prompt, system)
        return self._call_ollama(prompt, system)

    def _call_nvidia(self, prompt: str, system: str) -> str:
        """Call NVIDIA Llama via the OpenAI-compatible chat API."""
        try:
            with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                resp = client.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {NVIDIA_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 800,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("NVIDIA report generation failed: %s", exc)
            return f"Report generation failed: {exc}"

    def _call_ollama(self, prompt: str, system: str) -> str:
        """Call local Ollama for text generation (text-only model)."""
        try:
            full_prompt = f"{system}\n\n{prompt}"
            with httpx.Client(timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0)) as client:
                resp = client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 800,
                            "num_ctx": 4096,
                        },
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.error("Ollama report generation failed: %s", exc)
            return f"Report generation failed: {exc}"


# Module-level singleton
report_generator = ReportGenerator()
