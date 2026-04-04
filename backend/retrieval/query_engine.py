"""
CogniStream — Query Engine

End-to-end semantic retrieval pipeline.  Takes a natural-language query
and returns ranked video segments with timestamps.

Pipeline stages:
    1. **Embed**    — encode query with the same SentenceTransformer used
                      at index time.
    2. **Search**   — vector similarity search in ChromaDB with optional
                      video_id / source_type filters.
    3. **Re-rank**  — temporal proximity boost: segments near other
                      high-scoring segments are promoted.
    4. **Format**   — map raw results into SearchResult dataclasses.

Temporal re-ranking rationale:
    If a user asks "when did the red car arrive", three segments at
    t=33s, 35s, 37s all matching is stronger evidence than a lone match
    at t=200s.  A Gaussian decay kernel lets nearby segments contribute
    mutual score boosts, controlled by a configurable time window.

Usage:
    engine = QueryEngine(embedder, store)
    results = engine.search("when did the professor explain deadlocks")
    for r in results:
        print(f"[{r.start_time:.1f}s] {r.text} (score={r.score:.3f})")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from backend.config import DEFAULT_TOP_K
from backend.db.chroma_store import ChromaStore
from backend.db.models import SearchResult
from backend.fusion.multimodal_embedder import MultimodalEmbedder

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Re-ranking config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Temporal window (seconds): segments within this range of each other
# can exchange score boosts.
_TEMPORAL_WINDOW = 15.0

# Weight of the temporal boost relative to the base similarity score.
# 0.0 = no temporal effect, 1.0 = temporal signal equals similarity signal.
_TEMPORAL_WEIGHT = 0.25

# Gaussian sigma for temporal decay (in seconds).  Controls how quickly
# the boost falls off with distance.  sigma = window/3 gives near-zero
# contribution at the window edge.
_TEMPORAL_SIGMA = _TEMPORAL_WINDOW / 3.0

# Over-fetch factor: retrieve more candidates from ChromaDB than top_k
# so the re-ranker has room to promote temporally clustered results.
_OVERFETCH_FACTOR = 2


class QueryEngine:
    """Semantic search with temporal re-ranking."""

    def __init__(
        self,
        embedder: MultimodalEmbedder | None = None,
        store: ChromaStore | None = None,
        temporal_window: float = _TEMPORAL_WINDOW,
        temporal_weight: float = _TEMPORAL_WEIGHT,
    ):
        self.embedder = embedder or MultimodalEmbedder()
        self.store = store or ChromaStore()
        self.temporal_window = temporal_window
        self.temporal_weight = temporal_weight

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        video_id: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """Execute the full search pipeline.

        Args:
            query:         Natural-language search query.
            top_k:         Number of final results to return.
            video_id:      Scope search to one video (optional).
            source_filter: Restrict to a source type: "visual", "audio",
                           "fused", or "event" (optional).

        Returns:
            Up to *top_k* :class:`SearchResult` objects sorted by
            final score (descending).
        """
        if not query.strip():
            logger.warning("Empty query — returning no results.")
            return []

        # Stage 1: embed the query
        logger.info("Search query: '%s'", query)
        query_embedding = self.embedder.embed_query(query)

        # Stage 2: vector search (over-fetch for re-ranking headroom)
        fetch_k = min(top_k * _OVERFETCH_FACTOR, top_k + 20)
        raw_results = self.store.query(
            embedding=query_embedding,
            top_k=fetch_k,
            video_id=video_id,
            source_filter=source_filter,
        )

        if not raw_results:
            logger.info("No results found for query: '%s'", query)
            return []

        logger.debug("ChromaDB returned %d candidates.", len(raw_results))

        # Stage 3: temporal re-ranking
        reranked = self._temporal_rerank(raw_results)

        # Stage 4: format and trim to top_k
        results = self._format(reranked[:top_k])

        logger.info(
            "Search complete: %d results (top score=%.3f)",
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 3: Temporal re-ranking
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _temporal_rerank(self, candidates: list[dict]) -> list[dict]:
        """Boost scores when multiple candidates cluster temporally.

        For each candidate, sum the Gaussian-weighted scores of its
        temporal neighbours within ``temporal_window`` seconds.  The
        resulting boost is blended with the original similarity score:

            final = (1 - w) * similarity + w * normalised_boost

        Candidates are then re-sorted by final score.
        """
        n = len(candidates)
        if n <= 1 or self.temporal_weight <= 0:
            return candidates

        # Group by video_id — temporal proximity only makes sense within
        # the same video.
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(candidates):
            vid = c.get("video_id", "")
            groups.setdefault(vid, []).append(i)

        # Compute temporal boost for each candidate
        boosts = [0.0] * n
        sigma = self.temporal_window / 3.0

        for indices in groups.values():
            for i in indices:
                c_i = candidates[i]
                mid_i = (c_i["start_time"] + c_i["end_time"]) / 2.0

                for j in indices:
                    if i == j:
                        continue
                    c_j = candidates[j]
                    mid_j = (c_j["start_time"] + c_j["end_time"]) / 2.0

                    dt = abs(mid_i - mid_j)
                    if dt > self.temporal_window:
                        continue

                    # Gaussian decay weighted by the neighbour's similarity
                    weight = math.exp(-0.5 * (dt / sigma) ** 2)
                    boosts[i] += weight * c_j["score"]

        # Normalise boosts to [0, 1]
        max_boost = max(boosts) if boosts else 1.0
        if max_boost > 0:
            boosts = [b / max_boost for b in boosts]

        # Blend original score with temporal boost
        w = self.temporal_weight
        for i, c in enumerate(candidates):
            original = c["score"]
            c["_original_score"] = original
            c["_temporal_boost"] = round(boosts[i], 4)
            c["score"] = round((1 - w) * original + w * boosts[i], 4)

        # Re-sort descending by final score
        candidates.sort(key=lambda c: c["score"], reverse=True)

        logger.debug(
            "Temporal re-ranking applied (window=%.0fs, weight=%.2f)",
            self.temporal_window,
            self.temporal_weight,
        )
        return candidates

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 4: Format results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _format(candidates: list[dict]) -> list[SearchResult]:
        """Convert raw dicts into typed SearchResult dataclasses."""
        results: list[SearchResult] = []

        for c in candidates:
            frame_path = c.get("frame_path")
            video_id = c.get("video_id", "")

            # Build a URL-safe frame reference if a path exists
            frame_url = None
            if frame_path and video_id:
                # Extract just the filename from the stored path
                import os
                fname = os.path.basename(frame_path)
                frame_url = f"/video/{video_id}/frame/{fname}"

            results.append(
                SearchResult(
                    video_id=video_id,
                    segment_id=c.get("id", ""),
                    start_time=c.get("start_time", 0.0),
                    end_time=c.get("end_time", 0.0),
                    text=c.get("text", ""),
                    source_type=c.get("source_type", ""),
                    score=c.get("score", 0.0),
                    event_type=c.get("event_type"),
                    frame_url=frame_url,
                )
            )

        return results
