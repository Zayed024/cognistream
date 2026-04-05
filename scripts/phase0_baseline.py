#!/usr/bin/env python3
"""Phase 0 baseline runner for CogniStream.

This script repeatedly reprocesses one or more videos and captures:
- pipeline elapsed time and stage timings
- estimated VLM throughput (frames/min)
- retrieval diversity and score statistics
- process RSS start/peak/delta (from benchmark quality metrics)

Usage examples:
  python scripts/phase0_baseline.py --video-id <VIDEO_ID>
  python scripts/phase0_baseline.py --video-id <VIDEO_ID> --runs 3
  python scripts/phase0_baseline.py --discover-videos --max-videos 2 --runs 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _http_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _load_queries(args: argparse.Namespace) -> list[str]:
    if args.query:
        return args.query

    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    return [
        "person entering room",
        "vehicle movement",
        "crowd activity",
        "suspicious behavior",
        "object left behind",
    ]


def _safe_fmean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _safe_float(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return default


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except Exception:
        return None


def _trigger_processing(base_url: str, video_id: str, timeout_sec: int, poll_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    start_utc = datetime.now(timezone.utc)

    history_before = _http_json(f"{base_url}/video/{video_id}/benchmark/history?limit=1")
    baseline_count = int(history_before.get("count", 0))
    runs_before = history_before.get("runs") or []
    if runs_before:
        _parse_iso_datetime((runs_before[0] or {}).get("captured_at"))

    # Retry while queue is busy (429) instead of failing immediately.
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timeout waiting to start processing for video={video_id}")

        try:
            _http_json(
                f"{base_url}/process-video",
                method="POST",
                payload={"video_id": video_id},
            )
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (409, 429):
                print(f"  pipeline busy for {video_id} ({exc.code}), retrying...")
                time.sleep(poll_sec)
                continue
            raise RuntimeError(f"process-video failed: HTTP {exc.code} - {body}") from exc

    saw_processing = False
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timeout waiting for completion video={video_id}")

        meta = _http_json(f"{base_url}/video/{video_id}")
        status = (meta.get("status") or "").upper()
        if status == "PROCESSING":
            saw_processing = True
        if status == "FAILED":
            raise RuntimeError(f"Video processing failed for video={video_id}")

        history = _http_json(f"{base_url}/video/{video_id}/benchmark/history?limit=1")
        current_count = int(history.get("count", 0))
        latest_runs = history.get("runs") or []
        latest_run = latest_runs[0] if latest_runs else None
        latest_captured_at = _parse_iso_datetime((latest_run or {}).get("captured_at"))

        if current_count > baseline_count and latest_run:
            if latest_captured_at is None or latest_captured_at >= start_utc or saw_processing:
                return meta

        time.sleep(poll_sec)


def _run_retrieval_probe(base_url: str, video_id: str, queries: list[str], top_k: int) -> dict[str, Any]:
    all_sets: list[set[str]] = []
    all_scores: list[float] = []
    query_results: list[dict[str, Any]] = []

    for q in queries:
        data = _http_json(
            f"{base_url}/search",
            method="POST",
            payload={
                "query": q,
                "video_id": video_id,
                "top_k": top_k,
            },
        )

        results = data.get("results", [])
        ids = {
            (row.get("segment_id") or f"{row.get('video_id')}:{row.get('start_time')}:{row.get('end_time')}")
            for row in results
        }
        ids.discard(None)
        all_sets.append(ids)

        query_scores = [float(row.get("score", 0.0)) for row in results]
        all_scores.extend(query_scores)
        query_results.append(
            {
                "query": q,
                "hit_count": len(results),
                "unique_ids": len(ids),
                "scores": query_scores,
            }
        )

    pairwise = [_jaccard(a, b) for a, b in itertools.combinations(all_sets, 2)]
    avg_overlap = _safe_fmean(pairwise)

    return {
        "query_count": len(queries),
        "queries": query_results,
        "retrieval_summary": {
            "avg_pairwise_overlap_jaccard": round(avg_overlap, 4),
            "diversity_index": round(1.0 - avg_overlap, 4),
            "avg_result_score": round(_safe_fmean(all_scores), 4) if all_scores else None,
            "max_result_score": round(max(all_scores), 4) if all_scores else None,
        },
    }


def _estimate_vlm_fpm(benchmark: dict[str, Any]) -> float | None:
    stage = benchmark.get("stage_timings") or {}
    quality = benchmark.get("quality_metrics") or {}
    vlm_sec = _safe_float(stage, "vlm_sec", 0.0)
    kept = _safe_float(quality, "keyframes_kept", 0.0)
    if vlm_sec <= 0 or kept <= 0:
        return None
    return round((kept / vlm_sec) * 60.0, 3)


def _resolve_video_ids(args: argparse.Namespace) -> list[str]:
    if args.video_id:
        return args.video_id

    if not args.discover_videos:
        return []

    payload = _http_json(f"{args.base_url}/videos")
    videos = payload.get("videos") or []

    allowed = {"UPLOADED", "PROCESSED", "FAILED"}
    selected = [v.get("video_id") for v in videos if (v.get("status") or "").upper() in allowed]
    selected = [v for v in selected if v]

    return selected[: args.max_videos]


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Phase 0 Baseline Report")
    lines.append("")
    lines.append(f"Generated at: {summary['generated_at']}")
    lines.append(f"Base URL: {summary['base_url']}")
    lines.append(f"Runs per video: {summary['runs_per_video']}")
    lines.append("")

    for video in summary.get("videos", []):
        lines.append(f"## Video {video['video_id']}")
        lines.append("")
        agg = video.get("aggregate", {})
        lines.append(f"- runs_completed: {video.get('runs_completed', 0)}")
        lines.append(f"- elapsed_sec_mean: {agg.get('elapsed_sec_mean')}")
        lines.append(f"- elapsed_sec_stdev: {agg.get('elapsed_sec_stdev')}")
        lines.append(f"- throughput_vlm_frames_per_min_mean: {agg.get('vlm_frames_per_min_mean')}")
        lines.append(f"- throughput_vlm_frames_per_min_stdev: {agg.get('vlm_frames_per_min_stdev')}")
        lines.append(f"- process_rss_peak_mb_max: {agg.get('process_rss_peak_mb_max')}")
        lines.append(f"- retrieval_diversity_mean: {agg.get('retrieval_diversity_mean')}")
        lines.append(f"- retrieval_avg_score_mean: {agg.get('retrieval_avg_score_mean')}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0 baseline measurements")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--video-id", action="append", help="Video ID (repeatable)")
    parser.add_argument("--discover-videos", action="store_true", help="Auto-pick videos from /videos")
    parser.add_argument("--max-videos", type=int, default=2, help="Max discovered videos")
    parser.add_argument("--runs", type=int, default=3, help="Runs per video")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for retrieval probe")
    parser.add_argument("--query", action="append", help="Query text (repeatable)")
    parser.add_argument("--query-file", help="Path to newline-delimited queries")
    parser.add_argument("--poll-sec", type=float, default=2.0, help="Polling interval while processing")
    parser.add_argument("--timeout-sec", type=int, default=7200, help="Per-run timeout")
    parser.add_argument("--output-dir", default="reports/phase0", help="Directory for reports")
    args = parser.parse_args()

    queries = _load_queries(args)
    try:
        video_ids = _resolve_video_ids(args)
    except urllib.error.URLError as exc:
        print(f"Cannot reach backend at {args.base_url}: {exc}")
        print("Start the API server first, then rerun this command.")
        return 1

    if not video_ids:
        print("No videos selected. Use --video-id or --discover-videos.")
        return 2

    out_dir = Path(args.output_dir).resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    report_videos: list[dict[str, Any]] = []

    for video_id in video_ids:
        print(f"\n=== Baseline video={video_id} runs={args.runs} ===")
        per_run: list[dict[str, Any]] = []

        for idx in range(1, args.runs + 1):
            print(f"\nRun {idx}/{args.runs}")
            started = datetime.now(timezone.utc).isoformat()
            run_error: str | None = None

            try:
                _trigger_processing(args.base_url, video_id, timeout_sec=args.timeout_sec, poll_sec=args.poll_sec)
                benchmark = _http_json(f"{args.base_url}/video/{video_id}/benchmark")
                retrieval = _run_retrieval_probe(args.base_url, video_id, queries, args.top_k)
                vlm_fpm = _estimate_vlm_fpm(benchmark)
            except Exception as exc:
                benchmark = {}
                retrieval = {"query_count": len(queries), "queries": [], "retrieval_summary": {}}
                vlm_fpm = None
                run_error = str(exc)

            run_payload = {
                "run_index": idx,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "video_id": video_id,
                "benchmark": benchmark,
                "retrieval": retrieval,
                "vlm_frames_per_min": vlm_fpm,
                "error": run_error,
            }
            per_run.append(run_payload)

            run_path = raw_dir / f"{video_id}_run_{idx:03d}.json"
            run_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
            print(f"  saved {run_path}")
            if run_error:
                print(f"  error: {run_error}")
            else:
                print(f"  elapsed_sec={benchmark.get('elapsed_sec')} vlm_fpm={vlm_fpm}")

        successful = [r for r in per_run if not r.get("error")]

        elapsed_values = [
            _safe_float(r.get("benchmark", {}), "elapsed_sec", 0.0)
            for r in successful
        ]
        vlm_fpm_values = [
            float(r["vlm_frames_per_min"]) for r in successful if r.get("vlm_frames_per_min") is not None
        ]
        diversity_values = [
            _safe_float((r.get("retrieval", {}).get("retrieval_summary") or {}), "diversity_index", 0.0)
            for r in successful
        ]
        avg_score_values = []
        for r in successful:
            value = (r.get("retrieval", {}).get("retrieval_summary") or {}).get("avg_result_score")
            if value is not None:
                try:
                    avg_score_values.append(float(value))
                except Exception:
                    pass

        peak_mem_values = []
        for r in successful:
            q = (r.get("benchmark", {}).get("quality_metrics") or {})
            val = q.get("process_rss_peak_mb")
            if val is not None:
                try:
                    peak_mem_values.append(float(val))
                except Exception:
                    pass

        aggregate = {
            "elapsed_sec_mean": round(_safe_fmean(elapsed_values), 3) if elapsed_values else None,
            "elapsed_sec_stdev": round(_safe_stdev(elapsed_values), 3) if elapsed_values else None,
            "vlm_frames_per_min_mean": round(_safe_fmean(vlm_fpm_values), 3) if vlm_fpm_values else None,
            "vlm_frames_per_min_stdev": round(_safe_stdev(vlm_fpm_values), 3) if vlm_fpm_values else None,
            "process_rss_peak_mb_max": round(max(peak_mem_values), 2) if peak_mem_values else None,
            "retrieval_diversity_mean": round(_safe_fmean(diversity_values), 4) if diversity_values else None,
            "retrieval_avg_score_mean": round(_safe_fmean(avg_score_values), 4) if avg_score_values else None,
            "success_ratio": round(len(successful) / max(1, len(per_run)), 4),
        }

        report_videos.append(
            {
                "video_id": video_id,
                "runs_completed": len(successful),
                "runs_requested": args.runs,
                "aggregate": aggregate,
                "runs": per_run,
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "runs_per_video": args.runs,
        "query_count": len(queries),
        "queries": queries,
        "videos": report_videos,
    }

    out_json = out_dir / "phase0_summary.json"
    out_md = out_dir / "phase0_summary.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_summary(out_md, summary)

    print(f"\nSummary JSON: {out_json}")
    print(f"Summary MD:   {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
