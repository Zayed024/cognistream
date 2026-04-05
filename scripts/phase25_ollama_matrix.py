#!/usr/bin/env python3
"""Run Phase 2.5 Ollama/VLM matrix benchmarks.

Scenarios requested:
- vlm=4, ollama parallel=2
- vlm=2, ollama parallel=3
- vlm=4, ollama parallel=3
- vlm=2, ollama parallel=4
- two Ollama server endpoints (1 worker each), vlm=2

Failure policy:
- If a scenario fails, mark it failed and continue to next.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_ID = "ae3008ed225a4a63ba47e56008c6f779"
BASE_PORT_A = 11436
BASE_PORT_B = 11437
BACKEND_PORT = 8000


@dataclass
class Scenario:
    name: str
    vlm_workers: int
    ollama_parallel: int
    endpoints: int


DEFAULT_BASELINE_TIMEOUT_SEC = 7200
DEFAULT_PRUNE_SLOWER_THAN_PCT = 10.0


def _http_json(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url=url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _wait_http_ok(url: str, timeout_sec: int, step_sec: float = 1.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5).read(1)
            return True
        except Exception:
            time.sleep(step_sec)
    return False


def _kill_port(port: int) -> None:
    cmd = (
        "$conn = Get-NetTCPConnection -LocalPort "
        f"{port}"
        " -State Listen -ErrorAction SilentlyContinue; "
        "if ($conn) { "
        "  $pids = $conn | Select-Object -ExpandProperty OwningProcess -Unique; "
        "  foreach ($pid in $pids) { "
        "    try { Stop-Process -Id $pid -Force -ErrorAction Stop } catch {} "
        "  } "
        "}"
    )
    subprocess.run(["pwsh", "-NoProfile", "-Command", cmd], check=False)


def _start_ollama(port: int, parallel: int, log_file: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_NUM_PARALLEL"] = str(parallel)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def _start_backend(ollama_urls: list[str], vlm_workers: int, log_file: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["OLLAMA_BASE_URL"] = ",".join(ollama_urls)
    env["VLM_WORKERS"] = str(vlm_workers)
    env["MAX_KEYFRAMES"] = "140"
    env["KEYFRAME_NOVELTY_DIFF_THRESHOLD"] = "12.0"
    env["KEYFRAME_NOVELTY_MAX_SKIP"] = "10"
    env["KEYFRAME_NOVELTY_MIN_KEEP"] = "32"

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def _run_baseline(output_dir: Path, timeout_sec: int) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "phase0_baseline.py"),
        "--base-url",
        f"http://127.0.0.1:{BACKEND_PORT}",
        "--video-id",
        VIDEO_ID,
        "--runs",
        "1",
        "--timeout-sec",
        str(timeout_sec),
        "--output-dir",
        str(output_dir),
    ]
    done = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
    out = (done.stdout or "") + "\n" + (done.stderr or "")
    return done.returncode, out


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _snapshot_benchmark(output_dir: Path, scenario: Scenario) -> dict:
    bench = _http_json(f"http://127.0.0.1:{BACKEND_PORT}/video/{VIDEO_ID}/benchmark", timeout=30)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    bench_path = raw_dir / f"{VIDEO_ID}_benchmark.json"
    bench_path.write_text(json.dumps(bench, indent=2), encoding="utf-8")

    summary = {
        "scenario": scenario.name,
        "config": {
            "ollama_servers": (
                [f"http://127.0.0.1:{BASE_PORT_A}"]
                if scenario.endpoints == 1
                else [f"http://127.0.0.1:{BASE_PORT_A}", f"http://127.0.0.1:{BASE_PORT_B}"]
            ),
            "ollama_num_parallel": scenario.ollama_parallel,
            "backend_vlm_workers": scenario.vlm_workers,
        },
        "video_id": VIDEO_ID,
        "benchmark": bench,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _scenario_timeout_sec(best_elapsed_sec: float | None, prune_slow_pct: float) -> int:
    if best_elapsed_sec is None or best_elapsed_sec <= 0:
        return DEFAULT_BASELINE_TIMEOUT_SEC

    factor = 1.0 + max(0.0, prune_slow_pct) / 100.0
    timeout_sec = int(math.ceil(best_elapsed_sec * factor))
    return max(300, timeout_sec)


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_scenario(s: Scenario, prune_slow_pct: float, best_elapsed_sec: float | None) -> dict:
    scenario_dir = PROJECT_ROOT / "reports" / "phase25" / s.name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Clean ports from prior crashes/runs.
    _kill_port(BACKEND_PORT)
    _kill_port(BASE_PORT_A)
    _kill_port(BASE_PORT_B)

    ollama_a = None
    ollama_b = None
    backend = None

    started = datetime.now(timezone.utc).isoformat()

    try:
        ollama_a = _start_ollama(BASE_PORT_A, s.ollama_parallel if s.endpoints == 1 else 1, scenario_dir / "ollama_a.log")
        if not _wait_http_ok(f"http://127.0.0.1:{BASE_PORT_A}/api/tags", timeout_sec=60):
            return {"scenario": s.name, "status": "failed", "reason": "ollama_a_not_ready", "started_at": started}

        urls = [f"http://127.0.0.1:{BASE_PORT_A}"]
        if s.endpoints == 2:
            ollama_b = _start_ollama(BASE_PORT_B, 1, scenario_dir / "ollama_b.log")
            if not _wait_http_ok(f"http://127.0.0.1:{BASE_PORT_B}/api/tags", timeout_sec=60):
                return {"scenario": s.name, "status": "failed", "reason": "ollama_b_not_ready", "started_at": started}
            urls.append(f"http://127.0.0.1:{BASE_PORT_B}")

        backend = _start_backend(urls, s.vlm_workers, scenario_dir / "backend.log")
        if not _wait_http_ok(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout_sec=90):
            return {"scenario": s.name, "status": "failed", "reason": "backend_not_ready", "started_at": started}

        timeout_sec = _scenario_timeout_sec(best_elapsed_sec, prune_slow_pct)
        rc, out = _run_baseline(scenario_dir, timeout_sec)
        (scenario_dir / "benchmark_stdout.log").write_text(out, encoding="utf-8")
        if rc != 0:
            reason = f"baseline_exit_{rc}"
            if best_elapsed_sec is not None and best_elapsed_sec > 0:
                reason = f"pruned_slower_than_best_timeout_{timeout_sec}s"
            return {
                "scenario": s.name,
                "status": "failed",
                "reason": reason,
                "started_at": started,
            }

        phase0 = _read_json(scenario_dir / "phase0_summary.json")
        video = (phase0.get("videos") or [{}])[0]
        run = (video.get("runs") or [{}])[0]
        if video.get("runs_completed", 0) < 1 or run.get("error"):
            return {
                "scenario": s.name,
                "status": "failed",
                "reason": run.get("error") or "runs_not_completed",
                "started_at": started,
            }

        summary = _snapshot_benchmark(scenario_dir, s)
        return {
            "scenario": s.name,
            "status": "ok",
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": (summary.get("benchmark") or {}).get("elapsed_sec"),
        }

    finally:
        _stop_proc(backend)
        _stop_proc(ollama_b)
        _stop_proc(ollama_a)
        _kill_port(BACKEND_PORT)
        _kill_port(BASE_PORT_A)
        _kill_port(BASE_PORT_B)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2.5 Ollama/VLM matrix benchmarks")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run only the named scenario(s); repeatable. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--prune-slower-than-pct",
        type=float,
        default=DEFAULT_PRUNE_SLOWER_THAN_PCT,
        help="Prune scenarios that exceed the best elapsed time by this percentage.",
    )

    scenarios = [
        Scenario(name="single_np2_w2", vlm_workers=2, ollama_parallel=2, endpoints=1),
        Scenario(name="single_np2_w4", vlm_workers=4, ollama_parallel=2, endpoints=1),
        Scenario(name="single_np3_w2", vlm_workers=2, ollama_parallel=3, endpoints=1),
        Scenario(name="single_np3_w4", vlm_workers=4, ollama_parallel=3, endpoints=1),
        Scenario(name="single_np4_w1", vlm_workers=1, ollama_parallel=4, endpoints=1),
        Scenario(name="single_np4_w2", vlm_workers=2, ollama_parallel=4, endpoints=1),
        Scenario(name="single_np5_w1", vlm_workers=1, ollama_parallel=5, endpoints=1),
        Scenario(name="single_np5_w2", vlm_workers=2, ollama_parallel=5, endpoints=1),
        # Assumption for "2 ollama servers endpoints": one worker per server, backend workers=2.
        Scenario(name="two_servers_p1_w2", vlm_workers=2, ollama_parallel=1, endpoints=2),
    ]

    args = parser.parse_args()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.name in wanted]
        missing = sorted(wanted.difference({s.name for s in scenarios}))
        if missing:
            raise SystemExit(f"Unknown scenario(s): {', '.join(missing)}")
        if not scenarios:
            raise SystemExit("No scenarios selected")

    matrix_dir = PROJECT_ROOT / "reports" / "phase25"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_elapsed_sec: float | None = None
    prune_slow_pct = max(0.0, float(args.prune_slower_than_pct))
    for s in scenarios:
        print(f"\\n=== Running {s.name} ===")
        result = run_scenario(s, prune_slow_pct, best_elapsed_sec)
        print(json.dumps(result, indent=2))
        results.append(result)

        if result.get("status") == "ok":
            elapsed = result.get("elapsed_sec")
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                if best_elapsed_sec is None or elapsed < best_elapsed_sec:
                    best_elapsed_sec = float(elapsed)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video_id": VIDEO_ID,
        "prune_slower_than_pct": prune_slow_pct,
        "best_elapsed_sec": best_elapsed_sec,
        "results": results,
    }
    (matrix_dir / "matrix_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    failures = [r for r in results if r.get("status") != "ok"]
    print("\\nMatrix complete.")
    print(f"Scenarios: {len(results)}, failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
