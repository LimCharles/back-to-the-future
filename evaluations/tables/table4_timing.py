#!/usr/bin/env python
"""
Table 4 — Timing measurements.

Measures:
  1. Classifier training time (wall-clock for ``src/fit.py``).
  2. Per-token inference ratio: TRACE vs baseline generation time.

Usage::

    python -m evaluations.tables.table4_timing --hmm_variant hmm1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _time_classifier(data_path: str, output_path: str) -> float:
    """Time ``src/fit.py`` and return wall-clock seconds."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "fit.py"),
        "--data_path", data_path,
        "--output_path", output_path,
    ]
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), stderr=subprocess.PIPE)
    return time.perf_counter() - start


def _time_generation(
    hmm_variant: str,
    a: float,
    prompts_path: str,
    weights_path: str,
    device: str,
    max_len: int = 20,
    num_generations: int = 5,
    use_hmm: bool = True,
) -> float:
    """Time ``src/generate.py`` and return wall-clock seconds."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "generate.py"),
        "--hmm_variant", hmm_variant,
        "--a", str(a if use_hmm else 0.0),
        "--prompts_path", prompts_path,
        "--weights_path", weights_path,
        "--max_len", str(max_len),
        "--num_generations", str(num_generations),
        "--device", device,
    ]
    if not use_hmm:
        # a=0 still loads HMM; for true baseline, use --baseline without HMM
        cmd.extend(["--baseline"])

    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), stderr=subprocess.PIPE)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Table 4: timing measurements.")
    parser.add_argument("--hmm_variant", type=str, default="hmm1")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument("--train_data_path", type=str, default="data/RTP_train.jsonl")
    parser.add_argument("--weights_path", type=str, default="data/coefficients.csv")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of timing runs to average")
    parser.add_argument("--num_prompts", type=int, default=100,
                        help="Prompts for inference timing (uses first N from file)")
    parser.add_argument("--output_dir", type=str, default="evaluations/results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Classifier training time
    # ------------------------------------------------------------------
    print("Measuring classifier training time…")
    train_times = []
    tmp_coeff = str(PROJECT_ROOT / "data" / "_timing_tmp_coefficients.csv")
    for run in range(args.num_runs):
        t = _time_classifier(args.train_data_path, tmp_coeff)
        train_times.append(t)
        print(f"  Run {run + 1}/{args.num_runs}: {t:.2f}s")
    # Clean up temp file
    if os.path.exists(tmp_coeff):
        os.remove(tmp_coeff)

    avg_train = sum(train_times) / len(train_times)
    print(f"  Average: {avg_train:.2f}s")

    # ------------------------------------------------------------------
    # 2. Per-token inference ratio
    # ------------------------------------------------------------------
    print("\nMeasuring inference time…")
    trace_times, baseline_times = [], []

    for run in range(args.num_runs):
        t_trace = _time_generation(
            args.hmm_variant, 1.0, args.prompts_path,
            args.weights_path, args.device, use_hmm=True,
        )
        t_baseline = _time_generation(
            args.hmm_variant, 0.0, args.prompts_path,
            args.weights_path, args.device, use_hmm=False,
        )
        trace_times.append(t_trace)
        baseline_times.append(t_baseline)
        print(f"  Run {run + 1}: TRACE={t_trace:.2f}s  baseline={t_baseline:.2f}s")

    avg_trace = sum(trace_times) / len(trace_times)
    avg_baseline = sum(baseline_times) / len(baseline_times)
    ratio = avg_trace / avg_baseline if avg_baseline > 0 else float("inf")

    print(f"  Avg TRACE: {avg_trace:.2f}s, Avg baseline: {avg_baseline:.2f}s")
    print(f"  Ratio: {ratio:.2f}x")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    result = {
        "variant": args.hmm_variant,
        "classifier_training_time_s": avg_train,
        "classifier_training_runs": train_times,
        "trace_inference_time_s": avg_trace,
        "baseline_inference_time_s": avg_baseline,
        "per_token_ratio": ratio,
        "trace_runs": trace_times,
        "baseline_runs": baseline_times,
        "num_runs": args.num_runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    json_path = os.path.join(args.output_dir, "table4_timing.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    csv_path = os.path.join(args.output_dir, "table4_timing.csv")
    pd.DataFrame([{
        "variant": result["variant"],
        "classifier_training_time_s": result["classifier_training_time_s"],
        "per_token_ratio": result["per_token_ratio"],
        "trace_inference_time_s": result["trace_inference_time_s"],
        "baseline_inference_time_s": result["baseline_inference_time_s"],
    }]).to_csv(csv_path, index=False)

    print(f"\n✓ Table 4 → {json_path}")
    print(f"           → {csv_path}")


if __name__ == "__main__":
    main()
