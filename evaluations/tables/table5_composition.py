#!/usr/bin/env python
"""
Table 5 — Attribute composition (nontoxicity + nonpoliticalness).

Composes two attribute classifiers by element-wise addition of their
coefficient vectors.  Addition in log space = multiplication in
probability space, matching the paper's composition rule w' = w¹ · w².

Four rows: base LM, detox only, political only, detox+political.

Usage::

    python -m evaluations.tables.table5_composition \\
        --hmm_variant hmm1 --prompts_path data/rtp_10k.jsonl
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.generation_runner import GenerationRunner
from evaluations.metrics import aggregate_detox_run, parse_scored_csv


def _load_coefficients(csv_path: str) -> np.ndarray:
    """Load coefficient CSV into a numpy array."""
    df = pd.read_csv(csv_path)
    return df["Coefficient"].values.astype(float)


def _save_coefficients(coefficients: np.ndarray, output_path: str):
    """Write coefficient array to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Token ID", "Coefficient"])
        for tid, coeff in enumerate(coefficients):
            writer.writerow([tid, coeff])


def _score_political(scored_csv: str, device: str = "cuda") -> float:
    """Score generations for political content using DeBERTa zero-shot.

    Returns mean political score across all prompts.
    """
    from transformers import pipeline as hf_pipeline

    classifier = hf_pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device=0 if "cuda" in device else -1,
    )

    df = parse_scored_csv(scored_csv)
    gen_cols = [c for c in df.columns if c.startswith("trace_gen_")]

    political_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Political scoring", file=sys.stderr):
        for col in gen_cols:
            cell = row[col]
            if cell is None:
                continue
            text = cell.get("continuation", "") if isinstance(cell, dict) else ""
            if not text.strip():
                continue
            result = classifier(text, candidate_labels=["politics", "not politics"])
            idx = result["labels"].index("politics")
            political_scores.append(result["scores"][idx])

    return float(np.mean(political_scores)) if political_scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Table 5: attribute composition.")
    parser.add_argument("--hmm_variant", type=str, default="hmm1")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument(
        "--nontoxicity_weights", type=str,
        default="classifiers/coefficients_nontoxicity.csv",
    )
    parser.add_argument(
        "--nonpolitical_weights", type=str,
        default="classifiers/coefficients_nonpoliticalness.csv",
    )
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--num_generations", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="results/tables")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve paths
    for attr in ("nontoxicity_weights", "nonpolitical_weights"):
        p = getattr(args, attr)
        if not os.path.isabs(p):
            setattr(args, attr, str(PROJECT_ROOT / p))

    # Create composed weights:
    # Addition in log space = multiplication in probability space,
    # matching the paper's composition rule w' = w¹ · w².
    detox_coeffs = _load_coefficients(args.nontoxicity_weights)
    political_coeffs = _load_coefficients(args.nonpolitical_weights)
    composed_coeffs = detox_coeffs + political_coeffs

    composed_path = str(PROJECT_ROOT / "data" / "coefficients_composed.csv")
    _save_coefficients(composed_coeffs, composed_path)

    # Zero weights for baseline
    zero_coeffs = np.zeros_like(detox_coeffs)
    zero_path = str(PROJECT_ROOT / "data" / "coefficients_zero.csv")
    _save_coefficients(zero_coeffs, zero_path)

    runner = GenerationRunner(device=args.device)

    rows_config = [
        {"label": "base_lm", "weights_path": zero_path},
        {"label": "detox_only", "weights_path": args.nontoxicity_weights},
        {"label": "political_only", "weights_path": args.nonpolitical_weights},
        {"label": "detox_and_political", "weights_path": composed_path},
    ]

    all_results = []
    checkpoint_path = os.path.join(args.output_dir, "table5_checkpoint.json")
    done_labels = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for entry in json.load(f):
                done_labels.add(entry["label"])
                all_results.append(entry)

    for cfg in rows_config:
        if cfg["label"] in done_labels:
            print(f"Skipping {cfg['label']} (already done)")
            continue

        print(f"\n--- {cfg['label']} ---")
        scored_csv = runner.generate_and_score(
            hmm_variant=args.hmm_variant,
            a=args.a,
            prompts_path=args.prompts_path,
            weights_path=cfg["weights_path"],
            num_generations=args.num_generations,
            seed=args.seed,
        )
        df = parse_scored_csv(scored_csv)
        metrics = aggregate_detox_run(df, mode="trace")

        # Political scoring
        mean_political = _score_political(scored_csv, args.device)
        metrics["mean_political_score"] = mean_political

        entry = {
            "label": cfg["label"],
            "variant": args.hmm_variant,
            "a": args.a,
            "metrics": metrics,
            "weights_path": cfg["weights_path"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        all_results.append(entry)

        # Checkpoint
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(all_results, f, indent=2)
        os.replace(tmp, checkpoint_path)

    # ------------------------------------------------------------------
    # Write final outputs
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "table5_composition.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for entry in all_results:
        row = {"label": entry["label"], "variant": entry["variant"], "a": entry["a"]}
        row.update(entry["metrics"])
        rows.append(row)
    csv_path = os.path.join(args.output_dir, "table5_composition.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n✓ Table 5 → {json_path}")
    print(f"           → {csv_path}")


if __name__ == "__main__":
    main()
