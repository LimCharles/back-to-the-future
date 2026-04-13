#!/usr/bin/env python
"""
Table 7 — Conditional entropy.

Runs detox generation for baseline and each variant under top-p=0.9,
tokenises continuations, and computes conditional entropy via
``metrics.conditional_entropy``.

Entropy is **summed** across positions per prompt, then averaged over
prompts.  Target magnitude: ~52 for GPT-2-large baseline.

Usage::

    python -m evaluations.tables.table7_conditional_entropy \\
        --scored_csv results/evaluation/detox_hmm1_a1.0_scored.csv
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from transformers import GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.metrics import conditional_entropy, extract_continuations, parse_scored_csv
from evaluations.generation_runner import GenerationRunner


def main():
    parser = argparse.ArgumentParser(description="Table 7: conditional entropy.")
    parser.add_argument("--hmm_variant", type=str, default="hmm1")
    parser.add_argument("--scored_csv", type=str, default=None,
                        help="Pre-existing scored CSV (skip generation if set)")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument("--weights_path", type=str, default="data/coefficients.csv")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--num_generations", type=int, default=25)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/tables")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

    # ------------------------------------------------------------------
    # Get scored CSV (generate if needed)
    # ------------------------------------------------------------------
    if args.scored_csv and os.path.exists(args.scored_csv):
        scored_csv = args.scored_csv
    else:
        runner = GenerationRunner(device=args.device)
        scored_csv = runner.generate_and_score(
            hmm_variant=args.hmm_variant,
            a=args.a,
            prompts_path=args.prompts_path,
            weights_path=args.weights_path,
            num_generations=args.num_generations,
            max_len=args.max_len,
            seed=args.seed,
        )

    # ------------------------------------------------------------------
    # Compute conditional entropy for each mode
    # ------------------------------------------------------------------
    df = parse_scored_csv(scored_csv)
    results = {}

    for mode in ("trace", "baseline"):
        try:
            continuations = extract_continuations(df, mode=mode)
        except Exception:
            continue
        if not continuations or not continuations[0]:
            continue

        # Tokenise continuations
        token_ids_per_prompt = []
        for prompt_conts in continuations:
            prompt_tokens = []
            for text in prompt_conts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                prompt_tokens.append(ids)
            token_ids_per_prompt.append(prompt_tokens)

        h = conditional_entropy(
            token_ids_per_prompt,
            vocab_size=tokenizer.vocab_size,
            max_position=args.max_len,
        )
        results[mode] = h
        print(f"  {mode}: H = {h:.2f} bits")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    output = {
        "variant": args.hmm_variant,
        "a": args.a,
        "conditional_entropy": results,
        "max_position": args.max_len,
        "num_generations": args.num_generations,
        "scored_csv": scored_csv,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    json_path = os.path.join(args.output_dir, "table7_conditional_entropy.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    rows = [{"mode": mode, "conditional_entropy": h} for mode, h in results.items()]
    csv_path = os.path.join(args.output_dir, "table7_conditional_entropy.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n✓ Table 7 → {json_path}")
    print(f"           → {csv_path}")


if __name__ == "__main__":
    main()
