#!/usr/bin/env python
"""
HMM capacity vs toxicity plot (repurposed).

Originally this script swept training-step checkpoints. With finished models
only (hmm1 H=4096, hmm2 H=64, hmm2 H=256), the sweep is now across hidden-state
capacity instead.

Inputs per model:
  - hidden_size (read from <model_dir>/config.json)
  - avg max toxicity (TRACE-mode) from an existing scored CSV in
    ``results/evaluation/``; if absent, generate + score on the fly.
  - optional: validation log-likelihood on a held-out JSONL if --val_data is set.

Output:
  - ``results/figures/hmm_capacity_vs_toxicity.png``
  - ``results/figures/hmm_capacity_vs_toxicity.json``

Usage::

    python -m evaluations.plots.plot_hmm_quality_vs_toxicity \\
        --models "hmm1:models/hmm_gpt2-large_bttf,hmm2:models/hmm2_gpt2-large_64_bttf,hmm2:models/hmm2_gpt2-large_256_bttf"

    # With val-LL annotation (slower; loads each HMM/SOHMM):
    python -m evaluations.plots.plot_hmm_quality_vs_toxicity \\
        --models "hmm1:models/hmm_gpt2-large_bttf,hmm2:models/hmm2_gpt2-large_256_bttf" \\
        --val_data data/RTP_test.jsonl --val_num_samples 1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.generation_runner import GenerationRunner
from evaluations.metrics import compute_distinct_n, parse_scored_csv


def _parse_models_arg(spec: str) -> List[Tuple[str, Path]]:
    """Parse ``--models 'hmm1:path_a,hmm2:path_b'`` into tuples."""
    entries = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--models entry '{chunk}' must be formatted as 'variant:path'"
            )
        variant, path_str = chunk.split(":", 1)
        variant = variant.strip()
        if variant not in {"hmm1", "hmm2"}:
            raise ValueError(
                f"Unknown variant '{variant}' in --models; expected hmm1 or hmm2"
            )
        model_path = Path(path_str.strip())
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        entries.append((variant, model_path))
    if not entries:
        raise ValueError("--models parsed to empty list")
    return entries


def _read_hidden_size(model_dir: Path) -> int:
    with open(model_dir / "config.json", "r") as f:
        return int(json.load(f)["hidden_size"])


def _canonical_tag(variant: str, model_dir: Path, hidden_size: int) -> str:
    """Produce the filename tag used by existing scored CSVs.

    Convention (matches the files the user already has under
    ``results/evaluation/``):

      - hmm1 → ``hmm1``
      - hmm2 → ``hmm2_<hidden_size>``

    The hmm2 tag is keyed on hidden size because ``src/generate.py`` writes
    to ``results/generated/<comparison|detox>_<variant>_a<a>_generated.csv``
    which would clobber across hmm2 sizes. Checkpoints the user scored
    manually follow the ``hmm2_64`` / ``hmm2_256`` pattern.
    """
    if variant == "hmm1":
        return "hmm1"
    # Try to extract size from directory name first, fall back to config
    m = re.search(r"_(\d+)_", model_dir.name)
    if m:
        return f"hmm2_{m.group(1)}"
    return f"hmm2_{hidden_size}"


def _find_or_generate_scored_csv(
    runner: GenerationRunner,
    variant: str,
    tag: str,
    model_path: Path,
    a: float,
    scored_dir: Path,
    prompts_path: str,
    weights_path: str,
    num_generations: int,
    max_len: int,
    naming: str,
) -> Path:
    """Return a scored CSV for (variant, tag, a). Generate + score if missing."""
    stem = f"{naming}_{tag}_a{a}_scored.csv"
    candidate = scored_dir / stem
    if candidate.exists():
        return candidate
    # Fallback: regenerate. The runner writes scored CSVs to
    # results/evaluation/<basename>_scored.csv; we rename afterwards if
    # the tag carries extra info (e.g. hmm2_256) the runner doesn't know.
    print(f"[hmm_capacity] No cached scored CSV at {candidate}; running generate+score for {variant} ({tag}) …")
    scored = runner.generate_and_score(
        hmm_variant=variant,
        a=a,
        prompts_path=prompts_path,
        weights_path=weights_path,
        baseline=(naming == "comparison"),
        max_len=max_len,
        num_generations=num_generations,
        hmm_model_path=str(model_path),
    )
    scored_path = Path(scored)
    if scored_path.name != stem:
        target = scored_dir / stem
        scored_path.rename(target)
        return target
    return scored_path


def _per_mode_aggregate(df: pd.DataFrame, mode: str) -> Dict[str, float]:
    """Compute TRACE-only (or baseline-only) toxicity + fluency + dist-n.

    The union-mode columns written by ``src/score.py`` mix trace and baseline
    scores, so capacity-vs-detox comparisons need a per-mode recomputation
    from the JSON cells. Logic mirrors
    ``evaluations/tables/table1_detoxification._aggregate_mode``.
    """
    prefix = f"{mode}_gen_"
    cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not cols:
        raise ValueError(f"No '{mode}_gen_*' columns in scored CSV")

    max_tox, any_tox, flus = [], [], []
    d1, d2, d3 = [], [], []
    for _, row in df.iterrows():
        toxes, flu_vals, conts = [], [], []
        for c in cols:
            cell = row[c]
            if not isinstance(cell, dict):
                continue
            t = cell.get("toxicity")
            f = cell.get("fluency")
            if t is not None:
                toxes.append(float(t))
            if isinstance(f, (int, float)):
                flu_vals.append(float(f))
            conts.append(str(cell.get("continuation", "")))
        if not toxes:
            continue
        max_tox.append(max(toxes))
        any_tox.append(1.0 if max(toxes) > 0.5 else 0.0)
        if flu_vals:
            flus.append(sum(flu_vals) / len(flu_vals))
        d1.append(compute_distinct_n(conts, 1))
        d2.append(compute_distinct_n(conts, 2))
        d3.append(compute_distinct_n(conts, 3))

    return {
        "avg_max_tox": statistics.fmean(max_tox),
        "prob_tox_gt_0.5": statistics.fmean(any_tox),
        "mean_perplexity": statistics.fmean(flus) if flus else float("nan"),
        "dist1": statistics.fmean(d1),
        "dist2": statistics.fmean(d2),
        "dist3": statistics.fmean(d3),
    }


def _compute_val_ll_per_token(
    variant: str,
    model_path: Path,
    val_data: Path,
    num_samples: int,
    device: str,
) -> Optional[float]:
    """Average log-likelihood per token on the first ``num_samples`` continuations.

    Returns ``None`` on failure so the plot still renders without LL
    annotation for that point.
    """
    try:
        import torch
        from transformers import GPT2Tokenizer
        from src import utils
    except Exception as exc:
        print(f"[hmm_capacity] val-LL import failed: {exc}", file=sys.stderr)
        return None

    if variant == "hmm1":
        model = utils.load_hmm_model(str(model_path), device=device)
    else:
        model = utils.load_sohmm_model(str(model_path), device=device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

    token_ids: List[List[int]] = []
    with open(val_data, "r", encoding="utf-8") as f:
        for line in f:
            if len(token_ids) >= num_samples:
                break
            try:
                record = json.loads(line)
                text = record["continuation"]["text"]
            except Exception:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            token_ids.append(ids)

    if not token_ids:
        return None

    max_len = max(len(x) for x in token_ids)
    pad_id = -1  # both HMM.forward and SOHMM.forward mask input_ids == -1
    padded = torch.full((len(token_ids), max_len), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(token_ids):
        padded[i, : len(ids)] = torch.tensor(ids, device=device)

    total_tokens = sum(len(x) for x in token_ids)
    with torch.no_grad():
        try:
            ll = model.loglikelihood(padded, batch_size=8)
        except AttributeError:
            # HMM has no `loglikelihood` method — run forward and sum the last-layer
            # logsumexp.
            probs = model.forward(padded)
            ll = probs[-1].sum()
        return float(ll.item()) / float(total_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Plot HMM capacity vs avg-max-toxicity (repurposed from training-step sweep)."
    )
    parser.add_argument(
        "--models", type=str, required=True,
        help="Comma-separated 'variant:path' tuples, e.g. "
             "'hmm1:models/hmm_gpt2-large_bttf,hmm2:models/hmm2_gpt2-large_256_bttf'",
    )
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--naming", type=str, default="comparison",
                        choices=["comparison", "detox"],
                        help="Scored-CSV filename prefix to look for under --scored_dir")
    parser.add_argument("--scored_dir", type=str, default="results/evaluation")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument("--weights_path", type=str, default="data/coefficients.csv")
    parser.add_argument("--num_generations", type=int, default=25)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--val_data", type=str, default=None,
                        help="Optional JSONL (e.g. data/RTP_test.jsonl) to score val-LL per model")
    parser.add_argument("--val_num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    if not scored_dir.is_absolute():
        scored_dir = PROJECT_ROOT / scored_dir
    scored_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    val_data_path = None
    if args.val_data:
        val_data_path = Path(args.val_data)
        if not val_data_path.is_absolute():
            val_data_path = PROJECT_ROOT / val_data_path

    runner = GenerationRunner(device=args.device)
    entries = _parse_models_arg(args.models)

    records: List[Dict] = []
    for variant, model_path in tqdm(entries, desc="Capacity sweep"):
        hidden_size = _read_hidden_size(model_path)
        tag = _canonical_tag(variant, model_path, hidden_size)

        scored_csv = _find_or_generate_scored_csv(
            runner=runner,
            variant=variant,
            tag=tag,
            model_path=model_path,
            a=args.a,
            scored_dir=scored_dir,
            prompts_path=args.prompts_path,
            weights_path=args.weights_path,
            num_generations=args.num_generations,
            max_len=args.max_len,
            naming=args.naming,
        )

        df = parse_scored_csv(str(scored_csv))
        trace_agg = _per_mode_aggregate(df, mode="trace")

        val_ll = None
        if val_data_path is not None:
            val_ll = _compute_val_ll_per_token(
                variant=variant,
                model_path=model_path,
                val_data=val_data_path,
                num_samples=args.val_num_samples,
                device=args.device,
            )

        records.append({
            "variant": variant,
            "tag": tag,
            "hidden_size": hidden_size,
            "model_path": str(model_path),
            "scored_csv": str(scored_csv),
            "avg_max_tox": trace_agg["avg_max_tox"],
            "prob_tox_gt_0.5": trace_agg["prob_tox_gt_0.5"],
            "mean_perplexity": trace_agg["mean_perplexity"],
            "dist2": trace_agg["dist2"],
            "dist3": trace_agg["dist3"],
            "val_ll_per_token": val_ll,
        })

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax1 = plt.subplots(figsize=(8, 5))

    df_plot = pd.DataFrame(records)
    sns.scatterplot(
        data=df_plot, x="hidden_size", y="avg_max_tox",
        hue="variant", style="variant", s=140, ax=ax1,
    )
    for _, row in df_plot.iterrows():
        ax1.annotate(
            f"{row['tag']} (H={row['hidden_size']})",
            (row["hidden_size"], row["avg_max_tox"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Hidden states H (log scale)")
    ax1.set_ylabel("Avg Max Toxicity (TRACE)")
    ax1.set_title("HMM capacity vs TRACE detoxification")

    if df_plot["val_ll_per_token"].notna().any():
        ax2 = ax1.twinx()
        ll_df = df_plot.dropna(subset=["val_ll_per_token"])
        sns.lineplot(
            data=ll_df.sort_values("hidden_size"),
            x="hidden_size", y="val_ll_per_token",
            marker="s", ax=ax2, color="tab:green", legend=False,
        )
        ax2.set_ylabel("Val log-likelihood / token", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

    fig.tight_layout()
    out_path = output_dir / "hmm_capacity_vs_toxicity.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    json_path = output_dir / "hmm_capacity_vs_toxicity.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Plot -> {out_path}")
    print(f"Data -> {json_path}")


if __name__ == "__main__":
    main()
