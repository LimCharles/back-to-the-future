"""
Single source of truth for metric computation across all TRACE evaluations.
"""
import json
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def parse_scored_csv(csv_path: str) -> pd.DataFrame:
    """Load a scored CSV and parse JSON gen columns into dicts.

    Each ``trace_gen_*`` / ``baseline_gen_*`` cell is a JSON string like
    ``{"continuation": "...", "toxicity": 0.15, "fluency": 23.4}``.
    This function parses those cells in-place so downstream code can
    access the dicts directly.

    Returns:
        DataFrame with parsed generation columns.
    Raises:
        FileNotFoundError: if *csv_path* does not exist.
        ValueError: if no generation columns are found.
    """
    df = pd.read_csv(csv_path)

    gen_cols = [c for c in df.columns if re.match(r"(trace|baseline)_gen_\d+", c)]
    if not gen_cols:
        # Fall back to old-style gen_N columns
        gen_cols = [c for c in df.columns if re.match(r"gen_\d+$", c)]
    if not gen_cols:
        raise ValueError(f"No generation columns found in {csv_path}")

    for col in gen_cols:
        df[col] = df[col].apply(lambda x: json.loads(str(x)) if pd.notna(x) else None)

    return df


def extract_continuations(df: pd.DataFrame, mode: str = "trace") -> List[List[str]]:
    """Extract continuation text per prompt from a parsed DataFrame.

    Args:
        df: DataFrame from :func:`parse_scored_csv`.
        mode: ``"trace"`` or ``"baseline"`` — which gen columns to extract.

    Returns:
        List of lists; outer index = prompt, inner = continuations.
    """
    prefix = f"{mode}_gen_"
    gen_cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not gen_cols:
        # Try old-style gen_N
        gen_cols = sorted(
            [c for c in df.columns if re.match(r"gen_\d+$", c)],
            key=lambda c: int(c.split("_")[-1]),
        )

    result: List[List[str]] = []
    for _, row in df.iterrows():
        conts: List[str] = []
        for col in gen_cols:
            cell = row[col]
            if cell is None:
                continue
            if isinstance(cell, dict):
                conts.append(str(cell.get("continuation", "")))
            else:
                conts.append(str(cell))
        result.append(conts)
    return result


def compute_distinct_n(texts: List[str], n: int) -> float:
    """Compute Dist-N: ratio of unique n-grams to total words.

    Mirrors the logic in ``src/score.py`` (space-tokenised).

    Args:
        texts: list of generated strings (all continuations for one prompt).
        n: n-gram order (1, 2, or 3).

    Returns:
        Float between 0 and 1 (0 when *texts* is empty).
    """
    ngrams: set = set()
    total_words = 0

    for text in texts:
        tokens = text.strip().split()
        total_words += len(tokens)
        for i in range(len(tokens) - n + 1):
            ngrams.add("_".join(tokens[i : i + n]))

    if total_words == 0:
        return 0.0
    return len(ngrams) / total_words


def aggregate_detox_run(df: pd.DataFrame, mode: str = "trace") -> Dict[str, float]:
    """Aggregate metrics for a scored detox run.

    Reads from the per-row aggregate columns already present in scored
    CSVs produced by ``src/score.py``.

    Returns:
        Dict with keys ``avg_max_tox``, ``prob_tox_gt_0.5``,
        ``mean_perplexity``, ``dist2``, ``dist3``.
    """
    return {
        "avg_max_tox": float(df["max_toxicity"].mean()),
        "prob_tox_gt_0.5": float(df["any_toxicity_gt_0.5"].mean()),
        "mean_perplexity": float(df["mean_fluency"].mean()),
        "dist2": float(df["dist-2"].mean()),
        "dist3": float(df["dist-3"].mean()),
    }


def conditional_entropy(
    continuations_per_prompt: List[List[List[int]]],
    vocab_size: int,
    max_position: int,
) -> float:
    """Compute conditional entropy H(X_t | prompt, X_1..X_{t-1}).

    For each prompt *p* and each position *t* (0 … max_position-1):
      1. Collect all tokens at position *t* across the *K* generations
         (only from generations long enough to reach *t*).
      2. Estimate P(X_t | context) from empirical frequency.
      3. Compute H_t_p = −Σ_x P(x) log₂ P(x).

    **Sum** H across positions for each prompt, then average over prompts.
    The paper reports ~52 for GPT-2-large; per-position averaging gives ~2-3,
    so summing is required to match the paper's magnitude.

    Args:
        continuations_per_prompt: ``[num_prompts][num_gens][seq_len]`` token IDs.
        vocab_size: vocabulary size (unused for counting but documents intent).
        max_position: number of token positions to consider.

    Returns:
        Average (over prompts) of the sum-over-positions conditional entropy
        in bits.
    """
    per_prompt_sums: List[float] = []

    for gens in continuations_per_prompt:
        if not gens:
            continue
        pos_sum = 0.0
        for t in range(max_position):
            tokens_at_t = [g[t] for g in gens if len(g) > t]
            if len(tokens_at_t) < 2:
                continue
            counts = Counter(tokens_at_t)
            total = len(tokens_at_t)
            h = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    h -= p * np.log2(p)
            pos_sum += h
        per_prompt_sums.append(pos_sum)

    if not per_prompt_sums:
        return 0.0
    return float(np.mean(per_prompt_sums))


def cross_entropy_loss(
    log_probs_a: torch.Tensor,
    log_probs_b: torch.Tensor,
) -> float:
    """Compute cross-entropy CE(a ‖ b) = −Σ exp(log_probs_a) · log_probs_b.

    Args:
        log_probs_a: ``(N, V)`` log-probabilities from model *a* (the "true"
            distribution).
        log_probs_b: ``(N, V)`` log-probabilities from model *b* (the
            approximation being evaluated).

    Returns:
        Scalar cross-entropy in nats, averaged over *N* samples.
    """
    p_a = torch.exp(log_probs_a)
    ce = -(p_a * log_probs_b).sum(dim=-1).mean()
    return float(ce.item())
