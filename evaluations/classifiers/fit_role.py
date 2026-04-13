#!/usr/bin/env python
"""
Fit per-character role classifiers on RoleBench.

Uses RoleBench's per-character training split directly as positives
(target = 1.0 for in-character responses).  Optionally samples
out-of-character responses from other characters as negatives
(target = 0.0).

Usage::

    python -m evaluations.classifiers.fit_role
    python -m evaluations.classifiers.fit_role --output_dir data/coefficients_role/
"""
import json
import os
import re
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.special import logit, expit
from scipy.sparse import lil_matrix
from sklearn.linear_model import Lasso
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.fit import fit_lasso_model, save_coefficients


def _slugify(name: str) -> str:
    """Convert character name to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_rolebench():
    """Load RoleBench dataset from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("ZenMoore/RoleBench")
    return ds


def _build_character_data(ds) -> dict:
    """Group RoleBench data by character.

    Returns:
        Dict mapping character name to list of response texts.
    """
    characters = {}
    for split in ds:
        for row in ds[split]:
            char = row.get("role") or row.get("character", "unknown")
            response = row.get("response") or row.get("answer", "")
            if not response.strip():
                continue
            characters.setdefault(char, []).append(response)
    return characters


def main():
    parser = argparse.ArgumentParser(
        description="Fit per-character role classifiers on RoleBench."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/coefficients_role/",
        help="Directory for per-character coefficient CSVs",
    )
    parser.add_argument("--alpha", type=float, default=1e-6, help="Lasso regularization")
    parser.add_argument(
        "--neg_ratio", type=float, default=1.0,
        help="Ratio of negative (out-of-character) samples to positive samples",
    )
    parser.add_argument(
        "--b", type=float, default=10.0, help="Logit transform scaling"
    )
    parser.add_argument(
        "--c", type=float, default=3.0, help="Logit transform shift"
    )
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Role Classifier Fitting (RoleBench)")
    print("=" * 50)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    vocab_size = tokenizer.vocab_size

    print("Loading RoleBench dataset…")
    ds = _load_rolebench()
    characters = _build_character_data(ds)
    print(f"Found {len(characters)} characters")

    manifest = {}

    for char_name in tqdm(sorted(characters.keys()), desc="Fitting classifiers"):
        positives = characters[char_name]
        if len(positives) < 10:
            print(f"  Skipping '{char_name}' — only {len(positives)} samples")
            continue

        # Build training set: positives (target=1.0) + negatives (target=0.0)
        texts = list(positives)
        targets = [1.0] * len(positives)

        # Sample negatives from other characters
        n_neg = int(len(positives) * args.neg_ratio)
        if n_neg > 0:
            all_other = []
            for other_char, other_texts in characters.items():
                if other_char != char_name:
                    all_other.extend(other_texts)
            rng = np.random.RandomState(42)
            if len(all_other) > n_neg:
                neg_indices = rng.choice(len(all_other), n_neg, replace=False)
                neg_texts = [all_other[i] for i in neg_indices]
            else:
                neg_texts = all_other
            texts.extend(neg_texts)
            targets.extend([0.0] * len(neg_texts))

        targets_arr = np.array(targets)

        # Preprocess: apply logit transform (same as nontoxicity)
        eps = 1e-15
        clipped = np.clip(targets_arr, eps, 1 - eps)
        logit_scores = logit(clipped)
        modified = args.b * (logit_scores - args.c)
        transformed = expit(modified)
        clipped_tf = np.clip(transformed, eps, 1 - eps)
        log_targets = np.log(clipped_tf)

        # Create token count matrix
        tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        unique_ids = sorted(set(tid for toks in tokenized for tid in toks))
        tid_to_idx = {tid: idx for idx, tid in enumerate(unique_ids)}

        X = lil_matrix((len(tokenized), len(unique_ids)), dtype=int)
        for i, toks in enumerate(tokenized):
            for tid in toks:
                if tid in tid_to_idx:
                    X[i, tid_to_idx[tid]] += 1
        X = X.tocsr()

        # Fit
        coefficients, _ = fit_lasso_model(X, log_targets, args.alpha)

        # Save
        slug = _slugify(char_name)
        out_path = os.path.join(args.output_dir, f"{slug}.csv")
        save_coefficients(coefficients, tid_to_idx, vocab_size, out_path)

        manifest[char_name] = {
            "slug": slug,
            "csv_path": f"{slug}.csv",
            "n_positive": len(positives),
            "n_negative": len(texts) - len(positives),
        }

    # Write manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Fitted {len(manifest)} character classifiers → {args.output_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
