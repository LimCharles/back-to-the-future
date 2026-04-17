#!/usr/bin/env python
"""
Fit per-character role classifiers on RoleBench from local JSONL files.

Uses RoleBench's per-character training split directly as positives
(target = 1.0 for in-character responses). Optionally samples
out-of-character responses from other characters as negatives
(target = 0.0).

Usage::

    python -m evaluations.classifiers.fit_role
    python -m evaluations.classifiers.fit_role --train_data data/rolebench/train.jsonl --neg_ratio 1.0
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
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.fit import fit_lasso_model, save_coefficients


def _slugify(name: str) -> str:
    """Convert character name to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_jsonl(filepath: str) -> list:
    """Manually load a JSONL file into a list of dictionaries."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _build_character_data(train_data: list) -> dict:
    """Group RoleBench train split by character."""
    characters = {}
    for row in train_data:
        char = row.get("role") or row.get("character", "unknown")
        if char == "unknown":
            continue
            
        responses = row.get("generated") or row.get("response") or row.get("answer", [])
        
        # Normalize to list to handle both single strings and multiple generations
        if isinstance(responses, str):
            responses = [responses]
            
        if not isinstance(responses, list):
            continue
            
        for response in responses:
            if not response or not isinstance(response, str) or not response.strip():
                continue
            characters.setdefault(char, []).append(response)
            
    return characters


def main():
    parser = argparse.ArgumentParser(
        description="Fit per-character role classifiers on RoleBench from local JSONL."
    )
    parser.add_argument(
        "--train_data", type=str, default="data/rolebench/train.jsonl",
        help="Path to the training JSONL file",
    )
    parser.add_argument(
        "--test_data", type=str, default="data/rolebench/test.jsonl",
        help="Path to the test JSONL file (loaded but fitting uses train_data)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="classifiers/role/",
        help="Directory for per-character coefficient CSVs",
    )
    parser.add_argument("--alpha", type=float, default=1e-6, help="Lasso regularization")
    parser.add_argument(
        "--neg_ratio", type=float, default=1.0,
        help="Ratio of out-of-character negatives to positives. "
             "Default 0.0 matches literal paper text, but MUST BE > 0 to actually learn anything.",
    )
    parser.add_argument(
        "--b", type=float, default=10.0, help="Logit transform scaling"
    )
    parser.add_argument(
        "--c", type=float, default=3.0, help="Logit transform shift"
    )
    args = parser.parse_args()

    # Resolve paths relative to PROJECT_ROOT if not absolute
    if not os.path.isabs(args.train_data):
        args.train_data = str(PROJECT_ROOT / args.train_data)
    if not os.path.isabs(args.test_data):
        args.test_data = str(PROJECT_ROOT / args.test_data)
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
        
    os.makedirs(args.output_dir, exist_ok=True)

    print("Role Classifier Fitting (RoleBench Local JSONL)")
    print("=" * 50)
    
    if args.neg_ratio <= 0.0:
        print("\n" + "!"*60)
        print("WARNING: neg_ratio is 0.0 (Positives only).")
        print("Because the transformation pipeline uses np.log(target),")
        print("a target of 1.0 becomes 0.0. The Lasso model will perfectly")
        print("solve this by assigning exactly 0 to all coefficients.")
        print("To get usable steering vectors, re-run with --neg_ratio 1.0")
        print("!"*60 + "\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    vocab_size = tokenizer.vocab_size

    print(f"Loading local RoleBench datasets...")
    try:
        train_data = _load_jsonl(args.train_data)
        print(f"Loaded {len(train_data)} records from {args.train_data}")
        
        test_data = _load_jsonl(args.test_data)
        print(f"Loaded {len(test_data)} records from {args.test_data}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not load data. Ensure the JSONL files exist. Error: {e}")

    # Build character groupings strictly from the training set
    characters = _build_character_data(train_data)
    print(f"\nFound {len(characters)} characters in the training set.")

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

        # Transformation pipeline
        # NOTE: if targets_arr is all 1.0, log_targets becomes all 0.0
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
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Fitted {len(manifest)} character classifiers → {args.output_dir}")
    print(f"  Manifest: {manifest_path}")

if __name__ == "__main__":
    main()