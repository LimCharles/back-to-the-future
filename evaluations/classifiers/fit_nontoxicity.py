#!/usr/bin/env python
"""
Fit a nontoxicity classifier using the existing src/fit.py pipeline.

Reuses ``load_attribute_data``, ``preprocess_scores``, ``create_token_matrix``,
``fit_lasso_model``, and ``save_coefficients`` from ``src.fit``.

Usage::

    python -m evaluations.classifiers.fit_nontoxicity
    python -m evaluations.classifiers.fit_nontoxicity --b 1 --c 0  # no transform (ablation)
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fit import (
    load_attribute_data,
    preprocess_scores,
    create_token_matrix,
    fit_lasso_model,
    save_coefficients,
    evaluate_model,
    create_diagnostic_plot,
)
from transformers import GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Fit nontoxicity classifier with Lasso regression."
    )
    parser.add_argument(
        "--data_path", type=str, default="data/RTP_train.jsonl",
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--output_path", type=str, default="classifiers/coefficients_nontoxicity.csv",
        help="Output path for coefficients",
    )
    parser.add_argument("--b", type=float, default=10.0, help="Logit transform scaling factor")
    parser.add_argument("--c", type=float, default=3.0, help="Logit transform shift factor")
    parser.add_argument("--alpha", type=float, default=1e-6, help="Lasso regularization")
    args = parser.parse_args()

    # Make paths absolute
    if not os.path.isabs(args.data_path):
        args.data_path = str(PROJECT_ROOT / args.data_path)
    if not os.path.isabs(args.output_path):
        args.output_path = str(PROJECT_ROOT / args.output_path)

    print("Nontoxicity Classifier Fitting")
    print("=" * 50)
    print(f"Data: {args.data_path}")
    print(f"Transformation: b={args.b}, c={args.c}")
    print(f"Regularization: α={args.alpha}")
    print(f"Output: {args.output_path}")
    print()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    vocab_size = tokenizer.vocab_size

    df = load_attribute_data(args.data_path, "toxicity")
    toxicity_scores = df["continuation.toxicity"].values.astype(float)
    texts = df["continuation.text"].tolist()

    transformed_scores, log_scores, original_scores = preprocess_scores(
        toxicity_scores, args.b, args.c
    )

    X, token_id_to_index = create_token_matrix(texts, tokenizer)
    coefficients, fitted_log_scores = fit_lasso_model(X, log_scores, args.alpha)
    fitted_scores = np.exp(fitted_log_scores)

    save_coefficients(coefficients, token_id_to_index, vocab_size, args.output_path)
    evaluate_model(transformed_scores, fitted_scores, log_scores, fitted_log_scores)

    print(f"\n✓ Nontoxicity coefficients saved to {args.output_path}")


if __name__ == "__main__":
    main()
