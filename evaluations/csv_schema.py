"""Canonical scored-CSV schema for all outputs-only tables and plots.

Every table/plot that operates from a scored CSV alone consumes files that
follow this schema. The contract is enforced by :func:`validate_scored_csv`,
which raises ``SchemaError`` on any deviation.

Schema (one row per prompt):

    index                  int   — original prompt index
    prefix                 str   — the prompt text fed to the LM

    trace_gen_{k}          JSON  — TRACE (HMM-guided) generation #k, 1-indexed
    baseline_gen_{k}       JSON  — pure-LM generation #k, 1-indexed
      (at least one of the two families must be present)

    Each *_gen_{k} cell is a JSON-encoded object:
        {
          "continuation": str,     # generated text, required
          "toxicity":    float,    # Detoxify score in [0, 1], required if scored
          "fluency":     float,    # per-gen perplexity, optional
        }

    max_toxicity           float — max of per-gen toxicity for the row
    any_toxicity_gt_0.5    int   — 1 if any gen exceeds 0.5 toxicity else 0
    mean_fluency           float — mean per-gen perplexity for the row
    dist-1, dist-2, dist-3 float — distinct-n aggregated across the row's gens

The aggregate columns (max_toxicity, any_toxicity_gt_0.5, mean_fluency,
dist-1, dist-2, dist-3) mix BOTH trace and baseline gens — they are the union
aggregates produced by src/score.py. Per-mode aggregates are recomputed from
the JSON cells in table1_detoxification's `--scored_csv_override` path.

The schema is intentionally the same one ``src/score.py`` emits, so no
conversion is needed for files produced by this project. Files from other
sources must be adapted to match before passing them to the harness.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_BASE_COLUMNS = ("index", "prefix")
REQUIRED_AGGREGATE_COLUMNS = (
    "max_toxicity",
    "any_toxicity_gt_0.5",
    "mean_fluency",
    "dist-1",
    "dist-2",
    "dist-3",
)
REQUIRED_GEN_CELL_KEYS = ("continuation",)
OPTIONAL_GEN_CELL_KEYS = ("toxicity", "fluency")

_GEN_COL_RE = re.compile(r"^(trace|baseline)_gen_(\d+)$")


class SchemaError(ValueError):
    """Raised when a scored CSV does not conform to the canonical schema."""


@dataclass
class SchemaReport:
    path: str
    num_rows: int
    trace_gen_columns: List[str]
    baseline_gen_columns: List[str]
    has_aggregate_columns: bool

    def summary(self) -> str:
        return (
            f"{self.path}: {self.num_rows} rows, "
            f"{len(self.trace_gen_columns)} trace gens, "
            f"{len(self.baseline_gen_columns)} baseline gens, "
            f"aggregates={'yes' if self.has_aggregate_columns else 'no'}"
        )


def _check_gen_cell(cell: object, col: str, row_idx: int) -> None:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return
    try:
        obj = json.loads(str(cell))
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"row {row_idx}, column {col!r}: value is not valid JSON ({exc})"
        ) from exc
    if not isinstance(obj, dict):
        raise SchemaError(
            f"row {row_idx}, column {col!r}: JSON value must be an object, got {type(obj).__name__}"
        )
    for key in REQUIRED_GEN_CELL_KEYS:
        if key not in obj:
            raise SchemaError(
                f"row {row_idx}, column {col!r}: missing required key {key!r}"
            )


def validate_scored_csv(
    csv_path: str | Path,
    require_aggregates: bool = True,
    sample_rows: int = 5,
) -> SchemaReport:
    """Validate that *csv_path* conforms to the canonical scored-CSV schema.

    Args:
        csv_path: path to the CSV to check.
        require_aggregates: if True, enforce presence of the per-row aggregate
            columns produced by src/score.py. Set False for raw "generated"
            CSVs that haven't been scored yet.
        sample_rows: how many rows to deeply inspect for JSON cell contents.
            The remaining rows are only checked for column presence.

    Returns:
        A :class:`SchemaReport` describing the file.

    Raises:
        SchemaError: on any deviation.
        FileNotFoundError: if *csv_path* does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Scored CSV not found: {path}")

    df = pd.read_csv(path)

    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise SchemaError(
            f"{path}: missing required base column(s): {missing_base}. "
            f"Expected {list(REQUIRED_BASE_COLUMNS)}."
        )

    trace_cols, baseline_cols = [], []
    for col in df.columns:
        m = _GEN_COL_RE.match(col)
        if not m:
            continue
        (trace_cols if m.group(1) == "trace" else baseline_cols).append(col)
    trace_cols.sort(key=lambda c: int(c.rsplit("_", 1)[-1]))
    baseline_cols.sort(key=lambda c: int(c.rsplit("_", 1)[-1]))

    if not trace_cols and not baseline_cols:
        raise SchemaError(
            f"{path}: no generation columns found. Expected at least one of "
            "trace_gen_N or baseline_gen_N."
        )

    has_aggregates = all(c in df.columns for c in REQUIRED_AGGREGATE_COLUMNS)
    if require_aggregates and not has_aggregates:
        missing_agg = [c for c in REQUIRED_AGGREGATE_COLUMNS if c not in df.columns]
        raise SchemaError(
            f"{path}: missing aggregate column(s) {missing_agg}. "
            "Run src/score.py on the generated CSV, or pass require_aggregates=False "
            "if validating a pre-score file."
        )

    n_check = min(sample_rows, len(df))
    gen_cols = trace_cols + baseline_cols
    for i in range(n_check):
        for col in gen_cols:
            _check_gen_cell(df.iloc[i][col], col, i)

    return SchemaReport(
        path=str(path),
        num_rows=len(df),
        trace_gen_columns=trace_cols,
        baseline_gen_columns=baseline_cols,
        has_aggregate_columns=has_aggregates,
    )


def write_scored_csv_template(path: str | Path, num_prompts: int = 3, num_gens: int = 2) -> None:
    """Write a minimal example scored CSV so users can see the expected shape.

    The file is not useful for evaluation; it just shows the column layout
    and per-cell JSON structure other tools can be tested against.
    """
    rows = []
    for i in range(num_prompts):
        row = {"index": i, "prefix": f"Example prompt {i}"}
        toxes = []
        for k in range(1, num_gens + 1):
            cell = {"continuation": f"trace continuation {i}.{k}", "toxicity": 0.05 * k, "fluency": 20.0}
            row[f"trace_gen_{k}"] = json.dumps(cell)
            toxes.append(cell["toxicity"])
            bcell = {"continuation": f"baseline continuation {i}.{k}", "toxicity": 0.1 * k, "fluency": 18.0}
            row[f"baseline_gen_{k}"] = json.dumps(bcell)
            toxes.append(bcell["toxicity"])
        row["max_toxicity"] = max(toxes)
        row["any_toxicity_gt_0.5"] = int(max(toxes) > 0.5)
        row["mean_fluency"] = 19.0
        row["dist-1"] = 0.9
        row["dist-2"] = 0.85
        row["dist-3"] = 0.8
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m evaluations.csv_schema <scored_csv_path>")
        sys.exit(1)
    try:
        report = validate_scored_csv(sys.argv[1])
    except SchemaError as exc:
        print(f"INVALID: {exc}")
        sys.exit(1)
    print(f"OK  {report.summary()}")
