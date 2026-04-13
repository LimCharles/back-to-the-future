#!/usr/bin/env python
"""Table 3 — Qualitative role-play comparison.

For a small hardcoded set of (role, prompt) pairs, generate continuations
with every HMM variant whose model directory exists on disk **and** with a
pure GPT-2 baseline. Writes a human-readable markdown table plus the raw
JSON.

This is a *qualitative* table — the paper's Table 3 is hand-picked sample
outputs, not an aggregate metric. The point is to show side-by-side how the
different guidance setups respond to the same role+prompt.

Variants scanned (run whichever exist):

    --hmm_dirs hmm1=models/hmm_base,hmm2=...,chmm=...

Missing variants are silently skipped.

Usage::

    python -m evaluations.tables.table3_roles \\
        --hmm_dirs hmm1=models/hmm_base \\
        --num_gens 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ROLES: List[Dict[str, str]] = [
    {
        "name": "Pirate Captain",
        "system": "You are a gruff pirate captain in the Golden Age of sail. Speak in first person with nautical slang.",
    },
    {
        "name": "Shakespearean Actor",
        "system": "You are a 17th-century Shakespearean actor. Respond in iambic-flavoured Elizabethan English.",
    },
    {
        "name": "Enthusiastic Scientist",
        "system": "You are a physicist who is wildly excited about every question. Answer with scientific accuracy and boundless curiosity.",
    },
]

PROMPTS: List[str] = [
    "A stranger asks you about the weather today",
    "Someone offers you a new book as a gift",
    "A child asks you why the sky is blue",
]


def _build_prompt_line(role: Dict[str, str], user_prompt: str) -> str:
    return f"{role['system']}\n\nUser: {user_prompt}\n{role['name']}:"


def _write_prompts_jsonl(entries: List[Dict[str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps({"prompt": {"text": entry["text"]}}) + "\n")


def _run_generate(
    variant: str,
    hmm_dir: str | None,
    prompts_file: Path,
    output_dir: Path,
    num_gens: int,
    max_len: int,
    a: float,
    device: str,
) -> Path:
    """Invoke src/generate.py and return the output CSV path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()

    if hmm_dir is None:
        out_csv = output_dir / f"table3_{variant}.csv"
        cmd = [
            sys.executable, "src/generate.py",
            "--baseline", "--a", "0",
            "--hmm_variant", "hmm1",
            "--prompts_path", str(prompts_file.relative_to(PROJECT_ROOT)),
            "--num_generations", str(num_gens),
            "--max_len", str(max_len),
            "--device", device,
        ]
        default_out = PROJECT_ROOT / "results" / "comparison_hmm1_a0.0_generated.csv"
    else:
        out_csv = output_dir / f"table3_{variant}.csv"
        cmd = [
            sys.executable, "src/generate.py",
            "--hmm_model_path", hmm_dir,
            "--hmm_variant", variant,
            "--a", str(a),
            "--prompts_path", str(prompts_file.relative_to(PROJECT_ROOT)),
            "--num_generations", str(num_gens),
            "--max_len", str(max_len),
            "--device", device,
        ]
        default_out = PROJECT_ROOT / "results" / f"detox_{variant}_a{a}_generated.csv"

    print(f"[{variant}] running generate.py ...", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

    if default_out.exists():
        default_out.replace(out_csv)
    return out_csv


def _read_generations(csv_path: Path) -> List[List[str]]:
    """Return per-prompt list of continuations from a generate.py output CSV."""
    df = pd.read_csv(csv_path)
    gen_cols = sorted(
        [c for c in df.columns if c.startswith("trace_gen_") or c.startswith("baseline_gen_")],
        key=lambda c: (0 if c.startswith("trace_gen_") else 1, int(c.rsplit("_", 1)[-1])),
    )
    out: List[List[str]] = []
    for _, row in df.iterrows():
        conts: List[str] = []
        for col in gen_cols:
            val = row[col]
            if pd.isna(val):
                continue
            try:
                obj = json.loads(str(val))
                conts.append(str(obj.get("continuation", "")).strip())
            except json.JSONDecodeError:
                pass
        out.append(conts)
    return out


def _parse_hmm_dirs(spec: str) -> Dict[str, str]:
    """Parse 'hmm1=path/a,hmm2=path/b' → {hmm1: 'path/a', hmm2: 'path/b'}."""
    if not spec:
        return {}
    out: Dict[str, str] = {}
    for piece in spec.split(","):
        if "=" not in piece:
            raise ValueError(f"Invalid --hmm_dirs entry: {piece!r}. Expected name=path.")
        name, path = piece.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 3 — qualitative role-play comparison.")
    parser.add_argument(
        "--hmm_dirs", type=str, default="hmm1=models/hmm_base",
        help="Comma-separated variant=dir map. Missing dirs are skipped.",
    )
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="evaluations/results/table3_roles")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from evaluations.device import pick_device
    device = str(pick_device(args.device))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hmm_dirs = _parse_hmm_dirs(args.hmm_dirs)
    available: Dict[str, str | None] = {"gpt2_baseline": None}
    for variant, path in hmm_dirs.items():
        abs_path = path if os.path.isabs(path) else str(PROJECT_ROOT / path)
        if os.path.exists(abs_path):
            available[variant] = abs_path
        else:
            print(f"[skip] {variant}: {abs_path} not found", file=sys.stderr)

    prompt_entries: List[Dict[str, str]] = []
    for role in ROLES:
        for user_prompt in PROMPTS:
            prompt_entries.append({
                "role": role["name"],
                "user_prompt": user_prompt,
                "text": _build_prompt_line(role, user_prompt),
            })

    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False,
        dir=str(PROJECT_ROOT / "data"), prefix="table3_prompts_",
    ) as tmp:
        tmp_path = Path(tmp.name)
    _write_prompts_jsonl(prompt_entries, tmp_path)

    per_variant_outputs: Dict[str, List[List[str]]] = {}
    try:
        for variant, hmm_dir in available.items():
            csv_path = _run_generate(
                variant=variant,
                hmm_dir=hmm_dir,
                prompts_file=tmp_path,
                output_dir=output_dir,
                num_gens=args.num_gens,
                max_len=args.max_len,
                a=args.a,
                device=device,
            )
            per_variant_outputs[variant] = _read_generations(csv_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    records = []
    for i, entry in enumerate(prompt_entries):
        variant_outputs = {
            variant: gens[i] if i < len(gens) else []
            for variant, gens in per_variant_outputs.items()
        }
        records.append({
            "role": entry["role"],
            "user_prompt": entry["user_prompt"],
            "full_prompt": entry["text"],
            "continuations": variant_outputs,
        })

    json_path = output_dir / "table3_roles.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "a": args.a,
            "num_gens": args.num_gens,
            "max_len": args.max_len,
            "device": device,
            "variants": list(available.keys()),
            "records": records,
        }, f, indent=2)

    md_path = output_dir / "table3_roles.md"
    variant_names = list(available.keys())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 3 — Role-Play Qualitative Examples\n\n")
        f.write(
            f"*{args.num_gens} generation(s) per prompt, a={args.a}, "
            f"max_len={args.max_len}, device={device}*\n\n"
        )
        header = "| Role | Prompt | " + " | ".join(variant_names) + " |\n"
        sep = "|---|---|" + "---|" * len(variant_names) + "\n"
        f.write(header)
        f.write(sep)
        for rec in records:
            cells = []
            for variant in variant_names:
                gens = rec["continuations"].get(variant, [])
                text = " / ".join(g.replace("|", "\\|").replace("\n", " ") for g in gens) or "—"
                cells.append(text)
            row_cells = [rec["role"], rec["user_prompt"].replace("|", "\\|")] + cells
            f.write("| " + " | ".join(row_cells) + " |\n")

    print(f"\n[OK] Table 3 -> {json_path}")
    print(f"     Markdown -> {md_path}")


if __name__ == "__main__":
    main()
