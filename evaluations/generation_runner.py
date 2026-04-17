"""
Wraps ``src/generate.py`` and ``src/score.py`` as Python functions.

Uses :func:`subprocess.run` with ``check=True`` and streams stderr.
Caches Detoxify scores by SHA1 of text under ``.score_cache/``.
All other scripts import from here rather than calling subprocess directly.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GenerationRunner:
    """Wraps generate.py and score.py via subprocess with Detoxify score caching."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        verbose: bool = True,
    ):
        """
        Args:
            cache_dir: Directory for score cache.
                Defaults to ``PROJECT_ROOT / evaluations / .score_cache``.
            device: Torch device string forwarded to subprocess scripts.
            verbose: If True, stream stderr from subprocesses to console.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / "evaluations" / ".score_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        hmm_variant: str = "hmm1",
        a: float = 1.0,
        prompts_path: str = "data/prompts.jsonl",
        weights_path: str = "data/coefficients.csv",
        baseline: bool = False,
        max_len: int = 20,
        num_generations: int = 25,
        generation_batch_size: int = 5,
        prompt_batch_size: int = 1,
        seed: int = 42,
        model_path: str = "gpt2-large",
        hmm_model_path: Optional[str] = None,
        no_decode_transform: bool = False,
        dump_eap_path: Optional[str] = None,
    ) -> str:
        """Run ``src/generate.py`` and return the output CSV path.

        Returns:
            Absolute path to the generated CSV file.
        Raises:
            subprocess.CalledProcessError: on non-zero exit.
        """
        cmd: List[str] = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "generate.py"),
            "--hmm_variant", hmm_variant,
            "--a", str(a),
            "--prompts_path", prompts_path,
            "--weights_path", weights_path,
            "--max_len", str(max_len),
            "--num_generations", str(num_generations),
            "--generation_batch_size", str(generation_batch_size),
            "--prompt_batch_size", str(prompt_batch_size),
            "--seed", str(seed),
            "--model_path", model_path,
            "--device", self.device,
        ]
        if baseline:
            cmd.append("--baseline")
        if hmm_model_path:
            cmd.extend(["--hmm_model_path", hmm_model_path])
        if no_decode_transform:
            cmd.append("--no_decode_transform")
        if dump_eap_path:
            cmd.extend(["--dump_eap_path", dump_eap_path])

        self._run(cmd)

        # Determine output path (mirrors generate.py naming convention)
        if baseline:
            output_csv = PROJECT_ROOT / f"results/generated/comparison_{hmm_variant}_a{a}_generated.csv"
        else:
            output_csv = PROJECT_ROOT / f"results/generated/detox_{hmm_variant}_a{a}_generated.csv"
        return str(output_csv)

    def score(
        self,
        input_csv: str,
        output_csv: Optional[str] = None,
        perp_model: str = "gpt2-xl",
        batch_size: int = 10,
    ) -> str:
        """Run ``src/score.py`` and return the scored CSV path.

        After scoring, populates the Detoxify score cache for future de-dup.

        Returns:
            Absolute path to the scored CSV file.
        """
        if output_csv is None:
            input_name = Path(input_csv).name
            output_name = input_name.replace("_generated.csv", "_scored.csv")
            if output_name == input_name:
                output_name = input_name.replace(".csv", "_scored.csv")
            output_csv = str(PROJECT_ROOT / "results" / "evaluation" / output_name)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "score.py"),
            "--input_csv", input_csv,
            "--output_csv", output_csv,
            "--perp_model", perp_model,
            "--batch_size", str(batch_size),
            "--device", self.device,
        ]
        self._run(cmd)

        # Populate score cache from newly scored CSV
        self._populate_cache(output_csv)

        return output_csv

    def generate_and_score(
        self,
        hmm_variant: str = "hmm1",
        a: float = 1.0,
        **kwargs,
    ) -> str:
        """Convenience: run generate then score, return scored CSV path."""
        gen_csv = self.generate(hmm_variant=hmm_variant, a=a, **kwargs)
        return self.score(gen_csv)

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(text: str) -> str:
        """SHA1 hex digest of text, used as cache filename."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _get_cached_score(self, text: str) -> Optional[float]:
        """Look up Detoxify toxicity score from cache. Returns None on miss."""
        sha = self._cache_key(text)
        path = self.cache_dir / sha[:2] / f"{sha}.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f).get("toxicity")
            except (json.JSONDecodeError, KeyError):
                path.unlink(missing_ok=True)
        return None

    def _set_cached_score(self, text: str, score: float) -> None:
        """Write Detoxify toxicity score to cache.

        File layout: ``{cache_dir}/{sha1[:2]}/{sha1}.json``
        """
        sha = self._cache_key(text)
        bucket = self.cache_dir / sha[:2]
        bucket.mkdir(parents=True, exist_ok=True)
        path = bucket / f"{sha}.json"
        with open(path, "w") as f:
            json.dump({"text": text, "toxicity": score}, f)

    def _populate_cache(self, scored_csv: str) -> None:
        """Read a scored CSV and cache every continuation's toxicity score."""
        try:
            df = pd.read_csv(scored_csv)
        except Exception:
            return

        import re
        gen_cols = [c for c in df.columns if re.match(r"(trace|baseline|gen)_gen_?\d+", c)]
        for col in gen_cols:
            for cell in df[col].dropna():
                try:
                    data = json.loads(str(cell))
                    text = data.get("continuation", "")
                    tox = data.get("toxicity")
                    if text and tox is not None:
                        self._set_cached_score(text, float(tox))
                except (json.JSONDecodeError, TypeError):
                    continue

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------

    def _run(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a subprocess, streaming stderr if verbose."""
        stderr_opt = None if self.verbose else subprocess.PIPE
        return subprocess.run(
            cmd,
            check=True,
            cwd=str(PROJECT_ROOT),
            stderr=stderr_opt,
        )
