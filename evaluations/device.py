"""Runtime device selection with graceful fallback.

Usage:
    from evaluations.device import pick_device
    device = pick_device()                # auto: cuda -> mps -> cpu
    device = pick_device("cuda")          # explicit; falls back if unavailable
"""
from __future__ import annotations

import os
import torch


def pick_device(preferred: str | None = None) -> torch.device:
    """Return a usable torch.device with automatic fallback.

    Resolution order when preferred is None or "auto":
      cuda -> mps -> cpu
    When an explicit preference is given but unavailable, emit a warning
    to stderr and fall back along the same chain.
    """
    pref = (preferred or os.environ.get("TRACE_DEVICE") or "auto").lower()

    def _cuda_ok() -> bool:
        return torch.cuda.is_available()

    def _mps_ok() -> bool:
        return (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )

    chain = {
        "auto": ["cuda", "mps", "cpu"],
        "cuda": ["cuda", "mps", "cpu"],
        "gpu":  ["cuda", "mps", "cpu"],
        "mps":  ["mps", "cpu"],
        "cpu":  ["cpu"],
    }.get(pref, ["cuda", "mps", "cpu"])

    for candidate in chain:
        if candidate == "cuda" and _cuda_ok():
            return torch.device("cuda")
        if candidate == "mps" and _mps_ok():
            return torch.device("mps")
        if candidate == "cpu":
            return torch.device("cpu")

    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Short string label for logging / JSON output."""
    return str(device)
