from __future__ import annotations

from datetime import datetime

from .fit import Sample


def choose_anchor_index(
    samples: list[Sample], *, video_start: datetime, mode: str
) -> int:
    """
    Pick which sample becomes t0.

    Modes:
      - "start": use first sample
      - "first-visible": first sample at/after video_start
      - "first-row-visible": first sample at/after video_start with cadence > 0
    """
    if not samples:
        return 0
    if mode == "start":
        return 0
    if mode not in {"first-visible", "first-row-visible"}:
        raise ValueError(f"unknown anchor mode: {mode}")

    for i, s in enumerate(samples):
        if s.t < video_start:
            continue
        if mode == "first-visible":
            return i
        if mode == "first-row-visible":
            if (s.cadence or 0) > 0:
                return i
            continue

    return 0
