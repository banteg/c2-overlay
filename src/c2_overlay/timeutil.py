from __future__ import annotations

import math
import re
from datetime import UTC, datetime


ISO8601_RE = re.compile(
    r"^(?P<main>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?P<frac>\.\d+)?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?$"
)


def parse_iso8601(s: str) -> datetime:
    """
    Parse an ffprobe-style timestamp into an aware datetime in UTC.

    Expected inputs (examples from QuickTime/MOV metadata):
      - 2025-12-14T10:41:31.000000Z
      - 2025-12-14T10:41:31Z
      - 2025-12-14T14:41:31+0400
      - 2025-12-14T14:41:31+04:00
    """
    s = s.strip()
    if not s:
        raise ValueError("empty datetime string")

    m = ISO8601_RE.match(s)
    if m:
        main, frac, tz = m.group("main"), m.group("frac"), m.group("tz")
        if frac and len(frac) > 7:  # "." + 6 microsecond digits
            frac = frac[:7]
        s = main + (frac or "") + (tz or "")

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # Convert timezone offset like +0000 to +00:00 (Python expects a colon).
    m = re.match(r"^(.*)([+-]\d{2})(\d{2})$", s)
    if m:
        s = f"{m.group(1)}{m.group(2)}:{m.group(3)}"

    dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def format_elapsed(sec: float) -> str:
    """Format elapsed seconds like PM5: MM:SS or H:MM:SS."""
    total = int(math.floor(max(0.0, sec)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    # PM5 typically shows minutes without zero-padding (e.g. "0:03", not "00:03").
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


def format_remaining(sec: float) -> str:
    """Format remaining seconds like PM5: use ceil to avoid hitting 0 early."""
    total = int(math.ceil(max(0.0, sec)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


def format_pace(sec_per_500: float | None) -> str:
    """Format pace as M:SS.t (tenths), e.g., 2:05.3."""
    if sec_per_500 is None or not math.isfinite(sec_per_500) or sec_per_500 <= 0:
        return "--:--.-"
    tenths = int(round(sec_per_500 * 10))
    m = tenths // 600
    s_tenths = tenths % 600
    s = s_tenths // 10
    t = s_tenths % 10
    return f"{m}:{s:02d}.{t}"


def ass_time(sec: float) -> str:
    """ASS timestamps: H:MM:SS.cc (centiseconds)."""
    if sec < 0:
        sec = 0.0
    # Use round-half-up to avoid Python's banker's rounding at x.xx5, which can
    # create surprising timestamp ties at ASS's 1-centisecond resolution.
    cs = int(math.floor((sec * 100) + 0.5))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    cc = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cc:02d}"
