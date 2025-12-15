from __future__ import annotations

from datetime import datetime

__all__ = ["parse_iso8601"]


def parse_iso8601(s: str) -> datetime:
    # Lazy import: keep `import c2_overlay` lightweight.
    from .timeutil import parse_iso8601 as _parse_iso8601

    return _parse_iso8601(s)
