"""
Compatibility re-exports.

The project is organized into focused modules:
- `timeutil`: timestamp parsing and formatting helpers
- `fit`: FIT parsing + data models
- `video`: ffprobe metadata parsing
- `overlay`: ASS overlay generation
- `ffmpeg`: optional burn-in helpers
"""

from __future__ import annotations

from .ffmpeg import burn_in
from .fit import LapSegment, ParsedData, Sample, normalize_intensity, parse_data_file
from .fit import parse_fit_messages as parse_fit_messages
from .overlay import compute_visibility_range, generate_ass
from .timeutil import (
    ass_time,
    format_elapsed,
    format_pace,
    format_remaining,
    parse_iso8601,
    to_utc,
)
from .video import extract_creation_time_tag, get_video_metadata, run_ffprobe

__all__ = [
    "LapSegment",
    "ParsedData",
    "Sample",
    "ass_time",
    "burn_in",
    "compute_visibility_range",
    "extract_creation_time_tag",
    "format_elapsed",
    "format_pace",
    "format_remaining",
    "generate_ass",
    "get_video_metadata",
    "normalize_intensity",
    "parse_data_file",
    "parse_fit_messages",
    "parse_iso8601",
    "run_ffprobe",
    "to_utc",
]
