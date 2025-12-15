from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from .timeutil import parse_iso8601


def run_ffprobe(video_path: str, ffprobe_bin: str) -> dict[str, Any]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration:format_tags:stream=width,height:stream_tags",
        "-select_streams",
        "v:0",
        video_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        msg = f"ffprobe failed (code {p.returncode})."
        if p.stderr:
            msg += f"\nstderr:\n{p.stderr.strip()}"
        if p.stdout:
            msg += f"\nstdout:\n{p.stdout.strip()}"
        raise RuntimeError(msg)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        out = (p.stdout or "").strip()
        if len(out) > 800:
            out = out[:800] + "..."
        raise RuntimeError(f"Could not parse ffprobe JSON output: {e}\nstdout:\n{out}") from e


def extract_creation_time_tag(ffprobe_json: dict[str, Any]) -> str | None:
    keys = [
        "creation_time",
        "com.apple.quicktime.creationdate",
        "date",
        "creation_date",
        "encoded_date",
    ]

    fmt_tags = (ffprobe_json.get("format") or {}).get("tags") or {}
    for k in keys:
        v = fmt_tags.get(k)
        if v:
            return v

    for stream in ffprobe_json.get("streams") or []:
        tags = stream.get("tags") or {}
        for k in keys:
            v = tags.get(k)
            if v:
                return v

    return None


def get_video_metadata(
    video_path: str, ffprobe_bin: str
) -> tuple[int, int, float | None, datetime, str]:
    """
    Returns:
      width, height, duration_seconds (or None), creation_time_utc, source_string
    """
    data = run_ffprobe(video_path, ffprobe_bin=ffprobe_bin)

    streams = data.get("streams") or []
    if not streams:
        raise ValueError("No video stream found (ffprobe returned no streams).")

    w = int(streams[0].get("width") or 0)
    h = int(streams[0].get("height") or 0)

    dur_text = (data.get("format") or {}).get("duration")
    duration = float(dur_text) if dur_text else None

    tag = extract_creation_time_tag(data)
    source = "ffprobe:creation_time"
    creation_dt = None
    if tag:
        if not re.search(r"(Z|[+-]\d{2}:?\d{2})$", tag.strip()):
            print(
                f"WARNING: ffprobe creation_time has no timezone; assuming UTC: {tag}",
                file=sys.stderr,
            )
        try:
            creation_dt = parse_iso8601(tag)
        except Exception:
            creation_dt = None

    if creation_dt is None:
        # Fallback: filesystem mtime (UTC). Not always accurate, but better than nothing.
        ts = os.path.getmtime(video_path)
        creation_dt = datetime.fromtimestamp(ts, tz=UTC)
        source = "filesystem_mtime_utc"

    return w, h, duration, creation_dt, source
