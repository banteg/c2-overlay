#!/usr/bin/env python3
"""
c2_overlay.py

Generate a Concept2 PM5-style overlay (.ass subtitles) from a Concept2 FIT file and a video.

What it does
- Reads the video's absolute start timestamp from metadata (ffprobe creation_time tag).
- Reads absolute timestamps and laps from the FIT file.
- Computes the offset so FIT samples line up on the video timeline.
- Writes a modern PM5-style grid overlay bottom-left with lap/rest context.
- Optionally burns the overlay into a new video using ffmpeg.

Requirements
- Python 3.12+
- ffprobe + ffmpeg available on PATH (or pass --ffprobe-bin / --ffmpeg-bin)

Example
  python c2_overlay.py input.mp4 workout.fit -o input.ass --burn-in output.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from fitparse import FitFile


# -----------------------------
# Helpers: time parsing/format
# -----------------------------


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
    cs = int(round(sec * 100))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    cc = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cc:02d}"


@dataclass
class Sample:
    t: datetime
    distance_m: float | None = None
    hr: int | None = None
    cadence: int | None = None  # usually stroke rate (SPM) in Concept2 FIT
    watts: int | None = None
    speed: float | None = None  # m/s


@dataclass(frozen=True)
class LapSegment:
    index: int  # 1-based
    start: datetime
    end: datetime
    intensity: str  # "active" | "rest" | other
    start_distance_m: float | None
    total_elapsed_s: float | None
    total_distance_m: float | None
    avg_speed_m_s: float | None
    avg_power_w: int | None
    avg_cadence_spm: int | None
    avg_hr_bpm: int | None


INTENSITY_MAP = {0: "active", 1: "rest"}


def normalize_intensity(v: object) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, (int, float)):
        return INTENSITY_MAP.get(int(v), f"unknown({int(v)})")
    s = str(v).strip().lower()
    if s in {"active", "rest"}:
        return s
    return s or "unknown"


def parse_fit_messages(fit: FitFile) -> list[Sample]:
    samples: list[Sample] = []
    for msg in fit.get_messages("record"):
        fields = {f.name: f.value for f in msg}
        ts = fields.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        ts = to_utc(ts)

        distance_m = fields.get("distance")
        if isinstance(distance_m, (int, float)):
            distance_m = float(distance_m)
        else:
            distance_m = None

        hr = fields.get("heart_rate")
        hr = int(hr) if isinstance(hr, (int, float)) else None

        cadence = fields.get("cadence")
        cadence = int(cadence) if isinstance(cadence, (int, float)) else None

        watts = fields.get("power")
        watts = int(watts) if isinstance(watts, (int, float)) else None

        speed = fields.get("enhanced_speed", fields.get("speed"))
        speed = float(speed) if isinstance(speed, (int, float)) else None

        samples.append(
            Sample(
                t=ts,
                distance_m=distance_m,
                hr=hr,
                cadence=cadence,
                watts=watts,
                speed=speed,
            )
        )

    samples.sort(key=lambda s: s.t)

    # Fill in missing speeds from distance/time deltas (m/s)
    for i in range(1, len(samples)):
        if (
            samples[i].speed is None
            and samples[i].distance_m is not None
            and samples[i - 1].distance_m is not None
        ):
            dt = (samples[i].t - samples[i - 1].t).total_seconds()
            dd = samples[i].distance_m - samples[i - 1].distance_m
            if dt > 0 and dd >= 0:
                samples[i].speed = dd / dt
    if samples and samples[0].speed is None and len(samples) > 1:
        samples[0].speed = samples[1].speed

    return samples


def parse_fit(fit_path: str) -> list[Sample]:
    return parse_fit_messages(FitFile(fit_path))


@dataclass(frozen=True)
class ParsedData:
    samples: list[Sample]
    laps: list[LapSegment] | None = None


def parse_data_file(path: str) -> ParsedData:
    ext = Path(path).suffix.lower()
    if ext == ".fit":
        fit = FitFile(path)
        samples = parse_fit_messages(fit)
        laps: list[LapSegment] = []
        # Build a timestamp list for fast lookup of lap start distances.
        ts_list = [s.t for s in samples]
        dist_list = [s.distance_m for s in samples]

        def distance_at(t: datetime) -> float | None:
            idx = min(bisect_left(ts_list, t), len(ts_list) - 1)
            # Prefer the first sample at/after start; fallback to previous if missing distance.
            for j in (idx, idx - 1, idx + 1):
                if 0 <= j < len(dist_list) and dist_list[j] is not None:
                    return float(dist_list[j])
            return None

        for msg in fit.get_messages("lap"):
            fields = {f.name: f.value for f in msg}
            start = fields.get("start_time")
            end = fields.get("timestamp")
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                continue
            start = to_utc(start)
            end = to_utc(end)

            total_elapsed = fields.get("total_elapsed_time")
            total_distance = fields.get("total_distance")
            avg_speed = fields.get("enhanced_avg_speed", fields.get("avg_speed"))
            avg_power = fields.get("avg_power")
            avg_cadence = fields.get("avg_cadence")
            avg_hr = fields.get("avg_heart_rate")

            laps.append(
                LapSegment(
                    index=int(fields.get("message_index", len(laps))) + 1,
                    start=start,
                    end=end,
                    intensity=normalize_intensity(fields.get("intensity")),
                    start_distance_m=distance_at(start),
                    total_elapsed_s=float(total_elapsed)
                    if isinstance(total_elapsed, (int, float))
                    else None,
                    total_distance_m=float(total_distance)
                    if isinstance(total_distance, (int, float))
                    else None,
                    avg_speed_m_s=float(avg_speed)
                    if isinstance(avg_speed, (int, float))
                    else None,
                    avg_power_w=int(avg_power)
                    if isinstance(avg_power, (int, float))
                    else None,
                    avg_cadence_spm=int(avg_cadence)
                    if isinstance(avg_cadence, (int, float))
                    else None,
                    avg_hr_bpm=int(avg_hr)
                    if isinstance(avg_hr, (int, float))
                    else None,
                )
            )

        laps.sort(key=lambda l: l.start)
        return ParsedData(samples=samples, laps=laps or None)
    raise ValueError(f"Unsupported data file extension: {ext} (expected .fit)")


# -----------------------------
# Video probing
# -----------------------------


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


def extract_creation_time_tag(ffprobe_json: dict) -> str | None:
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
        if not re.search(r"(Z|[+-]\\d{2}:?\\d{2})$", tag.strip()):
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


def compute_visibility_range(
    *,
    start_times: list[float],
    end_times: list[float],
    video_duration: float | None,
    laps: list[LapSegment] | None,
    t0: datetime,
    offset_seconds: float,
) -> tuple[float, float]:
    has_overlap = False
    first_visible: float | None = None
    last_visible: float | None = None

    for st, et in zip(start_times, end_times):
        if et <= 0:
            continue
        if video_duration is not None and st >= video_duration:
            break
        st_clip = max(0.0, st)
        et_clip = min(et, video_duration) if video_duration is not None else et
        if et_clip <= st_clip:
            continue
        first_visible = st_clip if first_visible is None else min(first_visible, st_clip)
        last_visible = et_clip if last_visible is None else max(last_visible, et_clip)
        has_overlap = True

    # Include lap-based overlays too (REST intervals can have sparse/no records).
    if laps:
        for lap in laps:
            vt_start = (lap.start - t0).total_seconds() + offset_seconds
            vt_end = (lap.end - t0).total_seconds() + offset_seconds
            if vt_end <= 0:
                continue
            if video_duration is not None and vt_start >= video_duration:
                continue
            st = max(0.0, vt_start)
            et = min(vt_end, video_duration) if video_duration is not None else vt_end
            if et <= st:
                continue
            first_visible = st if first_visible is None else min(first_visible, st)
            last_visible = et if last_visible is None else max(last_visible, et)
            has_overlap = True

    if not has_overlap:
        raise ValueError(
            "No FIT samples overlap the video timeline. "
            "Check the video's creation_time tag or adjust with --offset/--anchor."
        )
    if first_visible is None or last_visible is None:
        raise RuntimeError("Internal error: visibility range not computed despite overlap.")

    return first_visible, last_visible


# -----------------------------
# ASS generation
# -----------------------------


def generate_ass(
    samples: list[Sample],
    out_ass: str,
    *,
    video_w: int,
    video_h: int,
    video_duration: float | None,
    offset_seconds: float,
    label_font: str,
    value_font: str,
    value_fs: int | None,
    left_margin: int | None,
    top_margin: int | None,
    bottom_margin: int | None,
    box_alpha: int,
    interpolate: bool = True,
    laps: list[LapSegment] | None = None,
) -> None:
    """
    Create a PM5-inspired overlay matching `input/pm5_overlay_modern_grid.ass`:
      - single bottom-left panel with 2 rows x 3 cols:
        TIME / SPLIT / SPM
        METERS / WATTS / BPM

    offset_seconds is the computed (or overridden) shift that maps:
      video_time = (sample_time - t0) + offset_seconds
    where t0 is the selected anchor sample time (samples[0].t).
    """
    if not samples:
        raise ValueError("No samples found.")

    if laps:
        laps = sorted(laps, key=lambda l: (l.start, l.index))

    if video_w <= 0 or video_h <= 0:
        # If ffprobe couldn't determine, choose a reasonable default
        video_w, video_h = 1280, 720

    # Baseline: the sample ASS was authored at 1920x1080 with these values.
    scale_x = video_w / 1920.0
    scale_y = video_h / 1080.0

    if value_fs is None:
        value_fs = max(18, int(round(52 * scale_y)))
    label_fs = max(10, int(round(24 * scale_y)))
    outline_4 = max(1, int(round(4 * scale_y)))
    shadow_2 = max(0, int(round(2 * scale_y)))

    if left_margin is None:
        left_margin = max(10, int(round(20 * scale_x)))
    if top_margin is None:
        top_margin = 0
    if bottom_margin is None:
        bottom_margin = max(10, int(round(20 * scale_y)))

    # Box size and placement (match sample proportions).
    box_w = max(1, int(round(420 * scale_x)))
    box_h = max(1, int(round(190 * scale_y)))

    origin_x = int(left_margin)
    if top_margin > 0:
        origin_y = int(top_margin)
    else:
        origin_y = int(video_h - bottom_margin - box_h)
    origin_y = max(0, origin_y)

    # Compute per-sample video times
    t0 = samples[0].t
    start_times: list[float] = []
    for s in samples:
        vt = (s.t - t0).total_seconds() + offset_seconds
        start_times.append(vt)

    end_times: list[float] = start_times[1:] + [start_times[-1] + 1.0]

    # Determine overlay visibility range
    first_visible, last_visible = compute_visibility_range(
        start_times=start_times,
        end_times=end_times,
        video_duration=video_duration,
        laps=laps,
        t0=t0,
        offset_seconds=offset_seconds,
    )

    # Clamp alpha to 0..255
    box_alpha = max(0, min(255, int(box_alpha)))

    lines: list[str] = []
    lines.append("[Script Info]")
    lines.append("Title: Concept2 PM5 Rowing Overlay (Modern)")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {video_w}")
    lines.append(f"PlayResY: {video_h}")
    lines.append("WrapStyle: 0")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("YCbCr Matrix: TV.709")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    # Match `input/pm5_overlay_modern_grid.ass` styles (fonts/sizes are scaled to resolution).
    lines.append(
        "Style: Box,Arial,1,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
    )
    lines.append(
        f"Style: Label,{label_font},{label_fs},&H88FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,1,0,1,0,0,8,0,0,0,1"
    )
    for name, color in (
        ("Time", "&H00FFFFFF"),
        ("Split", "&H00FFCC00"),
        ("SPM", "&H0066AAFF"),
        ("Distance", "&H00FFFFFF"),
        ("Watts", "&H0088FF88"),
        ("HeartRate", "&H004444FF"),
    ):
        lines.append(
            f"Style: {name},{value_font},{value_fs},{color},&H00FFFFFF,&HAA0B0B0B,&H66000000,-1,0,0,0,100,100,0,0,1,{outline_4},{shadow_2},8,0,0,0,1"
        )
    lines.append("")
    lines.append("[Events]")
    lines.append(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )

    def px(x_1080p: int) -> int:
        return int(round(x_1080p * scale_x))

    def py(y_1080p: int) -> int:
        return int(round(y_1080p * scale_y))

    blur_scale = min(scale_x, scale_y)
    blur_12 = max(0, int(round(12 * blur_scale)))
    blur_1 = max(0, int(round(1 * blur_scale)))
    bord_2 = max(0, int(round(2 * blur_scale)))
    hdr_bord = max(1, int(round(2 * blur_scale)))
    hdr_blur = max(0, int(round(1 * blur_scale)))
    hdr_fx = f"\\bord{hdr_bord}\\3c&H000000&\\3a&H88&\\shad0\\blur{hdr_blur}"

    alpha_main = box_alpha  # default ~0x70 to match sample
    alpha_shadow = max(0, min(255, alpha_main + 0x20))  # 0x90 when alpha_main=0x70
    alpha_border = max(0, min(255, alpha_main - 0x2B))  # 0x45 when alpha_main=0x70

    # Static background (shadow + panel) + grid lines spanning the overlay range.
    shadow_dx = px(6)
    shadow_dy = py(6)
    shadow_draw = (
        f"{{\\pos({origin_x + shadow_dx},{origin_y + shadow_dy})\\p1\\c&H000000&\\alpha&H{alpha_shadow:02X}&\\blur{blur_12}}}"
        f"m 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}{{\\p0}}"
    )
    lines.append(
        f"Dialogue: 0,{ass_time(first_visible)},{ass_time(last_visible)},Box,,0,0,0,,{shadow_draw}"
    )

    panel_draw = (
        f"{{\\pos({origin_x},{origin_y})\\p1\\c&H101010&\\alpha&H{alpha_border:02X}&\\blur{blur_1}\\bord{bord_2}\\3c&HFFFFFF&\\3a&HD0&}}"
        f"m 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}{{\\p0}}"
    )
    lines.append(
        f"Dialogue: 1,{ass_time(first_visible)},{ass_time(last_visible)},Box,,0,0,0,,{panel_draw}"
    )

    header_draw = (
        f"{{\\pos({origin_x},{origin_y})\\p1\\c&H1E1E1E&\\alpha&H{alpha_main:02X}&}}"
        f"m 0 0 l {box_w} 0 l {box_w} {py(95)} l 0 {py(95)}{{\\p0}}"
    )
    lines.append(
        f"Dialogue: 2,{ass_time(first_visible)},{ass_time(last_visible)},Box,,0,0,0,,{header_draw}"
    )

    def rect_path(x1: int, y1: int, x2: int, y2: int) -> str:
        return f"m {x1} {y1} l {x2} {y1} l {x2} {y2} l {x1} {y2}"

    grid_shapes = [
        # vertical separators
        ("&HFFFFFF&", "D8", 3, rect_path(px(140), py(12), px(142), py(178))),
        ("&HFFFFFF&", "D8", 3, rect_path(px(280), py(12), px(282), py(178))),
        # row divider
        ("&HFFFFFF&", "E0", 3, rect_path(px(12), py(95), px(408), py(97))),
        # accent bars row 1
        (
            "&HFFFFFF&",
            f"{alpha_main:02X}",
            4,
            rect_path(px(36), py(36), px(104), py(39)),
        ),
        (
            "&HFFCC00&",
            f"{alpha_main:02X}",
            4,
            rect_path(px(176), py(36), px(244), py(39)),
        ),
        (
            "&H66AAFF&",
            f"{alpha_main:02X}",
            4,
            rect_path(px(316), py(36), px(384), py(39)),
        ),
        # accent bars row 2
        (
            "&HFFFFFF&",
            f"{alpha_main:02X}",
            4,
            rect_path(px(36), py(126), px(104), py(129)),
        ),
        (
            "&H88FF88&",
            f"{alpha_main:02X}",
            4,
            rect_path(px(176), py(126), px(244), py(129)),
        ),
        (
            "&H4444FF&",
            f"{alpha_main:02X}",
            4,
            rect_path(px(316), py(126), px(384), py(129)),
        ),
    ]

    for color, alpha_hex, layer, path in grid_shapes:
        blur = f"\\blur{blur_1}" if layer == 3 else ""
        lines.append(
            f"Dialogue: {layer},{ass_time(first_visible)},{ass_time(last_visible)},Box,,0,0,0,,"
            f"{{\\pos({origin_x},{origin_y})\\p1\\c{color}\\alpha&H{alpha_hex}&{blur}}}{path}{{\\p0}}"
        )

    # Column anchors (baseline positions inside the box, relative to origin at 20px).
    col1_x = origin_x + px(70)
    col2_x = origin_x + px(210)
    col3_x = origin_x + px(350)
    label_row1_y = origin_y + py(12)
    value_row1_y = origin_y + py(38)
    label_row2_y = origin_y + py(102)
    value_row2_y = origin_y + py(128)
    header_y = max(0, origin_y - py(28))

    labels = [
        ("TIME", "&HFFFFFF&", col1_x, label_row1_y),
        ("SPLIT", "&HFFCC00&", col2_x, label_row1_y),
        ("S/M", "&H66AAFF&", col3_x, label_row1_y),
        ("METERS", "&HFFFFFF&", col1_x, label_row2_y),
        ("WATTS", "&H88FF88&", col2_x, label_row2_y),
        ("BPM", "&H4444FF&", col3_x, label_row2_y),
    ]
    for text, color, x, y in labels:
        lines.append(
            f"Dialogue: 5,{ass_time(first_visible)},{ass_time(last_visible)},Label,,0,0,0,,{{\\pos({x},{y})\\c{color}}}{text}"
        )

    # Per-sample values (6 lines per sample so each value can have its own style/color).
    upsampled_laps: set[int] = set()
    if laps:

        def video_time_to_abs(vt: float) -> datetime:
            return t0 + timedelta(seconds=(vt - offset_seconds))

        lap_starts = [l.start for l in laps]

        prev_active_map: dict[int, LapSegment | None] = {}
        last_active: LapSegment | None = None
        for lap in laps:
            prev_active_map[lap.index] = last_active
            if (lap.intensity or "").lower() == "active":
                last_active = lap

        def lap_for_abs(t: datetime) -> LapSegment | None:
            idx = bisect_left(lap_starts, t)
            for i in (idx - 1, idx):
                if 0 <= i < len(laps) and laps[i].start <= t < laps[i].end:
                    return laps[i]
            return None

        def lap_video_range(lap: LapSegment) -> tuple[float, float] | None:
            """
            Returns (start_v, end_v) clamped to the video timeline, or None if fully outside.
            """
            vt_start = (lap.start - t0).total_seconds() + offset_seconds
            vt_end = (lap.end - t0).total_seconds() + offset_seconds
            if vt_end <= 0:
                return None
            if video_duration is not None and vt_start >= video_duration:
                return None
            start_v = max(0.0, vt_start)
            end_v = (
                min(vt_end, video_duration) if video_duration is not None else vt_end
            )
            return (start_v, end_v) if end_v > start_v else None

        def emit_distance_dialogue(a: float, b: float, meters: int) -> None:
            if b <= a:
                return
            lines.append(
                f"Dialogue: 9,{ass_time(a)},{ass_time(b)},Distance,,0,0,0,,{{\\pos({col1_x},{value_row2_y})}}{meters:d}"
            )

        # Per-second TIME during WORK (smooth ticking, like REST).
        for lap in (laps if interpolate else []):
            if (lap.intensity or "").lower() == "rest":
                continue

            if (rng := lap_video_range(lap)) is None:
                continue
            start_v, end_v = rng

            t = start_v
            while t < end_v:
                tn = min(end_v, math.floor(t + 1.0))
                if tn <= t:
                    tn = min(end_v, t + 1.0)

                abs_t = video_time_to_abs(t)
                lap_elapsed = max(0.0, (abs_t - lap.start).total_seconds())
                lap_elapsed_str = format_elapsed(lap_elapsed)

                # Layer above per-sample values so this always "wins".
                lines.append(
                    f"Dialogue: 8,{ass_time(t)},{ass_time(tn)},Time,,0,0,0,,{{\\pos({col1_x},{value_row1_y})}}{lap_elapsed_str}"
                )
                t = tn

        # Per-meter DISTANCE during WORK (interpolated between FIT samples).
        # This makes the METERS field increment smoothly even if FIT records are sparse.
        ts_abs = [s.t for s in samples]
        dist_abs = [s.distance_m for s in samples]

        def interpolate_distance_at(t: datetime) -> float | None:
            idx = bisect_left(ts_abs, t)
            if idx <= 0:
                return dist_abs[0] if dist_abs else None
            if idx >= len(ts_abs):
                return dist_abs[-1] if dist_abs else None
            t0_i, t1_i = ts_abs[idx - 1], ts_abs[idx]
            d0_i, d1_i = dist_abs[idx - 1], dist_abs[idx]
            if d0_i is None or d1_i is None:
                return d1_i if d1_i is not None else d0_i
            dt = (t1_i - t0_i).total_seconds()
            if dt <= 0:
                return d1_i
            alpha = (t - t0_i).total_seconds() / dt
            return float(d0_i + (d1_i - d0_i) * alpha)

        def estimate_distance_at(
            t: datetime, *, lap: LapSegment, lap_start_dist: float, lap_end_dist: float
        ) -> float:
            if not ts_abs or not dist_abs:
                return lap_start_dist

            if t <= ts_abs[0]:
                d0 = dist_abs[0]
                if d0 is not None and t >= lap.start:
                    dt = (ts_abs[0] - lap.start).total_seconds()
                    if dt > 0:
                        alpha = (t - lap.start).total_seconds() / dt
                        return float(
                            lap_start_dist + (float(d0) - lap_start_dist) * alpha
                        )
                return lap_start_dist

            if t >= ts_abs[-1]:
                d1 = dist_abs[-1]
                if d1 is not None and t <= lap.end:
                    dt = (lap.end - ts_abs[-1]).total_seconds()
                    if dt > 0:
                        alpha = (t - ts_abs[-1]).total_seconds() / dt
                        return float(float(d1) + (lap_end_dist - float(d1)) * alpha)
                return lap_end_dist

            d = interpolate_distance_at(t)
            return float(d) if d is not None else lap_start_dist

        for lap in (laps if interpolate else []):
            if (lap.intensity or "").lower() == "rest":
                continue

            if (rng := lap_video_range(lap)) is None:
                continue
            start_v, end_v = rng

            # Find sample range for this lap.
            i0 = bisect_left(ts_abs, lap.start)
            i1 = bisect_left(ts_abs, lap.end)
            if i0 >= len(ts_abs):
                continue
            i1 = max(i0 + 1, min(i1 + 1, len(ts_abs)))

            # Prefer the lap message's start distance to avoid pre-lap interpolation artifacts.
            lap_start_dist = lap.start_distance_m
            if lap_start_dist is None:
                lap_start_dist = interpolate_distance_at(lap.start)
            if lap_start_dist is None:
                lap_start_dist = dist_abs[i0]
            if lap_start_dist is None:
                continue
            lap_start_dist = float(lap_start_dist)

            lap_total_m = None
            if lap.total_distance_m is not None and lap.total_distance_m > 0:
                lap_total_m = int(round(lap.total_distance_m))
            else:
                lap_end_dist = interpolate_distance_at(lap.end)
                if lap_end_dist is not None:
                    lap_total_m = int(
                        round(max(0.0, float(lap_end_dist) - lap_start_dist))
                    )
            if lap_total_m is None or lap_total_m <= 0:
                continue

            # The Distance field shows lap meters (relative to lap start).
            # Layer 9 sits above per-sample values so interpolation always wins.
            end_clip = max(
                start_v, end_v - 0.01
            )  # avoid inclusive-end overlap with REST
            final_hold_s = 0.05
            final_change_v = max(start_v, end_clip - final_hold_s)
            # Bias meter ticks towards the end of each record interval.
            # FIT distance samples often represent "distance after the stroke", so linear interpolation
            # can make meters tick too early within a stroke.
            interval_bias_start = 0.4  # 0.0 = linear, 1.0 = all-at-end

            lap_end_dist = lap_start_dist + float(lap_total_m)
            abs_seg_start = max(lap.start, video_time_to_abs(start_v))
            if abs_seg_start >= lap.end:
                continue
            prev_t = abs_seg_start
            prev_d = estimate_distance_at(
                prev_t,
                lap=lap,
                lap_start_dist=lap_start_dist,
                lap_end_dist=lap_end_dist,
            )
            prev_d = min(max(prev_d, lap_start_dist), lap_end_dist)
            eps = 1e-6
            m_start_v = max(0, int(math.floor((prev_d - lap_start_dist) + eps)))
            # Keep the final meter tick reserved for the end of the lap.
            m_start_v = min(m_start_v, max(0, lap_total_m - 1))

            # Build a list of (vt, meters) change points.
            changes: list[tuple[float, int]] = [(start_v, m_start_v)]

            # Iterate through distance samples in-lap, plus a synthetic endpoint at lap end.
            points: list[tuple[datetime, float]] = []
            j0 = bisect_left(ts_abs, prev_t)
            for i in range(j0, i1):
                d_i = dist_abs[i]
                if d_i is None:
                    continue
                points.append((ts_abs[i], float(d_i)))
            points.append((lap.end, lap_end_dist))
            points.sort(key=lambda x: x[0])
            if not points:
                continue

            for t_i, d_i in points:
                if t_i <= prev_t:
                    continue
                if d_i <= prev_d:
                    prev_t = t_i
                    prev_d = d_i
                    continue

                rel0 = max(0.0, prev_d - lap_start_dist)
                rel1 = max(0.0, d_i - lap_start_dist)
                m0 = int(math.floor(rel0 + eps))
                m1 = int(math.floor(rel1 + eps))
                m1 = min(m1, max(0, lap_total_m - 1))
                if m1 > m0:
                    for m in range(m0 + 1, m1 + 1):
                        target = lap_start_dist + float(m)
                        alpha = (target - prev_d) / (d_i - prev_d)
                        alpha = max(0.0, min(1.0, alpha))
                        alpha_b = (
                            interval_bias_start + (1.0 - interval_bias_start) * alpha
                        )
                        abs_t = prev_t + timedelta(
                            seconds=(t_i - prev_t).total_seconds() * alpha_b
                        )
                        vt = (abs_t - t0).total_seconds() + offset_seconds
                        if vt >= final_change_v:
                            break
                        changes.append((vt, m))

                prev_t = t_i
                prev_d = d_i

            # Delay showing the final lap distance until the lap end to avoid finishing early.
            changes.append((final_change_v, lap_total_m))

            # Emit segments between change points, clamped to the lap.
            changes.sort(key=lambda x: x[0])
            # Enforce monotonic timestamps and drop degenerate/duplicate changes.
            dedup: list[tuple[float, int]] = []
            last_vt = None
            last_m = None
            min_dt = 0.011  # 1 centisecond + epsilon (ASS timestamp resolution)
            for vt, m in changes:
                if vt >= end_clip:
                    continue
                if last_vt is not None and vt <= last_vt + min_dt:
                    vt = last_vt + min_dt
                if vt >= end_clip:
                    continue
                if last_m is not None and m <= last_m:
                    continue
                dedup.append((vt, m))
                last_vt = vt
                last_m = m
            changes = dedup

            did_emit = False
            for (vt, m), (vt2, _) in zip(changes, changes[1:] + [(end_clip, -1)]):
                a = max(start_v, vt)
                b = min(end_clip, vt2)
                if b > a:
                    emit_distance_dialogue(a, b, m)
                    did_emit = True

            if did_emit:
                upsampled_laps.add(lap.index)

        # Rest backdrop tint (changes the panel background color during rest intervals).
        rest_backdrop_color = "&H5A4636&"  # slightly brighter, blue-tinted (BGR)
        rest_border_color = "&HFFCC00&"  # bright blue/cyan (matches Split accent)
        rest_border_alpha = "40"  # 00 opaque .. FF transparent
        rest_border_th = max(1, int(round(3 * blur_scale)))
        for lap in laps:
            if (lap.intensity or "").lower() != "rest":
                continue

            if (rng := lap_video_range(lap)) is None:
                continue
            st_h, et_h = rng

            rest_backdrop_draw = (
                f"{{\\pos({origin_x},{origin_y})\\p1\\c{rest_backdrop_color}\\alpha&H{alpha_main:02X}&}}"
                f"m 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}{{\\p0}}"
            )
            # Layer 2 sits above the normal header strip but below grid/text.
            lines.append(
                f"Dialogue: 2,{ass_time(st_h)},{ass_time(et_h)},Box,,0,0,0,,{rest_backdrop_draw}"
            )

            # Bright border during rest (drawn as thin filled rectangles).
            top = rect_path(0, 0, box_w, rest_border_th)
            bottom = rect_path(0, box_h - rest_border_th, box_w, box_h)
            left = rect_path(0, 0, rest_border_th, box_h)
            right = rect_path(box_w - rest_border_th, 0, box_w, box_h)
            for path in (top, bottom, left, right):
                lines.append(
                    f"Dialogue: 4,{ass_time(st_h)},{ass_time(et_h)},Box,,0,0,0,,"
                    f"{{\\pos({origin_x},{origin_y})\\p1\\c{rest_border_color}\\alpha&H{rest_border_alpha}&\\blur{blur_1}}}"
                    f"{path}{{\\p0}}"
                )

        # Lap header (ensures lap number/state is visible even if samples are sparse).
        for lap in laps:
            if (rng := lap_video_range(lap)) is None:
                continue
            st_h, et_h = rng

            state = "REST" if (lap.intensity or "").lower() == "rest" else "WORK"
            header_left_text = f"{{\\an7\\pos({origin_x + px(12)},{header_y})\\c&HFFFFFF&{hdr_fx}}}LAP {lap.index:02d} \u00b7 {state}"
            lines.append(
                f"Dialogue: 10,{ass_time(st_h)},{ass_time(et_h)},Label,,0,0,0,,{header_left_text}"
            )

        # Per-second rest overlays (FIT often has sparse/no records during rest).
        for lap in laps:
            if (lap.intensity or "").lower() != "rest":
                continue

            if (rng := lap_video_range(lap)) is None:
                continue
            start_v, end_v = rng

            prev_active = prev_active_map.get(lap.index)

            prev_pace = "--:--.-"
            prev_spm = "--"
            prev_watts = "---"
            prev_hr = "---"
            if prev_active is not None:
                if prev_active.avg_speed_m_s and prev_active.avg_speed_m_s > 0:
                    prev_pace = format_pace(500.0 / prev_active.avg_speed_m_s)
                if prev_active.avg_cadence_spm is not None:
                    prev_spm = f"{prev_active.avg_cadence_spm:d}"
                if prev_active.avg_power_w is not None:
                    prev_watts = f"{prev_active.avg_power_w:d}"
                if prev_active.avg_hr_bpm is not None:
                    prev_hr = f"{prev_active.avg_hr_bpm:d}"

            # During REST, show the previous WORK interval summary (like the PM5 intervals table).
            if prev_active is not None:
                prev_elapsed_s = prev_active.total_elapsed_s
                if prev_elapsed_s is None:
                    prev_elapsed_s = (
                        prev_active.end - prev_active.start
                    ).total_seconds()
                time_str = format_elapsed(prev_elapsed_s)
                meters_str = (
                    f"{int(round(prev_active.total_distance_m)):d}"
                    if prev_active.total_distance_m is not None
                    else "---"
                )
                split_str = prev_pace
                spm_str = prev_spm
                watts_str = prev_watts
                hr_str = prev_hr
            else:
                time_str = "--:--"
                meters_str = "---"
                split_str = "--:--.-"
                spm_str = "--"
                watts_str = "---"
                hr_str = "---"

            # Constant fields over the whole REST lap (minimize dialogue spam).
            # Use high layer so these sit above any lingering work sample events.
            lines.append(
                f"Dialogue: 20,{ass_time(start_v)},{ass_time(end_v)},Time,,0,0,0,,{{\\pos({col1_x},{value_row1_y})}}{time_str}"
            )
            lines.append(
                f"Dialogue: 20,{ass_time(start_v)},{ass_time(end_v)},Split,,0,0,0,,{{\\pos({col2_x},{value_row1_y})}}{split_str}"
            )
            lines.append(
                f"Dialogue: 20,{ass_time(start_v)},{ass_time(end_v)},SPM,,0,0,0,,{{\\pos({col3_x},{value_row1_y})}}{spm_str}"
            )
            lines.append(
                f"Dialogue: 20,{ass_time(start_v)},{ass_time(end_v)},Distance,,0,0,0,,{{\\pos({col1_x},{value_row2_y})}}{meters_str}"
            )
            lines.append(
                f"Dialogue: 20,{ass_time(start_v)},{ass_time(end_v)},Watts,,0,0,0,,{{\\pos({col2_x},{value_row2_y})}}{watts_str}"
            )
            lines.append(
                f"Dialogue: 20,{ass_time(start_v)},{ass_time(end_v)},HeartRate,,0,0,0,,{{\\pos({col3_x},{value_row2_y})}}{hr_str}"
            )

            # Per-second REST countdown only (ticks smoothly).
            rest_total = lap.total_elapsed_s
            if rest_total is None:
                rest_total = (lap.end - lap.start).total_seconds()

            t = start_v
            while t < end_v:
                tn = min(end_v, math.floor(t + 1.0))
                if tn <= t:
                    tn = min(end_v, t + 1.0)

                abs_t = video_time_to_abs(t)
                lap_elapsed = max(0.0, (abs_t - lap.start).total_seconds())
                rest_remaining = max(0.0, rest_total - lap_elapsed)
                header_right_text = f"{{\\an9\\pos({origin_x + box_w - px(12)},{header_y})\\c&HFFFFFF&{hdr_fx}}}REST {format_remaining(rest_remaining)}"
                lines.append(
                    f"Dialogue: 21,{ass_time(t)},{ass_time(tn)},Label,,0,0,0,,{header_right_text}"
                )

                t = tn

    for s, st, et in zip(samples, start_times, end_times):
        if et <= 0:
            continue
        if video_duration is not None and st >= video_duration:
            break

        st_clip = max(0.0, st)
        et_clip = et
        if video_duration is not None:
            et_clip = min(et_clip, video_duration)
        if et_clip <= st_clip:
            continue

        current_lap: LapSegment | None = lap_for_abs(s.t) if laps else None

        lap_intensity = (current_lap.intensity if current_lap else "active").lower()
        is_rest = lap_intensity == "rest"

        if is_rest and laps:
            # Rest laps are rendered per-second above.
            continue
        if current_lap is not None:
            # Prevent a sparse/long sample interval from bleeding into the next lap (e.g. into REST),
            # which would cause WORK values to overlap with REST overlays.
            lap_end_vt = (current_lap.end - t0).total_seconds() + offset_seconds
            # Some renderers treat end timestamps as inclusive; end slightly before the lap boundary.
            et_clip = min(et_clip, lap_end_vt - 0.01)
            if et_clip <= st_clip:
                continue

        if current_lap:
            lap_elapsed = (s.t - current_lap.start).total_seconds()
            lap_elapsed_str = format_elapsed(lap_elapsed)
            if current_lap.start_distance_m is not None and s.distance_m is not None:
                lap_meters = max(
                    0, int(round(s.distance_m - current_lap.start_distance_m))
                )
                meters_str = f"{lap_meters:d}"
            else:
                meters_str = "---"
        else:
            elapsed = (s.t - t0).total_seconds()
            lap_elapsed_str = format_elapsed(elapsed)
            meters_str = (
                f"{int(round(s.distance_m)):d}" if s.distance_m is not None else "---"
            )

        pace_sec = 500.0 / s.speed if (s.speed is not None and s.speed > 0) else None
        pace_str = format_pace(pace_sec)

        spm_str = f"{s.cadence:d}" if s.cadence is not None else "--"
        watts_str = f"{s.watts:d}" if s.watts is not None else "---"
        hr_str = f"{s.hr:d}" if s.hr is not None else "---"

        if current_lap is None or not laps or not interpolate:
            # When FIT laps exist, the WORK TIME value is rendered per-second above.
            lines.append(
                f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Time,,0,0,0,,{{\\pos({col1_x},{value_row1_y})}}{lap_elapsed_str}"
            )
        lines.append(
            f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Split,,0,0,0,,{{\\pos({col2_x},{value_row1_y})}}{pace_str}"
        )
        lines.append(
            f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},SPM,,0,0,0,,{{\\pos({col3_x},{value_row1_y})}}{spm_str}"
        )
        if current_lap is None or not laps or current_lap.index not in upsampled_laps:
            lines.append(
                f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Distance,,0,0,0,,{{\\pos({col1_x},{value_row2_y})}}{meters_str}"
            )
        lines.append(
            f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Watts,,0,0,0,,{{\\pos({col2_x},{value_row2_y})}}{watts_str}"
        )
        lines.append(
            f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},HeartRate,,0,0,0,,{{\\pos({col3_x},{value_row2_y})}}{hr_str}"
        )

    Path(out_ass).write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# ffmpeg burn-in
# -----------------------------


def burn_in(
    video_in: str,
    ass_path: str,
    video_out: str,
    *,
    ffmpeg_bin: str,
    crf: int,
    preset: str,
    copy_audio: bool,
) -> None:
    """
    Burn the ASS subtitles into the video using libass (ffmpeg).

    We run ffmpeg with cwd set to the ASS directory so the filter can reference the file
    without Windows drive-letter escaping pain.
    """
    ass_abs = os.path.abspath(ass_path)
    ass_dir = os.path.dirname(ass_abs) or "."
    ass_name = os.path.basename(ass_abs)

    # Escape for ffmpeg filter syntax (not shell escaping).
    ass_name_escaped = ass_name
    for ch in ("\\", ":", "'", "[", "]", ",", ";", "="):
        ass_name_escaped = ass_name_escaped.replace(ch, f"\\{ch}")
    vf = f"ass=filename='{ass_name_escaped}'"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_in,
        "-vf",
        vf,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
    ]
    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    if Path(video_out).suffix.lower() in {".mp4", ".m4v"}:
        cmd += ["-movflags", "+faststart"]

    cmd += [video_out]

    proc = subprocess.Popen(
        cmd,
        cwd=ass_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    tail: deque[str] = deque(maxlen=80)
    for line in proc.stdout:
        sys.stderr.write(line)
        tail.append(line)
    code = proc.wait()
    if code != 0:
        last = "".join(tail).strip()
        msg = f"ffmpeg burn-in failed (code {code})."
        if last:
            msg += f"\nLast ffmpeg output:\n{last}"
        raise RuntimeError(msg)


# -----------------------------
# Alignment helpers
# -----------------------------


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


# -----------------------------
# CLI
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="c2_overlay.py",
        description="Create a PM5-style overlay (.ass subtitles) from a Concept2 FIT file and align it to a video using metadata timestamps.",
    )
    ap.add_argument("video", help="Input video file (mp4/mov/etc)")
    ap.add_argument("fit", help="Concept2 workout data file (.fit)")
    ap.add_argument(
        "-o",
        "--out-ass",
        default=None,
        help="Output .ass path (default: next to input video)",
    )

    ap.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Manual offset adjustment in seconds (added to the auto-computed alignment). "
        "Positive makes data appear later; negative earlier.",
    )
    ap.add_argument(
        "--font",
        default=None,
        help="Legacy alias: set both --label-font and --value-font to this font name.",
    )
    ap.add_argument(
        "--label-font",
        default="PragmataPro",
        help="Font for labels (must exist on your system)",
    )
    ap.add_argument(
        "--value-font",
        default="PragmataPro Mono",
        help="Font for values (must exist on your system)",
    )
    ap.add_argument(
        "--fontsize",
        type=int,
        default=None,
        help="Value font size (default: scaled from 52 @ 1080p)",
    )
    ap.add_argument(
        "--left-margin",
        type=int,
        default=None,
        help="Left margin in pixels (default: scaled from 20 @ 1080p)",
    )
    ap.add_argument(
        "--top-margin",
        type=int,
        default=None,
        help="Top margin in pixels; if set, positions the overlay from the top instead of the bottom.",
    )
    ap.add_argument(
        "--bottom-margin",
        type=int,
        default=None,
        help="Bottom margin in pixels (default: scaled from 20 @ 1080p)",
    )
    ap.add_argument(
        "--box-alpha",
        type=int,
        default=112,
        help="Background box transparency 0..255 (0=opaque, 255=fully transparent). Default: 112.",
    )
    ap.add_argument(
        "--no-interp",
        action="store_true",
        help="Disable per-second work time and per-meter distance interpolation (smaller ASS output).",
    )
    ap.add_argument(
        "--lint",
        action="store_true",
        help="Lint the generated .ass output and exit non-zero on errors.",
    )
    ap.add_argument(
        "--lint-strict",
        action="store_true",
        help="Like --lint, but also fails on warnings.",
    )

    ap.add_argument(
        "--burn-in",
        metavar="OUT_VIDEO",
        default=None,
        help="If set, burn the overlay into a new video using ffmpeg.",
    )
    ap.add_argument(
        "--crf", type=int, default=18, help="x264 CRF for burn-in (default: 18)"
    )
    ap.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset for burn-in (default: veryfast)",
    )
    ap.add_argument(
        "--reencode-audio",
        action="store_true",
        help="Re-encode audio to AAC instead of stream-copying it (use if -c:a copy fails).",
    )

    ap.add_argument(
        "--ffprobe-bin", default="ffprobe", help="Path to ffprobe (default: ffprobe)"
    )
    ap.add_argument(
        "--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg (default: ffmpeg)"
    )
    ap.add_argument(
        "--anchor",
        "--tcx-anchor",
        dest="anchor",
        choices=["start", "first-visible", "first-row-visible"],
        default="start",
        help="Which sample to treat as time 0 for overlay generation (default: start).",
    )

    args = ap.parse_args()

    # Tools check
    if shutil.which(args.ffprobe_bin) is None:
        print(f"ERROR: ffprobe not found: {args.ffprobe_bin}", file=sys.stderr)
        return 2
    if args.burn_in and shutil.which(args.ffmpeg_bin) is None:
        print(f"ERROR: ffmpeg not found: {args.ffmpeg_bin}", file=sys.stderr)
        return 2

    video_path = args.video
    data_path = args.fit
    out_ass = args.out_ass or str(Path(video_path).with_suffix(".ass"))

    # Parse data (.fit)
    try:
        parsed = parse_data_file(data_path)
    except Exception as e:
        print(f"ERROR: Could not parse data file: {data_path}\n{e}", file=sys.stderr)
        return 2
    samples_all = parsed.samples
    if not samples_all:
        print(f"ERROR: No samples found in data file: {data_path}", file=sys.stderr)
        return 2

    data_start = samples_all[0].t

    # Probe video
    w, h, duration, video_creation, source = get_video_metadata(
        video_path, ffprobe_bin=args.ffprobe_bin
    )

    # Choose anchor (t0) used for both alignment and displayed elapsed time.
    anchor_idx = choose_anchor_index(
        samples_all, video_start=video_creation, mode=args.anchor
    )
    samples = samples_all[anchor_idx:] if anchor_idx else samples_all
    anchor_time = samples[0].t

    # Auto offset: when does anchor occur on the video timeline?
    auto_offset = (anchor_time - video_creation).total_seconds()
    offset = auto_offset + float(args.offset)

    print("== Alignment ==")
    print(f"Video creation/start time (UTC): {video_creation.isoformat()}  [{source}]")
    if duration is not None:
        video_end = video_creation + timedelta(seconds=duration)
        print(
            f"Video end time (UTC):            {video_end.isoformat()}  [duration {duration:.2f} s]"
        )
    print(f"FIT file: {data_path}")
    print(f"FIT first timestamp (UTC):       {data_start.isoformat()}")
    delta0 = (data_start - video_creation).total_seconds()
    if abs(delta0) >= 1.0:
        when = "after" if delta0 > 0 else "before"
        print(
            f"FIT starts {abs(delta0):.1f} s {when} video start (based on absolute timestamps)."
        )
    first_row_visible = next(
        (s for s in samples_all if s.t >= video_creation and (s.cadence or 0) > 0), None
    )
    if first_row_visible is not None:
        tv = (first_row_visible.t - video_creation).total_seconds()
        print(
            f"First sample with cadence>0 during video: t={tv:.1f} s  [{first_row_visible.t.isoformat()}]"
        )
    if anchor_time != data_start:
        print(
            f"Data anchor ({args.anchor}) time (UTC): {anchor_time.isoformat()}  [idx {anchor_idx}]"
        )
    print(f"Auto offset (anchor - video_start): {auto_offset:+.3f} s")
    if args.offset:
        print(f"Manual adjustment: {args.offset:+.3f} s")
    print(f"Final offset used: {offset:+.3f} s")
    if duration is not None:
        print(f"Video: {w}x{h}, duration ~ {duration:.2f} s")
    else:
        print(f"Video: {w}x{h}, duration unknown")

    # Write ASS
    label_font = args.label_font
    value_font = args.value_font
    if args.font:
        label_font = args.font
        value_font = args.font
    try:
        generate_ass(
            samples=samples,
            out_ass=out_ass,
            video_w=w,
            video_h=h,
            video_duration=duration,
            offset_seconds=offset,
            label_font=label_font,
            value_font=value_font,
            value_fs=args.fontsize,
            left_margin=args.left_margin,
            top_margin=args.top_margin,
            bottom_margin=args.bottom_margin,
            box_alpha=args.box_alpha,
            interpolate=(not args.no_interp),
            laps=parsed.laps,
        )
    except Exception as e:
        print(f"ERROR: Could not generate ASS overlay:\n{e}", file=sys.stderr)
        return 2
    print(f"Wrote ASS overlay: {out_ass}")

    # Lint (optional)
    if args.lint or args.lint_strict:
        from c2_overlay.ass_lint import SEVERITY_RANK, lint_ass_file, print_issues

        issues = lint_ass_file(out_ass)
        if issues:
            print(f"== ASS Lint ({len(issues)} issue(s)) ==", file=sys.stderr)
            print_issues(issues)

        fail_on = "warn" if args.lint_strict else "error"
        fail_rank = SEVERITY_RANK[fail_on]
        if any(SEVERITY_RANK.get(i.severity, 0) >= fail_rank for i in issues):
            print(f"ERROR: ASS lint failed (fail-on {fail_on}).", file=sys.stderr)
            return 1

    # Burn-in (optional)
    if args.burn_in:
        print(f"Burning in subtitles to: {args.burn_in}")
        burn_in(
            video_in=video_path,
            ass_path=out_ass,
            video_out=args.burn_in,
            ffmpeg_bin=args.ffmpeg_bin,
            crf=args.crf,
            preset=args.preset,
            copy_audio=(not args.reencode_audio),
        )
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
