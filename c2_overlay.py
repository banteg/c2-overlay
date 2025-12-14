#!/usr/bin/env python3
"""
rowerg_overlay.py

Generate a Concept2 RowErg-style overlay (.ass subtitles) from a TCX file and a video.

What it does
- Reads the video's absolute start timestamp from metadata (ffprobe creation_time tag).
- Reads absolute timestamps from the TCX trackpoints.
- Computes the offset so TCX samples line up on the video timeline.
- Writes a PM5-inspired overlay in the top-left, with heart rate in the top-right.
- Optionally burns the overlay into a new video using ffmpeg.

Requirements
- Python 3.9+
- ffprobe + ffmpeg available on PATH (or pass --ffprobe-bin / --ffmpeg-bin)

Example
  python rowerg_overlay.py input.mp4 workout.tcx -o overlay.ass --burn-in output.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from fitparse import FitFile
except Exception:  # pragma: no cover
    FitFile = None  # type: ignore[misc,assignment]


# -----------------------------
# Helpers: time parsing/format
# -----------------------------

def parse_iso8601(s: str) -> datetime:
    """
    Parse an ISO-8601-ish timestamp into an aware datetime in UTC.

    Handles:
      - ...Z
      - ...+00:00
      - ...+0000
      - with or without fractional seconds
      - naive datetimes (assumed UTC)
    """
    s = s.strip()
    if not s:
        raise ValueError("empty datetime string")
    s = s.replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # Convert timezone offset like +0000 to +00:00
    m = re.match(r"^(.*)([+-]\d{2})(\d{2})$", s)
    if m and ":" not in (m.group(2) + m.group(3)):
        s = f"{m.group(1)}{m.group(2)}:{m.group(3)}"

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Last-ditch parse attempts
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            raise

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_elapsed(sec: float) -> str:
    """Format elapsed seconds like PM5: MM:SS or H:MM:SS."""
    if sec < 0:
        sec = 0.0
    total = int(round(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    # PM5 typically shows minutes without zero-padding (e.g. "0:03", not "00:03").
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


def format_pace(sec_per_500: Optional[float]) -> str:
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


# -----------------------------
# TCX parsing
# -----------------------------

@dataclass
class Sample:
    t: datetime
    distance_m: Optional[float] = None
    hr: Optional[int] = None
    cadence: Optional[int] = None  # usually stroke rate (SPM) in Concept2 TCX
    watts: Optional[int] = None
    speed: Optional[float] = None  # m/s


def parse_tcx(tcx_path: str) -> List[Sample]:
    """
    Parse a Concept2/TrainingCenterDatabase TCX file.

    We read all <Trackpoint> entries and extract:
      - Time (absolute)
      - DistanceMeters
      - HeartRateBpm/Value
      - Cadence (often SPM)
      - Extensions: Watts, Speed, StrokeRate (if present)
    """
    # Some Concept2 exports (or toolchains) prepend a UTF-8 BOM and/or whitespace
    # before the XML declaration, which trips up `ET.parse(...)`.
    raw = Path(tcx_path).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace").lstrip()
    root = ET.fromstring(text)

    samples: List[Sample] = []

    for tp in root.findall(".//{*}Trackpoint"):
        time_text = (tp.findtext("./{*}Time") or "").strip()
        if not time_text:
            continue
        try:
            t = parse_iso8601(time_text)
        except Exception:
            continue

        dist_text = (tp.findtext("./{*}DistanceMeters") or "").strip()
        distance_m = float(dist_text) if dist_text else None

        hr_text = (tp.findtext("./{*}HeartRateBpm/{*}Value") or "").strip()
        hr = int(hr_text) if hr_text.isdigit() else None

        cad_text = (tp.findtext("./{*}Cadence") or "").strip()
        cadence = int(cad_text) if cad_text.isdigit() else None

        watts: Optional[int] = None
        speed: Optional[float] = None
        stroke_rate: Optional[int] = None

        # Scan descendants for extensions (namespace-agnostic)
        for el in tp.iter():
            local = el.tag.split("}")[-1].lower()
            txt = (el.text or "").strip()
            if not txt:
                continue

            if local in ("watts", "power"):
                try:
                    watts = int(float(txt))
                except ValueError:
                    pass
            elif local == "speed":
                try:
                    speed = float(txt)
                except ValueError:
                    pass
            elif local in ("strokerate", "strokecadence"):
                try:
                    stroke_rate = int(float(txt))
                except ValueError:
                    pass

        if cadence is None and stroke_rate is not None:
            cadence = stroke_rate

        samples.append(Sample(t=t, distance_m=distance_m, hr=hr, cadence=cadence, watts=watts, speed=speed))

    samples.sort(key=lambda s: s.t)

    # Fill in missing speeds from distance/time deltas (m/s)
    for i in range(1, len(samples)):
        if samples[i].speed is None and samples[i].distance_m is not None and samples[i - 1].distance_m is not None:
            dt = (samples[i].t - samples[i - 1].t).total_seconds()
            dd = samples[i].distance_m - samples[i - 1].distance_m
            if dt > 0 and dd >= 0:
                samples[i].speed = dd / dt
    if samples and samples[0].speed is None and len(samples) > 1:
        samples[0].speed = samples[1].speed

    return samples


def parse_fit(fit_path: str) -> List[Sample]:
    if FitFile is None:
        raise RuntimeError("fitparse is not installed; add it to dependencies or install it to parse .fit files.")

    fit = FitFile(fit_path)
    samples: List[Sample] = []
    for msg in fit.get_messages("record"):
        fields = {f.name: f.value for f in msg}
        ts = fields.get("timestamp")
        if ts is None:
            continue
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

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

        samples.append(Sample(t=ts, distance_m=distance_m, hr=hr, cadence=cadence, watts=watts, speed=speed))

    samples.sort(key=lambda s: s.t)

    # Fill in missing speeds from distance/time deltas (m/s)
    for i in range(1, len(samples)):
        if samples[i].speed is None and samples[i].distance_m is not None and samples[i - 1].distance_m is not None:
            dt = (samples[i].t - samples[i - 1].t).total_seconds()
            dd = samples[i].distance_m - samples[i - 1].distance_m
            if dt > 0 and dd >= 0:
                samples[i].speed = dd / dt
    if samples and samples[0].speed is None and len(samples) > 1:
        samples[0].speed = samples[1].speed

    return samples


def parse_concept2_csv(csv_path: str) -> List[Sample]:
    """
    Parse Concept2 "result" CSV (stroke/summary-like samples).

    These files contain relative times that can reset across pieces/intervals.
    We stitch segments into a monotonic elapsed timeline and anchor it at a
    dummy epoch (UTC) so downstream rendering works.
    """
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    samples: List[Sample] = []

    time_base = 0.0
    dist_base = 0.0
    prev_time = None
    prev_dist = None

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            t_s = (row.get("Time (seconds)") or "").strip()
            d_m = (row.get("Distance (meters)") or "").strip()
            if not t_s or not d_m:
                continue
            try:
                t_rel = float(t_s)
                d_rel = float(d_m)
            except ValueError:
                continue

            if prev_time is not None and prev_dist is not None:
                if t_rel + 1e-6 < prev_time or d_rel + 1e-3 < prev_dist:
                    time_base += prev_time
                    dist_base += prev_dist
                    prev_time = None
                    prev_dist = None

            t_total = time_base + t_rel
            d_total = dist_base + d_rel

            watts = (row.get("Watts") or "").strip()
            watts_i = int(watts) if watts.isdigit() else None

            cadence = (row.get("Stroke Rate") or "").strip()
            cadence_i = int(cadence) if cadence.isdigit() else None

            hr = (row.get("Heart Rate") or "").strip()
            hr_i = int(hr) if hr.isdigit() else None

            pace = (row.get("Pace (seconds)") or "").strip()
            speed = None
            try:
                pace_s = float(pace)
                if pace_s > 0:
                    speed = 500.0 / pace_s
            except ValueError:
                pass

            samples.append(
                Sample(
                    t=epoch + timedelta(seconds=t_total),
                    distance_m=d_total,
                    hr=hr_i,
                    cadence=cadence_i,
                    watts=watts_i,
                    speed=speed,
                )
            )
            prev_time = t_rel
            prev_dist = d_rel

    samples.sort(key=lambda s: s.t)
    return samples


@dataclass(frozen=True)
class ParsedData:
    samples: List[Sample]
    timebase: str  # "absolute" | "relative"
    kind: str  # "tcx" | "fit" | "csv"


def parse_data_file(path: str) -> ParsedData:
    ext = Path(path).suffix.lower()
    if ext == ".tcx":
        return ParsedData(samples=parse_tcx(path), timebase="absolute", kind="tcx")
    if ext == ".fit":
        return ParsedData(samples=parse_fit(path), timebase="absolute", kind="fit")
    if ext == ".csv":
        return ParsedData(samples=parse_concept2_csv(path), timebase="relative", kind="csv")
    raise ValueError(f"Unsupported data file extension: {ext} (expected .tcx, .fit, or .csv)")


# -----------------------------
# Video probing
# -----------------------------

def run_ffprobe(video_path: str, ffprobe_bin: str) -> dict:
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_entries", "format=duration:format_tags:stream=width,height:stream_tags",
        "-select_streams", "v:0",
        video_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed (code {p.returncode}):\n{p.stderr.strip()}")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse ffprobe JSON output: {e}") from e


def extract_creation_time_tag(ffprobe_json: dict) -> Optional[str]:
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


def get_video_metadata(video_path: str, ffprobe_bin: str) -> Tuple[int, int, Optional[float], datetime, str]:
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
        try:
            creation_dt = parse_iso8601(tag)
        except Exception:
            creation_dt = None

    if creation_dt is None:
        # Fallback: filesystem mtime (UTC). Not always accurate, but better than nothing.
        ts = os.path.getmtime(video_path)
        creation_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        source = "filesystem_mtime_utc"

    return w, h, duration, creation_dt, source


# -----------------------------
# ASS generation
# -----------------------------

def generate_ass(
    samples: List[Sample],
    out_ass: str,
    *,
    video_w: int,
    video_h: int,
    video_duration: Optional[float],
    offset_seconds: float,
    label_font: str,
    value_font: str,
    value_fs: Optional[int],
    left_margin: Optional[int],
    top_margin: Optional[int],
    bottom_margin: Optional[int],
    box_alpha: int,
) -> None:
    """
    Create a PM5-inspired overlay matching `input/pm5_overlay_modern_grid.ass`:
      - single bottom-left panel with 2 rows x 3 cols:
        TIME / SPLIT / SPM
        METERS / WATTS / BPM

    offset_seconds is the computed (or overridden) shift that maps:
      video_time = (sample_time - tcx_first_time) + offset_seconds
    """
    if not samples:
        raise ValueError("No TCX trackpoints found.")

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
    start_times: List[float] = []
    for s in samples:
        vt = (s.t - t0).total_seconds() + offset_seconds
        start_times.append(vt)

    end_times: List[float] = start_times[1:] + [start_times[-1] + 1.0]

    # Determine overlay visibility range
    first_visible = None
    last_visible = None
    for st, et in zip(start_times, end_times):
        if et <= 0:
            continue
        if video_duration is not None and st >= video_duration:
            break
        if first_visible is None:
            first_visible = max(0.0, st)
        last_visible = et if last_visible is None else max(last_visible, et)

    if first_visible is None:
        first_visible = 0.0
    if last_visible is None:
        last_visible = video_duration if video_duration is not None else (end_times[-1] if end_times else 0.0)
    if video_duration is not None:
        last_visible = min(last_visible, video_duration)

    # Clamp alpha to 0..255
    box_alpha = max(0, min(255, int(box_alpha)))
    box_a = f"{box_alpha:02X}"  # for \alpha

    # ASS colours are &HAABBGGRR (alpha first)
    # Sample palette:
    # - text colors include alpha in the colour literals
    # - vector box fill uses \c + \alpha override
    black = "&H00000000"

    lines: List[str] = []
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
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    # Match `input/pm5_overlay_modern_grid.ass` styles (fonts/sizes are scaled to resolution).
    lines.append("Style: Box,Arial,1,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1")
    lines.append(f"Style: Label,{label_font},{label_fs},&H88FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,1,0,1,0,0,8,0,0,0,1")
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
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    def px(x_1080p: int) -> int:
        return int(round(x_1080p * scale_x))

    def py(y_1080p: int) -> int:
        return int(round(y_1080p * scale_y))

    blur_scale = min(scale_x, scale_y)
    blur_12 = max(0, int(round(12 * blur_scale)))
    blur_1 = max(0, int(round(1 * blur_scale)))
    bord_2 = max(0, int(round(2 * blur_scale)))

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
    lines.append(f"Dialogue: 0,{ass_time(first_visible)},{ass_time(last_visible)},Box,,0,0,0,,{shadow_draw}")

    panel_draw = (
        f"{{\\pos({origin_x},{origin_y})\\p1\\c&H101010&\\alpha&H{alpha_border:02X}&\\blur{blur_1}\\bord{bord_2}\\3c&HFFFFFF&\\3a&HD0&}}"
        f"m 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}{{\\p0}}"
    )
    lines.append(f"Dialogue: 1,{ass_time(first_visible)},{ass_time(last_visible)},Box,,0,0,0,,{panel_draw}")

    header_draw = (
        f"{{\\pos({origin_x},{origin_y})\\p1\\c&H1E1E1E&\\alpha&H{alpha_main:02X}&}}"
        f"m 0 0 l {box_w} 0 l {box_w} {py(95)} l 0 {py(95)}{{\\p0}}"
    )
    lines.append(f"Dialogue: 2,{ass_time(first_visible)},{ass_time(last_visible)},Box,,0,0,0,,{header_draw}")

    def rect_path(x1: int, y1: int, x2: int, y2: int) -> str:
        return f"m {x1} {y1} l {x2} {y1} l {x2} {y2} l {x1} {y2}"

    grid_shapes = [
        # vertical separators
        ("&HFFFFFF&", "D8", 3, rect_path(px(140), py(12), px(142), py(178))),
        ("&HFFFFFF&", "D8", 3, rect_path(px(280), py(12), px(282), py(178))),
        # row divider
        ("&HFFFFFF&", "E0", 3, rect_path(px(12), py(95), px(408), py(97))),
        # accent bars row 1
        ("&HFFFFFF&", f"{alpha_main:02X}", 4, rect_path(px(36), py(36), px(104), py(39))),
        ("&HFFCC00&", f"{alpha_main:02X}", 4, rect_path(px(176), py(36), px(244), py(39))),
        ("&H66AAFF&", f"{alpha_main:02X}", 4, rect_path(px(316), py(36), px(384), py(39))),
        # accent bars row 2
        ("&HFFFFFF&", f"{alpha_main:02X}", 4, rect_path(px(36), py(126), px(104), py(129))),
        ("&H88FF88&", f"{alpha_main:02X}", 4, rect_path(px(176), py(126), px(244), py(129))),
        ("&H4444FF&", f"{alpha_main:02X}", 4, rect_path(px(316), py(126), px(384), py(129))),
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

        elapsed = (s.t - t0).total_seconds()
        elapsed_str = format_elapsed(elapsed)

        pace_sec = 500.0 / s.speed if (s.speed is not None and s.speed > 0) else None
        pace_str = format_pace(pace_sec)

        spm_str = f"{s.cadence:d}" if s.cadence is not None else "--"
        watts_str = f"{s.watts:d}" if s.watts is not None else "---"
        hr_str = f"{s.hr:d}" if s.hr is not None else "---"
        meters_str = f"{int(round(s.distance_m)):d}" if s.distance_m is not None else "---"

        lines.append(f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Time,,0,0,0,,{{\\pos({col1_x},{value_row1_y})}}{elapsed_str}")
        lines.append(f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Split,,0,0,0,,{{\\pos({col2_x},{value_row1_y})}}{pace_str}")
        lines.append(f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},SPM,,0,0,0,,{{\\pos({col3_x},{value_row1_y})}}{spm_str}")
        lines.append(f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Distance,,0,0,0,,{{\\pos({col1_x},{value_row2_y})}}{meters_str}")
        lines.append(f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},Watts,,0,0,0,,{{\\pos({col2_x},{value_row2_y})}}{watts_str}")
        lines.append(f"Dialogue: 6,{ass_time(st_clip)},{ass_time(et_clip)},HeartRate,,0,0,0,,{{\\pos({col3_x},{value_row2_y})}}{hr_str}")

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

    vf = f"ass=filename='{ass_name}'"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_in,
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
    ]
    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    cmd += [video_out]

    p = subprocess.run(cmd, cwd=ass_dir)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg burn-in failed (code {p.returncode}).")


# -----------------------------
# Alignment helpers
# -----------------------------

def choose_tcx_anchor_index(samples: List[Sample], *, video_start: datetime, mode: str) -> int:
    """
    Pick which TCX sample becomes t0.

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
        raise ValueError(f"unknown tcx anchor mode: {mode}")

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
        prog="rowerg_overlay.py",
        description="Create a PM5-style overlay (.ass subtitles) from Concept2 workout data and align it to a video using metadata timestamps.",
    )
    ap.add_argument("video", help="Input video file (mp4/mov/etc)")
    ap.add_argument("tcx", help="Workout data file (.tcx, .fit, or Concept2 .csv)")
    ap.add_argument("-o", "--out-ass", default=None, help="Output .ass path (default: next to input video)")

    ap.add_argument("--offset", type=float, default=0.0,
                    help="Manual offset adjustment in seconds (added to the auto-computed alignment). "
                         "Positive makes data appear later; negative earlier.")
    ap.add_argument(
        "--font",
        default=None,
        help="Legacy alias: set both --label-font and --value-font to this font name.",
    )
    ap.add_argument("--label-font", default="PragmataPro", help="Font for labels (must exist on your system)")
    ap.add_argument("--value-font", default="PragmataPro Mono", help="Font for values (must exist on your system)")
    ap.add_argument("--fontsize", type=int, default=None, help="Value font size (default: scaled from 52 @ 1080p)")
    ap.add_argument("--left-margin", type=int, default=None, help="Left margin in pixels (default: scaled from 20 @ 1080p)")
    ap.add_argument("--top-margin", type=int, default=None,
                    help="Top margin in pixels; if set, positions the overlay from the top instead of the bottom.")
    ap.add_argument("--bottom-margin", type=int, default=None, help="Bottom margin in pixels (default: scaled from 20 @ 1080p)")
    ap.add_argument("--box-alpha", type=int, default=112,
                    help="Background box transparency 0..255 (0=opaque, 255=fully transparent). Default: 112.")

    ap.add_argument("--burn-in", metavar="OUT_VIDEO", default=None,
                    help="If set, burn the overlay into a new video using ffmpeg.")
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF for burn-in (default: 18)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset for burn-in (default: veryfast)")
    ap.add_argument("--reencode-audio", action="store_true",
                    help="Re-encode audio to AAC instead of stream-copying it (use if -c:a copy fails).")

    ap.add_argument("--ffprobe-bin", default="ffprobe", help="Path to ffprobe (default: ffprobe)")
    ap.add_argument("--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg (default: ffmpeg)")
    ap.add_argument(
        "--tcx-anchor",
        choices=["start", "first-visible", "first-row-visible"],
        default="start",
        help="Which data sample to treat as time 0 for overlay generation (default: start).",
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
    data_path = args.tcx
    out_ass = args.out_ass or str(Path(video_path).with_suffix(".ass"))

    # Parse data (tcx/fit/csv)
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
    w, h, duration, video_creation, source = get_video_metadata(video_path, ffprobe_bin=args.ffprobe_bin)

    # Choose anchor (t0) used for both alignment and displayed elapsed time.
    video_start_for_anchor = video_creation
    if parsed.timebase == "relative":
        video_start_for_anchor = datetime(1970, 1, 1, tzinfo=timezone.utc)

    anchor_idx = choose_tcx_anchor_index(samples_all, video_start=video_start_for_anchor, mode=args.tcx_anchor)
    samples = samples_all[anchor_idx:] if anchor_idx else samples_all
    tcx_anchor = samples[0].t

    # Auto offset: when does anchor occur on the video timeline?
    if parsed.timebase == "absolute":
        auto_offset = (tcx_anchor - video_creation).total_seconds()
    else:
        auto_offset = 0.0
    offset = auto_offset + float(args.offset)

    print("== Alignment ==")
    print(f"Video creation/start time (UTC): {video_creation.isoformat()}  [{source}]")
    if duration is not None:
        video_end = datetime.fromtimestamp(video_creation.timestamp() + duration, tz=timezone.utc)
        print(f"Video end time (UTC):            {video_end.isoformat()}  [duration {duration:.2f} s]")
    print(f"Data file: {data_path}  [{parsed.kind}, {parsed.timebase}]")
    if parsed.timebase == "absolute":
        print(f"Data first timestamp (UTC):      {data_start.isoformat()}")
        delta0 = (data_start - video_creation).total_seconds()
        if abs(delta0) >= 1.0:
            when = "after" if delta0 > 0 else "before"
            print(f"Data starts {abs(delta0):.1f} s {when} video start (based on absolute timestamps).")
        first_row_visible = next((s for s in samples_all if s.t >= video_creation and (s.cadence or 0) > 0), None)
        if first_row_visible is not None:
            tv = (first_row_visible.t - video_creation).total_seconds()
            print(f"First sample with cadence>0 during video: t={tv:.1f} s  [{first_row_visible.t.isoformat()}]")
    else:
        print("Data has relative timestamps; auto alignment is disabled (auto_offset=0). Use --offset to place it.")
    if tcx_anchor != data_start:
        print(f"Data anchor ({args.tcx_anchor}) time (UTC): {tcx_anchor.isoformat()}  [idx {anchor_idx}]")
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
    )
    print(f"Wrote ASS overlay: {out_ass}")

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
