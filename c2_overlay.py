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
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


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
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


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
    tree = ET.parse(tcx_path)
    root = tree.getroot()

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
    font: str,
    normal_fs: Optional[int],
    left_margin: Optional[int],
    top_margin: Optional[int],
    right_margin: Optional[int],
    box_alpha: int,
) -> None:
    """
    Create a PM5-ish overlay:
      - main panel top-left (time, distance, pace, spm, watts)
      - HR panel top-right

    offset_seconds is the computed (or overridden) shift that maps:
      video_time = (sample_time - tcx_first_time) + offset_seconds
    """
    if not samples:
        raise ValueError("No TCX trackpoints found.")

    if video_w <= 0 or video_h <= 0:
        # If ffprobe couldn't determine, choose a reasonable default
        video_w, video_h = 1280, 720

    if normal_fs is None:
        normal_fs = max(18, round(video_h * 0.04))
    big_fs = int(round(normal_fs * 1.55))

    if left_margin is None:
        left_margin = max(10, round(video_w * 0.02))
    if right_margin is None:
        right_margin = left_margin
    if top_margin is None:
        top_margin = max(10, round(video_h * 0.02))

    pad = int(round(normal_fs * 0.50))
    left_x = left_margin
    top_y = top_margin
    right_x = video_w - right_margin

    # Box sizes (rough PM5 proportions)
    left_box_w = int(round(video_w * 0.42))
    left_box_h = int(round(video_h * 0.22))
    hr_box_w = int(round(video_w * 0.18))
    hr_box_h = int(round(video_h * 0.09))

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
    box_a = f"{box_alpha:02X}"  # for \1a

    # ASS colours are &HAABBGGRR (alpha first)
    green = "&H0000FF00"  # opaque green
    red = "&H000000FF"    # opaque red
    black = "&H00000000"  # opaque black

    lines: List[str] = []
    lines.append("[Script Info]")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {video_w}")
    lines.append(f"PlayResY: {video_h}")
    lines.append("WrapStyle: 2")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    # Top-left main text
    lines.append(f"Style: PM5Left,{font},{normal_fs},{green},{green},{black},{black},-1,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1")
    # Top-right HR
    lines.append(f"Style: PM5HR,{font},{normal_fs},{red},{red},{black},{black},-1,0,0,0,100,100,0,0,1,2,0,9,0,0,0,1")
    # Box style (we'll draw rectangles with \p1)
    lines.append(f"Style: PM5Box,{font},1,{black},{black},{black},{black},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1")
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    # Static background boxes (one line each, spanning the whole overlay range)
    left_box_draw = (
        f"{{\\an7\\pos({left_x},{top_y})\\bord0\\shad0\\1c&H000000&\\1a&H{box_a}&\\p1}}"
        f"m 0 0 l {left_box_w} 0 {left_box_w} {left_box_h} 0 {left_box_h}"
        f"{{\\p0}}"
    )
    lines.append(f"Dialogue: 0,{ass_time(first_visible)},{ass_time(last_visible)},PM5Box,,0,0,0,,{left_box_draw}")

    hr_box_draw = (
        f"{{\\an9\\pos({right_x},{top_y})\\bord0\\shad0\\1c&H000000&\\1a&H{box_a}&\\p1}}"
        f"m 0 0 l {hr_box_w} 0 {hr_box_w} {hr_box_h} 0 {hr_box_h}"
        f"{{\\p0}}"
    )
    lines.append(f"Dialogue: 0,{ass_time(first_visible)},{ass_time(last_visible)},PM5Box,,0,0,0,,{hr_box_draw}")

    # Per-sample text
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

        dist_km = (s.distance_m / 1000.0) if s.distance_m is not None else None
        dist_str = f"{dist_km:0.2f} km" if dist_km is not None else "--.- km"

        spm_str = f"{s.cadence:2d} spm" if s.cadence is not None else "-- spm"
        watts_str = f"{s.watts:3d} W" if s.watts is not None else "--- W"

        pace_sec = 500.0 / s.speed if (s.speed is not None and s.speed > 0) else None
        pace_str = format_pace(pace_sec)

        left_text = (
            f"{{\\fs{big_fs}\\b1}}{elapsed_str}{{\\b0\\fs{normal_fs}}}    {dist_str}\\N"
            f"{{\\fs{big_fs}\\b1}}{pace_str}{{\\b0\\fs{normal_fs}}} /500m   {spm_str}\\N"
            f"{{\\fs{normal_fs}}}PWR {watts_str}"
        )
        left_text = f"{{\\an7\\pos({left_x + pad},{top_y + pad})}}{left_text}"
        lines.append(f"Dialogue: 1,{ass_time(st_clip)},{ass_time(et_clip)},PM5Left,,0,0,0,,{left_text}")

        if s.hr is None:
            hr_text = f"{{\\fs{normal_fs}}}HR {{\\fs{big_fs}\\b1}}---{{\\b0\\fs{normal_fs}}} bpm"
        else:
            hr_text = f"{{\\fs{normal_fs}}}HR{{\\fs{big_fs}\\b1}} {s.hr:3d}{{\\b0\\fs{normal_fs}}} bpm"
        hr_text = f"{{\\an9\\pos({right_x - pad},{top_y + pad})}}{hr_text}"
        lines.append(f"Dialogue: 2,{ass_time(st_clip)},{ass_time(et_clip)},PM5HR,,0,0,0,,{hr_text}")

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
# CLI
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="rowerg_overlay.py",
        description="Create a PM5-style overlay (.ass subtitles) from Concept2 TCX stroke data and align it to a video using metadata timestamps.",
    )
    ap.add_argument("video", help="Input video file (mp4/mov/etc)")
    ap.add_argument("tcx", help="Concept2 TCX file with absolute timestamps")
    ap.add_argument("-o", "--out-ass", default="overlay.ass", help="Output .ass path (default: overlay.ass)")

    ap.add_argument("--offset", type=float, default=0.0,
                    help="Manual offset adjustment in seconds (added to the auto-computed alignment). "
                         "Positive makes data appear later; negative earlier.")
    ap.add_argument("--font", default="DejaVu Sans Mono", help="Font name for the overlay (must exist on your system)")
    ap.add_argument("--fontsize", type=int, default=None, help="Base font size (default: 4%% of video height)")
    ap.add_argument("--left-margin", type=int, default=None, help="Left margin in pixels (default: 2%% of width)")
    ap.add_argument("--right-margin", type=int, default=None, help="Right margin in pixels (default: same as left)")
    ap.add_argument("--top-margin", type=int, default=None, help="Top margin in pixels (default: 2%% of height)")
    ap.add_argument("--box-alpha", type=int, default=128,
                    help="Background box transparency 0..255 (0=opaque, 255=fully transparent). Default: 128.")

    ap.add_argument("--burn-in", metavar="OUT_VIDEO", default=None,
                    help="If set, burn the overlay into a new video using ffmpeg.")
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF for burn-in (default: 18)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset for burn-in (default: veryfast)")
    ap.add_argument("--reencode-audio", action="store_true",
                    help="Re-encode audio to AAC instead of stream-copying it (use if -c:a copy fails).")

    ap.add_argument("--ffprobe-bin", default="ffprobe", help="Path to ffprobe (default: ffprobe)")
    ap.add_argument("--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg (default: ffmpeg)")

    args = ap.parse_args()

    # Tools check
    if shutil.which(args.ffprobe_bin) is None:
        print(f"ERROR: ffprobe not found: {args.ffprobe_bin}", file=sys.stderr)
        return 2
    if args.burn_in and shutil.which(args.ffmpeg_bin) is None:
        print(f"ERROR: ffmpeg not found: {args.ffmpeg_bin}", file=sys.stderr)
        return 2

    video_path = args.video
    tcx_path = args.tcx
    out_ass = args.out_ass

    # Parse TCX
    samples = parse_tcx(tcx_path)
    if not samples:
        print("ERROR: No trackpoints found in the TCX file.", file=sys.stderr)
        return 2

    tcx_start = samples[0].t

    # Probe video
    w, h, duration, video_creation, source = get_video_metadata(video_path, ffprobe_bin=args.ffprobe_bin)

    # Auto offset: when does the TCX start occur on the video timeline?
    # offset_seconds = (tcx_start - video_creation).total_seconds()
    auto_offset = (tcx_start - video_creation).total_seconds()
    offset = auto_offset + float(args.offset)

    print("== Alignment ==")
    print(f"Video creation/start time (UTC): {video_creation.isoformat()}  [{source}]")
    print(f"TCX first trackpoint time (UTC): {tcx_start.isoformat()}")
    print(f"Auto offset (tcx_start - video_start): {auto_offset:+.3f} s")
    if args.offset:
        print(f"Manual adjustment: {args.offset:+.3f} s")
    print(f"Final offset used: {offset:+.3f} s")
    if duration is not None:
        print(f"Video: {w}x{h}, duration ~ {duration:.2f} s")
    else:
        print(f"Video: {w}x{h}, duration unknown")

    # Write ASS
    generate_ass(
        samples=samples,
        out_ass=out_ass,
        video_w=w,
        video_h=h,
        video_duration=duration,
        offset_seconds=offset,
        font=args.font,
        normal_fs=args.fontsize,
        left_margin=args.left_margin,
        right_margin=args.right_margin,
        top_margin=args.top_margin,
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
