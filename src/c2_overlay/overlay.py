from __future__ import annotations

import math
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path

from .fit import LapSegment, Sample
from .timeutil import ass_time, format_elapsed, format_pace, format_remaining


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
        laps = sorted(laps, key=lambda lap: (lap.start, lap.index))

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
    use_top = top_margin is not None
    if top_margin is None:
        top_margin = 0
    if bottom_margin is None:
        bottom_margin = max(10, int(round(20 * scale_y)))

    # Box size and placement (match sample proportions).
    box_w = max(1, int(round(420 * scale_x)))
    box_h = max(1, int(round(190 * scale_y)))

    origin_x = int(left_margin)
    if use_top:
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

        lap_starts = [lap.start for lap in laps]

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

    open_values: dict[tuple[int, str, int, int], tuple[float, float, str]] = {}
    merge_eps = 1e-4

    def flush_dialogue(
        key: tuple[int, str, int, int], st: float, et: float, text: str
    ) -> None:
        layer, style, x, y = key
        lines.append(
            f"Dialogue: {layer},{ass_time(st)},{ass_time(et)},{style},,0,0,0,,{{\\pos({x},{y})}}{text}"
        )

    def emit_dialogue(
        layer: int, st: float, et: float, style: str, x: int, y: int, text: str
    ) -> None:
        if et <= st:
            return
        key = (layer, style, x, y)
        prev = open_values.get(key)
        if prev is not None:
            pst, pet, ptext = prev
            if ptext == text and abs(pet - st) <= merge_eps:
                open_values[key] = (pst, et, ptext)
                return
            flush_dialogue(key, pst, pet, ptext)
        open_values[key] = (st, et, text)

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
            emit_dialogue(
                6,
                st_clip,
                et_clip,
                "Time",
                col1_x,
                value_row1_y,
                lap_elapsed_str,
            )
        emit_dialogue(
            6,
            st_clip,
            et_clip,
            "Split",
            col2_x,
            value_row1_y,
            pace_str,
        )
        emit_dialogue(
            6,
            st_clip,
            et_clip,
            "SPM",
            col3_x,
            value_row1_y,
            spm_str,
        )
        if current_lap is None or not laps or current_lap.index not in upsampled_laps:
            emit_dialogue(
                6,
                st_clip,
                et_clip,
                "Distance",
                col1_x,
                value_row2_y,
                meters_str,
            )
        emit_dialogue(
            6,
            st_clip,
            et_clip,
            "Watts",
            col2_x,
            value_row2_y,
            watts_str,
        )
        emit_dialogue(
            6,
            st_clip,
            et_clip,
            "HeartRate",
            col3_x,
            value_row2_y,
            hr_str,
        )

    for key, (st, et, text) in open_values.items():
        flush_dialogue(key, st, et, text)

    Path(out_ass).write_text("\n".join(lines), encoding="utf-8")
