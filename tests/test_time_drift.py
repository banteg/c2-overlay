from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path

from c2_overlay.ass_lint import AssEvent, parse_ass_events
from c2_overlay.fit import Sample, parse_data_file
from c2_overlay.overlay import generate_ass
from c2_overlay.timeutil import format_elapsed, format_pace


def _quantize_ass_seconds(sec: float) -> float:
    # Match c2_overlay.timeutil.ass_time() centisecond rounding.
    return math.floor((sec * 100.0) + 0.5) / 100.0


def _events_by_style(events: list[AssEvent], *, style: str) -> tuple[list[float], list[AssEvent]]:
    evs = sorted(
        [e for e in events if e.layer == 6 and e.style == style and not e.is_drawing],
        key=lambda e: (e.start, e.end, e.line_no),
    )
    starts = [e.start for e in evs]
    return starts, evs


def _value_at(starts: list[float], evs: list[AssEvent], t: float) -> str:
    idx = bisect_right(starts, t) - 1
    assert idx >= 0
    ev = evs[idx]
    eps = 1e-6
    assert ev.start <= t + eps < ev.end + eps
    return ev.plain


def _format_sample_fields(s: Sample, *, t0: Sample) -> dict[str, str]:
    elapsed = (s.t - t0.t).total_seconds()
    time_str = format_elapsed(elapsed)
    meters_str = f"{int(round(s.distance_m)):d}" if s.distance_m is not None else "---"
    pace_sec = 500.0 / s.speed if (s.speed is not None and s.speed > 0) else None
    split_str = format_pace(pace_sec)
    spm_str = f"{s.cadence:d}" if s.cadence is not None else "--"
    watts_str = f"{s.watts:d}" if s.watts is not None else "---"
    hr_str = f"{s.hr:d}" if s.hr is not None else "---"
    return {
        "Time": time_str,
        "Distance": meters_str,
        "Split": split_str,
        "SPM": spm_str,
        "Watts": watts_str,
        "HeartRate": hr_str,
    }


def test_no_time_drift_between_fit_and_ass(sample_fit: Path, tmp_path: Path) -> None:
    parsed = parse_data_file(str(sample_fit))
    samples = parsed.samples
    assert samples

    # Generate per-sample overlays only (no laps/interpolation) so event timing is driven
    # directly by the FIT record timestamps.
    out_ass = tmp_path / "drift.ass"
    generate_ass(
        samples=samples,
        out_ass=str(out_ass),
        video_w=1920,
        video_h=1080,
        video_duration=None,
        offset_seconds=0.0,
        label_font="Arial",
        value_font="Arial",
        value_fs=52,
        left_margin=20,
        top_margin=None,
        bottom_margin=20,
        box_alpha=112,
        interpolate=False,
        laps=None,
    )

    text = out_ass.read_text(encoding="utf-8")
    events, issues = parse_ass_events(text)
    assert issues == []

    styles = ["Time", "Split", "SPM", "Distance", "Watts", "HeartRate"]
    indexed = {style: _events_by_style(events, style=style) for style in styles}

    # Spot-check a set of samples across the workout for drift.
    stride = max(1, len(samples) // 50)
    t0 = samples[0]
    for s in samples[::stride]:
        vt = (s.t - t0.t).total_seconds()
        vt_q = _quantize_ass_seconds(vt)
        expected = _format_sample_fields(s, t0=t0)
        for style in styles:
            starts, evs = indexed[style]
            assert _value_at(starts, evs, vt_q) == expected[style]

