from datetime import UTC, datetime, timedelta
from pathlib import Path

from c2_overlay.ass_lint import lint_ass_text, parse_ass_events
from c2_overlay.fit import LapSegment, Sample
from c2_overlay.overlay import generate_ass


def test_generate_ass_contract_lints_clean_with_laps(tmp_path: Path) -> None:
    t0 = datetime(2025, 12, 14, 10, 41, 31, tzinfo=UTC)

    # 10 seconds of samples, 5 m/s => 50m total.
    samples = [
        Sample(
            t=t0 + timedelta(seconds=i),
            distance_m=float(i * 5),
            speed=5.0,
            cadence=20,
            watts=200,
            hr=150,
        )
        for i in range(10)
    ]

    laps = [
        LapSegment(
            index=1,
            start=t0,
            end=t0 + timedelta(seconds=6),
            intensity="active",
            start_distance_m=0.0,
            total_elapsed_s=6.0,
            total_distance_m=30.0,
            avg_speed_m_s=5.0,
            avg_power_w=200,
            avg_cadence_spm=20,
            avg_hr_bpm=150,
        ),
        LapSegment(
            index=2,
            start=t0 + timedelta(seconds=6),
            end=t0 + timedelta(seconds=10),
            intensity="rest",
            start_distance_m=30.0,
            total_elapsed_s=4.0,
            total_distance_m=0.0,
            avg_speed_m_s=None,
            avg_power_w=None,
            avg_cadence_spm=None,
            avg_hr_bpm=None,
        ),
    ]

    out_ass = tmp_path / "out.ass"
    generate_ass(
        samples=samples,
        out_ass=str(out_ass),
        video_w=1920,
        video_h=1080,
        video_duration=10.0,
        offset_seconds=0.0,
        label_font="Arial",
        value_font="Arial",
        value_fs=52,
        left_margin=20,
        top_margin=None,
        bottom_margin=20,
        box_alpha=112,
        interpolate=True,
        laps=laps,
    )

    text = out_ass.read_text(encoding="utf-8")
    issues = lint_ass_text(text)
    assert issues == []

    events, parse_issues = parse_ass_events(text)
    assert parse_issues == []

    # Lap 2 REST should show lap 1 summary for the whole rest interval.
    def find_ev(layer: int, style: str, start: float, end: float) -> str:
        matches = [
            e
            for e in events
            if e.layer == layer and e.style == style and e.start == start and e.end == end
        ]
        assert len(matches) == 1
        return matches[0].plain

    assert find_ev(20, "Time", 6.0, 10.0) == "0:06"
    assert find_ev(20, "Distance", 6.0, 10.0) == "30"
    assert find_ev(20, "Split", 6.0, 10.0) == "1:40.0"
    assert find_ev(20, "SPM", 6.0, 10.0) == "20"
    assert find_ev(20, "Watts", 6.0, 10.0) == "200"
    assert find_ev(20, "HeartRate", 6.0, 10.0) == "150"

    assert find_ev(10, "Label", 6.0, 10.0) == "LAP 02 · REST"

    countdown = [
        e
        for e in events
        if e.layer == 21 and e.style == "Label" and e.plain.startswith("REST ")
    ]
    assert [e.plain for e in countdown] == [
        "REST 0:04",
        "REST 0:03",
        "REST 0:02",
        "REST 0:01",
    ]

