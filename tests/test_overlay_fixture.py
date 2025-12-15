from pathlib import Path

from c2_overlay.ass_lint import lint_ass_text, parse_ass_events
from c2_overlay.fit import parse_data_file
from c2_overlay.overlay import generate_ass


def test_generate_ass_with_laps_from_fixture(sample_fit: Path, tmp_path: Path) -> None:
    parsed = parse_data_file(str(sample_fit))
    assert parsed.laps is not None

    # Keep the duration short to exercise one work->rest transition without generating
    # an excessively large ASS file.
    video_duration = 150.0

    out_ass = tmp_path / "fixture.ass"
    generate_ass(
        samples=parsed.samples,
        out_ass=str(out_ass),
        video_w=1920,
        video_h=1080,
        video_duration=video_duration,
        offset_seconds=0.0,
        label_font="Arial",
        value_font="Arial",
        value_fs=None,
        left_margin=None,
        top_margin=None,
        bottom_margin=None,
        box_alpha=112,
        interpolate=True,
        laps=parsed.laps,
    )

    text = out_ass.read_text(encoding="utf-8")
    issues = lint_ass_text(text)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []

    events, parse_issues = parse_ass_events(text)
    assert parse_issues == []

    # Validate REST overlay uses previous lap summary for the first rest lap (lap 2).
    samples0 = parsed.samples[0].t
    lap2 = next(lap for lap in parsed.laps if lap.index == 2)
    lap2_start = (lap2.start - samples0).total_seconds()
    lap2_end = (lap2.end - samples0).total_seconds()

    assert lap2.intensity == "rest"
    assert 0 <= lap2_start < lap2_end <= video_duration

    def find_rest_constant(style: str) -> str:
        match = [
            e
            for e in events
            if e.layer == 20
            and e.style == style
            and abs(e.start - lap2_start) <= 0.01
            and abs(e.end - lap2_end) <= 0.01
        ]
        assert len(match) == 1
        return match[0].plain

    assert find_rest_constant("Time") == "1:00"
    assert find_rest_constant("Distance") == "236"
    assert find_rest_constant("Split") == "2:07.1"
    assert find_rest_constant("SPM") == "23"
    assert find_rest_constant("Watts") == "170"
    assert find_rest_constant("HeartRate") == "132"

    # REST countdown should tick once per second for the lap duration.
    countdown = [
        e
        for e in events
        if e.layer == 21 and e.style == "Label" and e.plain.startswith("REST ")
    ]
    assert len(countdown) == int(lap2_end - lap2_start)
    assert countdown[0].plain == "REST 1:00"
    assert countdown[-1].plain == "REST 0:01"

    # Lap header should be present for the same interval.
    header = [
        e
        for e in events
        if e.layer == 10
        and e.style == "Label"
        and e.plain == "LAP 02 · REST"
        and abs(e.start - lap2_start) <= 0.01
        and abs(e.end - lap2_end) <= 0.01
    ]
    assert len(header) == 1

