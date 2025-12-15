from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from c2_overlay.fit import Sample
from c2_overlay.overlay import compute_visibility_range, generate_ass


@pytest.fixture
def sample_workout() -> list[Sample]:
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return [
        Sample(
            t=base + timedelta(seconds=i),
            distance_m=float(i * 10),
            hr=140 + i,
            cadence=28,
            watts=200 + i * 5,
            speed=3.5,
        )
        for i in range(10)
    ]


def test_generates_valid_ass_file(sample_workout: list[Sample], tmp_path: Path) -> None:
    out_ass = tmp_path / "test.ass"
    generate_ass(
        samples=sample_workout,
        out_ass=str(out_ass),
        video_w=1920,
        video_h=1080,
        video_duration=15.0,
        offset_seconds=0.0,
        label_font="Arial",
        value_font="Arial",
        value_fs=None,
        left_margin=None,
        top_margin=None,
        bottom_margin=None,
        box_alpha=112,
        interpolate=False,
        laps=None,
    )

    content = out_ass.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content


def test_empty_samples_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No samples"):
        generate_ass(
            samples=[],
            out_ass=str(tmp_path / "empty.ass"),
            video_w=1920,
            video_h=1080,
            video_duration=10.0,
            offset_seconds=0.0,
            label_font="Arial",
            value_font="Arial",
            value_fs=None,
            left_margin=None,
            top_margin=None,
            bottom_margin=None,
            box_alpha=112,
            laps=None,
        )


def test_no_overlap_raises(sample_workout: list[Sample], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No FIT samples overlap"):
        generate_ass(
            samples=sample_workout,
            out_ass=str(tmp_path / "no_overlap.ass"),
            video_w=1920,
            video_h=1080,
            video_duration=10.0,
            offset_seconds=-100.0,  # All samples before video
            label_font="Arial",
            value_font="Arial",
            value_fs=None,
            left_margin=None,
            top_margin=None,
            bottom_margin=None,
            box_alpha=112,
            laps=None,
        )


def test_compute_visibility_range_basic() -> None:
    first, last = compute_visibility_range(
        start_times=[0.0, 1.0, 2.0],
        end_times=[1.0, 2.0, 3.0],
        video_duration=10.0,
        laps=None,
        t0=datetime(2025, 1, 1, tzinfo=UTC),
        offset_seconds=0.0,
    )
    assert first == 0.0
    assert last == 3.0


def test_compute_visibility_range_clamps_to_video_duration() -> None:
    first, last = compute_visibility_range(
        start_times=[0.0, 5.0, 10.0],
        end_times=[5.0, 10.0, 15.0],
        video_duration=8.0,
        laps=None,
        t0=datetime(2025, 1, 1, tzinfo=UTC),
        offset_seconds=0.0,
    )
    assert first == 0.0
    assert last == 8.0
