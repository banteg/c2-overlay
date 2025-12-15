from datetime import UTC, datetime, timedelta

import pytest

from c2_overlay.align import choose_anchor_index
from c2_overlay.fit import Sample


def make_sample(offset_sec: float, cadence: int | None = None) -> Sample:
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return Sample(t=base + timedelta(seconds=offset_sec), cadence=cadence)


def test_empty_samples_returns_zero() -> None:
    video_start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert choose_anchor_index([], video_start=video_start, mode="start") == 0


def test_mode_start_always_returns_zero() -> None:
    samples = [make_sample(float(i)) for i in range(5)]
    video_start = datetime(2025, 1, 1, 12, 0, 2, tzinfo=UTC)
    assert choose_anchor_index(samples, video_start=video_start, mode="start") == 0


def test_first_visible_skips_before_video() -> None:
    samples = [make_sample(float(i)) for i in range(5)]
    video_start = datetime(2025, 1, 1, 12, 0, 2, tzinfo=UTC)
    assert choose_anchor_index(samples, video_start=video_start, mode="first-visible") == 2


def test_first_row_visible_requires_cadence() -> None:
    samples = [
        make_sample(0.0, cadence=0),
        make_sample(1.0, cadence=0),
        make_sample(2.0, cadence=20),
        make_sample(3.0, cadence=22),
    ]
    video_start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert (
        choose_anchor_index(samples, video_start=video_start, mode="first-row-visible")
        == 2
    )


def test_invalid_mode_raises() -> None:
    samples = [make_sample(0.0)]
    video_start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="unknown anchor mode"):
        choose_anchor_index(samples, video_start=video_start, mode="invalid")
