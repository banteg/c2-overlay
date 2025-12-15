from datetime import UTC, datetime, timedelta

from c2_overlay.fit import LapSegment
from c2_overlay.overlay import compute_visibility_range


def test_visibility_range_includes_laps_when_samples_sparse() -> None:
    t0 = datetime(2025, 12, 14, 10, 0, 0, tzinfo=UTC)

    # Samples are entirely before the video timeline, but a lap overlaps.
    start_times = [-10.0, -9.0]
    end_times = [-9.0, -8.0]

    laps = [
        LapSegment(
            index=1,
            start=t0 + timedelta(seconds=5),
            end=t0 + timedelta(seconds=7),
            intensity="active",
            start_distance_m=0.0,
            total_elapsed_s=2.0,
            total_distance_m=10.0,
            avg_speed_m_s=5.0,
            avg_power_w=200,
            avg_cadence_spm=20,
            avg_hr_bpm=150,
        )
    ]

    first, last = compute_visibility_range(
        start_times=start_times,
        end_times=end_times,
        video_duration=10.0,
        laps=laps,
        t0=t0,
        offset_seconds=0.0,
    )
    assert first == 5.0
    assert last == 7.0

