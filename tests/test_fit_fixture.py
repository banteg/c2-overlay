from datetime import UTC, datetime
from pathlib import Path

import pytest

from c2_overlay.fit import parse_data_file


def test_parse_fit_fixture(sample_fit: Path) -> None:
    parsed = parse_data_file(str(sample_fit))

    assert len(parsed.samples) == 352
    assert parsed.laps is not None
    assert len(parsed.laps) == 13

    assert parsed.samples[0].t == datetime(2025, 12, 14, 10, 40, 0, tzinfo=UTC)
    assert parsed.samples[-1].t == datetime(2025, 12, 14, 11, 1, 57, tzinfo=UTC)

    assert [lap.index for lap in parsed.laps] == list(range(1, len(parsed.laps) + 1))

    intensities = [lap.intensity for lap in parsed.laps]
    assert intensities[0] == "active"
    assert "rest" in intensities
    assert set(intensities).issubset({"active", "rest", "unknown"})

    lap1 = parsed.laps[0]
    assert lap1.total_elapsed_s == pytest.approx(60.0, abs=0.01)
    assert lap1.total_distance_m == pytest.approx(235.99, abs=0.2)
    assert lap1.avg_cadence_spm == 23
    assert lap1.avg_power_w == 170
    assert lap1.avg_hr_bpm == 132

