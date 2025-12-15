import math

from c2_overlay.timeutil import ass_time, format_elapsed, format_pace, format_remaining


def test_format_elapsed_zero() -> None:
    assert format_elapsed(0) == "0:00"


def test_format_elapsed_seconds_only() -> None:
    assert format_elapsed(45) == "0:45"


def test_format_elapsed_minutes_and_seconds() -> None:
    assert format_elapsed(125) == "2:05"


def test_format_elapsed_hours() -> None:
    assert format_elapsed(3661) == "1:01:01"


def test_format_elapsed_negative_clamps_to_zero() -> None:
    assert format_elapsed(-10) == "0:00"


def test_format_elapsed_fractional_floors() -> None:
    assert format_elapsed(59.9) == "0:59"


def test_format_remaining_zero() -> None:
    assert format_remaining(0) == "0:00"


def test_format_remaining_fractional_ceils() -> None:
    assert format_remaining(0.1) == "0:01"
    assert format_remaining(59.1) == "1:00"


def test_format_pace_typical() -> None:
    # 2:00.0 per 500m
    assert format_pace(120.0) == "2:00.0"


def test_format_pace_with_tenths() -> None:
    assert format_pace(125.3) == "2:05.3"


def test_format_pace_placeholders() -> None:
    assert format_pace(None) == "--:--.-"
    assert format_pace(0) == "--:--.-"
    assert format_pace(math.inf) == "--:--.-"


def test_ass_time_zero() -> None:
    assert ass_time(0) == "0:00:00.00"


def test_ass_time_centiseconds() -> None:
    assert ass_time(1.5) == "0:00:01.50"


def test_ass_time_hours() -> None:
    assert ass_time(3723.04) == "1:02:03.04"


def test_ass_time_negative_clamps() -> None:
    assert ass_time(-5) == "0:00:00.00"


def test_ass_time_rounding_boundary() -> None:
    assert ass_time(0.004) == "0:00:00.00"
    assert ass_time(0.005) == "0:00:00.01"
