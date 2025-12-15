from c2_overlay.fit import normalize_intensity


def test_numeric_active() -> None:
    assert normalize_intensity(0) == "active"


def test_numeric_rest() -> None:
    assert normalize_intensity(1) == "rest"


def test_string_active() -> None:
    assert normalize_intensity("active") == "active"
    assert normalize_intensity("ACTIVE") == "active"


def test_string_rest() -> None:
    assert normalize_intensity("rest") == "rest"


def test_none() -> None:
    assert normalize_intensity(None) == "unknown"


def test_unknown_numeric() -> None:
    assert normalize_intensity(99) == "unknown(99)"
