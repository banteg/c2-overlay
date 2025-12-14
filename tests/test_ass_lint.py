from c2_overlay.ass_lint import lint_ass_text, parse_ass_time


ASS_HEADER = """[Script Info]
Title: test

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Distance,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def test_parse_ass_time() -> None:
    t0 = parse_ass_time("0:00:00.00")
    assert t0 is not None
    assert t0 == 0.0

    t1 = parse_ass_time("1:02:03.04")
    assert t1 is not None
    assert t1 == 3723.04

    assert parse_ass_time("nope") is None


def test_detect_overlap() -> None:
    ass = (
        ASS_HEADER
        + "Dialogue: 6,0:00:00.00,0:00:02.00,Distance,,0,0,0,,{\\pos(10,10)}100\n"
        + "Dialogue: 9,0:00:01.00,0:00:03.00,Distance,,0,0,0,,{\\pos(10,10)}101\n"
    )
    issues = lint_ass_text(ass)
    assert any(i.code == "ASS020" and i.severity == "error" for i in issues)


def test_no_overlap_for_touching_boundaries() -> None:
    ass = (
        ASS_HEADER
        + "Dialogue: 9,0:00:00.00,0:00:01.00,Distance,,0,0,0,,{\\pos(10,10)}0\n"
        + "Dialogue: 9,0:00:01.00,0:00:02.00,Distance,,0,0,0,,{\\pos(10,10)}1\n"
    )
    issues = lint_ass_text(ass)
    assert not any(i.code == "ASS020" for i in issues)


def test_detect_bad_duration() -> None:
    ass = (
        ASS_HEADER
        + "Dialogue: 9,0:00:01.00,0:00:01.00,Distance,,0,0,0,,{\\pos(10,10)}1\n"
    )
    issues = lint_ass_text(ass)
    assert any(i.code == "ASS010" and i.severity == "error" for i in issues)


def test_warn_on_missing_pos() -> None:
    ass = ASS_HEADER + "Dialogue: 9,0:00:00.00,0:00:01.00,Distance,,0,0,0,,100\n"
    issues = lint_ass_text(ass)
    assert any(i.code == "ASS012" and i.severity == "warn" for i in issues)


def test_pos_allows_spaces() -> None:
    ass = (
        ASS_HEADER
        + "Dialogue: 9,0:00:00.00,0:00:01.00,Distance,,0,0,0,,{\\pos(10, 10)}100\n"
    )
    issues = lint_ass_text(ass)
    assert not any(i.code == "ASS012" for i in issues)


def test_drawing_detection_any_scale() -> None:
    ass = (
        ASS_HEADER
        + "Dialogue: 9,0:00:00.00,0:00:01.00,Distance,,0,0,0,,{\\p4}m 0 0 l 10 0{\\p0}\n"
    )
    issues = lint_ass_text(ass)
    assert not any(i.code == "ASS012" for i in issues)
