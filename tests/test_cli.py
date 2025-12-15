from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import pytest

import c2_overlay.cli as cli
from c2_overlay.ass_lint import LintIssue
from c2_overlay.fit import ParsedData, Sample


def test_cli_errors_when_ffprobe_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _p: None)
    monkeypatch.setattr(sys, "argv", ["c2-overlay", "video.mp4", "workout.fit"])
    assert cli.main() == 2


def test_cli_errors_on_invalid_video_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _p: "/usr/bin/ffprobe")
    t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "parse_data_file", lambda _p: ParsedData(samples=[Sample(t=t0)]))
    monkeypatch.setattr(
        cli,
        "get_video_metadata",
        lambda *_a, **_kw: (1920, 1080, 10.0, t0, "ffprobe:creation_time"),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["c2-overlay", "video.mp4", "workout.fit", "--video-start", "nope"],
    )
    assert cli.main() == 2


def test_cli_computes_offset_and_default_out_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(p: str) -> str | None:
        return "/usr/bin/ffprobe" if p == "ffprobe" else None

    monkeypatch.setattr(cli.shutil, "which", fake_which)

    video_start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    anchor_time = video_start + timedelta(seconds=2)
    monkeypatch.setattr(
        cli,
        "parse_data_file",
        lambda _p: ParsedData(samples=[Sample(t=anchor_time, cadence=20)]),
    )
    monkeypatch.setattr(
        cli,
        "get_video_metadata",
        lambda *_a, **_kw: (1920, 1080, 10.0, video_start, "ffprobe:creation_time"),
    )

    captured: dict[str, object] = {}

    def fake_generate_ass(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "generate_ass", fake_generate_ass)

    monkeypatch.setattr(sys, "argv", ["c2-overlay", "video.mp4", "workout.fit", "--offset", "1.5"])
    assert cli.main() == 0

    assert captured["out_ass"] == "video.ass"
    assert captured["offset_seconds"] == pytest.approx(3.5)
    assert captured["interpolate"] is True


def test_cli_no_interp_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _p: "/usr/bin/ffprobe")
    t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "parse_data_file", lambda _p: ParsedData(samples=[Sample(t=t0)]))
    monkeypatch.setattr(
        cli,
        "get_video_metadata",
        lambda *_a, **_kw: (1920, 1080, 10.0, t0, "ffprobe:creation_time"),
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "generate_ass", lambda **kwargs: captured.update(kwargs))

    monkeypatch.setattr(sys, "argv", ["c2-overlay", "video.mp4", "workout.fit", "--no-interp"])
    assert cli.main() == 0
    assert captured["interpolate"] is False


def test_cli_lint_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _p: "/usr/bin/ffprobe")
    t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "parse_data_file", lambda _p: ParsedData(samples=[Sample(t=t0)]))
    monkeypatch.setattr(
        cli,
        "get_video_metadata",
        lambda *_a, **_kw: (1920, 1080, 10.0, t0, "ffprobe:creation_time"),
    )
    monkeypatch.setattr(cli, "generate_ass", lambda **_kw: None)

    def fake_lint_ass_file(_path: str) -> list[LintIssue]:
        return [LintIssue(code="ASS999", severity="error", message="boom", line_no=1)]

    import c2_overlay.ass_lint as ass_lint

    monkeypatch.setattr(ass_lint, "lint_ass_file", fake_lint_ass_file)

    monkeypatch.setattr(sys, "argv", ["c2-overlay", "video.mp4", "workout.fit", "--lint"])
    assert cli.main() == 1


def test_cli_burn_in_calls_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(p: str) -> str | None:
        if p == "ffprobe":
            return "/usr/bin/ffprobe"
        if p == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return None

    monkeypatch.setattr(cli.shutil, "which", fake_which)
    t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(cli, "parse_data_file", lambda _p: ParsedData(samples=[Sample(t=t0)]))
    monkeypatch.setattr(
        cli,
        "get_video_metadata",
        lambda *_a, **_kw: (1920, 1080, 10.0, t0, "ffprobe:creation_time"),
    )
    monkeypatch.setattr(cli, "generate_ass", lambda **_kw: None)

    burn_calls: list[dict[str, object]] = []

    def fake_burn_in(**kwargs: object) -> None:
        burn_calls.append(kwargs)

    monkeypatch.setattr(cli, "burn_in", fake_burn_in)

    monkeypatch.setattr(
        sys,
        "argv",
        ["c2-overlay", "video.mp4", "workout.fit", "--burn-in", "out.mp4"],
    )
    assert cli.main() == 0
    assert burn_calls and burn_calls[0]["video_out"] == "out.mp4"

