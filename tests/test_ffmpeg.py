from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from c2_overlay.ffmpeg import burn_in


def test_burn_in_builds_cmd_and_escapes_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ass_path = tmp_path / "workout's [data],;=.ass"
    ass_path.write_text("[Script Info]\n", encoding="utf-8")

    video_in_rel = os.path.relpath(tmp_path / "video.mov")
    video_out_rel = os.path.relpath(tmp_path / "out.mp4")

    cmd_out: list[str] | None = None
    cwd_out: str | None = None

    class DummyProc:
        def __init__(self, cmd: list[str], cwd: str) -> None:
            self.cmd = cmd
            self.cwd = cwd
            self.stdout = io.StringIO("")
            self._code = 0

        def wait(self) -> int:
            return self._code

    def fake_popen(cmd: list[str], *, cwd: str, **_kwargs: object) -> DummyProc:
        nonlocal cmd_out, cwd_out
        cmd_out = cmd
        cwd_out = cwd
        return DummyProc(cmd, cwd)

    import c2_overlay.ffmpeg as ffmpeg_mod

    monkeypatch.setattr(ffmpeg_mod.subprocess, "Popen", fake_popen)

    burn_in(
        video_in=video_in_rel,
        ass_path=str(ass_path),
        video_out=video_out_rel,
        ffmpeg_bin="ffmpeg",
        crf=23,
        preset="veryfast",
        copy_audio=True,
    )

    assert cmd_out is not None
    assert cwd_out is not None
    cmd = cmd_out
    assert cmd[0] == "ffmpeg"
    assert cwd_out == str(tmp_path)

    i_idx = cmd.index("-i")
    assert cmd[i_idx + 1] == os.path.abspath(video_in_rel)
    assert cmd[-1] == os.path.abspath(video_out_rel)

    vf_idx = cmd.index("-vf")
    vf = cmd[vf_idx + 1]
    assert "ass=filename='" in vf
    assert "\\'" in vf
    assert "\\[" in vf
    assert "\\]" in vf
    assert "\\," in vf
    assert "\\;" in vf
    assert "\\=" in vf

    # MP4 outputs should be faststart-friendly.
    mov_idx = cmd.index("-movflags")
    assert cmd[mov_idx + 1] == "+faststart"


def test_burn_in_error_includes_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ass_path = tmp_path / "in.ass"
    ass_path.write_text("[Script Info]\n", encoding="utf-8")

    class DummyProc:
        def __init__(self) -> None:
            self.stdout = io.StringIO("line1\nline2\n")

        def wait(self) -> int:
            return 1

    def fake_popen(*_a: object, **_kw: object) -> DummyProc:
        return DummyProc()

    import c2_overlay.ffmpeg as ffmpeg_mod

    monkeypatch.setattr(ffmpeg_mod.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="ffmpeg burn-in failed"):
        burn_in(
            video_in=str(tmp_path / "in.mov"),
            ass_path=str(ass_path),
            video_out=str(tmp_path / "out.mp4"),
            ffmpeg_bin="ffmpeg",
            crf=23,
            preset="veryfast",
            copy_audio=True,
        )
