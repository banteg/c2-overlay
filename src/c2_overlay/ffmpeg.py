from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path


def burn_in(
    video_in: str,
    ass_path: str,
    video_out: str,
    *,
    ffmpeg_bin: str,
    crf: int,
    preset: str,
    copy_audio: bool,
) -> None:
    """
    Burn the ASS subtitles into the video using libass (ffmpeg).

    We run ffmpeg with cwd set to the ASS directory so the filter can reference the file
    without Windows drive-letter escaping pain.
    """
    ass_abs = os.path.abspath(ass_path)
    video_in_abs = os.path.abspath(video_in)
    video_out_abs = os.path.abspath(video_out)
    ass_dir = os.path.dirname(ass_abs) or "."
    ass_name = os.path.basename(ass_abs)

    # Escape for ffmpeg filter syntax (not shell escaping).
    ass_name_escaped = ass_name
    for ch in ("\\", ":", "'", "[", "]", ",", ";", "="):
        ass_name_escaped = ass_name_escaped.replace(ch, f"\\{ch}")
    vf = f"ass=filename='{ass_name_escaped}'"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_in_abs,
        "-vf",
        vf,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
    ]
    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    if Path(video_out_abs).suffix.lower() in {".mp4", ".m4v"}:
        cmd += ["-movflags", "+faststart"]

    cmd += [video_out_abs]

    proc = subprocess.Popen(
        cmd,
        cwd=ass_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    tail: deque[str] = deque(maxlen=80)
    for line in proc.stdout:
        sys.stderr.write(line)
        tail.append(line)
    code = proc.wait()
    if code != 0:
        last = "".join(tail).strip()
        msg = f"ffmpeg burn-in failed (code {code})."
        if last:
            msg += f"\nLast ffmpeg output:\n{last}"
        raise RuntimeError(msg)
