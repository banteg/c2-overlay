import json
from datetime import UTC
from unittest.mock import MagicMock, patch

from c2_overlay.video import extract_creation_time_tag, get_video_metadata


def test_extract_creation_time_tag_from_format_tags() -> None:
    data = {"format": {"tags": {"creation_time": "2025-01-01T12:00:00Z"}}}
    assert extract_creation_time_tag(data) == "2025-01-01T12:00:00Z"


def test_extract_creation_time_tag_from_stream_tags() -> None:
    data = {
        "format": {},
        "streams": [{"tags": {"creation_time": "2025-01-01T12:00:00Z"}}],
    }
    assert extract_creation_time_tag(data) == "2025-01-01T12:00:00Z"


def test_extract_creation_time_tag_returns_none_when_missing() -> None:
    assert extract_creation_time_tag({}) is None


def test_get_video_metadata_parses_ffprobe_output() -> None:
    mock_output = json.dumps(
        {
            "streams": [{"width": 1920, "height": 1080}],
            "format": {
                "duration": "120.5",
                "tags": {"creation_time": "2025-01-01T12:00:00Z"},
            },
        }
    )

    with patch("c2_overlay.video.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_output, stderr="")
        w, h, dur, creation, source = get_video_metadata("test.mp4", "ffprobe")

    assert (w, h) == (1920, 1080)
    assert dur == 120.5
    assert source == "ffprobe:creation_time"
    assert creation.tzinfo == UTC
