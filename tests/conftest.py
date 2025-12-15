from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_fit(fixtures_dir: Path) -> Path:
    return fixtures_dir / "concept2-logbook-workout-109762845.fit"


@pytest.fixture
def sample_video(fixtures_dir: Path) -> Path:
    return fixtures_dir / "white_5s_1080p.mp4"
