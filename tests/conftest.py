from __future__ import annotations

from pathlib import Path

import pytest

from flight_prefs import paper_data

PAPER_DIR = Path(__file__).resolve().parents[1] / "data" / "paper"


@pytest.fixture(scope="session")
def data() -> paper_data.PaperData:
    if not (PAPER_DIR / paper_data.HELDOUT_FILE).exists():
        pytest.skip("paper data not downloaded (run scripts/download_paper_data.py)")
    return paper_data.load(PAPER_DIR)
