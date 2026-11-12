import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _no_unpack_reservation(monkeypatch):
    """Unpacks in tests are fakes. Holding real room would make them wait on this host's /tmp."""
    monkeypatch.setenv("DGM_UNPACK_RESERVE_BYTES", "0")
