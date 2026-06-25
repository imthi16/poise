"""Shared pytest fixtures + path setup for off-device (mock) testing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def cfg():
    """A fresh, validated default config (mock backend, off-device)."""
    from poise.config import load_config

    return load_config()


@pytest.fixture
def db():
    """An in-memory, migrated SQLite connection."""
    from poise.storage.db import init_db

    conn = init_db(":memory:")
    try:
        yield conn
    finally:
        conn.close()
