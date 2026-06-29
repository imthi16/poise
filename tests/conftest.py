"""Shared pytest fixtures + path setup for hermetic (host-independent) testing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hermetic test environment: force the mock telemetry backend and CPU device so the
# suite is deterministic and HOST-INDEPENDENT — it must pass identically off-device,
# on a Jetson, or on a GPU box WITHOUT depending on jtop / NVML / nvpmodel / real
# sensors. ``setdefault`` lets a developer opt into real hardware by exporting the var.
# Tests that exercise auto-detection do so by monkeypatching ``poise.hardware`` directly.
os.environ.setdefault("POISE_TELEMETRY_BACKEND", "mock")
os.environ.setdefault("POISE_DEVICE", "cpu")


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
