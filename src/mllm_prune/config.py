"""Load machine-specific paths from the project-local ``.env`` file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def path_from_env(name: str) -> Path | None:
    """Return a configured path, resolving relative values from the repository."""

    value = os.getenv(name)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def value_from_env(name: str, default: str | None = None) -> str | None:
    """Return a non-empty string setting from the project environment."""

    return os.getenv(name) or default

