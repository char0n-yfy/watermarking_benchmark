from __future__ import annotations

import os
from pathlib import Path

_LOADED = False
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def load_project_env(*, override: bool = False) -> bool:
    """Load ``.env`` from the repository root into ``os.environ``.

    Existing environment variables are kept unless ``override=True``.
    Returns True when a ``.env`` file was found and loaded.
    """
    global _LOADED
    if _LOADED and not override:
        return False

    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        _LOADED = True
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=override)
    except ImportError:
        _load_env_file_manual(env_file, override=override)

    _LOADED = True
    return True


def _load_env_file_manual(env_file: Path, *, override: bool) -> None:
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
