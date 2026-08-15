"""Load OpenRouter API key from secrets/openrouter.env (gitignored)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT / "secrets" / "openrouter.env"


def load_openrouter_api_key(*, env_path: Path | None = None) -> str:
    """Return OPENROUTER_API_KEY from environment or secrets file."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    path = env_path or DEFAULT_ENV_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"OpenRouter key not found. Set OPENROUTER_API_KEY or create {path}"
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("OPENROUTER_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                return key

    raise ValueError(f"OPENROUTER_API_KEY is empty in {path}")
