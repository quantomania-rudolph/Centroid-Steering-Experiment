"""Load secrets.env into os.environ (no values printed)."""

from __future__ import annotations

import os
from pathlib import Path

SECRETS_DIR = Path(__file__).resolve().parents[1] / "secrets"
DEFAULT_SECRETS = SECRETS_DIR / "secrets.env"


def load_secrets(path: Path | None = None) -> dict[str, str]:
    secrets_path = path or DEFAULT_SECRETS
    if not secrets_path.is_file():
        raise FileNotFoundError(
            f"Missing {secrets_path}. Copy secrets.env.example to secrets.env and fill in."
        )
    values: dict[str, str] = {}
    for line in secrets_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
            if val:
                os.environ[key] = val
    return values


def require(keys: list[str], values: dict[str, str] | None = None) -> dict[str, str]:
    data = values or load_secrets()
    missing = [k for k in keys if not data.get(k)]
    if missing:
        raise ValueError(f"Missing required secrets: {', '.join(missing)}")
    return data
