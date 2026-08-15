#!/usr/bin/env python3
"""Destroy a Vast.ai instance via REST API (stops compute billing)."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RENTAL_ROOT / "scripts"))

from load_secrets import load_secrets  # noqa: E402

API_BASE = "https://console.vast.ai/api/v0"


def destroy_instance(api_key: str, instance_id: int) -> dict:
    url = f"{API_BASE}/instances/{instance_id}/"
    req = urllib.request.Request(
        url,
        method="DELETE",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {"success": True}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Destroy Vast.ai instance")
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Override VAST_INSTANCE_ID from secrets.env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be destroyed; do not call API",
    )
    args = parser.parse_args()

    secrets = load_secrets()
    api_key = secrets.get("VAST_API_KEY", "").strip()
    raw_id = args.instance_id or secrets.get("VAST_INSTANCE_ID", "").strip()
    if not api_key:
        print("ERROR: VAST_API_KEY missing in secrets/secrets.env", flush=True)
        return 1
    if not raw_id:
        print("ERROR: VAST_INSTANCE_ID missing (pass --instance-id or set in secrets)", flush=True)
        return 1
    instance_id = int(raw_id)

    if args.dry_run:
        print(f"DRY RUN: would DELETE {API_BASE}/instances/{instance_id}/", flush=True)
        return 0

    print(f"Destroying Vast instance {instance_id}...", flush=True)
    try:
        result = destroy_instance(api_key, instance_id)
    except Exception as exc:
        print(f"DESTROY FAILED: {exc}", flush=True)
        return 1

    print(json.dumps(result, indent=2), flush=True)
    ok = bool(result.get("success", False)) or result.get("msg")
    if ok:
        print("Instance destroy requested — billing for compute should stop.", flush=True)
        return 0
    print("Unexpected response — check Vast console manually.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
