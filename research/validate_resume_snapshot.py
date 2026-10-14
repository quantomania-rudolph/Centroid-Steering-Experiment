"""CI guard: committed resume snapshot matches documented headline claims."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "docs/resume_metrics_snapshot.json"

REQUIRED = {
    "live_cast_pairs": 15,
    "demo_panic_pass": 12,
    "demo_neutral_pass": 5,
    "demo_panic_pct": 80.0,
    "demo_neutral_pct": 33.3,
    "demo_lift_pp": 46.7,
    "demo_relative_success_x": 2.4,
    "cast_alpha_0": 0.85,
    "caa_natural_pairs": 39,
    "production_ready": False,
}


def main() -> int:
    if not SNAP.is_file():
        print(f"missing {SNAP}", file=sys.stderr)
        return 1
    data = json.loads(SNAP.read_text(encoding="utf-8"))
    errors: list[str] = []
    for key, expected in REQUIRED.items():
        if data.get(key) != expected:
            errors.append(f"{key}: got {data.get(key)!r}, want {expected!r}")
    gates = data.get("gates_pass") or {}
    for name, val in gates.items():
        if name == "caa_steering_status":
            continue
        if val is not True:
            errors.append(f"gates_pass.{name}: {val!r}")
    if data.get("gates_pass", {}).get("caa_steering_status") is not True:
        errors.append("gates_pass.caa_steering_status must be true (master gate)")
    pair_ids = data.get("live_cast_pair_ids") or []
    if len(pair_ids) != 15:
        errors.append(f"live_cast_pair_ids length {len(pair_ids)} != 15")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("resume_metrics_snapshot.json OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
