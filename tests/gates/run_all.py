"""CLI to run phase gates and emit PASS/FAIL report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "gates"

GATE_MARKERS = {
    0: "gate0",
    1: "gate1",
    2: "gate2",
    3: "gate3",
    4: "gate4",
    5: "gate5",
    6: "gate6",
    7: "gate7",
}


def summarize_from_artifacts(through: int, start: int = 0) -> dict:
    """Build a report from existing GATE-*_latest.json without re-running tests."""
    results = []
    overall = True
    for n in range(start, through + 1):
        path = ARTIFACTS / f"GATE-{n}_latest.json"
        if not path.exists():
            results.append(
                {
                    "gate": f"GATE-{n}",
                    "pass": False,
                    "returncode": None,
                    "detail": "missing artifact",
                }
            )
            overall = False
            break
        data = json.loads(path.read_text(encoding="utf-8"))
        passed = bool(data.get("overall_pass"))
        results.append(
            {
                "gate": f"GATE-{n}",
                "pass": passed,
                "returncode": 0 if passed else 1,
                "artifact": str(path),
                "n_pass": data.get("n_pass"),
                "n_fail": data.get("n_fail"),
            }
        )
        if not passed:
            overall = False
            break
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "through": through,
        "start": start,
        "mode": "from_artifacts",
        "overall_pass": overall,
        "results": results,
    }


def run_gates(
    through: int,
    start: int = 0,
    extra_pytest_args: list[str] | None = None,
    *,
    from_artifacts: bool = False,
) -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if from_artifacts:
        report = summarize_from_artifacts(through, start=start)
        out = ARTIFACTS / "run_all_latest.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["overall_pass"] else 1

    results = []
    overall = True
    for n in range(start, through + 1):
        marker = GATE_MARKERS[n]
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests" / "gates"),
            "-m",
            marker,
            "-v",
            "--tb=short",
        ]
        if extra_pytest_args:
            cmd.extend(extra_pytest_args)
        print(f"\n=== Running GATE-{n} ({marker}) ===\n")
        proc = subprocess.run(cmd, cwd=str(ROOT))
        passed = proc.returncode == 0
        results.append(
            {"gate": f"GATE-{n}", "pass": passed, "returncode": proc.returncode}
        )
        if not passed:
            overall = False
            break

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "through": through,
        "start": start,
        "mode": "pytest",
        "overall_pass": overall,
        "results": results,
    }
    out = ARTIFACTS / "run_all_latest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if overall else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run J-Lens phase gates")
    parser.add_argument("--through", type=int, default=0, help="Run gates 0..N")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pytest-arg", action="append", default=[])
    parser.add_argument(
        "--from-artifacts",
        action="store_true",
        help="Summarize existing GATE-*_latest.json without re-running tests",
    )
    args = parser.parse_args()
    return run_gates(
        args.through,
        start=args.start,
        extra_pytest_args=args.pytest_arg,
        from_artifacts=args.from_artifacts,
    )


if __name__ == "__main__":
    raise SystemExit(main())
