#!/usr/bin/env python
"""Delete v1/v2 bad calm tails and reset curator state for v3 pilot."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.calm_tail_registry import find_calm_tail  # noqa: E402  # used by manifest docs

CALM_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_tails"
PROOF = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_hardening_proof"
CHECKPOINT = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_checkpoint.json"
BRIEFS_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_briefs"
PY = sys.executable

BAD_FILES = [
    "unicaa01_p01_panic__chatcmpl_8b67ddd03286__d16.txt",
    "unicaa01_p06_panic__chatcmpl_44fb68e7b5cd__d3.txt",
    "unicaa01_p12_panic__chatcmpl_9fcda159903f__d14.txt",
    "unicaa01_p18_panic__chatcmpl_2ba03f73cbb9__d3.txt",
]


def main() -> int:
    manifest: list[dict[str, str]] = []
    deleted = 0
    for name in BAD_FILES:
        txt = CALM_DIR / name
        meta = txt.with_suffix(".meta.json")
        reason = "v3 clean sweep — reject-quality tail"
        if txt.is_file():
            manifest.append({"file": name, "reason": reason, "gate_evidence": "pre-v3 quality"})
            txt.unlink()
            deleted += 1
        if meta.is_file():
            meta.unlink()

    (CALM_DIR / "_rejected_manifest.json").write_text(json.dumps({"rejected": manifest}, indent=2), encoding="utf-8")

    if CHECKPOINT.is_file():
        ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        ck["completed"] = []
        CHECKPOINT.write_text(json.dumps(ck, indent=2), encoding="utf-8")

    status_path = BRIEFS_DIR / "curation_status.json"
    if BRIEFS_DIR.is_dir():
        entries = {}
        for p in BRIEFS_DIR.glob("*.json"):
            if p.name in {"index.json", "skipped.json", "curation_status.json"}:
                continue
            bid = p.stem
            entries[bid] = {"status": "pending_v3"}
        status_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    n_txt = len(list(CALM_DIR.glob("*.txt")))
    proc = subprocess.run(
        [PY, "research/caa_mine_natural_pairs.py", "--unicaa01-corpus", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    miner_out = (proc.stdout or "").strip()
    n_admitted = 0
    try:
        miner_data = json.loads(miner_out)
        n_admitted = int(miner_data.get("n_admitted") or 0)
    except json.JSONDecodeError:
        for line in reversed(miner_out.splitlines()):
            if line.strip().startswith("{"):
                try:
                    miner_data = json.loads(line)
                    n_admitted = int(miner_data.get("n_admitted") or 0)
                    break
                except json.JSONDecodeError:
                    continue

    # verify former paths deleted
    former_none = all(not (CALM_DIR / name).is_file() for name in BAD_FILES)

    payload = {
        "pass": n_txt == 0 and n_admitted == 0 and former_none,
        "n_deleted": deleted,
        "n_calm_tail_txt": n_txt,
        "n_admitted": n_admitted,
        "former_paths_gone": former_none,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    (PROOF / "G3_clean_sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
