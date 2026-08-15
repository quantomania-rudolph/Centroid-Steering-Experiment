#!/usr/bin/env python
"""Verify OpenRouter connectivity — FREE generation models only (no paid unless --allow-paid-qa)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from openrouter_curator_client import (  # noqa: E402
    OpenRouterCuratorClient,
    generation_models,
    load_curator_config,
    qa_model,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Test OpenRouter free model connectivity")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "curator_models.yaml")
    ap.add_argument(
        "--test-qa",
        action="store_true",
        help="Also ping paid DeepSeek QA model (costs a few tokens)",
    )
    ap.add_argument("--allow-paid-qa", action="store_true", help="Required with --test-qa")
    args = ap.parse_args()

    cfg = load_curator_config(args.config)
    client = OpenRouterCuratorClient(config=cfg)

    free_models = generation_models(cfg)
    report: dict[str, object] = {
        "generation_free_only": True,
        "n_free_models": len(free_models),
        "free_models": [m.id for m in free_models],
        "generation_pings": client.ping_all_free_generation(),
    }

    n_ok = sum(1 for r in report["generation_pings"] if r["ok"])  # type: ignore[index]
    report["generation_ok"] = n_ok
    report["generation_pass"] = n_ok > 0

    if args.test_qa:
        if not args.allow_paid_qa:
            print("Use --allow-paid-qa to test paid DeepSeek QA model", file=sys.stderr)
            return 2
        qm = qa_model(cfg)
        qa_resp = client.ping_model(qm.id, role="qa")
        report["qa_model"] = qm.id
        report["qa_ping"] = {
            "ok": qa_resp.ok,
            "error_class": qa_resp.error_class,
            "detail": (qa_resp.detail or "")[:200],
            "preview": (qa_resp.content or "")[:80],
        }
        report["qa_pass"] = qa_resp.ok

    out_path = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "openrouter_connection_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Redact any key material from stdout
    print(json.dumps(report, indent=2))
    print(f"\nreport: {out_path}")

    if not report["generation_pass"]:
        return 1
    if args.test_qa and not report.get("qa_pass"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
