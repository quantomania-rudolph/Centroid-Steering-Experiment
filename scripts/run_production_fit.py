#!/usr/bin/env python
"""Run production Jacobian fit (layers 8-26, 100 prompts)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Production J-lens fit")
    parser.add_argument(
        "--smoke-lens",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens_L15.pt",
    )
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--source-layers", default="8-26")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "data" / "prompts" / "corpus_100.json",
    )
    parser.add_argument("--dim-batch", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "checkpoints" / "fit_production.pt",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip smoke lens gate (use only if L15 already verified)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.skip_verify:
        verify_proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_lens_smoke.py"),
                "--lens",
                str(args.smoke_lens),
                "--out",
                str(ROOT / "artifacts" / "gates" / "lens_smoke_verify.json"),
            ],
            cwd=str(ROOT),
        )
        if verify_proc.returncode != 0:
            print("SMOKE VERIFY FAILED — production fit blocked.", file=sys.stderr)
            return 1
        print("SMOKE VERIFY PASSED — starting production fit.")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fit_jacobian.py"),
        "--prompts",
        str(args.prompts),
        "--n-prompts",
        str(args.n_prompts),
        "--source-layers",
        args.source_layers,
        "--device",
        "cpu",
        "--max-seq-len",
        str(args.max_seq_len),
        "--no-compile",
        "--out",
        str(args.out),
        "--checkpoint",
        str(args.checkpoint),
    ]
    if args.dim_batch is not None:
        cmd.extend(["--dim-batch", str(args.dim_batch)])
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.aggressive:
        cmd.append("--aggressive")

    print("Production command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    log_path = ROOT / "artifacts" / "gates" / "fit_production.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n--- starting production fit 8-26 x 100 ---\n")
        log.write(" ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
