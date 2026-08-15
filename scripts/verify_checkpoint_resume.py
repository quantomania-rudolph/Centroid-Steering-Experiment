#!/usr/bin/env python
"""Prove checkpoint save/load and quarter-milestone schema."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from jlens_qwen.checkpoint_fit import (
    CHECKPOINT_VERSION,
    atomic_save_checkpoint,
    load_checkpoint,
    quarter_milestones,
)


def main() -> int:
    d = 16
    layers = [1, 2]
    jacobian_sum = {l: torch.randn(d, d) for l in layers}
    partial = {l: torch.randn(d, d) for l in layers}

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "ckpt.pt")
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "jacobian_sum": jacobian_sum,
            "n_done": 3,
            "next_idx": 3,
            "source_layers": layers,
            "target_layer": 7,
            "skip_first": 16,
            "dim_batch": 14,
            "max_seq_len": 128,
            "partial_prompt_index": 3,
            "partial_jacobians": partial,
            "partial_pass_done": 64,
            "backward_passes_total": 256,
        }
        atomic_save_checkpoint(path, payload)
        assert Path(path).exists(), "checkpoint file missing"
        loaded = load_checkpoint(path)
        assert loaded["checkpoint_version"] == CHECKPOINT_VERSION
        assert loaded["n_done"] == 3
        assert loaded["partial_pass_done"] == 64
        assert loaded["partial_prompt_index"] == 3
        for l in layers:
            assert torch.allclose(loaded["jacobian_sum"][l], jacobian_sum[l])
            assert torch.allclose(loaded["partial_jacobians"][l], partial[l])

    marks = quarter_milestones(256)
    assert marks == {64, 128, 192, 256}, f"unexpected milestones: {marks}"

    print("VERIFY CHECKPOINT: PASS")
    print(f"  quarter milestones for 256 passes: {sorted(marks)}")
    print("  atomic save/load round-trip: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
