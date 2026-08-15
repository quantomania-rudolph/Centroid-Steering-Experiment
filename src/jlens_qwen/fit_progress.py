"""Atomic JSON progress files for Jacobian fits (dashboard + ETA)."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PROGRESS_PATH = Path("artifacts/gates/fit_progress.json")
DEFAULT_JOB_PATH = Path("artifacts/gates/fit_job.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class FitJobMeta:
    job_id: str
    pid: int
    started_at: str
    script: str
    source_layers: str
    n_prompts: int
    dim_batch: int
    max_seq_len: int
    omp_threads: int
    compile_model: bool
    out_path: str
    checkpoint_path: str
    d_model: int = 3584
    status: str = "running"
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path = DEFAULT_JOB_PATH) -> None:
        _atomic_write_json(path, asdict(self))


@dataclass
class FitProgress:
    updated_at: str
    phase: str
    prompt_index: int
    n_prompts: int
    backward_pass: int
    backward_passes_total: int
    pass_rate_per_min: float | None
    eta_seconds: float | None
    elapsed_seconds: float
    notes: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        if self.backward_passes_total <= 0:
            return 0.0
        prompt_frac = self.prompt_index / max(1, self.n_prompts)
        pass_frac = self.backward_pass / max(1, self.backward_passes_total)
        # Weight current prompt backward work inside overall job
        per_prompt = 1.0 / max(1, self.n_prompts)
        return min(1.0, prompt_frac + pass_frac * per_prompt)

    def save(self, path: Path = DEFAULT_PROGRESS_PATH) -> None:
        payload = asdict(self)
        payload["fraction"] = round(self.fraction, 4)
        _atomic_write_json(path, payload)


class FitProgressTracker:
    """Tracks backward-pass progress via autograd.grad hook."""

    def __init__(
        self,
        *,
        n_prompts: int,
        dim_batch: int,
        d_model: int = 3584,
        job_started_at: float | None = None,
        progress_path: Path = DEFAULT_PROGRESS_PATH,
    ):
        self.n_prompts = n_prompts
        self.dim_batch = dim_batch
        self.d_model = d_model
        self.backward_passes_total = math.ceil(d_model / dim_batch)
        self.progress_path = progress_path
        self.job_started_at = job_started_at or time.time()
        self.prompt_index = 0
        self.backward_pass = 0
        self.phase = "init"
        self._pass_times: list[float] = []
        self._last_pass_wall = time.time()
        self._grad_hook_installed = False

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self._flush()

    def on_prompt_start(self, prompt_index: int) -> None:
        self.prompt_index = prompt_index
        self.backward_pass = 0
        self.phase = "backward"
        self._last_pass_wall = time.time()
        self._flush()

    def on_backward_pass(self) -> None:
        now = time.time()
        dt = now - self._last_pass_wall
        self._last_pass_wall = now
        self.backward_pass += 1
        if dt > 0:
            self._pass_times.append(dt)
            if len(self._pass_times) > 40:
                self._pass_times = self._pass_times[-40:]
        self._flush()

    def mark_done(self) -> None:
        self.phase = "done"
        self.backward_pass = self.backward_passes_total
        self.prompt_index = self.n_prompts
        self._flush()

    def _eta_seconds(self) -> float | None:
        if self.backward_pass < 2 or not self._pass_times:
            return None
        avg = sum(self._pass_times) / len(self._pass_times)
        remaining_passes = self.backward_passes_total - self.backward_pass
        remaining_prompts = max(0, self.n_prompts - self.prompt_index - 1)
        return remaining_passes * avg + remaining_prompts * self.backward_passes_total * avg

    def _pass_rate_per_min(self) -> float | None:
        if len(self._pass_times) < 2:
            return None
        avg = sum(self._pass_times) / len(self._pass_times)
        if avg <= 0:
            return None
        return 60.0 / avg

    def _flush(self) -> None:
        FitProgress(
            updated_at=_utc_now(),
            phase=self.phase,
            prompt_index=self.prompt_index,
            n_prompts=self.n_prompts,
            backward_pass=self.backward_pass,
            backward_passes_total=self.backward_passes_total,
            pass_rate_per_min=self._pass_rate_per_min(),
            eta_seconds=self._eta_seconds(),
            elapsed_seconds=time.time() - self.job_started_at,
            notes=[],
        ).save(self.progress_path)


def backward_passes_total(dim_batch: int, d_model: int = 3584) -> int:
    return math.ceil(d_model / dim_batch)


def install_autograd_progress_hook(tracker: FitProgressTracker) -> Any:
    """Count torch.autograd.grad calls during jlens backward loop."""
    import torch

    original = torch.autograd.grad

    def counting_grad(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if tracker.phase in ("fit", "backward"):
            tracker.on_backward_pass()
        return result

    torch.autograd.grad = counting_grad  # type: ignore[method-assign]
    tracker._grad_hook_installed = True
    return original


def restore_autograd_grad(original: Any) -> None:
    import torch

    torch.autograd.grad = original  # type: ignore[method-assign]
