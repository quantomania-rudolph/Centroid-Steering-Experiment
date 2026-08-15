"""CPU thread and memory limits for Jacobian fitting (machine-safe defaults)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

# Conservative peak RSS for Qwen2.5-Coder-7B fp16 on CPU during backward.
ESTIMATED_MODEL_RAM_GB = 16.0


@dataclass(frozen=True)
class CpuFitPlan:
    logical_cores: int
    omp_threads: int
    workers: int
    max_concurrent: int
    dim_batch: int
    compile_model: bool
    available_ram_gb: float
    notes: list[str]


def compile_supported() -> bool:
    """torch.compile on CPU needs a C++ toolchain (MSVC on Windows)."""
    import sys

    if sys.platform != "win32":
        return True
    import shutil

    return shutil.which("cl") is not None


def logical_cpu_count() -> int:
    return os.cpu_count() or 4


def available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / 1e9


def configure_cpu_threads(omp_threads: int, *, interop_threads: int = 1) -> None:
    """Set BLAS/OpenMP and PyTorch thread counts (call before heavy torch work)."""
    n = max(1, omp_threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(n)
    import torch

    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, interop_threads))


def safe_max_concurrent_models(
    *,
    reserve_gb: float = 6.0,
    per_model_gb: float = ESTIMATED_MODEL_RAM_GB,
    max_fraction_of_total: float = 0.82,
) -> int:
    """How many full-model fit workers can run without likely OOM."""
    mem = psutil.virtual_memory()
    by_available = max(0.0, mem.available / 1e9 - reserve_gb)
    by_total = max(0.0, mem.total / 1e9 * max_fraction_of_total)
    headroom_gb = min(by_available, by_total)
    if headroom_gb < per_model_gb:
        return 1
    return max(1, int(headroom_gb // per_model_gb))


def recommend_dim_batch(*, omp_threads: int, avail_gb: float | None = None) -> int:
    """Larger dim_batch = fewer backward passes but more activation RAM."""
    if avail_gb is None:
        avail_gb = available_ram_gb()
    if avail_gb >= 22:
        return min(16, omp_threads * 2)
    if avail_gb >= 14:
        return min(8, omp_threads)
    return 4


def build_fit_plan(
    *,
    workers: int | None = None,
    omp_threads: int | None = None,
    reserve_cores: int = 1,
    dim_batch: int | None = None,
    compile_model: bool = True,
    aggressive: bool = False,
) -> CpuFitPlan:
    """Pick thread/worker counts that use the machine hard without oversubscribing."""
    cores = logical_cpu_count()
    notes: list[str] = []

    if compile_model and not compile_supported():
        compile_model = False
        notes.append("torch.compile disabled (MSVC cl.exe not found on Windows)")

    omp = omp_threads if omp_threads is not None else max(1, cores - reserve_cores)
    omp = min(omp, cores)

    if workers is None:
        # On memory-bound 7B fits, parallel model copies rarely help; shard for resume.
        workers = 1

    reserve = 4.0 if aggressive else 6.0
    max_conc = safe_max_concurrent_models(reserve_gb=reserve)
    if aggressive and max_conc < 2 and total_ram_gb() >= 30:
        # User opted in: try 2 shards if total RAM can plausibly hold them.
        if total_ram_gb() >= 2 * ESTIMATED_MODEL_RAM_GB + 4:
            max_conc = 2
            notes.append("aggressive: allowing up to 2 concurrent model workers")

    max_conc = min(workers, max_conc)
    if workers > 1 and max_conc == 1:
        notes.append(
            f"{workers} prompt shards will run sequentially (one ~{ESTIMATED_MODEL_RAM_GB:.0f}GB "
            f"model at a time); OMP uses {omp}/{cores} cores per shard"
        )
    elif max_conc > 1:
        notes.append(f"up to {max_conc} model workers in parallel ({omp} OMP threads each)")

    batch = dim_batch if dim_batch is not None else recommend_dim_batch(
        omp_threads=omp, avail_gb=available_ram_gb()
    )

    if cores < 16:
        notes.append(f"machine has {cores} logical CPUs (not 16); using {omp} compute threads")

    return CpuFitPlan(
        logical_cores=cores,
        omp_threads=omp,
        workers=max(1, workers),
        max_concurrent=max(1, max_conc),
        dim_batch=batch,
        compile_model=compile_model,
        available_ram_gb=round(available_ram_gb(), 2),
        notes=notes,
    )


def log_fit_plan(plan: CpuFitPlan) -> None:
    logger.info(
        "CPU fit plan: cores=%d omp_threads=%d workers=%d max_concurrent=%d "
        "dim_batch=%d compile=%s avail_ram=%.1fGB",
        plan.logical_cores,
        plan.omp_threads,
        plan.workers,
        plan.max_concurrent,
        plan.dim_batch,
        plan.compile_model,
        plan.available_ram_gb,
    )
    for note in plan.notes:
        logger.info("  %s", note)
