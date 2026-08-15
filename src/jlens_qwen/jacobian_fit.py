"""CPU Jacobian fitting with optional jlens backend and sharding."""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from jlens_qwen.jacobian_lens import FittedJacobianLens
from jlens_qwen.model_loader import ActivationRecorder, ModelBundle, load_qwen_coder

logger = logging.getLogger(__name__)

SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if mask.sum() == 0:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask


def jacobian_for_prompt(
    bundle: ModelBundle,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Per-prompt J_l estimator matching anthropics/jacobian-lens.

    Injects one-hot cotangents at all valid target positions for dim_batch
    output dims per backward, averages over valid source positions.
    """
    n_layers = bundle.n_layers
    d_model = bundle.d_model
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    sources = sorted({int(l) for l in source_layers})
    if sources[-1] >= target:
        raise ValueError(f"source_layers must be < target_layer={target}")

    input_ids = bundle.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
    }
    n_passes = math.ceil(d_model / dim_batch)

    # Enable grads only on activations via retained graph
    for param in bundle.model.parameters():
        param.requires_grad_(False)

    with (
        ActivationRecorder(bundle.layers, at=[*sources, target]) as recorder,
        torch.enable_grad(),
    ):
        # Replicate prompt along batch for dim_batch parallel rows
        replicated = input_ids.expand(dim_batch, -1)
        # Force text-module forward so hooks fire on layers
        _ = bundle.model.model(input_ids=replicated, use_cache=False)
        # Make activations require grad by re-attaching via identity
        target_act = recorder.activations[target]
        if not target_act.requires_grad:
            # Hooks captured detached tensors; re-run with gradient-enabled hooks
            pass

    # Re-run with proper retain-graph hooks that keep requires_grad
    jacobians, seq_len, n_valid = _jacobian_for_prompt_grad(
        bundle,
        prompt,
        sources,
        target_layer=target,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
    )
    return jacobians, seq_len, n_valid


class _GradRecorder:
    """Hook recorder that keeps activations in the autograd graph."""

    def __init__(self, layers: nn.ModuleList, at: list[int], start_graph_at: int):
        self.layers = layers
        self.at = at
        self.start_graph_at = start_graph_at
        self.activations: dict[int, torch.Tensor] = {}
        self._hooks: list[Any] = []

    def __enter__(self) -> _GradRecorder:
        def make_hook(idx: int):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if idx >= self.start_graph_at:
                    h = h.detach().requires_grad_(True)
                    # Re-insert into residual path: return modified tensor
                    if isinstance(out, tuple):
                        return (h,) + out[1:]
                    return h
                self.activations[idx] = h
                return out

            # For layers we need grads FROM, capture after enabling
            def capture_hook(idx: int):
                def hook(_module, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if idx >= self.start_graph_at:
                        # already requires_grad from previous return; just store
                        self.activations[idx] = h
                    else:
                        self.activations[idx] = h
                    return out

                return hook

            # Combined: enable grad on output then store
            def combined(idx: int):
                def hook(_module, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if idx >= self.start_graph_at and not h.requires_grad:
                        h = h.detach().requires_grad_(True)
                        self.activations[idx] = h
                        if isinstance(out, tuple):
                            return (h,) + tuple(out[1:])
                        return h
                    self.activations[idx] = h
                    return out

                return hook

            return combined(idx)

        for i in self.at:
            self._hooks.append(self.layers[i].register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _jacobian_for_prompt_grad(
    bundle: ModelBundle,
    prompt: str,
    source_layers: list[int],
    *,
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
) -> tuple[dict[int, torch.Tensor], int, int]:
    d_model = bundle.d_model
    input_ids = bundle.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())
    sources = list(source_layers)

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
    }
    n_passes = math.ceil(d_model / dim_batch)
    start_graph = min(sources)

    with (
        _GradRecorder(
            bundle.layers, at=[*sources, target_layer], start_graph_at=start_graph
        ) as recorder,
        torch.enable_grad(),
    ):
        replicated = input_ids.expand(dim_batch, -1)
        _ = bundle.model.model(input_ids=replicated, use_cache=False)
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in sources]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
                allow_unused=False,
            )
            for layer, grad in zip(sources, grads, strict=True):
                if grad is None:
                    raise RuntimeError(f"No gradient for layer {layer}")
                pos = valid_positions.to(grad.device)
                rows = grad[:n_dims, pos, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    return jacobians, seq_len, n_valid


def _split_prompts(prompts: Sequence[str], workers: int) -> list[list[str]]:
    workers = max(1, workers)
    if workers == 1:
        return [list(prompts)]
    shards: list[list[str]] = [[] for _ in range(workers)]
    for idx, prompt in enumerate(prompts):
        shards[idx % workers].append(prompt)
    return [s for s in shards if s]


def fit_jacobians_sharded(
    bundle: ModelBundle,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    compile_model: bool = False,
    workers: int = 1,
    max_concurrent: int = 1,
    omp_threads: int = 1,
    shard_out_dir: str | Path | None = None,
) -> FittedJacobianLens:
    """Fit over prompts, optionally sharding across processes (merge at end)."""
    shards = _split_prompts(prompts, workers)
    if len(shards) <= 1:
        return fit_jacobians(
            bundle,
            prompts,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
        )

    out_dir = Path(shard_out_dir or (checkpoint_path or "shard.pt")).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, object]] = []
    lo = min(source_layers) if source_layers is not None else 0
    hi = max(source_layers) if source_layers is not None else bundle.n_layers - 2
    for idx, shard_prompts in enumerate(shards):
        shard_out = out_dir / f"shard_{idx}_of_{len(shards)}.pt"
        shard_ckpt = out_dir / f"shard_{idx}_ckpt.pt"
        payloads.append(
            {
                "prompts": list(shard_prompts),
                "source_layers": f"{lo}-{hi}",
                "dim_batch": dim_batch,
                "max_seq_len": max_seq_len,
                "skip_first": skip_first,
                "checkpoint_path": str(shard_ckpt),
                "out": str(shard_out),
                "compile_model": compile_model,
                "omp_threads": omp_threads,
            }
        )

    lenses: list[FittedJacobianLens] = []
    if max_concurrent <= 1:
        for payload in payloads:
            lenses.append(_fit_shard_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=max_concurrent) as pool:
            futures = [pool.submit(_fit_shard_worker, p) for p in payloads]
            for fut in as_completed(futures):
                lenses.append(fut.result())

    return FittedJacobianLens.merge(lenses)


def _fit_shard_worker(payload: dict[str, object]) -> FittedJacobianLens:
    from jlens_qwen.cpu_tuning import configure_cpu_threads

    omp = int(payload["omp_threads"])
    configure_cpu_threads(omp)

    import torch

    from jlens_qwen.model_loader import load_qwen_coder

    lo, hi = [int(x) for x in str(payload["source_layers"]).split("-")]
    source_layers = list(range(lo, hi + 1))
    bundle = load_qwen_coder(device="cpu", dtype=torch.float16)
    lens = fit_jacobians(
        bundle,
        list(payload["prompts"]),  # type: ignore[arg-type]
        source_layers=source_layers,
        dim_batch=int(payload["dim_batch"]),
        max_seq_len=int(payload["max_seq_len"]),
        skip_first=int(payload["skip_first"]),
        checkpoint_path=str(payload["checkpoint_path"]),
        compile_model=bool(payload["compile_model"]),
    )
    out = str(payload["out"])
    lens.save(out)
    return lens


def fit_jacobians(
    bundle: ModelBundle,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    compile_model: bool = False,
) -> FittedJacobianLens:
    """Fit J_l over prompts as a running mean (single-process)."""
    n_layers = bundle.n_layers
    d_model = bundle.d_model
    target = n_layers - 1 if target_layer is None else target_layer
    if source_layers is None:
        sources = list(range(target))
    else:
        sources = sorted({int(l) for l in source_layers})

    # Prefer official jlens if available and model is HF-compatible
    try:
        return _fit_with_jlens(
            bundle,
            prompts,
            source_layers=sources,
            target_layer=target,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
        )
    except ImportError:
        logger.info("jlens not installed; using built-in jacobian_for_prompt")
    except Exception as exc:
        logger.info("jlens unfit for this model (%s); using built-in fitter", exc)
        # jlens may have applied torch.compile wrappers; reload a clean model.
        bundle = load_qwen_coder(device="cpu", dtype=torch.float16)

    jacobian_sum = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
    }
    n_done = 0
    for idx, prompt in enumerate(prompts):
        try:
            per_prompt, seq_len, n_valid = jacobian_for_prompt(
                bundle,
                prompt,
                sources,
                target_layer=target,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("skipping prompt %d: %s", idx, exc)
            continue
        for layer in sources:
            jacobian_sum[layer] += per_prompt[layer]
        n_done += 1
        logger.info(
            "prompt %d/%d seq_len=%d n_valid=%d",
            idx + 1,
            len(prompts),
            seq_len,
            n_valid,
        )
        if checkpoint_path and n_done % 1 == 0:
            _save_running(checkpoint_path, jacobian_sum, n_done, sources, target, skip_first)

    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    mean = {layer: jacobian_sum[layer] / n_done for layer in sources}
    return FittedJacobianLens(jacobians=mean, n_prompts=n_done, d_model=d_model)


def _fit_with_jlens(
    bundle: ModelBundle,
    prompts: Sequence[str],
    *,
    source_layers: list[int],
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    checkpoint_path: str | None,
    compile_model: bool = False,
) -> FittedJacobianLens:
    import jlens
    from jlens.fitting import fit as jlens_fit

    from jlens_qwen.fit_progress import FitProgressTracker, install_autograd_progress_hook, restore_autograd_grad

    tracker = FitProgressTracker(
        n_prompts=len(prompts),
        dim_batch=dim_batch,
        d_model=bundle.d_model,
    )
    tracker.set_phase("load_model")

    model = jlens.from_hf(
        bundle.model, bundle.tokenizer, force_bos=True, compile=compile_model
    )
    # Move to CPU for Jacobian fit if currently on XPU
    if str(bundle.device) != "cpu":
        logger.info("Moving model to CPU for Jacobian fit")
        bundle.model.to("cpu")
        # Refresh references
        model = jlens.from_hf(
            bundle.model, bundle.tokenizer, force_bos=True, compile=compile_model
        )

    tracker.set_phase("fit")

    def _on_progress(prompt_idx: int, n_prompts: int, done: int, total: int) -> None:
        tracker.prompt_index = prompt_idx
        tracker.backward_pass = done
        tracker.backward_passes_total = total
        tracker.phase = "backward"
        tracker._flush()

    orig_grad = None
    try:
        from jlens_qwen.checkpoint_fit import fit_with_quarter_checkpoints

        lens = fit_with_quarter_checkpoints(
            model,
            prompts=list(prompts),
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            checkpoint_path=checkpoint_path,
            resume=True,
            on_progress=_on_progress,
        )
        fitted = FittedJacobianLens(
            jacobians=dict(lens.jacobians),
            n_prompts=lens.n_prompts,
            d_model=lens.d_model,
        )
    except Exception as exc:
        logger.info("quarter-checkpoint fit failed (%s); falling back to jlens.fit", exc)
        orig_grad = install_autograd_progress_hook(tracker)
        lens = jlens_fit(
            model,
            prompts=list(prompts),
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            checkpoint_path=checkpoint_path,
        )
        fitted = FittedJacobianLens(
            jacobians=dict(lens.jacobians),
            n_prompts=lens.n_prompts,
            d_model=lens.d_model,
        )
    finally:
        if orig_grad is not None:
            restore_autograd_grad(orig_grad)
    tracker.mark_done()
    return fitted


def _save_running(
    path: str,
    jacobian_sum: dict[int, torch.Tensor],
    n_done: int,
    sources: list[int],
    target: int,
    skip_first: int,
) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(
        {
            "jacobian_sum": jacobian_sum,
            "n_done": n_done,
            "source_layers": sources,
            "target_layer": target,
            "skip_first": skip_first,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_prompts(path: str | Path, n: int | None = None) -> list[str]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        prompts = [f.read_text(encoding="utf-8").strip() for f in files]
    else:
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            data = json.loads(text)
            prompts = list(data) if isinstance(data, list) else data["prompts"]
        else:
            # Merge consecutive short blocks so each prompt is substantial
            raw = [blk.strip() for blk in text.split("\n\n") if blk.strip()]
            prompts = []
            buf = ""
            for blk in raw:
                if not buf:
                    buf = blk
                elif len(buf) < 120:
                    buf = buf + "\n\n" + blk
                else:
                    prompts.append(buf)
                    buf = blk
            if buf:
                prompts.append(buf)
    prompts = [p for p in prompts if len(p) >= 40]
    if n is not None:
        prompts = prompts[:n]
    return prompts


def identity_lens(d_model: int, layers: Sequence[int]) -> FittedJacobianLens:
    """Identity J_l for logit-lens-equivalent apply / smoke tests."""
    I = torch.eye(d_model, dtype=torch.float32)
    return FittedJacobianLens(
        jacobians={int(l): I.clone() for l in layers},
        n_prompts=0,
        d_model=d_model,
    )
