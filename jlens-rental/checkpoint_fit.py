"""Resumable Jacobian fitting with per-prompt and quarter-pass checkpoints."""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)


class FittedJacobianLens:
    """Minimal stand-in for rental GPU fit (same fields as jlens_qwen)."""

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ):
        self.jacobians = {int(k): v for k, v in jacobians.items()}
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.source_layers = sorted(self.jacobians)

CHECKPOINT_VERSION = 1


def atomic_save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def quarter_milestones(n_passes: int) -> set[int]:
    if n_passes <= 0:
        return set()
    marks = {n_passes}
    for frac in (0.25, 0.5, 0.75):
        marks.add(max(1, min(n_passes, int(round(n_passes * frac)))))
    return marks


def jacobian_for_prompt_resumable(
    model: Any,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    partial_jacobians: dict[int, torch.Tensor] | None = None,
    start_pass: int = 0,
    on_pass: Callable[[int, int, dict[int, torch.Tensor]], None] | None = None,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Like jlens.jacobian_for_prompt but can resume mid-pass loop."""
    from jlens.fitting import valid_position_mask
    from jlens.hooks import ActivationRecorder

    n_layers, d_model = model.n_layers, model.d_model
    sources = list(source_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())

    if partial_jacobians is None:
        jacobians = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
        }
    else:
        jacobians = {layer: partial_jacobians[layer].clone() for layer in sources}

    n_passes = math.ceil(d_model / dim_batch)
    start_pass = max(0, min(start_pass, n_passes))

    with (
        ActivationRecorder(
            model.layers,
            at=[*sources, target_layer],
            start_graph_at=min(sources),
        ) as recorder,
        torch.enable_grad(),
    ):
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in sources]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            if pass_idx < start_pass:
                continue
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
            )
            for layer, grad in zip(sources, grads, strict=True):
                positions_on_device = valid_positions.to(grad.device)
                rows = grad[:n_dims, positions_on_device, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads
            if on_pass is not None:
                on_pass(pass_idx + 1, n_passes, jacobians)

    return jacobians, seq_len, n_valid


def fit_with_quarter_checkpoints(
    model: Any,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int],
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    checkpoint_path: str | None,
    resume: bool = True,
    on_progress: Callable[[int, int, int, int], None] | None = None,
) -> FittedJacobianLens:
    """Fit with checkpoints after each prompt and at 25/50/75/100% of each prompt."""
    sources = list(source_layers)
    d_model = model.d_model
    n_passes = math.ceil(d_model / dim_batch)
    milestones = quarter_milestones(n_passes)

    jacobian_sum: dict[int, torch.Tensor]
    n_done = 0
    next_idx = 0
    partial_prompt_index: int | None = None
    partial_jacobians: dict[int, torch.Tensor] | None = None
    partial_pass_done = 0

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        state = load_checkpoint(checkpoint_path)
        if state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {state.get('checkpoint_version')}")
        for key, expected in (
            ("source_layers", sources),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
            ("dim_batch", dim_batch),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint {key}={state[key]!r} != expected {expected!r}"
                )
        jacobian_sum = state["jacobian_sum"]
        n_done = int(state["n_done"])
        next_idx = int(state["next_idx"])
        partial_prompt_index = state.get("partial_prompt_index")
        partial_jacobians = state.get("partial_jacobians")
        partial_pass_done = int(state.get("partial_pass_done", 0))
        logger.info(
            "resuming checkpoint: prompts %d/%d partial_prompt=%s partial_pass=%d/%d",
            next_idx,
            len(prompts),
            partial_prompt_index,
            partial_pass_done,
            n_passes,
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
        }

    def write_checkpoint(
        *,
        partial_idx: int | None,
        partial_j: dict[int, torch.Tensor] | None,
        pass_done: int,
    ) -> None:
        if checkpoint_path is None:
            return
        atomic_save_checkpoint(
            checkpoint_path,
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "jacobian_sum": jacobian_sum,
                "n_done": n_done,
                "next_idx": next_idx,
                "source_layers": sources,
                "target_layer": target_layer,
                "skip_first": skip_first,
                "dim_batch": dim_batch,
                "max_seq_len": max_seq_len,
                "partial_prompt_index": partial_idx,
                "partial_jacobians": partial_j,
                "partial_pass_done": pass_done,
                "backward_passes_total": n_passes,
                "saved_at": time.time(),
            },
        )
        logger.info(
            "checkpoint saved: n_done=%d next_idx=%d partial_prompt=%s pass=%d/%d",
            n_done,
            next_idx,
            partial_idx,
            pass_done,
            n_passes,
        )

    sqrt_d = math.sqrt(d_model)

    # Resume mid-prompt if needed
    if partial_prompt_index is not None and partial_jacobians is not None:
        prompt_idx = partial_prompt_index
        if prompt_idx < len(prompts):

            def _on_pass(done: int, total: int, partial: dict[int, torch.Tensor]) -> None:
                if on_progress is not None:
                    on_progress(prompt_idx, len(prompts), done, total)
                if done in milestones:
                    write_checkpoint(
                        partial_idx=prompt_idx,
                        partial_j=partial,
                        pass_done=done,
                    )

            try:
                per_prompt_J, seq_len, n_valid = jacobian_for_prompt_resumable(
                    model,
                    prompts[prompt_idx],
                    sources,
                    target_layer=target_layer,
                    dim_batch=dim_batch,
                    max_seq_len=max_seq_len,
                    skip_first=skip_first,
                    partial_jacobians=partial_jacobians,
                    start_pass=partial_pass_done,
                    on_pass=_on_pass,
                )
            except ValueError as exc:
                logger.warning("skipping partial prompt %d on resume: %s", prompt_idx, exc)
                next_idx = prompt_idx + 1
                partial_prompt_index = None
                partial_jacobians = None
                partial_pass_done = 0
                write_checkpoint(partial_idx=None, partial_j=None, pass_done=0)
            else:
                for layer in sources:
                    jacobian_sum[layer] += per_prompt_J[layer]
                n_done += 1
                next_idx = prompt_idx + 1
                partial_prompt_index = None
                partial_jacobians = None
                partial_pass_done = 0
                write_checkpoint(partial_idx=None, partial_j=None, pass_done=0)
                logger.info(
                    "  prompt %d/%d (resumed) seq_len=%d n_valid=%d",
                    prompt_idx + 1,
                    len(prompts),
                    seq_len,
                    n_valid,
                )

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()

        def _on_pass(
            done: int,
            total: int,
            partial: dict[int, torch.Tensor],
            *,
            _prompt_idx: int = prompt_idx,
        ) -> None:
            if on_progress is not None:
                on_progress(_prompt_idx, len(prompts), done, total)
            if done in milestones:
                write_checkpoint(
                    partial_idx=_prompt_idx,
                    partial_j={k: v.clone() for k, v in partial.items()},
                    pass_done=done,
                )

        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt_resumable(
                model,
                prompt,
                sources,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                on_pass=_on_pass,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            write_checkpoint(partial_idx=None, partial_j=None, pass_done=0)
            continue

        prompt_norm = max(per_prompt_J[l].norm().item() for l in sources) / sqrt_d
        for layer in sources:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1
        write_checkpoint(partial_idx=None, partial_j=None, pass_done=0)
        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  max||J||/sqrt(d)=%.3f",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
        )

    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in sources}
    return FittedJacobianLens(
        jacobians=jacobian_mean,
        n_prompts=n_done,
        d_model=d_model,
    )
