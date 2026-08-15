"""Custom KV-cache decode loop with live A–G chaos instrumentation."""

from __future__ import annotations

import time
from typing import Any

import torch

from jlens_qwen.chaos.bus import ChaosBus
from jlens_qwen.chaos.monitor_bus import MonitorChaosBus
from jlens_qwen.chaos.constants import F2_STRIDE, PROBE_LAYERS, WINDOW_C
from jlens_qwen.chaos.flags import THRESH
from jlens_qwen.chaos.metrics_c import TrajectoryTracker
from jlens_qwen.chaos.metrics_d import cross_position
from jlens_qwen.chaos.metrics_e import layer_curve_metrics
from jlens_qwen.chaos.metrics_f import lens_disagreement
from jlens_qwen.chaos.metrics_g import sparse_residual
from jlens_qwen.chaos.probe import ProbeEngine
from jlens_qwen.model_loader import ModelBundle


def _probe_f_metrics(
    probe: ProbeEngine,
    layer: int,
    sample_ids: list[int],
) -> dict[str, float] | None:
    """J-lens vs sampling-head overlap; naive logit-lens kept as diagnostic only."""
    bj = probe.b_jlens_topk(layer)
    if not bj:
        return None
    ids_j = [int(x) for x in bj["ids"].tolist()]
    vs_sample = lens_disagreement(ids_j, sample_ids)
    ll = probe.logit_lens_topk(layer)
    overlap_naive = (
        lens_disagreement(ids_j, ll["id_list"])["overlap"] if ll else None
    )
    return {
        "overlap": vs_sample["overlap"],
        "overlap_sample": vs_sample["overlap"],
        "overlap_naive": overlap_naive,
        "kendall_tau": vs_sample["kendall_tau"],
        "m": vs_sample["m"],
    }


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    eos_id: int | None,
) -> int:
    """Greedy if temperature≈0; otherwise multinomial."""
    row = logits.float().flatten()
    if temperature is None or temperature <= 0.01:
        return int(row.argmax().item())
    scaled = row / float(temperature)
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def live_generate(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    chaos_enabled: bool = True,
    probe: ProbeEngine | None = None,
    bus: ChaosBus | None = None,
    session_id: str | None = None,
    invocation_id: str | None = None,
    meta: dict[str, Any] | None = None,
    forced_token_ids: list[int] | None = None,
    cast_session: Any | None = None,
) -> dict[str, Any]:
    """Generate with optional chaos; returns text + usage + chaos record.

    Sampling happens immediately after each forward; chaos uses detached tensors.
    F4 heavy curves are lightweight probe means here; full L8–L26 is async finalize.

    When ``forced_token_ids`` is set, each completion step replays that token
    (teacher forcing) while still recording live chaos — used for Method B
    matched-prefix calm/bad pair replay.
    """
    device = bundle.device
    eos_id = bundle.tokenizer.eos_token_id

    if cast_session is not None and (cast_session.shadow or cast_session.inject):
        if not chaos_enabled or probe is None or bus is None:
            raise RuntimeError("CAST requires chaos_enabled=True with probe and bus")
    if bus is not None and chaos_enabled:
        bus.begin_invocation(
            session_id or "nosession",
            invocation_id or "noid",
            meta,
            device=str(device),
        )
    if cast_session is not None:
        cast_session.reset(max_new_tokens=max_new_tokens)
        if cast_session.inject:
            cast_session.register_hook(bundle)

    traj_vocab = TrajectoryTracker(window=WINDOW_C)
    traj_probes = {L: TrajectoryTracker(window=WINDOW_C) for L in PROBE_LAYERS}

    anchor_prompt: dict[str, Any] | None = None
    anchor_roll: dict[str, Any] | None = None
    pending_g_state: dict[str, Any] | None = None  # deferred F5 G snapshot

    t0 = time.perf_counter()

    # Prefill
    if probe is not None and chaos_enabled:
        probe.clear_residuals()
    out = bundle.model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :]

    # F3 prompt_end
    if chaos_enabled and probe is not None and bus is not None:
        t_f3 = time.perf_counter()
        vocab = probe.vocab_ab(logits, full_entropy=True)
        probe_snap: dict[str, Any] = {}
        for L in probe.probe_layers:
            hot = probe.hot_metrics_at(L)
            bj = probe.b_jlens_topk(L)
            if hot:
                probe_snap[str(L)] = {
                    "B_transport": hot.get("B_transport"),
                    "A": hot.get("A"),
                    "B_jlens": (bj or {}).get("B_jlens"),
                }
        anchor_prompt = {
            "entropy": vocab["A_vocab"]["entropy"],
            "code_mass": vocab["B_vocab"]["code_mass"],
            "ids": vocab["ids"],
            "probs": vocab["probs"],
        }
        bus.record_event(
            "prompt_end",
            {
                "A_vocab": vocab["A_vocab"],
                "B_vocab": vocab["B_vocab"],
                "probes": probe_snap,
            },
        )
        bus.record_timing("F3", (time.perf_counter() - t_f3) * 1e6)

    generated: list[int] = []
    gen_start_done = False

    n_steps = int(max_new_tokens)
    if forced_token_ids is not None:
        if len(forced_token_ids) == 0:
            n_steps = 0
        else:
            n_steps = min(n_steps, len(forced_token_ids))
    if n_steps < 1:
        if forced_token_ids is not None and len(forced_token_ids) == 0:
            # Teacher-forced empty completion — skip generation loop.
            pass
        else:
            n_steps = 1

    for step in range(n_steps):
        if forced_token_ids is not None and step < len(forced_token_ids):
            next_id = int(forced_token_ids[step])
        else:
            next_id = _sample_next(logits, temperature=temperature, eos_id=eos_id)
        generated.append(next_id)

        if chaos_enabled and probe is not None and bus is not None:
            t_tok = time.perf_counter()
            t_f0 = time.perf_counter()
            vocab = probe.vocab_ab(logits, full_entropy=False)
            c_vocab = traj_vocab.push(vocab["top1_id"])
            bus.record_timing("F0", (time.perf_counter() - t_f0) * 1e6)

            t_f1 = time.perf_counter()
            probes_payload: dict[str, Any] = {}
            # F1 hot: L16 only (Arc latency). L10/L22 filled on F2/F3.
            hot_layers = [16] if not bus.degraded_mode else [16]
            for L in hot_layers:
                hot = probe.hot_metrics_at(L)
                bt = (hot or {}).get("B_transport")
                a_sc = (hot or {}).get("A")
                if bt is not None:
                    class_ids = {
                        "code_mass": 1,
                        "distress_mass": 2,
                        "stub_mass": 3,
                        "junk_mass": 4,
                    }
                    winner = max(bt, key=lambda k: bt[k])
                    c_p = traj_probes[L].push(class_ids.get(winner, 0))
                else:
                    c_p = traj_probes[L].metrics()
                probes_payload[str(L)] = {
                    "A": a_sc,
                    "B_transport": bt,
                    "C": c_p,
                }
            bus.record_timing("F1", (time.perf_counter() - t_f1) * 1e6)

            payload: dict[str, Any] = {
                "A_vocab": vocab["A_vocab"],
                "B_vocab": vocab["B_vocab"],
                "C_vocab": c_vocab,
                "probes": probes_payload,
            }

            if not gen_start_done:
                t_gs = time.perf_counter()
                d_payload = None
                if anchor_prompt is not None:
                    # Match prompt_end entropy basis (full vocab) for boundary D.
                    vocab_d = probe.vocab_ab(logits, full_entropy=True)
                    d_payload = cross_position(
                        entropy_a=anchor_prompt["entropy"],
                        entropy_b=vocab_d["A_vocab"]["entropy"],
                        code_mass_a=anchor_prompt["code_mass"],
                        code_mass_b=vocab_d["B_vocab"]["code_mass"],
                        ids_a=anchor_prompt["ids"],
                        probs_a=anchor_prompt["probs"],
                        ids_b=vocab_d["ids"],
                        probs_b=vocab_d["probs"],
                    )
                    payload["D"] = d_payload
                    payload["D_boundary"] = True
                bus.record_event(
                    "gen_start",
                    {
                        "D": d_payload,
                        "A_vocab": vocab["A_vocab"],
                        "B_vocab": vocab["B_vocab"],
                    },
                )
                bus.record_timing("F3", (time.perf_counter() - t_gs) * 1e6)
                gen_start_done = True
                if not bus.degraded_mode:
                    f_recs: dict[str, Any] = {}
                    sample_ids = [int(x) for x in vocab["ids"].tolist()]
                    for L in probe.probe_layers:
                        fm = _probe_f_metrics(probe, L, sample_ids)
                        if fm is not None:
                            f_recs[str(L)] = fm
                    if f_recs:
                        bus.record_f({"step": step, "layers": f_recs})

            do_f2 = (step % F2_STRIDE == 0) and not (bus.degraded_mode and step > 0)
            if do_f2:
                t_f2 = time.perf_counter()
                if anchor_roll is not None:
                    d_roll = cross_position(
                        entropy_a=anchor_roll["entropy"],
                        entropy_b=vocab["A_vocab"]["entropy"],
                        code_mass_a=anchor_roll["code_mass"],
                        code_mass_b=vocab["B_vocab"]["code_mass"],
                        ids_a=anchor_roll["ids"],
                        probs_a=anchor_roll["probs"],
                        ids_b=vocab["ids"],
                        probs_b=vocab["probs"],
                    )
                    payload["D_roll"] = d_roll
                for L in probe.probe_layers:
                    if bus.degraded_mode and L != 16:
                        continue
                    # fill F1-style for L10/L22 on stride
                    if str(L) not in probes_payload:
                        hot = probe.hot_metrics_at(L)
                        if hot:
                            probes_payload[str(L)] = {
                                "A": hot.get("A"),
                                "B_transport": hot.get("B_transport"),
                                "C": traj_probes[L].metrics(),
                            }
                    bj = probe.b_jlens_topk(L)
                    if bj:
                        probes_payload.setdefault(str(L), {})
                        probes_payload[str(L)]["B_jlens"] = bj["B_jlens"]
                        probes_payload[str(L)]["A_jlens"] = bj["A"]
                    sample_ids = [int(x) for x in vocab["ids"].tolist()]
                    fm = _probe_f_metrics(probe, L, sample_ids)
                    if fm is not None:
                        probes_payload.setdefault(str(L), {})
                        probes_payload[str(L)]["F"] = fm
                anchor_roll = {
                    "entropy": vocab["A_vocab"]["entropy"],
                    "code_mass": vocab["B_vocab"]["code_mass"],
                    "ids": vocab["ids"],
                    "probs": vocab["probs"],
                }
                bus.record_timing("F2", (time.perf_counter() - t_f2) * 1e6)
                if isinstance(bus, MonitorChaosBus):
                    cliff = bus.estimate_cliff_metrics()
                    layer = cliff.get("cliff_layer")
                    bus.set_cliff_metrics(
                        int(layer) if layer is not None else None,
                        float(cliff.get("cliff_mag", 0)),
                    )

            flags = bus.record_token(step, next_id, payload)

            monitor_event = None
            if isinstance(bus, MonitorChaosBus):
                monitor_event = bus.record_monitor_event(step, next_id, flags, payload)

            if cast_session is not None and (cast_session.shadow or cast_session.inject):
                from jlens_qwen.caa.cast_injector import patch_last_token_fields

                recipe_id = (monitor_event or {}).get("recipe_id")
                severity = (monitor_event or {}).get("severity")
                h16 = probe.residuals.get(16)
                cast_row = cast_session.process_token(
                    step,
                    distress_mass=float((payload.get("B_vocab") or {}).get("distress_mass") or 0),
                    stub_mass=float((payload.get("B_vocab") or {}).get("stub_mass") or 0),
                    flags=flags,
                    recipe_id=recipe_id,
                    severity=severity,
                    h_l16=h16,
                    max_new_tokens=max_new_tokens,
                )
                patch_last_token_fields(bus, {"cast": cast_row})
                if monitor_event is not None and isinstance(bus, MonitorChaosBus):
                    bus.publish_cast_update(step, cast_row)
                    bus.timing.record(
                        "cast_eval",
                        float(cast_row.get("cast_eval_us") or 0),
                    )

            # G is alarm-gated and expensive — defer to post-generation (never block token loop)
            if bus.should_fire_g(flags, step):
                h16 = probe.residuals.get(16) if probe is not None else None
                j16 = probe.J_xpu.get(16) if probe is not None else None
                if h16 is not None and j16 is not None:
                    pending_g_state = {
                        "step": step,
                        "flags": list(flags),
                        "h": h16[0].float().cpu().clone(),
                        "J": j16.float().cpu().clone(),
                    }

            # Hot-path only (F0+F1) for overhead SLO — exclude F2/F5
            f0 = ((bus.timing._samples.get("F0") or [0])[-1])
            f1 = ((bus.timing._samples.get("F1") or [0])[-1])
            bus.record_timing("token_total", f0 + f1)

        if eos_id is not None and next_id == eos_id:
            break

        next_tensor = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        if probe is not None and chaos_enabled:
            probe.clear_residuals()
        out = bundle.model(
            input_ids=next_tensor, past_key_values=past, use_cache=True
        )
        if cast_session is not None and (cast_session.shadow or cast_session.inject):
            from jlens_qwen.caa.cast_injector import patch_last_token_fields

            applied = cast_session.note_post_forward()
            if cast_session.inject and bus is not None:
                with bus._lock:
                    if bus._tokens:
                        cast_block = dict(bus._tokens[-1].get("cast") or {})
                        cast_block["inject_applied"] = applied
                        bus._tokens[-1]["cast"] = cast_block
                if isinstance(bus, MonitorChaosBus):
                    bus.publish_cast_update(step, cast_block)
        past = out.past_key_values
        logits = out.logits[0, -1, :]

    if str(device).startswith("xpu"):
        try:
            torch.xpu.synchronize()
        except Exception:
            pass
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Deferred F5 G (post hot path, single probe L16)
    if (
        chaos_enabled
        and bus is not None
        and probe is not None
        and probe.lens is not None
        and pending_g_state is not None
    ):
        t_g = time.perf_counter()
        w_u = probe.W_U_cpu
        if w_u is None:
            w_u = bundle.W_U.detach().to("cpu").float()
        g_out = {
            "16": sparse_residual(
                pending_g_state["h"],
                w_u,
                pending_g_state["J"],
                tokenizer=bundle.tokenizer,
            )
        }
        g_payload = {
            "step": pending_g_state["step"],
            "layers": g_out,
            "flags": pending_g_state["flags"],
        }
        bus.record_g(g_payload)
        if isinstance(bus, MonitorChaosBus):
            g16 = g_out.get("16") or {}
            rel_mse = float(g16.get("relMSE", g16.get("rel_mse", 0)))
            oob = rel_mse >= THRESH["G_high_relmse"]
            bus.set_g_oob(oob)
            if generated:
                bus.record_monitor_event(
                    len(generated) - 1,
                    generated[-1],
                    list(pending_g_state["flags"]),
                    {},
                )
        bus.record_timing("F5_G", (time.perf_counter() - t_g) * 1e6)

    chaos_rec = None
    if chaos_enabled and bus is not None:
        bus.set_gen_ms(elapsed_ms)
        curves = {}
        for name, by_L in bus._probe_series.items():
            series = {
                L: (sum(xs) / len(xs) if xs else 0.0) for L, xs in by_L.items()
            }
            curves[name] = layer_curve_metrics(series)
        cliff = curves.get("entropy", {})
        cliff_layer = cliff.get("cliff_layer")
        if isinstance(bus, MonitorChaosBus) and cliff_layer is not None:
            bus.set_cliff_metrics(int(cliff_layer), float(cliff.get("cliff_mag", 0)))
        bus.record_invocation_summary(
            {
                "curves_E": {
                    "series": curves,
                    "cliff_layer": cliff_layer,
                    "cliff_mag": cliff.get("cliff_mag"),
                }
            }
        )
        if cast_session is not None:
            bus.record_timing("cast_eval_p95", cast_session.timing_summary().get("p95_us", 0))
        chaos_rec = bus.finalize()

    text = bundle.tokenizer.decode(generated, skip_special_tokens=True)
    # Canonical token stream lives on chaos_rec (same list, single source for tracer).
    canon_ids = chaos_rec.get("token_ids") if chaos_rec else generated
    return {
        "text": text,
        "prompt_tokens": int(input_ids.shape[1]),
        "completion_tokens": len(generated),
        "seconds": elapsed_ms / 1000.0,
        "token_ids": canon_ids,
        "chaos": chaos_rec,
    }
