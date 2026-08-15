#!/usr/bin/env python
"""OpenAI-compatible chat server for local Qwen2.5-Coder on Intel XPU.

Logs every Aider/LLM invocation to a JSONL session file so J-space can
replay each call after the agentic run finishes.
"""

import argparse
import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SESSIONS = ROOT / "experiments" / "out" / "aider_sessions"
DEFAULT_LENS = ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt"
OUT_DIR = ROOT / "experiments" / "out"
JSPACE_TOP_K = 20

# Global model state
_bundle = None
_lens = None
_probe = None
_bus = None
_chaos_enabled = True
_cast_shadow = False
_cast_inject = False
_cast_session = None
_lock = threading.Lock()
_session_id: str | None = None
_session_path: Path | None = None
_session_meta: dict[str, Any] = {}
_invocation_idx = 0
_chaos_by_id: dict[str, Any] = {}
_stream_queues: list[Any] = []
_stream_lock = threading.Lock()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_bundle(device: str = "xpu", *, chaos: bool = True):
    global _bundle, _lens, _probe, _bus, _chaos_enabled, _cast_shadow, _cast_inject, _cast_session
    import os
    import torch
    from jlens_qwen.caa.cast_injector import CastLiveSession
    from jlens_qwen.chaos.monitor_bus import MonitorChaosBus
    from jlens_qwen.chaos.probe import ProbeEngine
    from jlens_qwen.jacobian_lens import FittedJacobianLens
    from jlens_qwen.model_loader import load_qwen_coder

    _chaos_enabled = bool(chaos)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    print(f"Loading Qwen on {device} ...", flush=True)
    t0 = time.perf_counter()
    if device == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError(
                "XPU requested but torch.xpu is unavailable — refusing CPU fallback"
            )
    _bundle = load_qwen_coder(device=device, dtype=torch.float16, prefer_local=True)
    if device == "xpu" and str(_bundle.device) != "xpu":
        raise RuntimeError(
            f"Refusing to serve: requested xpu but model landed on {_bundle.device}"
        )
    if DEFAULT_LENS.is_file():
        _lens = FittedJacobianLens.load(str(DEFAULT_LENS))
        print(f"Loaded J-lens layers {_lens.source_layers[0]}-{_lens.source_layers[-1]}", flush=True)
    else:
        print(f"WARNING: no lens at {DEFAULT_LENS}", flush=True)
    _bus = MonitorChaosBus()
    _probe = ProbeEngine(_bundle, _lens)
    if _chaos_enabled:

        def _on_monitor_event(ev: dict[str, Any]) -> None:
            ev_out = {**ev, "sse_enqueued_ns": time.perf_counter_ns()}
            with _stream_lock:
                dead: list[Any] = []
                for q in _stream_queues:
                    try:
                        q.put_nowait(ev_out)
                    except Exception:
                        dead.append(q)
                for q in dead:
                    if q in _stream_queues:
                        _stream_queues.remove(q)

        _bus.subscribe(_on_monitor_event)
        info = _probe.setup()
        print(f"Chaos probes ready: {info}", flush=True)
    _cast_shadow = os.environ.get("CAA_CAST_SHADOW", "0") == "1"
    _cast_inject = os.environ.get("CAA_CAST_INJECT", "0") == "1"
    if _cast_inject and not _cast_shadow:
        raise RuntimeError("CAA_CAST_INJECT=1 requires CAA_CAST_SHADOW=1")
    _cast_session = None
    caa_root = ROOT / "artifacts" / "caa" / "UNI_CAA01"
    if _cast_shadow or _cast_inject:
        meta = caa_root / "metadata.json"
        cond = caa_root / "condition_L16.pt"
        beh = caa_root / "behavior_L16.pt"
        if not (meta.is_file() and cond.is_file() and beh.is_file()):
            raise RuntimeError(f"CAST vectors missing under {caa_root}")
        _alpha_0_override = None
        if os.environ.get("CAA_ALPHA_0"):
            _alpha_0_override = float(os.environ["CAA_ALPHA_0"])
        _cast_session = CastLiveSession(
            meta,
            cond,
            beh,
            shadow=_cast_shadow,
            inject=_cast_inject,
            alpha_0_override=_alpha_0_override,
        )
        print(
            f"CAST live session: shadow={_cast_shadow} inject={_cast_inject}",
            flush=True,
        )
    if str(_bundle.device).startswith("xpu"):
        try:
            name = torch.xpu.get_device_name(0)
            print(f"Arc device: {name}", flush=True)
            torch.xpu.empty_cache()
        except Exception:
            pass
    print(f"Loaded on {_bundle.device} in {time.perf_counter() - t0:.1f}s", flush=True)
    return _bundle


def _topk(bundle, logits_row, k: int = JSPACE_TOP_K) -> list[dict]:
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    top = probs.topk(min(k, probs.shape[-1]))
    return [
        {
            "id": int(tid),
            "token": bundle.tokenizer.decode([int(tid)]),
            "prob": float(score),
        }
        for score, tid in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]


def _jspace_one(inv: dict, layers: list[int], max_seq_len: int = 256) -> dict:
    import torch
    from jlens_qwen.logit_lens import LogitLens

    assert _bundle is not None and _lens is not None
    chat = inv.get("chat_text") or ""
    assistant = inv.get("assistant") or ""
    # Prefer assistant-focused window: keep finalize tractable on Arc XPU
    full = (chat + assistant)[-8000:]
    with torch.inference_mode():
        j_result = _lens.apply(
            _bundle,
            full,
            layers=layers,
            max_seq_len=max_seq_len,
            use_jacobian=True,
        )
        ll = LogitLens(_bundle)
        ll_result = ll.project_layers(
            chat[-4000:] or full, layers=layers, max_seq_len=min(max_seq_len, 256)
        )
    input_ids = j_result["input_ids"][0]
    seq_len = int(input_ids.shape[0])
    prompt_len = min(
        int(
            _bundle.tokenizer(
                chat or full, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).input_ids.shape[1]
        ),
        seq_len,
    )
    pe = max(0, prompt_len - 1)
    layer_rows = []
    for layer in layers:
        j_logits = j_result["lens_logits"][layer]
        ll_logits = ll_result["lens_logits"][layer]
        ll_pe = min(pe, ll_logits.shape[0] - 1)
        traj = [
            _bundle.tokenizer.decode([int(j_logits[t].argmax())])
            for t in range(j_logits.shape[0])
        ]
        mean_probs = torch.softmax(j_logits.float(), dim=-1).mean(dim=0)
        mean_top = mean_probs.topk(min(JSPACE_TOP_K, mean_probs.numel()))
        layer_rows.append(
            {
                "layer": layer,
                "seq_len": int(j_logits.shape[0]),
                "trajectory_top1": traj,
                "mean_top20": [
                    {
                        "id": int(i),
                        "token": _bundle.tokenizer.decode([int(i)]),
                        "prob": float(p),
                    }
                    for p, i in zip(
                        mean_top.values.tolist(), mean_top.indices.tolist(), strict=True
                    )
                ],
                "prompt_end_top20": _topk(_bundle, j_logits[pe]),
                "prompt_end_logit_top20": _topk(_bundle, ll_logits[ll_pe]),
                "gen_start_top20": _topk(
                    _bundle, j_logits[min(prompt_len, seq_len - 1)]
                ),
                "gen_mid_top20": _topk(
                    _bundle,
                    j_logits[
                        min(
                            prompt_len + max(1, (seq_len - prompt_len) // 2),
                            seq_len - 1,
                        )
                    ],
                ),
            }
        )
    return {
        "seq_len": seq_len,
        "prompt_len_tokens": prompt_len,
        "layers": layer_rows,
        "key_positions": {
            "prompt_end": pe,
            "gen_start": min(prompt_len, seq_len - 1),
            "gen_mid": min(
                prompt_len + max(1, (seq_len - prompt_len) // 2), seq_len - 1
            ),
        },
    }


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _finalize_session(layers: list[int] | None = None, *, skip_jspace: bool = False) -> dict:
    assert _bundle is not None
    if layers is None:
        layers = list(range(8, 27))
    invs = _load_jsonl(_session_path) if _session_path else []
    nested = []
    for inv in invs:
        print(f"  J-space inv#{inv.get('invocation_index')} ...", flush=True)
        if skip_jspace:
            js = {"skipped": True, "reason": "chaos_only"}
        else:
            try:
                js = _jspace_one(inv, layers=layers) if _lens is not None else {"error": "no lens"}
            except Exception as exc:  # noqa: BLE001
                js = {"error": str(exc)}
        nested.append(
            {
                "invocation_index": inv.get("invocation_index"),
                "id": inv.get("id"),
                "stamp": inv.get("stamp"),
                "prompt_tokens": inv.get("prompt_tokens"),
                "completion_tokens": inv.get("completion_tokens"),
                "seconds": inv.get("seconds"),
                "assistant_preview": (inv.get("assistant") or "")[:800],
                "messages_preview": [
                    {
                        "role": m.get("role"),
                        "content": (m.get("content") or "")[:400],
                    }
                    for m in (inv.get("messages") or [])[:8]
                ],
                "jspace": js,
                "chaos": inv.get("chaos"),
            }
        )

    experiment = _session_meta.get("experiment") or (_session_id or "aider_run")
    task_slug = _session_meta.get("task_slug") or "unknown"
    message = _session_meta.get("message") or ""
    stamp = _utc()
    last_js = next(
        (
            n["jspace"]
            for n in reversed(nested)
            if isinstance(n.get("jspace"), dict) and n["jspace"].get("layers")
        ),
        None,
    )
    workspace: dict[str, Any] = {}
    # Full workspace band detection is expensive on Arc; summarize from last jspace instead.
    if last_js and last_js.get("layers"):
        workspace = {
            "source": "last_invocation_jspace",
            "n_layers": len(last_js["layers"]),
            "seq_len": last_js.get("seq_len"),
            "prompt_len_tokens": last_js.get("prompt_len_tokens"),
        }

    checks = [
        {
            "id": "A1",
            "name": "at least one LLM invocation logged",
            "pass": len(nested) >= 1,
            "detail": f"n_invocations={len(nested)}",
        },
        {
            "id": "A3",
            "name": "J-space attached to invocations",
            "pass": skip_jspace
            or sum(1 for n in nested if n.get("jspace", {}).get("layers")) >= 1,
            "detail": (
                f"skip_jspace={skip_jspace} "
                f"with_jspace={sum(1 for n in nested if n.get('jspace', {}).get('layers'))}/{len(nested)}"
            ),
        },
        {
            "id": "A4",
            "name": "nested invocations indexed",
            "pass": all("invocation_index" in n for n in nested) and len(nested) >= 1,
            "detail": f"indexed={len(nested)}",
        },
        {
            "id": "A5",
            "name": "layer readouts for browsing",
            "pass": skip_jspace
            or bool(last_js and len(last_js.get("layers") or []) >= 1),
            "detail": f"skip_jspace={skip_jspace} n_layers={len((last_js or {}).get('layers') or [])}",
        },
        {
            "id": "C1",
            "name": "chaos attached to invocations",
            "pass": sum(1 for n in nested if n.get("chaos")) >= 1 or not _chaos_enabled,
            "detail": (
                f"with_chaos={sum(1 for n in nested if n.get('chaos'))}/{len(nested)} "
                f"enabled={_chaos_enabled}"
            ),
        },
    ]
    # Cross-invocation chaos deltas
    chaos_separation: dict[str, Any] = {}
    plds_vals = []
    for n in nested:
        c = n.get("chaos") or {}
        comp = c.get("composites") or {}
        if "PLDS" in comp:
            plds_vals.append(float(comp["PLDS"]))
    if len(plds_vals) >= 2:
        chaos_separation = {
            "clean_plds": plds_vals[0],
            "distressed_plds": max(plds_vals),
            "invocation_deltas": [
                plds_vals[i] - plds_vals[0] for i in range(1, len(plds_vals))
            ],
        }
    evidence = {
        "checks": checks,
        "passed": sum(1 for c in checks if c["pass"]),
        "total": len(checks),
    }
    evidence["overall_pass"] = evidence["passed"] == evidence["total"]

    payload = {
        "experiment": experiment,
        "stamp": stamp,
        "kind": "aider_agentic",
        "task_slug": task_slug,
        "message": message,
        "session_id": _session_id,
        "session_path": str(_session_path) if _session_path else None,
        "device": str(_bundle.device),
        "n_invocations": len(nested),
        "invocations": nested,
        "layers": (last_js or {}).get("layers") or [],
        "seq_len": (last_js or {}).get("seq_len"),
        "prompt_len_tokens": (last_js or {}).get("prompt_len_tokens"),
        "key_positions": (last_js or {}).get("key_positions"),
        "workspace": workspace,
        "evidence": evidence,
        "chaos_separation": chaos_separation,
        "chaos_enabled": _chaos_enabled,
        "lens_path": str(DEFAULT_LENS),
        "n_prompts_fitted": getattr(_lens, "n_prompts", None) if _lens else None,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{experiment}_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / f"{experiment}_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Rebuild dashboard
    try:
        import subprocess

        subprocess.call(
            [sys.executable, str(ROOT / "experiments" / "build_dashboard.py")],
            cwd=str(ROOT),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"dashboard rebuild failed: {exc}", flush=True)
    return {"path": str(out), "n_invocations": len(nested), "evidence": evidence}


def _messages_to_chat(messages: list[dict]) -> str:
    assert _bundle is not None
    # Prefer chat template when roles are present
    roles = {m.get("role") for m in messages}
    if roles <= {"system", "user", "assistant"} and "user" in roles:
        try:
            return _bundle.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback: concatenate
    parts = []
    for m in messages:
        parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
    parts.append("assistant:")
    return "\n".join(parts)


def _generate(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    invocation_id: str | None = None,
) -> dict:
    import torch
    from jlens_qwen.chaos.live_generate import live_generate

    assert _bundle is not None
    chat = _messages_to_chat(messages)
    input_ids = _bundle.tokenizer(
        chat,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).input_ids.to(_bundle.device)

    result = live_generate(
        _bundle,
        input_ids,
        max_new_tokens=max(1, int(max_tokens or 1024)),
        temperature=float(temperature if temperature is not None else 0.0),
        chaos_enabled=_chaos_enabled,
        probe=_probe,
        bus=_bus,
        session_id=_session_id,
        invocation_id=invocation_id,
        meta=dict(_session_meta),
        cast_session=_cast_session,
    )
    return {
        "text": result["text"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "seconds": result["seconds"],
        "chat_text": chat,
        "chaos": result.get("chaos"),
    }


def _append_invocation(record: dict) -> None:
    global _invocation_idx
    if _session_path is None:
        return
    _session_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _invocation_idx += 1
        record["invocation_index"] = _invocation_idx
        record["session_id"] = _session_id
        with _session_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="Qwen local OpenAI server for Aider")

    class ChatMessage(BaseModel):
        role: str
        content: str | None = ""

    class ChatRequest(BaseModel):
        model: str = "qwen-local"
        messages: list[ChatMessage]
        max_tokens: int | None = 1024
        temperature: float | None = 0.0
        stream: bool | None = False

    @app.get("/health")
    def health():
        import torch

        gpu_name = None
        if _bundle is not None and str(_bundle.device).startswith("xpu"):
            try:
                gpu_name = torch.xpu.get_device_name(0)
            except Exception:
                gpu_name = "xpu"
        snap = _bus.snapshot() if _bus is not None else {}
        vector_id = None
        cast_alpha_0 = None
        if _cast_session is not None:
            vector_id = getattr(_cast_session.cast, "vector_id", None)
            cast_alpha_0 = getattr(_cast_session.cast, "alpha_0", None)
        return {
            "ok": _bundle is not None,
            "device": str(_bundle.device) if _bundle else None,
            "gpu_name": gpu_name,
            "refuses_cpu_fallback": True,
            "session_id": _session_id,
            "invocations": _invocation_idx,
            "session_path": str(_session_path) if _session_path else None,
            "chaos_enabled": _chaos_enabled,
            "cast_shadow": _cast_shadow,
            "cast_inject": _cast_inject,
            "cast_vector_id": vector_id,
            "cast_alpha_0": cast_alpha_0,
            "degraded_mode": bool(snap.get("degraded_mode")),
            "probe_layers": snap.get("probe_layers"),
            "chaos_timing": snap.get("timing"),
        }

    @app.get("/chaos/live")
    def chaos_live():
        if _bus is None:
            raise HTTPException(503, "chaos bus not ready")
        return _bus.snapshot()

    @app.get("/chaos/recipes")
    def chaos_recipes():
        from jlens_qwen.chaos.recipes import recipe_catalog

        return {"recipes": recipe_catalog(), "schema": "monitor_v1"}

    @app.get("/chaos/stream")
    async def chaos_stream(request: Request):
        import asyncio
        import queue

        if _bus is None:
            raise HTTPException(503, "chaos bus not ready")

        q: queue.Queue = queue.Queue(maxsize=512)
        with _stream_lock:
            _stream_queues.append(q)

        async def event_generator():
            last_seq = -1
            try:
                # Push current live state immediately
                snap = _bus.live_state()
                yield f"data: {json.dumps({'kind': 'snapshot', **snap})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.to_thread(q.get, True, 0.05)
                    except queue.Empty:
                        # heartbeat + catch-up poll
                        missed = _bus.monitor_events_since(last_seq)
                        for ev in missed:
                            last_seq = max(last_seq, int(ev.get("seq", -1)))
                            yield f"data: {json.dumps(ev)}\n\n"
                        yield ": heartbeat\n\n"
                        continue
                    last_seq = max(last_seq, int(ev.get("seq", -1)))
                    yield f"data: {json.dumps(ev)}\n\n"
            finally:
                with _stream_lock:
                    if q in _stream_queues:
                        _stream_queues.remove(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/chaos/invocation/{inv_id}")
    def chaos_invocation(inv_id: str):
        rec = _chaos_by_id.get(inv_id)
        if rec is None:
            raise HTTPException(404, f"no chaos for {inv_id}")
        return rec

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {"id": "qwen-local", "object": "model", "owned_by": "local"},
                {
                    "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "object": "model",
                    "owned_by": "local",
                },
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if _bundle is None:
            raise HTTPException(503, "model not loaded")
        raw = await request.json()
        try:
            req = ChatRequest.model_validate(raw)
        except Exception as exc:
            raise HTTPException(422, f"invalid body: {exc}") from exc
        if req.stream:
            raise HTTPException(400, "streaming not supported; set --no-stream in aider")

        messages = [{"role": m.role, "content": m.content or ""} for m in req.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        with _lock:
            result = _generate(
                messages,
                max_tokens=req.max_tokens or 1024,
                temperature=req.temperature if req.temperature is not None else 0.0,
                invocation_id=completion_id,
            )

        record = {
            "id": completion_id,
            "stamp": _utc(),
            "model": req.model,
            "messages": messages,
            "assistant": result["text"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "seconds": result["seconds"],
            "chat_text_preview": result["chat_text"][:800],
            "chat_text": result["chat_text"],
            "device": str(_bundle.device),
            "chaos": result.get("chaos"),
        }
        if result.get("chaos"):
            _chaos_by_id[completion_id] = result["chaos"]
        _append_invocation(record)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }

    @app.post("/session/start")
    async def session_start(request: Request):
        global _session_id, _session_path, _invocation_idx, _session_meta
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        name = payload.get("name") or "session"
        stamp = _utc()
        _session_id = f"{name}_{stamp}"
        _invocation_idx = 0
        _session_meta = {
            "experiment": payload.get("experiment") or name,
            "task_slug": payload.get("task_slug") or name,
            "message": payload.get("message") or "",
        }
        out_dir = Path(payload.get("out_dir") or DEFAULT_SESSIONS)
        out_dir.mkdir(parents=True, exist_ok=True)
        _session_path = out_dir / f"{_session_id}.jsonl"
        _session_path.write_text("", encoding="utf-8")
        return {
            "session_id": _session_id,
            "path": str(_session_path),
            "meta": _session_meta,
            "device": str(_bundle.device) if _bundle else None,
        }

    @app.get("/session/status")
    def session_status():
        return {
            "session_id": _session_id,
            "invocations": _invocation_idx,
            "path": str(_session_path) if _session_path else None,
            "meta": _session_meta,
            "device": str(_bundle.device) if _bundle else None,
        }

    @app.post("/session/finalize")
    async def session_finalize(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if payload.get("experiment"):
            _session_meta["experiment"] = payload["experiment"]
        if payload.get("task_slug"):
            _session_meta["task_slug"] = payload["task_slug"]
        if payload.get("message"):
            _session_meta["message"] = payload["message"]
        layers = payload.get("layers")
        skip_jspace = bool(payload.get("skip_jspace") or payload.get("chaos_only"))
        if isinstance(layers, str) and "-" in layers:
            lo, hi = [int(x) for x in layers.split("-")]
            layer_list = list(range(lo, hi + 1))
        elif isinstance(layers, str) and layers.strip().isdigit():
            layer_list = [int(layers.strip())]
        elif isinstance(layers, list):
            layer_list = [int(x) for x in layers]
        elif isinstance(layers, int):
            layer_list = [int(layers)]
        else:
            layer_list = list(range(8, 27))
        with _lock:
            result = _finalize_session(layers=layer_list, skip_jspace=skip_jspace)
        return result

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--session-name", default="boot")
    parser.add_argument(
        "--chaos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live A–G chaos instrumentation (default: on)",
    )
    args = parser.parse_args()

    global _session_id, _session_path, _invocation_idx
    _load_bundle(args.device, chaos=args.chaos)
    stamp = _utc()
    _session_id = f"{args.session_name}_{stamp}"
    _invocation_idx = 0
    DEFAULT_SESSIONS.mkdir(parents=True, exist_ok=True)
    _session_path = DEFAULT_SESSIONS / f"{_session_id}.jsonl"
    _session_path.write_text("", encoding="utf-8")
    print(f"Session log: {_session_path}", flush=True)
    print(f"Chaos enabled: {_chaos_enabled}", flush=True)

    import uvicorn

    app = create_app()
    print(f"Serving OpenAI API at http://{args.host}:{args.port}/v1", flush=True)
    print(f"DEVICE CONFIRMED: {_bundle.device}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
