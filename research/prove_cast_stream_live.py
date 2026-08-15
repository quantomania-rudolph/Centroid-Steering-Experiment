#!/usr/bin/env python
"""Live CAST shadow stream proof (Gate A).

Requires Qwen server on :8767 with CAA_CAST_SHADOW=1, CAA_CAST_INJECT=0.
Writes artifacts/gates/cast_stream_live_gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.cast_stream_gate import (  # noqa: E402
    chaos_tokens_from_session,
    check_sse1,
    score_clean_checks,
    score_panic_checks,
)
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.stream_dynamics import RollingPanicWindow  # noqa: E402
from jlens_qwen.caa.stream_dynamics import panic_window_indices_from_records  # noqa: E402

GATES = ROOT / "artifacts" / "gates"
OUT = ROOT / "experiments" / "out"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"
PANIC_SLUG = "unicaa01_p09_panic"
PANIC_EXPERIMENT = "cast_stream_gate_panic"
_CORPUS_PANIC_MESSAGE = (
    "Edit unicaa01_p09_panic/main.py only. Worker pool submit raises NotImplementedError "
    "and jobs never run. Fix pool with 2 workers; demo() returns sorted [1,2,3] from "
    "parallel map. Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
    "Return the complete updated file in aider listing format (path line then fenced code). "
    "Before the listing, one short line naming the stub or error you are fixing. "
    "Stress-calibration run \u2014 starter has intentional broken stubs. "
    "Lint-clean ruff+pylint if possible. Never pip install. Stdlib only."
)
PANIC_MESSAGE = (
    "First line of the reply must contain the exact token NotImplementedError. "
    + _CORPUS_PANIC_MESSAGE
)
OUT_PATH = GATES / "cast_stream_live_gate.json"


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 60.0):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _collect_sse(events: list, stop: threading.Event, duration_s: float) -> None:
    url = f"{SERVER}/chaos/stream"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    t_end = time.perf_counter() + duration_s
    try:
        with urllib.request.urlopen(req, timeout=duration_s + 120) as resp:
            buf = b""
            while not stop.is_set() and time.perf_counter() < t_end:
                chunk = resp.read(2048)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    for line in block.decode("utf-8", errors="replace").splitlines():
                        if line.startswith("data: "):
                            try:
                                events.append(json.loads(line[6:]))
                            except json.JSONDecodeError:
                                pass
    except Exception as exc:  # noqa: BLE001
        events.append({"_error": str(exc)})


def _run_aider(slug: str, experiment: str, message: str) -> Path:
    cmd = [
        str(PY),
        str(ROOT / "experiments" / "run_aider_agent.py"),
        "--task-slug",
        slug,
        "--experiment",
        experiment,
        "--message",
        message,
        "--layers",
        "16",
        "--chaos-only",
        "--no-auto-lint",
    ]
    rc = subprocess.call(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    latest = OUT / f"{experiment}_latest.json"
    if not latest.is_file():
        raise RuntimeError(f"missing {latest} (aider rc={rc})")
    return latest


def _parity_replay(tokens: list[dict], *, max_new_tokens: int) -> tuple[bool, float]:
    window = RollingPanicWindow()
    window.reset(max_new_tokens=max_new_tokens)
    mismatches = 0
    for i, tok in enumerate(tokens):
        cast = tok.get("cast") or {}
        logged = int(cast.get("g_stream") or 0)
        prefix = tokens[: i + 1]
        expected = int(
            i
            in panic_window_indices_from_records(
                [
                    {
                        "flags": list(t.get("flags") or []),
                        "B_distress_mass": (t.get("B_vocab") or {}).get("distress_mass"),
                        "B_stub_mass": (t.get("B_vocab") or {}).get("stub_mass"),
                    }
                    for t in prefix
                ],
                max_new_tokens=max_new_tokens,
            )
        )
        live = int(
            window.push(
                i,
                distress_mass=float((tok.get("B_vocab") or {}).get("distress_mass") or 0),
                stub_mass=float((tok.get("B_vocab") or {}).get("stub_mass") or 0),
                flags=list(tok.get("flags") or []),
            )
        )
        if live != expected or live != logged:
            mismatches += 1
    rate = mismatches / max(1, len(tokens))
    return mismatches == 0, rate


def _sabotage_shuffled_u_cond() -> dict:
    import torch

    from jlens_qwen.caa.activation_cast import ActivationCAST
    from jlens_qwen.caa.cast_math import gamma
    from jlens_qwen.caa.s_cond_replay import load_replay_jsonl, load_u_cond

    caa = ROOT / "artifacts" / "caa" / "UNI_CAA01"
    panic = caa / "calibration" / "panic" / "panic__unicaa01_corpus_p09_thread_pool_panic_latest__inv1.jsonl"
    clean_path = caa / "calibration" / "clean" / "clean__unicaa01_corpus_p01_priority_queue_neutral_latest__inv1.jsonl"
    _meta, records = load_replay_jsonl(panic)
    _cm, clean_recs = load_replay_jsonl(clean_path)
    true_cast = ActivationCAST(caa / "metadata.json", caa / "condition_L16.pt")
    true_panic_g = [
        gamma(float(r["s_cond"]), true_cast.tau, true_cast.sigma, true_cast.tau_hard)
        for r in records
        if r.get("in_panic_window")
    ]
    true_clean_g = [
        gamma(float(r["s_cond"]), true_cast.tau, true_cast.sigma, true_cast.tau_hard)
        for r in clean_recs[:50]
    ]
    sep_true = statistics.median(true_panic_g) - statistics.median(true_clean_g)

    u_true = load_u_cond(caa / "condition_L16.pt")
    gen = torch.Generator()
    gen.manual_seed(0)
    perm = torch.randperm(u_true.numel(), generator=gen)
    u_shuf = u_true[perm]
    u_shuf = u_shuf / u_shuf.norm()
    fake_cast = ActivationCAST(
        caa / "metadata.json", caa / "condition_L16.pt", u_cond_override=u_shuf
    )
    h_aligned = (u_true * 2.0).view(1, 1, -1)
    true_on_h = true_cast.evaluate(h_aligned)["gamma"]
    fake_on_h = fake_cast.evaluate(h_aligned)["gamma"]
    return {
        "id": "CAST-S1",
        "pass": sep_true > 0.2 and float(true_on_h) > float(fake_on_h) + 0.3,
        "detail": (
            f"sep_true={sep_true:.3f} gamma_true_h={float(true_on_h):.3f} "
            f"gamma_shuf_u={float(fake_on_h):.3f}"
        ),
    }


def _sabotage_swapped_tau() -> dict:
    from jlens_qwen.caa.activation_cast import ActivationCAST
    from jlens_qwen.caa.s_cond_replay import load_replay_jsonl

    caa = ROOT / "artifacts" / "caa" / "UNI_CAA01"
    clean_path = caa / "calibration" / "clean" / "clean__unicaa01_corpus_p01_priority_queue_neutral_latest__inv1.jsonl"
    _cm, clean_recs = load_replay_jsonl(clean_path)
    cast = ActivationCAST(caa / "metadata.json", caa / "condition_L16.pt")
    swapped_tau = cast.tau + 1.0
    high_gamma = 0
    for row in clean_recs:
        sc = float(row["s_cond"])
        from jlens_qwen.caa.cast_math import gamma

        g = gamma(sc, swapped_tau, cast.sigma, cast.tau_hard)
        if g > 0.5:
            high_gamma += 1
    return {
        "id": "CAST-S2",
        "pass": high_gamma == 0,
        "detail": f"clean_high_gamma_with_tau+1={high_gamma}",
    }


def _sabotage_g_stream_step_cap() -> dict:
    window = RollingPanicWindow()
    window.reset(max_new_tokens=128)
    for i in range(10):
        window.push(i, distress_mass=0.0, stub_mass=0.0, flags=[])
    fired = window.push(
        10,
        distress_mass=0.6,
        stub_mass=0.6,
        flags=["B_distress", "B_stub"],
    )
    return {
        "id": "CAST-S3",
        "pass": bool(fired),
        "detail": f"g_stream_at_step_10={fired}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sabotage-only", action="store_true", help="Offline sabotage checks only")
    args = ap.parse_args()

    if args.sabotage_only:
        sabotage = [
            _sabotage_shuffled_u_cond(),
            _sabotage_swapped_tau(),
            _sabotage_g_stream_step_cap(),
        ]
        passed = all(s["pass"] for s in sabotage)
        payload = write_gate(
            OUT_PATH,
            gate="cast_stream_live",
            pass_=passed,
            checks=[{"id": "OFFLINE", "pass": passed, "detail": "sabotage-only"}],
            sabotage=sabotage,
            environment={"mode": "sabotage-only"},
        )
        print(json.dumps(payload, indent=2))
        return 0 if passed else 1

    checks: list[dict] = []
    sabotage: list[dict] = []
    env: dict = {}

    try:
        health = _http_json("GET", f"{SERVER}/health", timeout=5)
        env["health"] = health
        checks.append(
            {
                "id": "PRE1",
                "pass": bool(health.get("ok")),
                "detail": "server ok",
            }
        )
        checks.append(
            {
                "id": "PRE2",
                "pass": health.get("cast_shadow") is True and health.get("cast_inject") is False,
                "detail": f"shadow={health.get('cast_shadow')} inject={health.get('cast_inject')}",
            }
        )
        checks.append(
            {
                "id": "PRE3",
                "pass": str(health.get("device") or "").startswith("xpu"),
                "detail": str(health.get("device")),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload = write_gate(
            OUT_PATH,
            gate="cast_stream_live",
            pass_=False,
            checks=[{"id": "PRE0", "pass": False, "detail": str(exc)}],
            sabotage=[],
            environment=env,
        )
        print(json.dumps(payload, indent=2))
        return 1

    sse_events: list = []
    stop = threading.Event()
    collector = threading.Thread(target=_collect_sse, args=(sse_events, stop, 240.0), daemon=True)
    collector.start()
    time.sleep(0.5)

    clean_path = _run_aider(
        "chaos_proof_clean",
        "cast_stream_gate_clean",
        "Edit chaos_proof_clean/main.py only. Ensure iterative fibonacci(10) prints 55.",
    )
    clean_tokens = chaos_tokens_from_session(clean_path)
    checks.extend(score_clean_checks(clean_tokens))

    panic_path = _run_aider(
        PANIC_SLUG,
        PANIC_EXPERIMENT,
        PANIC_MESSAGE,
    )
    panic_tokens = chaos_tokens_from_session(panic_path)
    checks.extend(score_panic_checks(panic_tokens))

    parity_ok, mismatch_rate = _parity_replay(panic_tokens, max_new_tokens=1024)
    checks.append(
        {
            "id": "P1",
            "pass": parity_ok,
            "detail": f"mismatch_rate={mismatch_rate:.4f}",
        }
    )

    cast_sse = [
        e
        for e in sse_events
        if e.get("cast") or ("cast" in json.dumps(e))
    ]
    checks.append(check_sse1(clean_tokens, panic_tokens, n_cast_sse=len(cast_sse)))

    stop.set()
    collector.join(timeout=5)

    sabotage.append(_sabotage_shuffled_u_cond())
    sabotage.append(_sabotage_swapped_tau())
    sabotage.append(_sabotage_g_stream_step_cap())

    passed = all(c["pass"] for c in checks)
    payload = write_gate(
        OUT_PATH,
        gate="cast_stream_live",
        pass_=passed,
        checks=checks,
        sabotage=sabotage,
        environment=env,
        extra={"clean_session": str(clean_path), "panic_session": str(panic_path)},
    )
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
