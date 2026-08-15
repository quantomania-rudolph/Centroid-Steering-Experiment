#!/usr/bin/env python
"""Final review: Aider harness + J-space dashboard + chaos gates in one pass."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"
OUT = ROOT / "experiments" / "out"
GATES = ROOT / "artifacts" / "gates"
DASH = ROOT / "experiments" / "dashboard"
EXPERIMENT = "aider_final_review_topo_sort"
TASK = "final_review_topo_sort"

PROMPT = (
    "Edit final_review_topo_sort/main.py only. Fix topo_sort to use Kahn's algorithm correctly "
    "(deque popleft, proper indegree). main() must print a valid topological order for the "
    "given DAG and pass its asserts. Lint-clean ruff+pylint. Never pip. Stdlib only."
)


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 30.0):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_pytest(args: list[str]) -> tuple[int, str]:
    cmd = [str(PY), "-m", "pytest", *args, "-q", "--tb=line"]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def assert_chaos_inv(inv: dict, label: str) -> None:
    c = inv.get("chaos") or {}
    assert c.get("version") == 1, f"{label}: missing chaos version"
    toks = c.get("tokens") or []
    comp = inv.get("completion_tokens") or 1
    assert len(toks) >= int(0.85 * comp), f"{label}: token drop {len(toks)}/{comp}"
    assert toks, f"{label}: no chaos tokens"
    t0 = toks[0]
    for k in ("A_vocab", "B_vocab", "C_vocab"):
        assert k in t0, f"{label}: missing {k}"
    assert (c.get("timing") or {}).get("chaos_ms_per_token_p95", 999) <= 120.0, (
        f"{label}: chaos p95 too high"
    )
    events = [e.get("kind") for e in (c.get("events") or [])]
    assert "prompt_end" in events, f"{label}: no prompt_end"
    assert "gen_start" in events, f"{label}: no gen_start"
    cliff = (c.get("curves_E") or {}).get("cliff_layer")
    assert cliff is not None and 8 <= int(cliff) <= 26, f"{label}: bad cliff {cliff}"
    assert "PLDS" in (c.get("composites") or {}), f"{label}: no PLDS"


def assert_jspace_inv(inv: dict, label: str, *, require_layers: bool) -> None:
    js = inv.get("jspace") or {}
    if js.get("skipped"):
        if require_layers:
            raise AssertionError(f"{label}: jspace skipped unexpectedly")
        return
    layers = js.get("layers") or []
    if require_layers:
        assert len(layers) >= 10, f"{label}: jspace layers={len(layers)}"
        assert js.get("seq_len", 0) > 0, f"{label}: bad seq_len"
        row = layers[0]
        for k in ("layer", "trajectory_top1", "mean_top20", "prompt_end_top20"):
            assert k in row, f"{label}: jspace row missing {k}"


def main() -> int:
    failures: list[str] = []
    h: dict = {}

    print("=== 1) Server health ===", flush=True)
    try:
        h = _http_json("GET", f"{SERVER}/health")
        assert h.get("device") == "xpu", h
        assert h.get("chaos_enabled") is True, h
        assert h.get("refuses_cpu_fallback") is True, h
        print(f"OK device={h['device']} chaos={h.get('chaos_enabled')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"health: {exc}")

    print("\n=== 2) Unit chaos tests ===", flush=True)
    rc, out = run_pytest(["tests/unit/test_chaos_metrics.py", "tests/unit/test_chaos_bus.py"])
    print(out[-800:], flush=True)
    if rc != 0:
        failures.append(f"unit chaos rc={rc}")

    print("\n=== 3) Gate chaos tests (CHAOS_LIVE=1) ===", flush=True)
    env = {**dict(__import__("os").environ), "CHAOS_LIVE": "1"}
    p = subprocess.run(
        [str(PY), "-m", "pytest", "tests/gates/test_gate_chaos_A.py", "tests/gates/test_gate_chaos_B_vocab.py",
         "tests/gates/test_gate_chaos_B_jlens.py", "tests/gates/test_gate_chaos_B_transport.py",
         "tests/gates/test_gate_chaos_C.py", "tests/gates/test_gate_chaos_D.py", "tests/gates/test_gate_chaos_E.py",
         "tests/gates/test_gate_chaos_F.py", "tests/gates/test_gate_chaos_G.py", "tests/gates/test_gate_chaos_live.py",
         "-q", "--tb=line"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    gate_out = (p.stdout or "") + (p.stderr or "")
    print(gate_out[-1200:], flush=True)
    if p.returncode != 0:
        failures.append(f"gate chaos rc={p.returncode}")

    print("\n=== 4) Live XPU chaos probe tests ===", flush=True)
    # Cannot load a second model copy while serve_qwen_openai holds Arc VRAM.
    if h.get("ok") and h.get("chaos_enabled"):
        print("SKIP: server already owns XPU model (covered by prove_chaos_live + gate artifacts)", flush=True)
    else:
        rc, out = run_pytest(["tests/live/test_chaos_probe_hooks.py"])
        print(out[-1200:], flush=True)
        if rc != 0:
            failures.append(f"live probe rc={rc}")

    print("\n=== 5) Complex Aider harness run ===", flush=True)
    cmd = [
        str(PY), str(ROOT / "experiments" / "run_aider_agent.py"),
        "--task-slug", TASK,
        "--experiment", EXPERIMENT,
        "--message", PROMPT,
        "--layers", "16",
        "--chaos-only",
    ]
    print(" ".join(cmd[-6:]), flush=True)
    ar = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"},
        timeout=900,
    )
    if ar.returncode != 0:
        failures.append(f"aider harness rc={ar.returncode}")

    latest = OUT / f"{EXPERIMENT}_latest.json"
    if not latest.is_file():
        failures.append(f"missing {latest}")
    else:
        data = json.loads(latest.read_text(encoding="utf-8"))
        invs = data.get("invocations") or []
        print(f"invocations={len(invs)} evidence_pass={data.get('evidence',{}).get('overall_pass')}", flush=True)
        if not invs:
            failures.append("no invocations logged")
        else:
            for inv in invs:
                idx = inv.get("invocation_index")
                try:
                    assert_chaos_inv(inv, f"inv#{idx}")
                    assert_jspace_inv(inv, f"inv#{idx}", require_layers=False)
                except AssertionError as exc:
                    failures.append(str(exc))
            sep = data.get("chaos_separation") or {}
            if sep and "clean_plds" in sep and "distressed_plds" in sep:
                print(f"chaos_separation clean={sep['clean_plds']:.3f} distressed={sep['distressed_plds']:.3f}", flush=True)

        # functional check on generated code
        main_py = ROOT / "qwen code" / TASK / "main.py"
        if main_py.is_file():
            fr = subprocess.run([str(PY), str(main_py)], cwd=str(main_py.parent), capture_output=True, text=True)
            if fr.returncode != 0:
                failures.append(f"main.py run failed: {fr.stderr[:300]}")
            else:
                print(f"main.py output: {fr.stdout.strip()[:200]}", flush=True)

    print("\n=== 6) Dashboard rebuild + catalog ===", flush=True)
    dr = subprocess.run([str(PY), str(ROOT / "experiments" / "build_dashboard.py")], cwd=str(ROOT))
    if dr.returncode != 0:
        failures.append(f"build_dashboard rc={dr.returncode}")
    catalog = json.loads((DASH / "catalog.json").read_text(encoding="utf-8"))
    entry = next((c for c in catalog if c.get("experiment") == EXPERIMENT), None)
    if not entry:
        failures.append(f"catalog missing {EXPERIMENT}")
    else:
        print(
            f"catalog: chaos={entry.get('chaos')} inv={entry.get('invocations')} "
            f"layers={entry.get('layers')}",
            flush=True,
        )
        if not entry.get("chaos"):
            failures.append("dashboard catalog chaos=false")
        src = entry.get("source")
        dash_path = OUT / src if src else DASH / "data.json"
        if src and dash_path.is_file():
            dash = json.loads(dash_path.read_text(encoding="utf-8"))
        else:
            all_runs = json.loads((DASH / "data.json").read_text(encoding="utf-8"))
            dash = next((r for r in all_runs if r.get("experiment") == EXPERIMENT), {})
        if not any((i.get("chaos") or {}).get("tokens") for i in (dash.get("invocations") or [])):
            failures.append("dashboard data missing chaos tokens")

    jspace_ref = next((c for c in catalog if (c.get("layers") or 0) >= 10), None)
    if jspace_ref:
        src = jspace_ref.get("source")
        ref_path = OUT / src if src else None
        if ref_path and ref_path.is_file():
            ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        else:
            all_runs = json.loads((DASH / "data.json").read_text(encoding="utf-8"))
            ref_data = next((r for r in all_runs if r.get("experiment") == jspace_ref["experiment"]), {})
        ref_inv = (ref_data.get("invocations") or [{}])[-1]
        try:
            assert_jspace_inv(ref_inv, f"catalog:{jspace_ref['experiment']}", require_layers=True)
            print(f"jspace_ref OK: {jspace_ref['experiment']} layers={jspace_ref.get('layers')}", flush=True)
        except AssertionError as exc:
            failures.append(str(exc))
    else:
        failures.append("no catalog entry with jspace layers for dashboard check")

    print("\n=== 7) /chaos/live endpoint ===", flush=True)
    try:
        live = _http_json("GET", f"{SERVER}/chaos/live")
        assert live.get("probe_layers"), live
        print(f"chaos/live n_tokens={live.get('n_tokens')} degraded={live.get('degraded_mode')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"chaos/live: {exc}")

    print("\n" + "=" * 60, flush=True)
    if failures:
        print("FINAL REVIEW: FAIL", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    print("FINAL REVIEW: PASS — Aider + J-space + Chaos all green", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
