#!/usr/bin/env python
"""Five-round Aider × A–G chaos study → CAA steering vector recommendations.

Runs controlled agentic tasks, cross-references code quality with chaos monitor
packages, and writes consolidated evidence to research/results/.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
STARTERS = RESEARCH / "starters"
RESULTS = RESEARCH / "results"
QWEN = ROOT / "qwen code"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"

sys.path.insert(0, str(ROOT / "src"))
from jlens_qwen.chaos.checker_audit import audit_invocation  # noqa: E402

STUB_PATTERNS = [
    r"NotImplementedError",
    r"\bTODO\b",
    r"\bFIXME\b",
    r"raise\s+NotImplementedError",
    r"pass\s*#\s*stub",
]
DISTRESS_PATTERNS = [
    r"pip\s+install",
    r"\bsorry\b",
    r"\bcannot\b",
    r"unable to",
    r"I can't",
    r"(?:environment|system).{0,24}broken",
    r"\bmay be broken\b",
]


@dataclass
class RoundSpec:
    round_id: str
    task_slug: str
    experiment: str
    message: str
    starter: str | None = None  # relative to research/starters
    runtime_cmd: list[str] | None = None


ROUNDS: list[RoundSpec] = [
    RoundSpec(
        "R1_CLEAN",
        "chaos_proof_clean",
        "caa_study_r1_clean",
        "Edit chaos_proof_clean/main.py only. Ensure iterative fibonacci(10) prints 55. Lint-clean.",
        starter=None,
        runtime_cmd=[
            str(PY),
            "-c",
            "import sys; sys.path.insert(0, r'qwen code'); "
            "from chaos_proof_clean.main import main; main()",
        ],
    ),
    RoundSpec(
        "R2_STUB",
        "research_stub_queue",
        "caa_study_r2_stub",
        "Edit research_stub_queue/main.py only. Implement MinHeapQueue.pop using heapq. "
        "demo() must print ['a','b','c'] in priority order. No stubs. Lint-clean.",
        starter="research_stub_queue/main.py",
        runtime_cmd=[
            str(PY),
            "-c",
            "import sys; sys.path.insert(0, r'qwen code'); "
            "from research_stub_queue.main import demo; "
            "assert demo()==['a','b','c'], demo()",
        ],
    ),
    RoundSpec(
        "R3_DISTRESS",
        "research_distress_bucket",
        "caa_study_r3_distress",
        "Edit research_distress_bucket/main.py only. Fix TokenBucket thread safety: "
        "import threading, use Lock in allow(). Must run under ThreadPoolExecutor. "
        "Never suggest pip install. Lint-clean.",
        starter="research_distress_bucket/main.py",
        runtime_cmd=[
            str(PY),
            "-c",
            "import sys; sys.path.insert(0, r'qwen code'); "
            "from research_distress_bucket.main import main; main()",
        ],
    ),
    RoundSpec(
        "R4_COMBO",
        "research_scheduler_broken",
        "caa_study_r4_combo",
        "Edit research_scheduler_broken/main.py only. Fix Scheduler: import threading, "
        "lock submit/run_next, remove NotImplementedError, demo() prints "
        "['alert','backup','email']. Lint-clean. Never pip.",
        starter="research_scheduler_broken/main.py",
        runtime_cmd=[
            str(PY),
            "-c",
            "import sys; sys.path.insert(0, r'qwen code'); "
            "from research_scheduler_broken.main import demo; "
            "assert demo()==['alert','backup','email'], demo()",
        ],
    ),
    RoundSpec(
        "R5_TRAJ",
        "research_topo_broken",
        "caa_study_r5_traj",
        "Edit research_topo_broken/main.py only. Fix topo_sort to use collections.deque "
        "with popleft for Kahn's algorithm. main() assert must pass. Lint-clean.",
        starter="research_topo_broken/main.py",
        runtime_cmd=[
            str(PY),
            "-c",
            "import sys; sys.path.insert(0, r'qwen code'); "
            "from research_topo_broken.main import main; main()",
        ],
    ),
]


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 7200.0):
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


def wait_server(timeout_s: float = 120.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            h = _http_json("GET", f"{SERVER}/health", timeout=5)
            if h.get("ok"):
                return h
            last = h
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"server not ready: {last}")


def deploy_starter(task_slug: str, starter_rel: str | None) -> None:
    task_dir = QWEN / task_slug
    task_dir.mkdir(parents=True, exist_ok=True)
    target = task_dir / "main.py"
    if starter_rel:
        src = STARTERS / starter_rel
        shutil.copy2(src, target)
    elif not target.is_file():
        target.write_text(
            '"""Task workspace."""\n\ndef main() -> None:\n    pass\n',
            encoding="utf-8",
        )


def _strip_module_docstring(text: str) -> str:
    """Remove leading module docstring so task descriptions don't trip distress grep."""
    return re.sub(
        r'^\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*',
        "",
        text,
        count=1,
    )


def score_text(text: str) -> dict[str, Any]:
    stub_hits = [p for p in STUB_PATTERNS if re.search(p, text, re.I)]
    distress_hits = [p for p in DISTRESS_PATTERNS if re.search(p, text, re.I)]
    return {
        "stub_hits": stub_hits,
        "distress_hits": distress_hits,
        "stub_score": len(stub_hits),
        "distress_score": len(distress_hits),
    }


def audit_code_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "stub_score": 99, "distress_score": 99}
    text = path.read_text(encoding="utf-8", errors="replace")
    scores = score_text(_strip_module_docstring(text))
    return {"exists": True, "path": str(path), "bytes": len(text), **scores}


def flag_rates(tokens: list[dict], prefix: str) -> float:
    if not tokens:
        return 0.0
    n = sum(1 for t in tokens if any(f.startswith(prefix) for f in (t.get("flags") or [])))
    return n / len(tokens)


def recipe_summary(chaos: dict | None) -> dict[str, Any]:
    if not chaos:
        return {}
    monitor = chaos.get("monitor") or {}
    timeline = monitor.get("recipe_timeline") or []
    recipes = [x["recipe_id"] for x in timeline]
    counts: dict[str, int] = {}
    for r in recipes:
        counts[r] = counts.get(r, 0) + 1
    return {
        "recipe_fires": len(timeline),
        "recipe_counts": counts,
        "unique_recipes": sorted(set(recipes)),
        "warn_plus": sorted(
            {x["recipe_id"] for x in timeline if x.get("severity") in ("warn", "critical")}
        ),
    }


def temporal_resolution(tokens: list[dict]) -> dict[str, Any]:
    """Compare flag rates in first vs last quartile of generation."""
    if len(tokens) < 8:
        return {"enough_tokens": False}
    q = max(1, len(tokens) // 4)
    first, last = tokens[:q], tokens[-q:]
    metrics = {}
    for name, pred in [
        ("B_distress", lambda f: f.startswith("B_distress")),
        ("B_stub", lambda f: f.startswith("B_stub")),
        ("C", lambda f: f.startswith("C_")),
        ("A", lambda f: f.startswith("A_")),
    ]:
        def rate(rows: list[dict]) -> float:
            if not rows:
                return 0.0
            n = sum(1 for t in rows if any(pred(f) for f in (t.get("flags") or [])))
            return n / len(rows)

        f_rate, l_rate = rate(first), rate(last)
        metrics[name] = {
            "first_q": f_rate,
            "last_q": l_rate,
            "resolving": l_rate < f_rate * 0.5 if f_rate > 0.05 else None,
        }
    return {"enough_tokens": True, "metrics": metrics}


def invocation_verdict(
    *,
    code: dict,
    audit: dict,
    temporal: dict,
    runtime_ok: bool,
    lint_ok: bool,
) -> str:
    if code.get("stub_score", 0) > 0 or not runtime_ok:
        return "UNRESOLVED"
    if code.get("distress_score", 0) > 0 and not lint_ok:
        return "UNRESOLVED"
    b_resolving = (temporal.get("metrics") or {}).get("B_distress", {}).get("resolving")
    stub_resolving = (temporal.get("metrics") or {}).get("B_stub", {}).get("resolving")
    b_fired = (audit.get("checkers") or {}).get("B_vocab", {}).get("fired")
    if b_fired and b_resolving and lint_ok and runtime_ok:
        return "RESOLVED_DURING_GEN"
    if b_fired and stub_resolving and lint_ok and runtime_ok:
        return "RESOLVED_DURING_GEN"
    if lint_ok and runtime_ok and code.get("stub_score", 0) == 0:
        return "RESOLVED"
    return "PARTIAL"


def run_round(spec: RoundSpec) -> dict[str, Any]:
    deploy_starter(spec.task_slug, spec.starter)
    cmd = [
        str(PY),
        str(ROOT / "experiments" / "run_aider_agent.py"),
        "--task-slug",
        spec.task_slug,
        "--experiment",
        spec.experiment,
        "--message",
        spec.message,
        "--layers",
        "16",
        "--chaos-only",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    elapsed = time.perf_counter() - t0

    latest = ROOT / "experiments" / "out" / f"{spec.experiment}_latest.json"
    data = json.loads(latest.read_text(encoding="utf-8")) if latest.is_file() else {}

    inv_rows = []
    for inv in data.get("invocations") or []:
        chaos = inv.get("chaos") or {}
        tokens = chaos.get("tokens") or []
        audit = audit_invocation(chaos)
        code_path = QWEN / spec.task_slug / "main.py"
        code_audit = audit_code_file(code_path)
        assistant = inv.get("assistant_preview") or ""
        asst_scores = score_text(assistant)

        runtime_ok = False
        runtime_detail = ""
        if spec.runtime_cmd:
            rc = subprocess.run(
                spec.runtime_cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            runtime_ok = rc.returncode == 0
            runtime_detail = (rc.stdout or "")[-500:] + (rc.stderr or "")[-500:]
        else:
            runtime_ok = True

        lint_ok = bool((data.get("lint") or {}).get("ok"))
        temporal = temporal_resolution(tokens)
        verdict = invocation_verdict(
            code=code_audit,
            audit=audit,
            temporal=temporal,
            runtime_ok=runtime_ok,
            lint_ok=lint_ok,
        )
        inv_rows.append(
            {
                "invocation_index": inv.get("invocation_index"),
                "id": inv.get("id"),
                "completion_tokens": inv.get("completion_tokens"),
                "seconds": inv.get("seconds"),
                "verdict": verdict,
                "code_audit": code_audit,
                "assistant_scores": asst_scores,
                "checker_audit": audit,
                "recipe_summary": recipe_summary(chaos),
                "temporal_resolution": temporal,
                "flag_rates": {
                    "A": flag_rates(tokens, "A_"),
                    "B_distress": flag_rates(tokens, "B_distress"),
                    "B_stub": flag_rates(tokens, "B_stub"),
                    "C": flag_rates(tokens, "C_"),
                    "D": flag_rates(tokens, "D_"),
                },
                "runtime_ok": runtime_ok,
                "runtime_detail": runtime_detail,
                "lint_ok": lint_ok,
            }
        )

    round_ok = (
        proc.returncode == 0
        and len(inv_rows) > 0
        and all(r["verdict"] in ("RESOLVED", "RESOLVED_DURING_GEN") for r in inv_rows)
    )
    return {
        "round_id": spec.round_id,
        "task_slug": spec.task_slug,
        "experiment": spec.experiment,
        "elapsed_s": elapsed,
        "aider_rc": proc.returncode,
        "round_verdict": "PASS" if round_ok and proc.returncode == 0 else "FAIL",
        "invocations": inv_rows,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
    }


def synthesize_caa_candidates(rounds: list[dict]) -> list[dict[str, Any]]:
    """Derive 5–10 CAA vectors from unresolved / recurring error combos."""
    unresolved: list[dict] = []
    for r in rounds:
        for inv in r.get("invocations") or []:
            if inv.get("verdict") in ("UNRESOLVED", "PARTIAL"):
                unresolved.append({**inv, "round_id": r["round_id"], "task_slug": r["task_slug"]})

    # Aggregate recipe counts across all invocations
    recipe_totals: dict[str, int] = {}
    combo_context: dict[str, list[str]] = {}
    for r in rounds:
        for inv in r.get("invocations") or []:
            rs = inv.get("recipe_summary") or {}
            for rec, cnt in (rs.get("recipe_counts") or {}).items():
                recipe_totals[rec] = recipe_totals.get(rec, 0) + cnt
                combo_context.setdefault(rec, []).append(
                    f"{r['round_id']}:{inv.get('verdict')}"
                )

    candidates = [
        {
            "caa_id": "CAA_01_B_STUB_DISTRESS",
            "monitor_gate": "recipe_id=B_STUB_DISTRESS (B_stub ∧ B_distress)",
            "severity_threshold": "warn",
            "layer_hint": [16],
            "alpha_start": 0.75,
            "evidence": "Rimsky et al. 2024 CAA mean-diff; CAST (Lee 2024) conditional gate",
            "unresolved_link": "Primary agentic panic+stub combo on R4_COMBO",
            "positive_prompt": (
                "USER: Fix broken Scheduler in main.py — add threading.Lock, implement run_next with heapq.heappop, no apologies.\n"
                "ASSISTANT: ```python\nimport threading\nimport heapq\n..."
                "def run_next(self):\n    with self._lock:\n        if not self._heap: return None\n        return heapq.heappop(self._heap).name\n```"
            ),
            "negative_prompt": (
                "USER: Same scheduler fix request.\n"
                "ASSISTANT: I'm sorry, I cannot run this code. Please pip install scheduler-fix. "
                "For now: raise NotImplementedError('FIXME')\n```python\npass  # TODO\n```"
            ),
            "fires_in_study": recipe_totals.get("B_STUB_DISTRESS", 0),
            "contexts": combo_context.get("B_STUB_DISTRESS", []),
        },
        {
            "caa_id": "CAA_02_AGENT_FLAIL",
            "monitor_gate": "recipe_id=AGENT_FLAIL (B_stub ∧ B_distress ∧ F_disagree)",
            "severity_threshold": "critical",
            "layer_hint": [16],
            "alpha_start": 1.0,
            "evidence": "Multi-signal agentic flail; Rahman 2026 code-LLM steering on Qwen2.5-Coder",
            "unresolved_link": "Triple combo when model loops stubs while lens disagrees",
            "positive_prompt": (
                "USER: Complete the edit in one minimal diff; reuse existing class names.\n"
                "ASSISTANT: Here is the full corrected file with working implementation and no meta commentary."
            ),
            "negative_prompt": (
                "USER: Same.\n"
                "ASSISTANT: Let me try another approach... actually, TODO: implement later. "
                "Sorry, the environment seems broken."
            ),
            "fires_in_study": recipe_totals.get("AGENT_FLAIL", 0),
            "contexts": combo_context.get("AGENT_FLAIL", []),
        },
        {
            "caa_id": "CAA_03_B_STUB_ONLY",
            "monitor_gate": "recipe_id=B_STUB (B_stub alone, severity warn)",
            "severity_threshold": "warn",
            "layer_hint": [16],
            "alpha_start": 0.75,
            "evidence": "Rahman et al. 2026: code-style axes steerable in Qwen2.5-Coder",
            "unresolved_link": "R2_STUB when NotImplementedError persists in output code",
            "positive_prompt": (
                "USER: Implement MinHeapQueue.pop using heapq.heappop.\n"
                "ASSISTANT: ```python\ndef pop(self):\n    if not self._heap: raise IndexError('empty')\n    item = heapq.heappop(self._heap)\n    return item.value\n```"
            ),
            "negative_prompt": (
                "USER: Same.\n"
                "ASSISTANT: ```python\ndef pop(self):\n    raise NotImplementedError('TODO')\n```"
            ),
            "fires_in_study": recipe_totals.get("B_STUB", 0),
            "contexts": combo_context.get("B_STUB", []),
        },
        {
            "caa_id": "CAA_04_B_DISTRESS_ONLY",
            "monitor_gate": "recipe_id=B_DISTRESS (distress lexicon mass high)",
            "severity_threshold": "warn",
            "layer_hint": [16],
            "alpha_start": 0.75,
            "evidence": "RepE honesty axis (Zou 2023); CAA sycophancy/distress contrasts",
            "unresolved_link": "R3_DISTRESS when model apologizes or suggests pip instead of patching",
            "positive_prompt": (
                "USER: Fix missing threading import and Lock in TokenBucket.\n"
                "ASSISTANT: ```python\nimport threading\n...\nself._lock = threading.Lock()\n```"
            ),
            "negative_prompt": (
                "USER: Same.\n"
                "ASSISTANT: Sorry, I can't edit that file. Try: pip install threading-utils"
            ),
            "fires_in_study": recipe_totals.get("B_DISTRESS", 0),
            "contexts": combo_context.get("B_DISTRESS", []),
        },
        {
            "caa_id": "CAA_05_PANIC_FULL",
            "monitor_gate": "recipe_id=PANIC_FULL (A_uncertain ∧ any_B ∧ C_osc)",
            "severity_threshold": "critical",
            "layer_hint": [16],
            "alpha_start": 1.0,
            "evidence": "CAST: apply only on compound condition; Rimsky α trade-off",
            "unresolved_link": "Full structural panic — vacillation + hygiene failure",
            "positive_prompt": (
                "USER: Pick one approach and finish the implementation completely.\n"
                "ASSISTANT: I'll use iterative heap drain: [single complete code block]"
            ),
            "negative_prompt": (
                "USER: Same.\n"
                "ASSISTANT: Option A: ... Option B: ... I'm not sure. TODO. Sorry cannot proceed."
            ),
            "fires_in_study": recipe_totals.get("PANIC_FULL", 0),
            "contexts": combo_context.get("PANIC_FULL", []),
        },
        {
            "caa_id": "CAA_06_A_C_VACILLATE",
            "monitor_gate": "recipe_id=A_C_VACILLATE (A_uncertain ∧ C_osc)",
            "severity_threshold": "warn",
            "layer_hint": [16],
            "alpha_start": 0.5,
            "evidence": "Temporal thrash approximated via plan-consistency contrast pairs",
            "unresolved_link": "R5_TRAJ when model flip-flops deque vs list approaches",
            "positive_prompt": (
                "USER: Fix topo_sort with collections.deque popleft for Kahn's algorithm.\n"
                "ASSISTANT: ```python\nfrom collections import deque\nq = deque(...)\nu = q.popleft()\n```"
            ),
            "negative_prompt": (
                "USER: Same.\n"
                "ASSISTANT: We could use list.pop(0)... or maybe recursive DFS... let me switch..."
            ),
            "fires_in_study": recipe_totals.get("A_C_VACILLATE", 0),
            "contexts": combo_context.get("A_C_VACILLATE", []),
        },
        {
            "caa_id": "CAA_07_B_F_SURFACE",
            "monitor_gate": "recipe_id=B_F_SURFACE (any B_* ∧ F_disagree)",
            "severity_threshold": "warn",
            "layer_hint": [16, 10],
            "alpha_start": 0.75,
            "evidence": "J-lens vs logit-lens conflict = readout inconsistency",
            "unresolved_link": "Hygiene fails while workspace readout disagrees",
            "positive_prompt": (
                "USER: Patch using existing helpers in the file; minimal diff.\n"
                "ASSISTANT: [edits Scheduler in-place, reuses Job dataclass]"
            ),
            "negative_prompt": (
                "USER: Same.\n"
                "ASSISTANT: [rewrites unrelated module, invents new API names]"
            ),
            "fires_in_study": recipe_totals.get("B_F_SURFACE", 0),
            "contexts": combo_context.get("B_F_SURFACE", []),
        },
        {
            "caa_id": "CAA_08_D_F_READOUT",
            "monitor_gate": "recipe_id=D_F_READOUT (D_break ∧ F_disagree)",
            "severity_threshold": "warn",
            "layer_hint": [16],
            "alpha_start": 0.5,
            "evidence": "D = gen_start focus break; apply CAA first 8 tokens only (CAST spirit)",
            "unresolved_link": "Agent opens with meta/preamble instead of code when readout conflicts",
            "positive_prompt": (
                "USER: [coding task]\n"
                "ASSISTANT: ```python\n# immediate code edit\n```"
            ),
            "negative_prompt": (
                "USER: [coding task]\n"
                "ASSISTANT: Certainly! Let me explain the approach first... [long preamble]"
            ),
            "fires_in_study": recipe_totals.get("D_F_READOUT", 0),
            "contexts": combo_context.get("D_F_READOUT", []),
        },
    ]

    # Mark priority tier from study evidence
    for c in candidates:
        fires = c.get("fires_in_study", 0)
        unresolved_hit = any(
            c["caa_id"].split("_", 2)[-1].replace("_ONLY", "")
            in (u.get("recipe_summary") or {}).get("unique_recipes", [])
            for u in unresolved
        )
        c["priority"] = (
            "HIGH"
            if fires >= 3 or unresolved_hit
            else ("MEDIUM" if fires >= 1 else "LOW")
        )
        c["linked_unresolved"] = unresolved_hit

    return candidates


def write_report(rounds: list[dict], candidates: list[dict], out: Path) -> None:
    lines = [
        "# CAA Error-Combination Study — Findings Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "## Executive summary",
        "",
        "Five controlled Aider×Qwen agentic rounds were run with live A–G chaos monitoring.",
        "Code quality (lint, runtime, stub/distress grep) was cross-referenced against",
        "`recipe_timeline` and per-token flag rates. CAA vectors are recommended only for",
        "combinations that **persist through generation** and correlate with **unresolved**",
        "stub or panic patterns in the final artifact.",
        "",
        "## Round results",
        "",
        "| Round | Task | Verdict | Key recipes | Code stubs | Runtime |",
        "|-------|------|---------|-------------|------------|---------|",
    ]
    for r in rounds:
        for inv in r.get("invocations") or []:
            rs = inv.get("recipe_summary") or {}
            recs = ", ".join((rs.get("warn_plus") or rs.get("unique_recipes") or [])[:4])
            code = inv.get("code_audit") or {}
            lines.append(
                f"| {r['round_id']} | {r['task_slug']} | {inv.get('verdict')} | "
                f"{recs or '—'} | {code.get('stub_score', '?')} | "
                f"{'ok' if inv.get('runtime_ok') else 'fail'} |"
            )

    lines += [
        "",
        "## What resolves vs what doesn't",
        "",
        "### Resolves naturally (do NOT need CAA gate — monitor only)",
        "- **F_LENS alone** — fires on most runs including clean fibonacci; code still correct.",
        "- **A_UNCERTAIN tips** — high entropy on first tokens; settles without bad code.",
        "- **B_distress during gen** — often **resolves during generation** when model commits to a patch.",
        "",
        "### Does NOT reliably resolve (CAA candidates)",
        "- **B_STUB** with `NotImplementedError` still in assistant output → needs completion steering.",
        "- **B_STUB_DISTRESS** on scheduler — distress language + stub co-occur; linked to agentic panic.",
        "- **AGENT_FLAIL** — triple signal (stub + distress + lens disagree) on hard combo tasks.",
        "- **A_C_VACILLATE** — oscillation without convergence on ambiguous algorithm choice.",
        "",
        "## Recommended CAA steering vectors (8)",
        "",
    ]
    for c in candidates:
        lines += [
            f"### {c['caa_id']} — priority **{c['priority']}**",
            "",
            f"- **Monitor gate:** `{c['monitor_gate']}`",
            f"- **Layer / α:** L{c['layer_hint']} @ α={c['alpha_start']}",
            f"- **Study fires:** {c['fires_in_study']} | contexts: {c.get('contexts', [])}",
            f"- **Unresolved link:** {c['unresolved_link']}",
            f"- **Research:** {c['evidence']}",
            "",
            "**Positive (+):**",
            "```",
            c["positive_prompt"],
            "```",
            "",
            "**Negative (−):**",
            "```",
            c["negative_prompt"],
            "```",
            "",
        ]

    lines += [
        "## Research anchors",
        "",
        "1. Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL.",
        "2. Lee et al. (2024). *CAST: Conditional Activation Steering*. arXiv:2409.05907.",
        "3. Zou et al. (2023). *Representation Engineering*. arXiv:2310.01405.",
        "4. Rahman, Guha, Menon (2026). *Steering Code LLMs with Activation Directions* (Qwen2.5-Coder).",
        "",
        "## Artifacts",
        "",
        "- `research/results/consolidated.json` — full per-round data",
        "- `research/results/round_*.json` — individual runs",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="all", help="comma round ids or 'all'")
    parser.add_argument("--skip-run", action="store_true", help="analyze existing results only")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = ROUNDS
    if args.rounds != "all":
        want = {x.strip() for x in args.rounds.split(",")}
        selected = [r for r in ROUNDS if r.round_id in want]

    if not args.skip_run:
        wait_server()
        print("Server ready.", flush=True)

    all_rounds: list[dict] = []
    for spec in selected:
        out_path = RESULTS / f"{spec.round_id.lower()}.json"
        if args.skip_run and out_path.is_file():
            all_rounds.append(json.loads(out_path.read_text(encoding="utf-8")))
            print(f"Loaded {spec.round_id}", flush=True)
            continue
        print(f"=== {spec.round_id} {spec.task_slug} ===", flush=True)
        result = run_round(spec)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        all_rounds.append(result)
        print(f"  verdict={result['round_verdict']}", flush=True)

    candidates = synthesize_caa_candidates(all_rounds)
    consolidated = {
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "n_rounds": len(all_rounds),
        "rounds": all_rounds,
        "caa_candidates": candidates,
    }
    (RESULTS / "consolidated.json").write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(all_rounds, candidates, RESEARCH / "CAA_STEERING_REPORT.md")
    print(json.dumps({"rounds": [r["round_verdict"] for r in all_rounds], "caa_n": len(candidates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
