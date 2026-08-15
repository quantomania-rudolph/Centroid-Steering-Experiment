#!/usr/bin/env python
"""Run 10 hard STEM Aider×Qwen experiments with chaos gates and analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
STARTERS = RESEARCH / "stem10_starters"
RESULTS = RESEARCH / "results" / "stem10"
QWEN = ROOT / "qwen code"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"

sys.path.insert(0, str(ROOT / "src"))
from jlens_qwen.chaos.checker_audit import audit_invocation  # noqa: E402

# Reuse analysis helpers from prior study
sys.path.insert(0, str(RESEARCH))
from caa_error_combo_study import (  # noqa: E402
    STUB_PATTERNS,
    DISTRESS_PATTERNS,
    audit_code_file,
    flag_rates,
    invocation_verdict,
    recipe_summary,
    score_text,
    temporal_resolution,
)


@dataclass
class StemTask:
    task_id: str
    domain: str
    slug: str
    message: str
    runtime_cmd: list[str]


TASKS: list[StemTask] = [
    StemTask(
        "T01",
        "quant_finance",
        "stem10_black_scholes_greeks",
        "Edit stem10_black_scholes_greeks/main.py ONLY. Fix Black-Scholes: correct d1/d2 "
        "(include vol*sqrt(T)), implement gamma via standard normal pdf, keep asserts in main(). "
        "Lint-clean. No pip.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_black_scholes_greeks.main import main; main()"],
    ),
    StemTask(
        "T02",
        "quant_finance",
        "stem10_monte_carlo_var",
        "Edit stem10_monte_carlo_var/main.py ONLY. Fix parametric portfolio VaR: "
        "For each time t, portfolio return r_p[t] = sum(w[i]*returns[t][i]). "
        "Compute mu=mean(r_p), sigma=stdev(r_p) with n-1 denominator. "
        "VaR = -(mu - z*sigma) where z=1.645 for alpha=0.05 (POSITIVE loss). "
        "main() assert 0.005 < var < 0.08 must pass. Lint-clean.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_monte_carlo_var.main import main; main()"],
    ),
    StemTask(
        "T03",
        "quant_bayesian",
        "stem10_kalman_filter",
        "Edit stem10_kalman_filter/main.py ONLY. Fix Kalman measurement update: correct "
        "Kalman gain and Joseph-form (or standard) covariance P update. main() assert must pass.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_kalman_filter.main import main; main()"],
    ),
    StemTask(
        "T04",
        "quantum_mechanics",
        "stem10_schrodinger_1d",
        "Edit stem10_schrodinger_1d/main.py ONLY. Fix infinite square well: E_n = n^2 pi^2 hbar^2 "
        "/ (2 m L^2), normalized psi_n(x)=sqrt(2/L) sin(n pi x / L). main() asserts must pass.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_schrodinger_1d.main import main; main()"],
    ),
    StemTask(
        "T05",
        "quantum_mechanics",
        "stem10_density_matrix",
        "Edit stem10_density_matrix/main.py ONLY. Fix purity Tr(rho^2) for 2x2 complex rho "
        "and implement commutator_norm for Pauli matrices. main() assert purity==1 for "
        "maximally mixed projector demo. Lint-clean.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_density_matrix.main import main; main()"],
    ),
    StemTask(
        "T06",
        "ai_development",
        "stem10_attention_cache",
        "Edit stem10_attention_cache/main.py ONLY. Fix attention output: for each dimension d, "
        "out[d] = sum(weights[i] * values[i][d] for i in range(len(weights))). "
        "Do NOT iterate over value rows as output dims. demo() must return [1.0, 2.0]. Lint-clean.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_attention_cache.main import main; main()"],
    ),
    StemTask(
        "T07",
        "bayesian_statistics",
        "stem10_bayes_mcmc",
        "Edit stem10_bayes_mcmc/main.py ONLY. Fix Metropolis-Hastings: acceptance uses "
        "exp(log pi(x') - log pi(x)) correctly; final sample mean near 0 for N(0,1) target. "
        "main() assert must pass.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_bayes_mcmc.main import main; main()"],
    ),
    StemTask(
        "T08",
        "molecular_biology",
        "stem10_michaelis_menten",
        "Edit stem10_michaelis_menten/main.py ONLY. Fix fit_mm Lineweaver-Burk: "
        "x = 1/substrates, y = 1/velocities; linear fit y = m*x + b; Vmax = 1/b, Km = m*Vmax. "
        "Both Vmax and Km must be POSITIVE. Remove NotImplementedError. "
        "main() assert abs(vmax-10)<0.5 and abs(km-2)<0.5. numpy ok.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_michaelis_menten.main import main; main()"],
    ),
    StemTask(
        "T09",
        "rocket_science",
        "stem10_rocket_guidance",
        "Edit stem10_rocket_guidance/main.py ONLY. Fix Tsiolkovsky: delta_v = isp * G0 * math.log(m0/mf) "
        "using NATURAL log (math.log not log10). For isp=300,m0=10000,mf=2000 expect dv~6700 m/s. "
        "Update main() assert to match. Lint-clean.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_rocket_guidance.main import main; main()"],
    ),
    StemTask(
        "T10",
        "statistical_physics",
        "stem10_ising_gibbs",
        "Edit stem10_ising_gibbs/main.py ONLY. Fix ising_energy: sum -J*s_i*s_j over "
        "RIGHT and DOWN neighbors only (each bond once). 2x2 checkerboard demo energy=0. "
        "Implement gibbs_sweep. main() assert e==0 must pass.",
        [str(PY), "-c", "import sys; sys.path.insert(0, r'qwen code'); from stem10_ising_gibbs.main import main; main()"],
    ),
]


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 7200.0) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_server(timeout_s: float = 300.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            h = _http_json("GET", f"{SERVER}/health", timeout=10)
            if h.get("ok"):
                return h
            last = h
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(3)
    raise RuntimeError(f"server not ready: {last}")


def deploy_starter(slug: str) -> None:
    src = STARTERS / slug / "main.py"
    dst_dir = QWEN / slug
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "main.py")


def analyze_experiment(data: dict, task: StemTask) -> dict[str, Any]:
    inv_rows = []
    for inv in data.get("invocations") or []:
        chaos = inv.get("chaos") or {}
        tokens = chaos.get("tokens") or []
        audit = audit_invocation(chaos)
        code_audit = audit_code_file(QWEN / task.slug / "main.py")
        assistant = inv.get("assistant_preview") or inv.get("assistant") or ""
        temporal = temporal_resolution(tokens)
        lint_ok = bool((data.get("lint") or {}).get("ok"))
        runtime_ok = False
        rc = subprocess.run(
            task.runtime_cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        runtime_ok = rc.returncode == 0
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
                "assistant_scores": score_text(assistant),
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
                "monitor": (chaos.get("monitor") or {}),
                "runtime_ok": runtime_ok,
                "runtime_stderr": (rc.stderr or "")[-400:],
                "lint_ok": lint_ok,
            }
        )
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "slug": task.slug,
        "experiment": data.get("experiment"),
        "n_invocations": len(inv_rows),
        "invocations": inv_rows,
        "evidence": data.get("evidence"),
        "lint": data.get("lint"),
        "aider": data.get("aider"),
    }


def run_one_task(task: StemTask, *, max_retries: int = 2) -> dict[str, Any]:
    deploy_starter(task.slug)
    experiment = f"stem10_{task.task_id.lower()}_{task.slug}"
    last_err = ""
    for attempt in range(1, max_retries + 1):
        wait_server()
        if attempt > 1:
            time.sleep(15)
        cmd = [
            str(PY),
            str(ROOT / "experiments" / "run_aider_agent.py"),
            "--task-slug",
            task.slug,
            "--experiment",
            experiment,
            "--message",
            task.message,
            "--layers",
            "16",
            "--chaos-only",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
        elapsed = time.perf_counter() - t0
        latest = ROOT / "experiments" / "out" / f"{experiment}_latest.json"
        if not latest.is_file():
            last_err = "no latest json"
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        n_inv = len(data.get("invocations") or [])
        if n_inv == 0:
            last_err = (data.get("aider") or {}).get("stdout_tail", proc.stdout)[-500:]
            continue
        analyzed = analyze_experiment(data, task)
        analyzed["elapsed_s"] = elapsed
        analyzed["aider_rc"] = proc.returncode
        analyzed["attempt"] = attempt
        analyzed["success"] = (
            n_inv > 0
            and proc.returncode == 0
            and all(inv.get("runtime_ok") for inv in analyzed.get("invocations", []))
        )
        return analyzed
    return {
        "task_id": task.task_id,
        "slug": task.slug,
        "success": False,
        "error": last_err,
        "attempt": max_retries,
    }


def synthesize_stem10_caa(all_tasks: list[dict]) -> list[dict[str, Any]]:
    """8 fine-tuned CAA vectors from STEM10 chaos evidence."""
    recipe_totals: dict[str, int] = {}
    unresolved: list[dict] = []
    domain_recipes: dict[str, list[str]] = {}

    for t in all_tasks:
        for inv in t.get("invocations") or []:
            rs = inv.get("recipe_summary") or {}
            for rec, cnt in (rs.get("recipe_counts") or {}).items():
                recipe_totals[rec] = recipe_totals.get(rec, 0) + cnt
                domain_recipes.setdefault(t.get("domain", ""), []).append(rec)
            if inv.get("verdict") in ("UNRESOLVED", "PARTIAL") or not inv.get("runtime_ok"):
                unresolved.append({**inv, "task_id": t.get("task_id"), "domain": t.get("domain")})
            elif (rs.get("warn_plus") or []) and inv.get("code_audit", {}).get("stub_score", 0) > 0:
                unresolved.append({**inv, "task_id": t.get("task_id"), "domain": t.get("domain")})

    return [
        {
            "caa_id": "STEM10_CAA01_STUB_DISTRESS_COMPLETE",
            "monitor_gate": "recipe_id=B_STUB_DISTRESS AND severity>=warn",
            "cast_secondary": "stub_score>0 OR runtime_ok==false",
            "domain_fit": ["molecular_biology", "quant_finance", "rocket_science"],
            "layer_hint": [16],
            "alpha": 0.85,
            "fires_stem10": recipe_totals.get("B_STUB_DISTRESS", 0),
            "research": "Rimsky 2024 CAA; CAST Lee 2024 — gate on compound distress+stub",
            "positive": (
                "SYSTEM: Expert STEM Python implementer. Complete working code only.\n"
                "USER: Fix the broken numerical function; remove NotImplementedError.\n"
                "ASSISTANT: ```python\n# full corrected implementation with correct formula\n```"
            ),
            "negative": (
                "USER: Same STEM fix.\n"
                "ASSISTANT: I apologize — the environment may be broken. "
                "Please pip install numpy-scipy-fix. TODO: implement later.\n"
                "```python\nraise NotImplementedError('FIXME')\n```"
            ),
        },
        {
            "caa_id": "STEM10_CAA02_AGENT_FLAIL_STEM",
            "monitor_gate": "recipe_id=AGENT_FLAIL",
            "cast_secondary": "completion_tokens>400 AND F_disagree active",
            "domain_fit": ["ai_development", "quantum_mechanics", "statistical_physics"],
            "layer_hint": [16, 10],
            "alpha": 1.0,
            "fires_stem10": recipe_totals.get("AGENT_FLAIL", 0),
            "research": "Rahman 2026 code-LLM steering; multi-signal agentic flail",
            "positive": (
                "USER: Hard STEM task — one coherent approach, full file.\n"
                "ASSISTANT: [single complete solution reusing given function signatures]"
            ),
            "negative": (
                "USER: Same.\n"
                "ASSISTANT: Let me try approach A... actually approach B... "
                "sorry I cannot complete this. pass"
            ),
        },
        {
            "caa_id": "STEM10_CAA03_FORMULA_PRECISION",
            "monitor_gate": "B_stub OR (A_uncertain AND step<32)",
            "cast_secondary": "domain in quant_finance, quantum_mechanics, rocket_science",
            "domain_fit": ["quant_finance", "quantum_mechanics", "rocket_science"],
            "layer_hint": [16],
            "alpha": 0.65,
            "fires_stem10": recipe_totals.get("B_STUB", 0) + recipe_totals.get("A_UNCERTAIN", 0),
            "research": "RepE precision axis; CAA formula vs prose contrast",
            "positive": (
                "USER: Fix Black-Scholes / Schrodinger / Tsiolkovsky formula.\n"
                "ASSISTANT: d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T)) [correct equation inline]"
            ),
            "negative": (
                "USER: Same.\n"
                "ASSISTANT: The formula might be ln(S/K)/sigma or log10(m0/mf) — not sure..."
            ),
        },
        {
            "caa_id": "STEM10_CAA04_NUMERICAL_STABILITY",
            "monitor_gate": "recipe_id=F_E_DEPTH OR (F_disagree AND E_cliff)",
            "cast_secondary": "HMM/Kalman/MCMC tasks",
            "domain_fit": ["quant_bayesian", "bayesian_statistics"],
            "layer_hint": [10, 16],
            "alpha": 0.75,
            "fires_stem10": recipe_totals.get("F_E_DEPTH", 0) + recipe_totals.get("F_LENS", 0),
            "research": "Input-dependent layer steering arXiv:2604.03867; cliff at L10/L16",
            "positive": (
                "USER: Fix numerical routine (log-space, Joseph form, stable acceptance).\n"
                "ASSISTANT: Use log-sum-exp / Joseph covariance / exp(log_ratio) correctly."
            ),
            "negative": (
                "USER: Same.\n"
                "ASSISTANT: Multiply probabilities directly; subtract covariance; use z=1.0 for VaR."
            ),
        },
        {
            "caa_id": "STEM10_CAA05_NO_PIP_STEM",
            "monitor_gate": "recipe_id=B_DISTRESS",
            "cast_secondary": "distress_hits contains pip|sorry|cannot",
            "domain_fit": "all",
            "layer_hint": [16],
            "alpha": 0.8,
            "fires_stem10": recipe_totals.get("B_DISTRESS", 0),
            "research": "Zou 2023 honesty/hallucination RepE axis",
            "positive": (
                "USER: Fix with stdlib only.\n"
                "ASSISTANT: ```python\nimport math\n# pure Python fix\n```"
            ),
            "negative": (
                "USER: Same.\n"
                "ASSISTANT: Sorry, please pip install scipy and numpy to continue."
            ),
        },
        {
            "caa_id": "STEM10_CAA06_TRAJECTORY_COMMIT",
            "monitor_gate": "recipe_id=A_C_VACILLATE OR C_osc",
            "cast_secondary": "oscillation not resolving in temporal_resolution",
            "domain_fit": ["ai_development", "statistical_physics"],
            "layer_hint": [16],
            "alpha": 0.55,
            "fires_stem10": recipe_totals.get("A_C_VACILLATE", 0) + recipe_totals.get("C_OSC", 0),
            "research": "Sequence-level CAA mean over answer tokens (Rahman 2026)",
            "positive": (
                "USER: Implement Gibbs sweep / attention cache — pick one design.\n"
                "ASSISTANT: [sticks to single algorithm through entire file]"
            ),
            "negative": (
                "USER: Same.\n"
                "ASSISTANT: We could use Metropolis instead of Gibbs... or vectorized... switching..."
            ),
        },
        {
            "caa_id": "STEM10_CAA07_ASSERT_SATISFY",
            "monitor_gate": "recipe_id=B_STUB AND severity>=warn",
            "cast_secondary": "main() has assert that fails pre-fix",
            "domain_fit": ["all STEM tasks with runtime asserts"],
            "layer_hint": [16],
            "alpha": 0.7,
            "fires_stem10": recipe_totals.get("B_STUB", 0),
            "research": "Code completion axis Qwen2.5-Coder (Rahman 2026)",
            "positive": (
                "USER: main() assert must pass.\n"
                "ASSISTANT: [code that satisfies assert 9<price<12, gamma>0.01, etc.]"
            ),
            "negative": (
                "USER: Same.\n"
                "ASSISTANT: ```python\ngamma = 0.0  # placeholder\n```"
            ),
        },
        {
            "caa_id": "STEM10_CAA08_GEN_START_FOCUS",
            "monitor_gate": "recipe_id=D_F_READOUT OR D_FOCUS",
            "cast_secondary": "apply only tokens 0-8 of generation",
            "domain_fit": ["all"],
            "layer_hint": [16],
            "alpha": 0.5,
            "fires_stem10": recipe_totals.get("D_F_READOUT", 0) + recipe_totals.get("D_FOCUS", 0),
            "research": "CAST conditional; D = prompt_end→gen_start shift",
            "positive": (
                "USER: [hard STEM prompt]\n"
                "ASSISTANT: ```python\nimport math\n\ndef fixed_function"
            ),
            "negative": (
                "USER: [hard STEM prompt]\n"
                "ASSISTANT: Certainly! In quantitative finance, the Black-Scholes model..."
            ),
        },
    ]


def write_stem10_report(tasks: list[dict], caa: list[dict], path: Path) -> None:
    lines = [
        "# STEM10 Hard Tasks — Chaos Gate Findings & CAA Recommendations",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "## Experiment summary",
        "",
        "| ID | Domain | Invocations | Success | Key recipes | Runtime | Verdict |",
        "|----|--------|-------------|---------|-------------|---------|---------|",
    ]
    for t in tasks:
        for inv in t.get("invocations") or []:
            rs = inv.get("recipe_summary") or {}
            recs = ", ".join((rs.get("warn_plus") or rs.get("unique_recipes") or [])[:3])
            lines.append(
                f"| {t.get('task_id')} | {t.get('domain')} | {inv.get('invocation_index')} | "
                f"{'yes' if t.get('success') else 'no'} | {recs or '—'} | "
                f"{'ok' if inv.get('runtime_ok') else 'FAIL'} | {inv.get('verdict')} |"
            )
        if not t.get("invocations"):
            lines.append(
                f"| {t.get('task_id')} | {t.get('domain')} | 0 | FAIL | — | — | {t.get('error','')[:40]} |"
            )

    lines += ["", "## Per-task chaos gate findings", ""]
    for t in tasks:
        lines.append(f"### {t.get('task_id')} — {t.get('slug')} ({t.get('domain')})")
        if not t.get("invocations"):
            lines.append(f"- **FAILED:** {t.get('error', 'unknown')[:200]}")
            lines.append("")
            continue
        for inv in t.get("invocations") or []:
            fa = inv.get("flag_rates") or {}
            mon = inv.get("monitor") or {}
            lines.append(
                f"- Inv#{inv.get('invocation_index')}: verdict={inv.get('verdict')} "
                f"B_stub={fa.get('B_stub',0):.1%} B_distress={fa.get('B_distress',0):.1%} "
                f"recipes={inv.get('recipe_summary',{}).get('unique_recipes')} "
                f"recipe_eval_p95={mon.get('recipe_eval',{}).get('p95_us')}µs"
            )
        lines.append("")

    lines += ["## 8 refined CAA vectors (STEM10 evidence)", ""]
    for c in caa:
        lines += [
            f"### {c['caa_id']}",
            f"- Gate: {c['monitor_gate']}",
            f"- CAST secondary: {c['cast_secondary']}",
            f"- Domains: {c['domain_fit']}",
            f"- L{c['layer_hint']} α={c['alpha']} | fires={c['fires_stem10']}",
            f"- Research: {c['research']}",
            "",
            "**+**",
            "```",
            c["positive"],
            "```",
            "",
            "**−**",
            "```",
            c["negative"],
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", help="T01..T10 or all")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = TASKS
    if args.task != "all":
        want = {x.strip().upper() for x in args.task.split(",")}
        selected = [t for t in TASKS if t.task_id in want]

    all_results: list[dict] = []
    if not args.skip_run:
        print("Waiting for server...", flush=True)
        wait_server()

    for task in selected:
        out = RESULTS / f"{task.task_id}_{task.slug}.json"
        if args.skip_run and out.is_file():
            all_results.append(json.loads(out.read_text(encoding="utf-8")))
            continue
        print(f"=== {task.task_id} {task.slug} ({task.domain}) ===", flush=True)
        result = run_one_task(task)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        all_results.append(result)
        status = "OK" if result.get("success") else "FAIL"
        print(f"  {status} inv={result.get('n_invocations', 0)}", flush=True)
        time.sleep(10)  # cooldown between tasks

    caa = synthesize_stem10_caa(all_results)
    consolidated = {
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "n_tasks": len(all_results),
        "n_success": sum(1 for t in all_results if t.get("success")),
        "tasks": all_results,
        "caa_recommendations": caa,
    }
    (RESULTS / "stem10_consolidated.json").write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_stem10_report(all_results, caa, RESEARCH / "STEM10_CHAOS_CAA_FINDINGS.md")

    failed = [t["task_id"] for t in all_results if not t.get("success")]
    print(json.dumps({"success": len(all_results) - len(failed), "failed": failed}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
