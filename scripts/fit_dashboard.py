#!/usr/bin/env python
"""Live fit dashboard: process health, progress, ETA. Writes HTML + JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GATES = ROOT / "artifacts" / "gates"
DASHBOARD_DIR = ROOT / "artifacts" / "dashboard"
PROGRESS_PATH = GATES / "fit_progress.json"
JOB_PATH = GATES / "fit_job.json"
PLAN_PATH = GATES / "cpu_fit_plan.json"
L15_LENS = ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens_L15.pt"
READOUT_TXT = ROOT / "artifacts" / "readouts" / "L15_readout.txt"
LOG_PATH = GATES / "fit_L15_maxcpu.log"

# Heuristic from observed L15 smoke on this machine (dim_batch=14, ~30s/pass wall)
DEFAULT_SEC_PER_PASS = 30.0


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _find_fit_process() -> dict | None:
    import psutil

    candidates: list[dict] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if proc.info["name"] != "python.exe":
                continue
            cmd = " ".join(proc.info["cmdline"] or [])
            if "fit_jacobian.py" not in cmd:
                continue
            mem = proc.memory_info().rss / 1e9
            cpu = proc.cpu_times()
            candidates.append(
                {
                    "pid": proc.info["pid"],
                    "cmdline": cmd,
                    "ram_gb": round(mem, 2),
                    "cpu_user_sec": round(cpu.user, 1),
                    "cpu_system_sec": round(cpu.system, 1),
                    "started_at": datetime.fromtimestamp(
                        proc.info["create_time"], tz=timezone.utc
                    ).isoformat(),
                    "runtime_sec": round(time.time() - proc.info["create_time"], 1),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["ram_gb"])


def _sample_cpu_rate(pid: int, interval: float = 10.0) -> float | None:
    import psutil

    try:
        proc = psutil.Process(pid)
        t0 = time.time()
        c0 = sum(proc.cpu_times()[:2])
        time.sleep(interval)
        c1 = sum(proc.cpu_times()[:2])
        wall = time.time() - t0
        if wall <= 0:
            return None
        return round((c1 - c0) / wall, 2)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds):
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_status(*, sample_cpu: bool = False, cpu_sample_sec: float = 8.0) -> dict:
    from jlens_qwen.fit_progress import backward_passes_total

    job = _read_json(JOB_PATH)
    progress = _read_json(PROGRESS_PATH)
    plan = _read_json(PLAN_PATH)
    proc = _find_fit_process()

    dim_batch = int((job or plan or {}).get("dim_batch", 14))
    n_prompts = int((job or {}).get("n_prompts", 1))
    total_passes = backward_passes_total(dim_batch)
    layers = (job or {}).get("source_layers", "15-15")

    cpu_rate = None
    health = "unknown"
    if proc:
        if sample_cpu:
            cpu_rate = _sample_cpu_rate(proc["pid"], cpu_sample_sec)
        elif proc["runtime_sec"] > 0:
            cpu_rate = round(
                (proc["cpu_user_sec"] + proc["cpu_system_sec"]) / proc["runtime_sec"], 2
            )
        if cpu_rate is not None and cpu_rate >= 2.0:
            health = "active"
        elif cpu_rate is not None and cpu_rate >= 0.5:
            health = "low"
        else:
            health = "stalled"
    elif L15_LENS.exists():
        health = "complete"
    else:
        health = "not_running"

    eta_sec = None
    eta_source = None
    fraction = None
    backward_pass = None

    if progress:
        backward_pass = progress.get("backward_pass")
        fraction = progress.get("fraction")
        eta_sec = progress.get("eta_seconds")
        eta_source = "measured (per-pass timing)"
    elif proc and proc["runtime_sec"] > 60:
        # No per-pass file (pre-progress run): estimate from runtime vs typical pass rate
        assumed_pass = min(
            total_passes - 1,
            max(1, int(proc["runtime_sec"] / DEFAULT_SEC_PER_PASS)),
        )
        backward_pass = assumed_pass
        fraction = min(0.99, assumed_pass / total_passes)
        remaining = max(0, total_passes - assumed_pass)
        eta_sec = remaining * DEFAULT_SEC_PER_PASS
        eta_source = f"heuristic (~{DEFAULT_SEC_PER_PASS:.0f}s/pass, no per-pass telemetry)"

    log_tail = ""
    if LOG_PATH.exists():
        raw = LOG_PATH.read_bytes()
        if raw.startswith(b"\xff\xfe") or b"\x00" in raw[:200]:
            text = raw.decode("utf-16", errors="replace")
        else:
            text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        log_tail = "\n".join(lines[-8:])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job": job,
        "plan": plan,
        "progress": progress,
        "process": proc,
        "health": health,
        "cpu_cores_equiv": cpu_rate,
        "layers": layers,
        "n_prompts": n_prompts,
        "dim_batch": dim_batch,
        "backward_passes_total": total_passes,
        "backward_pass_estimate": backward_pass,
        "fraction_estimate": fraction,
        "eta_seconds": eta_sec,
        "eta_human": _format_duration(eta_sec),
        "eta_source": eta_source,
        "l15_lens_exists": L15_LENS.exists(),
        "readout_exists": READOUT_TXT.exists(),
        "log_tail": log_tail,
        "pipeline": [
            {"step": "L15 smoke fit", "done": L15_LENS.exists(), "active": health == "active"},
            {"step": "verify_lens_smoke", "done": (GATES / "lens_smoke_verify.json").exists()},
            {"step": "show_layer_readout", "done": READOUT_TXT.exists()},
            {"step": "production fit", "done": (ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt").exists()},
        ],
    }


def render_html(status: dict, *, refresh_sec: int = 30) -> str:
    def row(label: str, value: str, ok: bool | None = None) -> str:
        badge = ""
        if ok is True:
            badge = " <span class='ok'>✓</span>"
        elif ok is False:
            badge = " <span class='bad'>✗</span>"
        return f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}{badge}</td></tr>"

    health = status["health"]
    health_class = {"active": "ok", "complete": "ok", "low": "warn", "stalled": "bad"}.get(
        health, ""
    )

    frac = status.get("fraction_estimate")
    frac_pct = f"{100 * frac:.1f}%" if frac is not None else "—"
    pass_info = "—"
    if status.get("backward_pass_estimate") is not None:
        pass_info = (
            f"~{status['backward_pass_estimate']} / {status['backward_passes_total']} "
            f"backward passes"
        )
    if status.get("progress"):
        p = status["progress"]
        pass_info = f"{p['backward_pass']} / {p['backward_passes_total']} backward passes"

    proc = status.get("process") or {}
    runtime = _format_duration(proc.get("runtime_sec"))

    pipeline_rows = ""
    for step in status["pipeline"]:
        state = "done" if step.get("done") else ("running" if step.get("active") else "pending")
        cls = {"done": "ok", "running": "warn", "pending": ""}[state]
        pipeline_rows += (
            f"<li class='{cls}'><strong>{html.escape(step['step'])}</strong> — {state}</li>"
        )

    bar_w = int(100 * (frac or 0))

    job_label = "L15 smoke (no job file — started before dashboard)"
    if status.get("job"):
        job_label = str(status["job"].get("job_id", job_label))

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh_sec}">
<title>J-Lens Fit Dashboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #0f1117; color: #e6e8ef; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #9aa3b2; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; margin: 12px 0 24px; }}
  th, td {{ border: 1px solid #2a3142; padding: 8px 12px; text-align: left; }}
  th {{ background: #1a1f2e; width: 220px; }}
  .ok {{ color: #3dd68c; }} .warn {{ color: #f5c542; }} .bad {{ color: #ff6b6b; }}
  .bar {{ background: #2a3142; height: 18px; border-radius: 4px; overflow: hidden; }}
  .bar > i {{ display: block; height: 100%; background: linear-gradient(90deg,#3d7eff,#3dd68c); width: {bar_w}%; }}
  pre {{ background: #1a1f2e; padding: 12px; overflow-x: auto; font-size: 12px; }}
  ul {{ line-height: 1.8; }}
</style>
</head><body>
<h1>J-Lens Fit Dashboard</h1>
<p class="sub">Auto-refresh every {refresh_sec}s · {html.escape(status['generated_at'])}</p>

<h2>Health: <span class="{health_class}">{html.escape(health.upper())}</span></h2>
<div class="bar" title="progress estimate"><i></i></div>
<p>{frac_pct} — {html.escape(pass_info)}</p>

<table>
{row("Job", job_label)}
{row("Layers / prompts", f"{status['layers']} · {status['n_prompts']} prompt(s)")}
{row("dim_batch", str(status['dim_batch']))}
{row("Process PID", str(proc.get('pid', '—')))}
{row("Runtime", runtime)}
{row("RAM", f"{proc.get('ram_gb', '—')} GB" if proc else "—")}
{row("CPU (cores equiv)", str(status.get('cpu_cores_equiv', '—')))}
{row("ETA", f"{status.get('eta_human', '—')} ({status.get('eta_source') or 'n/a'})")}
{row("L15 lens file", "yes" if status['l15_lens_exists'] else "not yet", status['l15_lens_exists'])}
{row("Layer readout", "yes" if status['readout_exists'] else "not yet", status['readout_exists'])}
</table>

<h2>Pipeline</h2>
<ul>{pipeline_rows}</ul>

<h2>Log tail</h2>
<pre>{html.escape(status.get('log_tail') or '(no log)')}</pre>

<p class="sub">CLI: <code>python scripts/fit_dashboard.py --watch</code> ·
<code>python scripts/fit_dashboard.py --serve</code></p>
</body></html>"""


def write_dashboard(status: dict, *, refresh_sec: int = 30) -> tuple[Path, Path]:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DASHBOARD_DIR / "status.json"
    html_path = DASHBOARD_DIR / "index.html"
    json_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    html_path.write_text(render_html(status, refresh_sec=refresh_sec), encoding="utf-8")
    return json_path, html_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit status dashboard")
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds (0=once)")
    parser.add_argument("--serve", type=int, default=0, help="Serve HTTP on port N")
    parser.add_argument("--sample-cpu", action="store_true", help="Live CPU sample (slower)")
    parser.add_argument("--refresh-html", type=int, default=30)
    args = parser.parse_args()

    def refresh_once() -> tuple[Path, Path]:
        status = build_status(sample_cpu=args.sample_cpu)
        paths = write_dashboard(status, refresh_sec=args.refresh_html)
        proc = status.get("process")
        print(
            f"[{status['generated_at']}] health={status['health']} "
            f"eta={status.get('eta_human')} pass={status.get('backward_pass_estimate')} "
            f"pid={proc.get('pid') if proc else None}"
        )
        return paths

    if args.serve:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                status = build_status(sample_cpu=False)
                write_dashboard(status, refresh_sec=args.refresh_html)
                body = (DASHBOARD_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        print(f"Serving dashboard at http://127.0.0.1:{args.serve}/")
        refresh_once()
        HTTPServer(("127.0.0.1", args.serve), Handler).serve_forever()

    if args.watch:
        while True:
            _, html_path = refresh_once()
            print(f"  -> {html_path}")
            time.sleep(args.watch)

    _, html_path = refresh_once()
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
