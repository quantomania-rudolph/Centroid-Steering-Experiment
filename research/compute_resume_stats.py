"""One-off stats for resume doc (no torch import)."""
import ast
import json
import subprocess
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/Scripts/python.exe"
OUT = ROOT / "experiments/unicaa01_corpus/out"
QWEN = ROOT / "qwen code"
ALPHA = 0.85

ORACLES: dict[str, str] = {
    "p07_csv_agg": "60.0",
    "p08_rate_limiter": "['ok', 'no', 'ok']",
    "p09_thread_pool": "[1, 2, 3]",
    "p10_config_validate": "False",
    "p11_palindrome": "True",
    "p12_fibonacci": "55",
    "p14_regex_email": "True",
    "p15_http_retry": "3",
    "p17_deque_queue": "[1, 2, 3]",
    "p27_base64_pad": "'aGk='",
    "p28_glob_match": "True",
    "p29_cron_field": "True",
    "p30_ini_parse": "'localhost'",
    "p32_xml_escape": "'a&amp;b'",
    "p33_html_strip": "'hi'",
    "p36_hmac_sign": "64",
    "p37_hash_chain": "3",
    "p45_producer_consumer": "[1, 2]",
}


def inject_count(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    invs = data.get("invocations") or []
    if not invs:
        return 0
    best = max(invs, key=lambda i: len((i.get("chaos") or {}).get("tokens") or []))
    toks = (best.get("chaos") or {}).get("tokens") or []
    return sum(1 for t in toks if (t.get("cast") or {}).get("inject_applied"))


def has_live(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("cast_enabled"):
        return False
    if float(data.get("cast_alpha_0") or -1) != ALPHA:
        return False
    return inject_count(path) >= 1


def run_demo(slug: str, pair_id: str) -> bool | None:
    if pair_id not in ORACLES:
        return None
    expected = ast.literal_eval(ORACLES[pair_id])
    code = "import json\nfrom main import demo\nv=demo()\nprint(json.dumps(v))\n"
    proc = subprocess.run(
        [str(PY), "-c", code],
        cwd=QWEN / slug,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        return False
    val = json.loads(proc.stdout.strip().splitlines()[-1])
    return val == expected


def main() -> None:
    live: list[str] = []
    for p in sorted(OUT.glob("unicaa01_corpus_*_panic_latest.json")):
        pid = p.name.replace("unicaa01_corpus_", "").replace("_panic_latest.json", "")
        if has_live(p):
            live.append(pid)

    pan_ok = neu_ok = 0
    pan_tok: list[int] = []
    neu_tok: list[int] = []
    for pid in live:
        pan_p = OUT / f"unicaa01_corpus_{pid}_panic_latest.json"
        neu_p = OUT / f"unicaa01_corpus_{pid}_neutral_latest.json"
        pan_d = json.loads(pan_p.read_text(encoding="utf-8"))
        pan_tok.append((pan_d.get("invocations") or [{}])[0].get("completion_tokens") or 0)
        slug_p = pan_d.get("task_slug") or ""
        r = run_demo(slug_p, pid)
        if r is True:
            pan_ok += 1
        if neu_p.is_file():
            neu_d = json.loads(neu_p.read_text(encoding="utf-8"))
            neu_tok.append((neu_d.get("invocations") or [{}])[0].get("completion_tokens") or 0)
            slug_n = neu_d.get("task_slug") or ""
            if run_demo(slug_n, pid) is True:
                neu_ok += 1

    n = len(live)
    print("live_cast_pairs", n)
    print("demo_panic", pan_ok, n)
    print("demo_neutral", neu_ok, n)
    if n:
        print("demo_panic_pct", round(100 * pan_ok / n, 1))
        print("demo_neutral_pct", round(100 * neu_ok / n, 1))
        print("demo_lift_pp", round(100 * (pan_ok - neu_ok) / n, 1))
    ap = an = None
    if pan_tok and neu_tok:
        ap, an = statistics.mean(pan_tok), statistics.mean(neu_tok)
        print("avg_tokens_panic", round(ap))
        print("avg_tokens_neutral", round(an))
        print("token_reduction_pct", round(100 * (1 - ap / an), 1))

    scan_path = ROOT / "artifacts/caa/UNI_CAA01_corpus/scan_report.json"
    scan = {}
    if scan_path.is_file():
        sr = json.loads(scan_path.read_text(encoding="utf-8"))
        scan = {
            "n_files": sr.get("n_files"),
            "n_panic_ok": sr.get("n_panic_ok"),
            "n_neutral_ok": sr.get("n_neutral_ok"),
        }
    meta_path = ROOT / "artifacts/caa/UNI_CAA01/metadata.json"
    alpha_0 = None
    n_pairs = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        alpha_0 = meta.get("alpha_0")
        n_pairs = meta.get("n_pairs")

    gates = {
        "cast_stream_live": _gate_pass(ROOT / "artifacts/gates/cast_stream_live_gate.json"),
        "cast_inject_live": _gate_pass(ROOT / "artifacts/gates/cast_inject_live_gate.json"),
        "caa_alpha0_4c": _gate_pass(ROOT / "artifacts/gates/caa_alpha0_4c_gate.json"),
        "cast_oneoff_panic": _gate_pass(ROOT / "artifacts/gates/cast_oneoff_panic_live_proof.json"),
        "cast_oneoff_b02": _gate_pass(ROOT / "artifacts/gates/cast_oneoff_b02_live_proof.json"),
    }

    snapshot = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_cast_pairs": n,
        "demo_panic_pass": pan_ok,
        "demo_neutral_pass": neu_ok,
        "demo_panic_pct": round(100 * pan_ok / n, 1) if n else None,
        "demo_neutral_pct": round(100 * neu_ok / n, 1) if n else None,
        "demo_lift_pp": round(100 * (pan_ok - neu_ok) / n, 1) if n else None,
        "demo_relative_success_x": round(pan_ok / neu_ok, 2) if neu_ok else None,
        "avg_tokens_panic_inv1": round(ap) if ap is not None else None,
        "avg_tokens_neutral_inv1": round(an) if an is not None else None,
        "token_reduction_pct": round(100 * (1 - ap / an), 1) if ap and an else None,
        "cast_alpha_0": alpha_0,
        "caa_natural_pairs": n_pairs,
        "corpus_scan": scan,
        "gates_pass": gates,
        "repro_command": "python research/compute_resume_stats.py",
    }
    out = ROOT / "docs/resume_metrics_snapshot.json"
    out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


def _gate_pass(path: Path) -> bool | None:
    if not path.is_file():
        return None
    return bool(json.loads(path.read_text(encoding="utf-8")).get("pass"))


if __name__ == "__main__":
    main()
