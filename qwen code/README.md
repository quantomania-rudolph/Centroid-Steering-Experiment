# Qwen Code Experiments

Each subfolder is one Aider task.

## Live A–G Chaos Checker

The OpenAI-compatible server (`experiments/serve_qwen_openai.py`) instruments every
generation with A–G chaos metrics on Arc XPU (default `--chaos`; disable with `--no-chaos`).

Per-token hot path: F0 (vocab A/B/C) + F1 (L16 transport) ≤ ~10ms typical.
F2 every 8 tokens; F3 at boundaries; F4 curves after finalize; F5 G sparse on ≥2 flag categories.

### Prove + sign-off

1. Start server: `../.venv/Scripts/python.exe experiments/serve_qwen_openai.py --device xpu --port 8767 --chaos`
2. Run: `../.venv/Scripts/python.exe experiments/prove_chaos_live.py`
3. Gates: `$env:CHAOS_LIVE=1; pytest tests/gates/test_gate_chaos_*.py -q`

Artifacts land in `artifacts/gates/chaos_checker_*.json` and `chaos_live_proof.json`.
Chaos is also nested under `invocations[].chaos` in finalize JSON and shown on the dashboard radar.

