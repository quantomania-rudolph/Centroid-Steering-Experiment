# Chaos checker gate artifacts

Produced by `experiments/prove_chaos_live.py` on Arc XPU (live Aider, no synthetic alarms).

| File | Meaning |
|------|---------|
| `chaos_checker_{A,B_vocab,B_jlens,B_transport,C,D,E,F,G}.json` | Per-checker sign-off (speed, verifiability, usefulness) |
| `chaos_live_proof.json` | Master aggregate + overhead + separation |

Gate tests: `tests/gates/test_gate_chaos_*.py` (require `CHAOS_LIVE=1`).
