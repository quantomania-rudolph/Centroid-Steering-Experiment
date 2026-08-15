# CAA Error-Combination Research

Empirical study linking A–G chaos monitor packages to agentic code outcomes and CAA steering vector design.

## Run the study

```bash
# Server must be running: experiments/serve_qwen_openai.py
python research/caa_error_combo_study.py
```

## Outputs

| File | Description |
|------|-------------|
| `results/consolidated.json` | All 5 rounds + CAA candidates |
| `results/r1_clean.json` … `r5_traj.json` | Per-round analysis |
| `CAA_STEERING_REPORT.md` | Human-readable findings |

## Starters

Broken task templates in `starters/` are copied into `qwen code/<task_slug>/` before each round.
