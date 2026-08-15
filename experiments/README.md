# Experiments

## J-space workspace (primary)

Full generation + J-space readouts across all layers:

```powershell
cd "C:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It"
.\.venv\Scripts\Activate.ps1
python experiments\run_jspace_workspace.py --experiment new_full_3d --prompt-file experiments\prompts\explore_3d.txt --device xpu --max-new-tokens 4096 --max-seq-len 8192
```

Outputs: `experiments/out/new_full_3d_{stamp}.json` with:
- Top-20 J-space representations per layer (mean, prompt-end, gen-start, gen-mid)
- Per-position trajectory (869 tokens)
- Sparse J-space decomposition at L20/L22/L24
- Workspace band detection (onset/motor layers)
- 7 automated evidence checks

## Dashboard

```powershell
python experiments\serve_dashboard.py --rebuild
```

Open `http://127.0.0.1:<port>/index.html` — **do not** use `file://`.

## Verify evidence

```powershell
python experiments\verify_jspace_evidence.py new_full_3d
```

## Legacy next-token experiments

```powershell
python experiments\run_arc_jlens_explore.py --prompt-file experiments\prompts\explore_3d.txt
```
