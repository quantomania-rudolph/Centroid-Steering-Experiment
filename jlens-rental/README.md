# H100 Rental Kit — production J-lens in ~1 hour

Same production lens as the local CPU weeks-long run: layers **8–26**, **100 prompts**, decode-ready `.pt` + readouts in your repo.

**Audit:** see [`AUDIT.md`](AUDIT.md) — verdict is yes, same J-space decoding capability when verify passes.

**Auto-destroy:** `run_full_pipeline.py` destroys the Vast instance in `finally` (and via watchdog) so billing stops if you fall asleep.

---

## Your checklist (tonight)

### A. Kill local load (done if fit PID gone)

Fit process was killed. Leftover watchers were killed. If Task Manager still shows high CPU, it is usually **Cursor / Edge / Java**, not the Jacobian fit (Python was ~0 GB after kill).

### B. Fill secrets

Open:

`jlens-rental/secrets/secrets.env`

Fill:

1. `HF_TOKEN` — Hugging Face read token  
2. `VAST_API_KEY` — from [cloud.vast.ai/cli](https://cloud.vast.ai/cli/)  
3. Leave SSH + `VAST_INSTANCE_ID` blank until after launch  

Keep `AUTO_DESTROY=1` and `MAX_BILLING_MINUTES=90`.

### C. Launch H100 on Vast (~5 min)

1. Go to [vast.ai](https://vast.ai) → Create / Search  
2. Filters: **H100 80GB**, reliability ≥ 0.9, disk ≥ 50 GB, RAM ≥ 32 GB, price ≤ ~$2.50/hr  
3. Image: **PyTorch 2.x + CUDA 12.x** (stock template is fine)  
4. Launch → wait until **Running**  
5. Copy into `secrets.env`:
   - `VAST_INSTANCE_ID` = numeric ID on the instance card  
   - `RENTAL_SSH_HOST` / `RENTAL_SSH_PORT` from `ssh -p PORT root@HOST`  

Details: [`vast/LAUNCH_CHECKLIST.md`](vast/LAUNCH_CHECKLIST.md)

### D. One command from your laptop

```powershell
cd "C:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It\jlens-rental"
python scripts/run_full_pipeline.py
```

That will: prepare → upload → fit on H100 → download → production verify → L8–L26 readouts → **destroy instance**.

Emergency destroy if anything hangs:

```powershell
python scripts/destroy_vast_instance.py
```

### E. Confirm results locally

- `artifacts/lenses/qwen_jacobian_lens.pt`
- `artifacts/gates/lens_production_verify.json` → `"overall_pass": true`
- `artifacts/readouts/production/L8_readout.txt` … `L26_readout.txt`

Then you can decode / apply / workspace tools on that lens exactly like a finished local production fit.

---

## Manual step-by-step (if you prefer)

```powershell
python scripts/prepare_bundle.py
python scripts/upload_to_instance.py
python scripts/run_remote_fit.py
python scripts/download_from_instance.py
python scripts/integrate_local.py
python scripts/destroy_vast_instance.py   # do not skip
```

---

## Package layout

```
jlens-rental/
  AUDIT.md
  PLAN.md
  fit_config.yaml          # L8-26, 100 prompts, dim_batch=64
  fit_on_gpu.py
  run_fit.sh
  Dockerfile               # optional
  secrets/secrets.env      # YOU fill this
  scripts/
    run_full_pipeline.py   # preferred entry
    destroy_vast_instance.py
    verify_production_lens.py
    prepare_bundle.py
    upload_to_instance.py
    run_remote_fit.py
    download_from_instance.py
    integrate_local.py
```

## Cost

~$1–2 for H100 ~40–60 min + destroy. With `AUTO_DESTROY=1` you should not wake up to a running bill.
