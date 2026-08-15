# H100 Rental Operation Plan

Goal: fit the production Jacobian lens (layers **8–26**, **100 prompts**) on a rented **H100 80GB**, then bring the `.pt` file home for local readouts and decoding — same artifacts as the CPU run, ~1000× faster.

## What we are fitting

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Source layers | 8–26 (19 layers) |
| Target layer | 27 |
| Prompts | `data/prompts/corpus_100.json` (100 prompts) |
| `max_seq_len` | 128 |
| `skip_first` | 16 |
| `dim_batch` | 64 (H100 80GB) |
| Output | `qwen_jacobian_lens.pt` (~650 MB fp16) |

This matches `scripts/run_production_fit.py` except device=GPU and larger `dim_batch`.

## Phase 0 — You (one-time setup, ~10 min)

1. Create accounts:
   - [Vast.ai](https://vast.ai) — GPU marketplace (cheapest H100)
   - [Hugging Face](https://huggingface.co) — model download token
2. Copy `secrets/secrets.env.example` → `secrets/secrets.env`
3. Fill in `HF_TOKEN` in `secrets/secrets.env`
4. Add payment method on Vast.ai

## Phase 1 — Agent / scripts prepare bundle (automated)

```powershell
cd jlens-rental
python scripts/prepare_bundle.py
```

Creates `bundle/rental_bundle.zip` containing:

- `corpus_100.json`
- `fit_config.yaml`
- `fit_on_gpu.py`
- `run_fit.sh`

## Phase 2 — You launch GPU instance (~5 min)

On [vast.ai/console/create](https://vast.ai/console/create):

| Filter | Value |
|--------|-------|
| GPU | H100 80GB (SXM or PCIe) |
| Reliability | > 0.90 |
| Disk | ≥ 50 GB |
| RAM | ≥ 32 GB |
| Template | PyTorch 2.x + CUDA 12.x |

After **Running**, copy from the instance page into `secrets/secrets.env`:

- `RENTAL_SSH_HOST`
- `RENTAL_SSH_PORT`
- `RENTAL_SSH_USER` (usually `root`)

## Phase 3 — Upload + run fit (~40 min GPU time)

```powershell
python scripts/upload_to_instance.py
python scripts/run_remote_fit.py
```

Or manually: Jupyter terminal on the instance → paste commands from `README.md`.

The fit script:

1. `pip install` jlens + transformers
2. `huggingface-cli login` (uses your token)
3. Loads model in bf16 on CUDA
4. Fits with checkpoints every 10 prompts
5. Saves `qwen_jacobian_lens.pt` in our `FittedJacobianLens` format

## Phase 4 — Download results (~2 min)

```powershell
python scripts/download_from_instance.py
```

Downloads to `jlens-rental/output/qwen_jacobian_lens.pt`.

## Phase 5 — Integrate locally (~30 min readouts)

```powershell
python scripts/integrate_local.py
```

This:

1. Copies lens → `artifacts/lenses/qwen_jacobian_lens.pt`
2. Runs `verify_lens_smoke.py` (format check)
3. Runs `run_all_layer_readouts.py --layers 8-26`
4. Writes manifest to `artifacts/readouts/production_manifest.json`

## Phase 6 — You stop the instance

**Destroy/stop the Vast instance immediately** after download. Billing is per-second.

## Cost estimate

| Item | Estimate |
|------|----------|
| H100 80GB @ ~$1.50/hr | ~$1.00 (40 min fit + setup) |
| Disk | ~$0.05 |
| **Total** | **~$1–2** |

## What stays the same vs local CPU run

| Artifact | Same? |
|----------|-------|
| `qwen_jacobian_lens.pt` format | Yes (`J`, `n_prompts`, `d_model`) |
| Layer readouts L8–L26 | Yes |
| Apply / steering / decode | Yes |
| Gate verification | Yes |

## Fallback

If GPU OOM: lower `DIM_BATCH=32` in `secrets/secrets.env` and re-run. Checkpoints resume from `ckpt.pt`.

## What the agent cannot do for you

- Vast.ai billing / account login
- Paste HF token (you keep it in `secrets/secrets.env` only)
- Click "Launch" in Vast console
- Stop instance (you must click Destroy)
