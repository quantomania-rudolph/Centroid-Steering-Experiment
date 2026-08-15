# Vast.ai H100 launch checklist

## Filters

| Setting | Value |
|---------|-------|
| GPU | H100 80GB |
| System RAM | ≥ 32 GB |
| Disk | ≥ 50 GB |
| Reliability | ≥ 0.90 |
| Max $/hr | ~2.50 (often ~1.50) |
| Image | PyTorch 2.x + CUDA 12.x |

## After Running — paste into `secrets/secrets.env`

From the instance page:

```
VAST_INSTANCE_ID=<number on the card>
RENTAL_SSH_HOST=<ip>
RENTAL_SSH_PORT=<port from ssh -p>
RENTAL_SSH_USER=root
```

Also ensure:

```
HF_TOKEN=...
VAST_API_KEY=...
AUTO_DESTROY=1
MAX_BILLING_MINUTES=90
```

## Run

```powershell
cd jlens-rental
python scripts/run_full_pipeline.py
```

## If SSH fails

Use Jupyter on the instance: upload `bundle/rental_bundle.zip`, unzip, `bash run_fit.sh`, download `qwen_jacobian_lens.pt` to `jlens-rental/output/`, then locally:

```powershell
python scripts/integrate_local.py
python scripts/destroy_vast_instance.py
```

## Billing rule

Destroy = stop compute billing. **Destroy** (not just Stop) is what `destroy_vast_instance.py` does.
