# J-Lens on Qwen2.5-Coder-7B-Instruct

Hybrid pipeline: **fp16 inference on Intel Arc (XPU)** + **CPU Jacobian fitting**.

## Setup

```powershell
cd "c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
pip install -e .
pip install "git+https://github.com/anthropics/jacobian-lens.git"
```

## Gates

```powershell
python -m tests.gates.run_all --through 0   # hardware
python scripts\probe_hardware.py
python scripts\download_model.py
python scripts\smoke_infer_xpu.py
python -m tests.gates.run_all --through 6
```

## Smoke → production workflow

**Step 1 — smoke (running now):** 1 layer, 1 prompt
```powershell
python scripts\fit_jacobian.py --n-prompts 1 --source-layers 15-15 --device cpu --dim-batch 8 --max-seq-len 64 --out artifacts\lenses\qwen_jacobian_lens_L15.pt
```

**Step 2 — verify smoke passed:**
```powershell
python scripts\verify_lens_smoke.py
```

**Step 3 — production (only if verify PASS):** 27 layers (0–26), 100 prompts
```powershell
python scripts\run_production_fit.py
```

Or wait for smoke file then auto-verify + launch:
```powershell
python scripts\wait_and_verify_smoke.py --launch-production
```

Production fit is **~27×100× slower** than smoke; expect **many hours** on CPU.

## Slice viewer

```powershell
python scripts\slice_view.py
```

Open `artifacts\slices\slice.html`.
