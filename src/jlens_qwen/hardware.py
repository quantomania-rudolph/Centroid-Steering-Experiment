"""Hardware probing for LG Gram + Intel Arc J-Lens pipeline."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class GpuInfo:
    name: str
    type: str
    driver: str = ""


def _classify_gpu(name: str) -> str:
    lower = name.lower()
    if "arc" in lower:
        return "discrete"
    if "xe" in lower or "iris" in lower or "uhd" in lower:
        return "integrated"
    if "intel" in lower:
        return "integrated"
    return "unknown"


def _ensure_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    for root in (sys.base_prefix, sys.prefix):
        dll_dir = os.path.join(root, "Library", "bin")
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
            path = os.environ.get("PATH", "")
            if dll_dir not in path.split(os.pathsep):
                os.environ["PATH"] = dll_dir + os.pathsep + path


def probe_intel_gpus_windows() -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, DriverVersion | ConvertTo-Json",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        data = json.loads(out)
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            name = str(entry.get("Name", ""))
            if "intel" in name.lower():
                gpus.append(
                    GpuInfo(
                        name=name,
                        type=_classify_gpu(name),
                        driver=str(entry.get("DriverVersion", "")),
                    )
                )
    except (
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ):
        pass
    return gpus


def probe_sycl_devices() -> list[dict[str, str]]:
    _ensure_dll_dirs()
    try:
        import dpctl
    except Exception:
        return []
    devices: list[dict[str, str]] = []
    try:
        for dev in dpctl.get_devices():
            dtype = str(dev.device_type).split(".")[-1]
            devices.append(
                {
                    "name": getattr(dev, "name", str(dev)),
                    "type": dtype,
                    "filter_string": getattr(dev, "filter_string", ""),
                }
            )
    except Exception:
        return []
    return devices


def probe_torch_xpu() -> dict[str, Any]:
    _ensure_dll_dirs()
    result: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "xpu_available": False,
        "device_name": "",
        "matmul_ok": False,
        "error": None,
    }
    try:
        import torch

        result["torch_installed"] = True
        result["torch_version"] = torch.__version__
        ok = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        result["xpu_available"] = ok
        if ok:
            result["device_name"] = torch.xpu.get_device_name(0)
            x = torch.randn(1024, 1024, device="xpu", dtype=torch.float16)
            y = x @ x.T
            torch.xpu.synchronize()
            result["matmul_ok"] = bool(torch.isfinite(y).all().item())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        return 0.0


def probe_hardware() -> dict[str, Any]:
    """Full hardware report for GATE-0."""
    intel_gpus = probe_intel_gpus_windows() if platform.system() == "Windows" else []
    torch_xpu = probe_torch_xpu()
    sycl = probe_sycl_devices()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": {
            "model": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count() or 0,
            "platform": platform.platform(),
        },
        "total_ram_gb": round(probe_ram_gb(), 2),
        "intel_gpus": [asdict(g) for g in intel_gpus],
        "sycl_devices": sycl,
        "torch_xpu": torch_xpu,
        "python": sys.version,
    }


def save_hardware_report(path: str | Path) -> dict[str, Any]:
    report = probe_hardware()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_gate0(report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate GATE-0 thresholds. Returns {id, name, pass, detail} list + overall."""
    report = report or probe_hardware()
    tx = report.get("torch_xpu", {})
    intel_gpus = report.get("intel_gpus", [])
    sycl = report.get("sycl_devices", [])
    has_arc = any("arc" in g.get("name", "").lower() for g in intel_gpus)
    has_intel = bool(intel_gpus)
    sycl_gpu = [d for d in sycl if d.get("type", "").lower() == "gpu"]

    checks = [
        {
            "id": "G0.1",
            "name": "XPU available",
            "pass": bool(tx.get("xpu_available")),
            "detail": f"xpu_available={tx.get('xpu_available')} device={tx.get('device_name')!r}",
        },
        {
            "id": "G0.2",
            "name": "XPU matmul smoke",
            "pass": bool(tx.get("matmul_ok")),
            "detail": f"matmul_ok={tx.get('matmul_ok')} error={tx.get('error')}",
        },
        {
            "id": "G0.3",
            "name": "Intel Arc / Intel GPU identity",
            "pass": has_arc or has_intel,
            "detail": f"gpus={intel_gpus}",
        },
        {
            "id": "G0.4",
            "name": "CPU cores >= 8 (measured LG Gram)",
            "pass": int(report.get("cpu", {}).get("logical_cores", 0)) >= 8,
            "detail": f"logical_cores={report.get('cpu', {}).get('logical_cores')}",
        },
        {
            "id": "G0.5",
            "name": "System RAM >= 30 GB (32GB class, OS reserve OK)",
            "pass": float(report.get("total_ram_gb", 0)) >= 30.0,
            "detail": f"total_ram_gb={report.get('total_ram_gb')}",
        },
        {
            "id": "G0.6",
            "name": "SYCL GPU runtime",
            "pass": len(sycl_gpu) >= 1 or bool(tx.get("xpu_available")),
            "detail": f"sycl_gpus={len(sycl_gpu)} xpu={tx.get('xpu_available')}",
        },
    ]
    overall = all(c["pass"] for c in checks)
    return {"gate": "GATE-0", "overall_pass": overall, "checks": checks, "report": report}
