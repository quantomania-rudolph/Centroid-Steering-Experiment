"""J-Lens pipeline for Qwen2.5-Coder-7B-Instruct."""

from jlens_qwen.hardware import probe_hardware, save_hardware_report
from jlens_qwen.model_loader import ModelBundle, load_qwen_coder
from jlens_qwen.logit_lens import LogitLens, rmsnorm
from jlens_qwen.jacobian_lens import apply_jacobian_lens, transport

__all__ = [
    "probe_hardware",
    "save_hardware_report",
    "ModelBundle",
    "load_qwen_coder",
    "LogitLens",
    "rmsnorm",
    "apply_jacobian_lens",
    "transport",
]
