"""Load Qwen2.5-Coder-7B-Instruct for J-Lens analysis."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "Qwen2.5-Coder-7B-Instruct"
)
EXPECTED = {
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}


def _ensure_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    for root in (sys.base_prefix, sys.prefix):
        dll_dir = os.path.join(root, "Library", "bin")
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    model_id: str
    config: dict[str, Any]

    @property
    def n_layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    @property
    def d_model(self) -> int:
        return int(self.config["hidden_size"])

    @property
    def vocab_size(self) -> int:
        return int(self.config["vocab_size"])

    @property
    def W_U(self) -> torch.Tensor:
        return self.model.lm_head.weight

    @property
    def gamma(self) -> torch.Tensor:
        return self.model.model.norm.weight

    @property
    def rms_eps(self) -> float:
        return float(self.model.model.norm.variance_epsilon)

    @property
    def final_norm(self) -> nn.Module:
        return self.model.model.norm

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def encode(
        self,
        text: str,
        *,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return encoded.input_ids.to(self.device)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm then lm_head. Matches jlens HFLensModel.unembed."""
        target_device = self.W_U.device
        target_dtype = self.W_U.dtype
        return self.model.lm_head(
            self.final_norm(residual.to(target_dtype).to(target_device))
        )


def _load_weight_from_shards(bundle: ModelBundle, weight_key: str) -> torch.Tensor:
    """Load a single weight tensor from local safetensors shards (CPU float32)."""
    model_dir = Path(resolve_model_source(bundle.model_id, prefer_local=True))
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"{weight_key} on meta device and no shard index at {index_path}")

    from safetensors.torch import load_file

    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map") or {}
    shard_name = weight_map.get(weight_key)
    if not shard_name:
        raise RuntimeError(f"{weight_key} missing from model.safetensors.index.json")

    shard_path = model_dir / shard_name
    if not shard_path.is_file():
        raise RuntimeError(f"{weight_key} shard missing: {shard_path}")

    return load_file(str(shard_path), device="cpu")[weight_key].float()


def materialize_weight_cpu(bundle: ModelBundle, weight: torch.Tensor, *, weight_key: str) -> torch.Tensor:
    """Return weight on CPU float32; load from disk if device_map left param on meta."""
    if not getattr(weight, "is_meta", False):
        return weight.detach().to("cpu").float()
    return _load_weight_from_shards(bundle, weight_key)


def materialize_w_u_cpu(bundle: ModelBundle) -> torch.Tensor:
    """Return unembedding on CPU float32; load from disk if device_map left lm_head meta."""
    return materialize_weight_cpu(bundle, bundle.W_U, weight_key="lm_head.weight")


def materialize_gamma_cpu(bundle: ModelBundle) -> torch.Tensor:
    """Return final RMSNorm gamma on CPU float32; load from disk if offloaded to meta."""
    return materialize_weight_cpu(bundle, bundle.gamma, weight_key="model.norm.weight")


def resolve_device(prefer: str = "auto") -> torch.device:
    _ensure_dll_dirs()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("torch.xpu requested but not available")
        return torch.device("xpu")
    # auto
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def resolve_model_source(
    model_id: str = MODEL_ID,
    *,
    prefer_local: bool = True,
) -> str:
    """Prefer local snapshot under artifacts/models when present."""
    if prefer_local and LOCAL_MODEL_DIR.is_dir() and (
        (LOCAL_MODEL_DIR / "config.json").exists()
    ):
        return str(LOCAL_MODEL_DIR)
    return model_id


def load_qwen_tokenizer(
    *,
    model_id: str = MODEL_ID,
    prefer_local: bool = True,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load tokenizer only (no model weights) for offline decode / divergence tracing."""
    from transformers import AutoTokenizer

    source = resolve_model_source(model_id, prefer_local=prefer_local)
    use_local_only = local_files_only or Path(source).is_dir()
    return AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        local_files_only=use_local_only,
        cache_dir=cache_dir,
    )


def _inplace_to_device(module: nn.Module, device: torch.device) -> None:
    """Move params/buffers onto device, releasing CPU storage as we go."""
    for child in module.children():
        _inplace_to_device(child, device)
    for _, param in list(module.named_parameters(recurse=False)):
        if param.device != device:
            param.data = param.data.to(device, non_blocking=False)
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf is None or not hasattr(buf, "device"):
            continue
        if buf.device != device:
            module._buffers[name] = buf.to(device, non_blocking=False)


def _move_model_to_xpu(model: nn.Module, torch_device: torch.device) -> nn.Module:
    """Layer-wise stream to XPU. Raises RuntimeError on OOM (no full .to() fallback)."""
    import gc

    gc.collect()
    try:
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    except Exception:
        pass

    core = getattr(model, "model", None)
    if core is None or not hasattr(core, "layers"):
        raise RuntimeError("unexpected model layout for XPU streaming move")

    if hasattr(core, "embed_tokens"):
        _inplace_to_device(core.embed_tokens, torch_device)
        gc.collect()
        torch.xpu.empty_cache()
    for i, layer in enumerate(core.layers):
        _inplace_to_device(layer, torch_device)
        if i % 4 == 3:
            gc.collect()
            try:
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            except Exception:
                pass
    if hasattr(core, "norm"):
        _inplace_to_device(core.norm, torch_device)
    for name, child in core.named_children():
        if name in {"embed_tokens", "layers", "norm"}:
            continue
        _inplace_to_device(child, torch_device)
    if hasattr(model, "lm_head"):
        _inplace_to_device(model.lm_head, torch_device)
    _inplace_to_device(model, torch_device)
    return model


def load_qwen_coder(
    *,
    model_id: str = MODEL_ID,
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    prefer_local: bool = True,
) -> ModelBundle:
    """Download/load Qwen2.5-Coder-7B-Instruct in float16.

    On Intel Arc 140V (shared system memory), prefer accelerate ``device_map='auto'``
    with an XPU budget so weights fill XPU then CPU-offload the overflow. A full CPU
    materialization followed by ``model.to(xpu)`` routinely OOMs when free RAM is
    tight, because CPU + XPU copies compete for the same memory pool.
    """
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_dll_dirs()
    torch_device = resolve_device(device)
    if str(torch_device).startswith("xpu") and hasattr(torch, "xpu"):
        os.environ.setdefault("PYTORCH_XPU_ALLOC_CONF", "max_split_size_mb:256")
        try:
            gc.collect()
            torch.xpu.empty_cache()
        except Exception:
            pass
    source = resolve_model_source(model_id, prefer_local=prefer_local)
    use_local_only = local_files_only or Path(source).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        local_files_only=use_local_only,
        cache_dir=cache_dir,
    )

    model: nn.Module
    if str(torch_device).startswith("xpu") and hasattr(torch, "xpu"):
        xpu_budget = os.environ.get("JLENS_XPU_MAX_MEMORY", "12GiB")
        cpu_budget = os.environ.get("JLENS_CPU_MAX_MEMORY", "14GiB")
        try:
            # Primary: fill XPU then CPU-offload remainder (proven on Arc 140V).
            model = AutoModelForCausalLM.from_pretrained(
                source,
                dtype=dtype,
                trust_remote_code=True,
                local_files_only=use_local_only,
                cache_dir=cache_dir,
                device_map="auto",
                max_memory={0: xpu_budget, "cpu": cpu_budget},
            )
        except Exception as exc:
            try:
                gc.collect()
                torch.xpu.empty_cache()
            except Exception:
                pass
            model = AutoModelForCausalLM.from_pretrained(
                source,
                dtype=dtype,
                trust_remote_code=True,
                local_files_only=use_local_only,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
            )
            try:
                model = _move_model_to_xpu(model, torch_device)
            except RuntimeError as move_exc:
                raise RuntimeError(
                    "XPU load failed via device_map and streaming move. "
                    "Free system RAM (close heavy apps / hung python probes), "
                    "set JLENS_XPU_MAX_MEMORY (default 12GiB), then retry. "
                    f"device_map_error={exc!r}; stream_error={move_exc!r}"
                ) from move_exc
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=use_local_only,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        model = model.to(torch_device)

    model.eval()
    for param in model.parameters():
        if param.device.type != "meta":
            param.requires_grad_(False)
    if str(torch_device).startswith("xpu") and hasattr(torch, "xpu"):
        try:
            gc.collect()
            torch.xpu.empty_cache()
        except Exception:
            pass

    cfg = model.config
    config = {
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(cfg.vocab_size),
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-6)),
        "architectures": list(getattr(cfg, "architectures", []) or []),
        "model_type": getattr(cfg, "model_type", ""),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "device": str(torch_device),
    }
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        dtype=dtype,
        model_id=model_id,
        config=config,
    )


def save_model_manifest(bundle: ModelBundle, path: str | Path) -> dict[str, Any]:
    manifest = {
        "model_id": bundle.model_id,
        "config": bundle.config,
        "W_U_shape": list(bundle.W_U.shape),
        "gamma_shape": list(bundle.gamma.shape),
        "rms_eps": bundle.rms_eps,
        "expected": EXPECTED,
        "matches_expected": {
            k: bundle.config.get(k) == v for k, v in EXPECTED.items()
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class ActivationRecorder:
    """Record post-block residual activations (pre final norm)."""

    def __init__(self, layers: nn.ModuleList, at: list[int] | None = None):
        self.layers = layers
        self.at = at if at is not None else list(range(len(layers)))
        self.activations: dict[int, torch.Tensor] = {}
        self._hooks: list[Any] = []

    def __enter__(self) -> ActivationRecorder:
        def make_hook(idx: int):
            def hook(_module, _inp, out):
                # Decoder layer returns tensor (or tuple in some variants)
                h = out[0] if isinstance(out, tuple) else out
                self.activations[idx] = h

            return hook

        for i in self.at:
            self._hooks.append(self.layers[i].register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
