from __future__ import annotations

from contextlib import contextmanager, nullcontext
from importlib.util import find_spec
from typing import Any, Iterator


ATTENTION_BACKEND_CHOICES = [
    ("Auto (Transformers default)", "auto"),
    ("Eager (manual attention)", "eager"),
    ("SDPA Auto (PyTorch)", "sdpa"),
    ("SDPA Flash kernel", "sdpa_flash"),
    ("SDPA Efficient / xFormers-style", "sdpa_efficient"),
    ("SDPA cuDNN", "sdpa_cudnn"),
    ("SDPA Math", "sdpa_math"),
    ("FlashAttention 2 package", "flash_attention_2"),
]

DEFAULT_QWEN_ATTENTION = "sdpa"
DEFAULT_JOY_ATTENTION = "sdpa"

_ALIASES = {
    "": "auto",
    "default": "auto",
    "transformers_default": "auto",
    "torch": "sdpa",
    "sdpa_auto": "sdpa",
    "flash": "sdpa_flash",
    "flash_kernel": "sdpa_flash",
    "triton": "sdpa_flash",
    "triton_flash": "sdpa_flash",
    "memory_efficient": "sdpa_efficient",
    "mem_efficient": "sdpa_efficient",
    "efficient": "sdpa_efficient",
    "xformers": "sdpa_efficient",
    "xformers_memory_efficient": "sdpa_efficient",
    "xformers_triton": "sdpa_efficient",
    "triton_splitk": "sdpa_efficient",
    "math": "sdpa_math",
    "cudnn": "sdpa_cudnn",
    "fa2": "flash_attention_2",
    "flash_attn": "flash_attention_2",
    "flash_attn_2": "flash_attention_2",
}

_LOAD_IMPL = {
    "eager": "eager",
    "sdpa": "sdpa",
    "sdpa_flash": "sdpa",
    "sdpa_efficient": "sdpa",
    "sdpa_cudnn": "sdpa",
    "sdpa_math": "sdpa",
    "flash_attention_2": "flash_attention_2",
}

_SDPA_KERNELS = {
    "sdpa_flash": "FLASH_ATTENTION",
    "sdpa_efficient": "EFFICIENT_ATTENTION",
    "sdpa_cudnn": "CUDNN_ATTENTION",
    "sdpa_math": "MATH",
}

_LABELS = {value: label for label, value in ATTENTION_BACKEND_CHOICES}


def normalize_attention_backend(settings: dict[str, Any] | str | None) -> str:
    if isinstance(settings, str):
        raw = settings
        legacy_sdpa = False
    else:
        raw = "" if settings is None else str(settings.get("attention_backend") or "")
        legacy_sdpa = bool(settings and settings.get("use_sdpa_attention", False))
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if not key and legacy_sdpa:
        return "sdpa"
    key = _ALIASES.get(key, key)
    valid = {value for _label, value in ATTENTION_BACKEND_CHOICES}
    if key not in valid:
        raise ValueError(f"Unknown attention backend: {raw}")
    return key


def attention_backend_label(settings: dict[str, Any] | str | None) -> str:
    backend = normalize_attention_backend(settings)
    return _LABELS.get(backend, backend)


def _validate_flash_attention_2(quant: str | None) -> None:
    quant = str(quant or "").lower()
    if quant and quant not in {"bf16", "fp16"}:
        raise RuntimeError("FlashAttention 2 can only be selected with bf16 or fp16 model loading.")
    if find_spec("flash_attn") is None:
        raise RuntimeError("FlashAttention 2 is not installed in this venv. Use an SDPA backend instead.")


def attention_load_kwargs(settings: dict[str, Any], quant: str | None = None) -> dict[str, Any]:
    backend = normalize_attention_backend(settings)
    if backend == "auto":
        return {}
    if backend == "flash_attention_2":
        _validate_flash_attention_2(quant)
    return {"attn_implementation": _LOAD_IMPL[backend]}


@contextmanager
def attention_runtime_context(settings: dict[str, Any]) -> Iterator[None]:
    backend = normalize_attention_backend(settings)
    kernel_name = _SDPA_KERNELS.get(backend)
    if not kernel_name:
        with nullcontext():
            yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError(f"PyTorch SDPA backend forcing is unavailable: {exc}") from exc
    kernel = getattr(SDPBackend, kernel_name, None)
    if kernel is None:
        raise RuntimeError(f"PyTorch does not expose SDPBackend.{kernel_name}.")
    with sdpa_kernel(kernel):
        yield


def attention_status_text(settings: dict[str, Any]) -> str:
    backend = normalize_attention_backend(settings)
    label = _LABELS.get(backend, backend)
    if backend == "sdpa_efficient":
        return f"Attention: {label} (requested through PyTorch SDPA; direct xFormers replacement is not used for HF VLM layers)"
    if backend in _SDPA_KERNELS:
        return f"Attention: {label} (forced with torch.nn.attention.sdpa_kernel)"
    return f"Attention: {label}"
