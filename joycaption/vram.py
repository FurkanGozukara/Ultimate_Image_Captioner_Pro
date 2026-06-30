from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .attention import DEFAULT_JOY_ATTENTION, DEFAULT_QWEN_ATTENTION


VRAM_PRESET_VALUES = [6, 8, 10, 12, 16, 24, 32]
VRAM_PRESET_CHOICES = [f"{value} GB" for value in VRAM_PRESET_VALUES]
VRAM_TOLERANCE_GB = 0.5
LOW_VRAM_QWEN_ATTENTION = "sdpa_cudnn"


@dataclass(frozen=True)
class GpuInfo:
    device_id: int
    name: str
    total_gb: float
    selected_preset: str


def _preset_label(value: int) -> str:
    return f"{value} GB"


def preset_value(label: str | int | None) -> int:
    if isinstance(label, int):
        return min(VRAM_PRESET_VALUES, key=lambda value: abs(value - label))
    text = str(label or "").strip().lower().replace("gb", "").replace("gib", "")
    try:
        value = int(float(text))
    except ValueError:
        return 6
    return min(VRAM_PRESET_VALUES, key=lambda candidate: abs(candidate - value))


def preset_label_for_vram(total_gb: float | int | None) -> str:
    if total_gb is None:
        return _preset_label(VRAM_PRESET_VALUES[0])
    available = float(total_gb)
    selected = VRAM_PRESET_VALUES[0]
    for preset in VRAM_PRESET_VALUES:
        if available + VRAM_TOLERANCE_GB >= preset:
            selected = preset
    return _preset_label(selected)


def detect_gpus() -> list[GpuInfo]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        gpus: list[GpuInfo] = []
        for device_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device_id)
            total_gb = props.total_memory / (1024**3)
            gpus.append(
                GpuInfo(
                    device_id=device_id,
                    name=str(props.name),
                    total_gb=total_gb,
                    selected_preset=preset_label_for_vram(total_gb),
                )
            )
        return gpus
    except Exception:
        return []


def default_vram_preset(device_id: int = 0) -> str:
    gpus = detect_gpus()
    if not gpus:
        return _preset_label(VRAM_PRESET_VALUES[0])
    for gpu in gpus:
        if gpu.device_id == device_id:
            return gpu.selected_preset
    return gpus[0].selected_preset


def gpu_summary_html() -> str:
    gpus = detect_gpus()
    if not gpus:
        return '<div class="jc-info">CUDA GPU was not detected. Defaulting VRAM preset to 6 GB.</div>'
    lines = [
        f"GPU {gpu.device_id}: {gpu.name} - {gpu.total_gb:.2f} GiB -> {gpu.selected_preset}"
        for gpu in gpus
    ]
    return '<div class="jc-info"><pre>' + "\n".join(lines) + "</pre></div>"


def legacy_vram_settings(label: str | int | None) -> dict[str, Any]:
    value = preset_value(label)
    if value <= 8:
        return {"use_4bit": True, "use_fp16": True, "max_resolution": 768, "batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}
    if value <= 12:
        return {"use_4bit": True, "use_fp16": False, "max_resolution": 1024, "batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}
    if value <= 16:
        return {"use_4bit": False, "use_fp16": True, "max_resolution": 1280, "batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}
    if value <= 24:
        return {"use_4bit": False, "use_fp16": True, "max_resolution": 1536, "batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}
    return {"use_4bit": False, "use_fp16": False, "max_resolution": 1536, "batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}


def beta_vram_settings(label: str | int | None) -> dict[str, Any]:
    value = preset_value(label)
    if value <= 10:
        return {"model_quantization": "nf4", "downscale_max_res": "768", "zip_batch_size": 1, "folder_batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}
    if value <= 16:
        return {"model_quantization": "int8", "downscale_max_res": "1024", "zip_batch_size": 1, "folder_batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}
    if value <= 24:
        return {"model_quantization": "bf16", "downscale_max_res": "1536", "zip_batch_size": 1, "folder_batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}
    return {"model_quantization": "bf16", "downscale_max_res": "2048", "zip_batch_size": 1, "folder_batch_size": 1, "attention_backend": DEFAULT_JOY_ATTENTION}


def qwen_vram_settings(label: str | int | None) -> dict[str, Any]:
    value = preset_value(label)
    if value <= 8:
        return {
            "model_quantization": "nf4",
            "image_long_edge": 512,
            "attention_backend": LOW_VRAM_QWEN_ATTENTION,
            "file_batch_size": 1,
            "folder_batch_size": 1,
            "max_new_tokens": 2048,
        }
    if value <= 12:
        return {
            "model_quantization": "nf4",
            "image_long_edge": 768,
            "attention_backend": LOW_VRAM_QWEN_ATTENTION,
            "file_batch_size": 1,
            "folder_batch_size": 1,
            "max_new_tokens": 3072,
        }
    if value <= 16:
        return {
            "model_quantization": "int8",
            "image_long_edge": 768,
            "attention_backend": DEFAULT_QWEN_ATTENTION,
            "file_batch_size": 1,
            "folder_batch_size": 1,
            "max_new_tokens": 4096,
        }
    if value <= 24:
        return {
            "model_quantization": "bf16",
            "image_long_edge": 1024,
            "attention_backend": DEFAULT_QWEN_ATTENTION,
            "file_batch_size": 1,
            "folder_batch_size": 1,
            "max_new_tokens": 4096,
        }
    return {
        "model_quantization": "bf16",
        "image_long_edge": 1280,
        "attention_backend": DEFAULT_QWEN_ATTENTION,
        "file_batch_size": 1,
        "folder_batch_size": 1,
        "max_new_tokens": 6144,
    }
