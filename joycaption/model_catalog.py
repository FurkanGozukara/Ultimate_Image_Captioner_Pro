from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import BASE_DIR


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo_id: str
    subdir: str
    family: str
    architecture: str
    startup_default: bool = False
    supports_thinking: bool = False
    required_files: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return BASE_DIR / self.subdir


MODEL_SPECS: dict[str, ModelSpec] = {
    "joycaption_pre_alpha": ModelSpec(
        key="joycaption_pre_alpha",
        label="JoyCaption Pre-Alpha",
        repo_id="MonsterMMORPG/CapFiles",
        subdir="model_files_pre_alpha",
        family="joycaption",
        architecture="legacy_siglip",
        required_files=("image_adapter.pt", "wpkklhc6_config.yaml"),
    ),
    "joycaption_alpha_one": ModelSpec(
        key="joycaption_alpha_one",
        label="JoyCaption Alpha One",
        repo_id="MonsterMMORPG/CapFiles",
        subdir="model_files_alpha_one",
        family="joycaption",
        architecture="legacy_siglip",
        required_files=(
            "config.yaml",
            "clip_model.pt",
            "image_adapter.pt",
            "text_model/adapter_config.json",
            "text_model/adapter_model.safetensors",
        ),
    ),
    "joycaption_alpha_two": ModelSpec(
        key="joycaption_alpha_two",
        label="JoyCaption Alpha Two",
        repo_id="MonsterMMORPG/CapFiles",
        subdir="model_files_alpha_two",
        family="joycaption",
        architecture="legacy_siglip",
        required_files=(
            "config.yaml",
            "clip_model.pt",
            "image_adapter.pt",
            "text_model/adapter_config.json",
            "text_model/adapter_model.safetensors",
        ),
    ),
    "joycaption_beta_one": ModelSpec(
        key="joycaption_beta_one",
        label="JoyCaption Beta One",
        repo_id="MonsterMMORPG/CapFiles",
        subdir="model_files_beta_one",
        family="joycaption",
        architecture="llava",
        startup_default=True,
        required_files=("config.json", "model.safetensors.index.json", "processor_config.json", "tokenizer.json"),
    ),
    "qwen3_vl_2b_instruct": ModelSpec(
        key="qwen3_vl_2b_instruct",
        label="Qwen3-VL 2B Instruct",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        subdir="model_files_qwen3_vl_2b_instruct",
        family="qwen",
        architecture="Qwen3VLForConditionalGeneration",
    ),
    "qwen3_vl_4b_instruct": ModelSpec(
        key="qwen3_vl_4b_instruct",
        label="Qwen3-VL 4B Instruct",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        subdir="model_files_qwen3_vl_4b_instruct",
        family="qwen",
        architecture="Qwen3VLForConditionalGeneration",
    ),
    "qwen3_vl_8b_instruct": ModelSpec(
        key="qwen3_vl_8b_instruct",
        label="Qwen3-VL 8B Instruct",
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        subdir="model_files_qwen3_vl3_8b_instruct",
        family="qwen",
        architecture="Qwen3VLForConditionalGeneration",
        startup_default=True,
    ),
    "huihui_qwen3_vl_8b_abliterated": ModelSpec(
        key="huihui_qwen3_vl_8b_abliterated",
        label="Huihui Qwen3-VL 8B Instruct Abliterated",
        repo_id="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        subdir="model_files_huihui_qwen3_vl_8b_instruct_abliterated",
        family="qwen",
        architecture="Qwen3VLForConditionalGeneration",
    ),
    "qwen3_vl_30b_a3b_instruct": ModelSpec(
        key="qwen3_vl_30b_a3b_instruct",
        label="Qwen3-VL 30B-A3B Instruct",
        repo_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
        subdir="model_files_qwen3_vl_30b_a3b_instruct",
        family="qwen",
        architecture="Qwen3VLMoeForConditionalGeneration",
    ),
    "qwen3_6_27b": ModelSpec(
        key="qwen3_6_27b",
        label="Qwen3.6 27B",
        repo_id="Qwen/Qwen3.6-27B",
        subdir="model_files_qwen3_6_27b",
        family="qwen",
        architecture="Qwen3_5ForConditionalGeneration",
        supports_thinking=True,
    ),
    "huihui_qwen3_6_27b_abliterated": ModelSpec(
        key="huihui_qwen3_6_27b_abliterated",
        label="Huihui Qwen3.6 27B Abliterated",
        repo_id="huihui-ai/Huihui-Qwen3.6-27B-abliterated",
        subdir="model_files_huihui_qwen3_6_27b_abliterated",
        family="qwen",
        architecture="Qwen3_5ForConditionalGeneration",
        supports_thinking=True,
    ),
}

DEFAULT_QWEN_MODEL_KEY = "qwen3_vl_8b_instruct"
QWEN_MODEL_SPECS = tuple(spec for spec in MODEL_SPECS.values() if spec.family == "qwen")


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[str(model_key)]
    except KeyError as exc:
        raise ValueError(f"Unknown caption model: {model_key}") from exc


def qwen_model_choices() -> list[tuple[str, str]]:
    return [(spec.label, spec.key) for spec in QWEN_MODEL_SPECS]


def selected_qwen_model(settings: dict[str, Any] | None) -> ModelSpec:
    key = str((settings or {}).get("model_key") or DEFAULT_QWEN_MODEL_KEY)
    spec = get_model_spec(key)
    if spec.family != "qwen":
        raise ValueError(f"{spec.label} is not a Qwen vision caption model.")
    return spec


def _indexed_weights_are_complete(model_dir: Path, index_path: Path) -> bool:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    filenames = {str(filename) for filename in weight_map.values() if filename}
    return bool(filenames) and all((model_dir / filename).is_file() for filename in filenames)


def _has_complete_weights(model_dir: Path) -> bool:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        return _indexed_weights_are_complete(model_dir, index_path)
    return (model_dir / "model.safetensors").is_file() or any(model_dir.glob("pytorch_model*.bin"))


def model_readiness_error(model_key: str) -> str | None:
    spec = get_model_spec(model_key)
    model_dir = spec.path
    if not model_dir.is_dir():
        return f"model directory does not exist: {model_dir}"
    for relative_path in spec.required_files:
        if not (model_dir / relative_path).is_file():
            return f"required file is missing: {model_dir / relative_path}"
    if spec.architecture in {"llava", "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration"}:
        if not (model_dir / "config.json").is_file():
            return f"config.json is missing: {model_dir}"
        if not _has_complete_weights(model_dir):
            return f"model weights are incomplete: {model_dir}"
    return None


def model_is_ready(model_key: str) -> bool:
    return model_readiness_error(model_key) is None
