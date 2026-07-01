from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import BASE_DIR


SYSTEM_PRESETS_DIR = BASE_DIR / "system_presets"
QWEN_SYSTEM_PRESETS_DIR = SYSTEM_PRESETS_DIR / "qwen"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OFFICIAL_V1_PRESET_ID = "i4_official_v1_app_compare"

VARIABLE_DEFAULTS = {
    "TRIGGER_PHRASE": "",
    "OUTPUT_LANGUAGE": "English",
    "CAPTION_LENGTH": "",
    "DATASET_GOAL": "",
    "KNOWN_SUBJECT_CLASS": "",
    "BRAND_POLICY": "Name only if visually certain; never guess.",
    "TEXT_POLICY": "Copy exact readable text only; mark unclear text unreadable.",
    "BBOX_POLICY": "Major localizable objects and readable text only.",
    "EXISTING_CAPTION": "",
    "EXISTING_JSON_CAPTION": "",
    "VALIDATION_ERROR": "",
    "INVALID_JSON_TEXT": "",
    "IDEOGRAM_JSON_CAPTION": "",
}


@dataclass(frozen=True)
class QwenPreset:
    id: str
    label: str
    output_format: str
    extension: str
    system_prompt: str
    user_prompt_template: str
    max_new_tokens: int
    temperature: float
    image_long_edge: int
    raw_meta: dict[str, Any]
    app_side_only: bool = False


def _fallback_presets() -> dict[str, QwenPreset]:
    prompt = "Describe this image in detail. Output only the caption."
    system = "You are a precise image captioning engine. Output only the requested caption."
    preset = QwenPreset(
        id="txt_flux2_general",
        label="Prompt Text - General",
        output_format="txt",
        extension=".txt",
        system_prompt=system,
        user_prompt_template=prompt,
        max_new_tokens=512,
        temperature=0.1,
        image_long_edge=768,
        raw_meta={"id": "txt_flux2_general", "output_format": "txt"},
    )
    return {preset.id: preset}


def _int_value(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_value(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _extension_for(output_format: str) -> str:
    return ".json" if output_format == "json" else ".txt"


def _preset_label(preset_id: str) -> str:
    return preset_id.replace("_", " ").title()


def _preset_from_json(path: Path) -> tuple[int, QwenPreset]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Qwen system preset file: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Qwen system preset must be a JSON object: {path}")

    preset_id = str(data.get("id") or path.stem).strip()
    if not preset_id:
        raise ValueError(f"Qwen system preset is missing an id: {path}")

    output_format = str(data.get("output_format") or ("json" if preset_id.startswith(("i4_", "qc_")) else "txt"))
    extension = str(data.get("extension") or _extension_for(output_format))
    system_prompt = str(data.get("system_prompt") or "")
    user_prompt_template = str(data.get("user_prompt_template") or data.get("prompt") or "")
    app_side_only = bool(data.get("app_side_only", False))
    if not user_prompt_template and not app_side_only:
        raise ValueError(f"Qwen system preset is missing a prompt: {path}")

    raw_meta = data.get("raw_meta") if isinstance(data.get("raw_meta"), dict) else {}
    order = _int_value(data.get("order"), 100000)
    preset = QwenPreset(
        id=preset_id,
        label=str(data.get("label") or _preset_label(preset_id)),
        output_format=output_format,
        extension=extension,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        max_new_tokens=_int_value(data.get("max_new_tokens"), 4096 if output_format == "json" else 512),
        temperature=_float_value(data.get("temperature"), 0.1),
        image_long_edge=_int_value(data.get("image_long_edge"), 1024 if output_format == "json" else 768),
        raw_meta=raw_meta,
        app_side_only=app_side_only,
    )
    return order, preset


@lru_cache(maxsize=1)
def load_qwen_presets() -> dict[str, QwenPreset]:
    preset_files = sorted(QWEN_SYSTEM_PRESETS_DIR.glob("*.json"))
    if not preset_files:
        return _fallback_presets()

    loaded = sorted((_preset_from_json(path) for path in preset_files), key=lambda item: (item[0], item[1].id))
    return {preset.id: preset for _, preset in loaded}


def qwen_preset_choices() -> list[tuple[str, str]]:
    return [(f"{index}. {preset.label}", preset.id) for index, preset in enumerate(load_qwen_presets().values(), start=1)]


def default_qwen_preset_id() -> str:
    preferred = OFFICIAL_V1_PRESET_ID
    presets = load_qwen_presets()
    return preferred if preferred in presets else next(iter(presets))


def get_qwen_preset(preset_id: str | None) -> QwenPreset:
    presets = load_qwen_presets()
    if preset_id in presets:
        return presets[str(preset_id)]
    return presets[default_qwen_preset_id()]


def render_prompt(template: str, variables: dict[str, Any] | None = None) -> str:
    values = {**VARIABLE_DEFAULTS, **{key: "" if value is None else str(value) for key, value in (variables or {}).items()}}
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    if not values.get("TRIGGER_PHRASE"):
        rendered = re.sub(r"\n\s*\.\s*\n", "\n\n", rendered)
    return rendered.strip()


def preset_payload(preset_id: str | None, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    preset = get_qwen_preset(preset_id)
    return {
        "id": preset.id,
        "label": preset.label,
        "output_format": preset.output_format,
        "extension": preset.extension,
        "system_prompt": render_prompt(preset.system_prompt, variables),
        "prompt": render_prompt(preset.user_prompt_template, variables),
        "max_new_tokens": preset.max_new_tokens,
        "temperature": preset.temperature,
        "image_long_edge": preset.image_long_edge,
        "app_side_only": preset.app_side_only,
    }
