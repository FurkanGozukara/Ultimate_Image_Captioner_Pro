from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import BASE_DIR


README_PATH = BASE_DIR.parent / "qwen3_vl_8b_caption_presets_README.md"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

SPECIAL_TEXT_PROMPTS = {
    "txt_legacy_comma_tags",
    "txt_accessibility_alt_text",
    "txt_ocr_transcription",
}

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
    raw_meta: dict[str, str]
    app_side_only: bool = False


def _read_readme() -> str:
    if README_PATH.exists():
        text = README_PATH.read_text(encoding="utf-8")
        return text.replace("[y_min, x_min, y_max, x_max]", "[x_min, y_min, x_max, y_max]")
    return ""


def _first_text_fence(block: str) -> str:
    match = re.search(r"```text\n(.*?)\n```", block, re.S)
    return match.group(1).strip() if match else ""


def _all_yaml_fences(block: str) -> list[str]:
    return [item.strip() for item in re.findall(r"```yaml\n(.*?)\n```", block, re.S)]


def _section_text(text: str, start_heading: str, end_heading: str) -> str:
    start = text.find(start_heading)
    if start < 0:
        return ""
    end = text.find(end_heading, start + len(start_heading))
    return text[start : end if end >= 0 else len(text)]


def _parse_yaml_scalars(yaml_text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in yaml_text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("-") or raw_line.startswith(" "):
            continue
        key, sep, value = raw_line.partition(":")
        if sep:
            values[key.strip()] = value.strip()
    return values


def _numbers(value: str) -> list[float]:
    return [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value))]


def _int_default(value: str, fallback: int, maximum: int | None = None) -> int:
    numbers = _numbers(value)
    if not numbers:
        return fallback
    selected = int(max(numbers))
    if maximum is not None:
        selected = min(selected, maximum)
    return selected


def _float_default(value: str, fallback: float) -> float:
    numbers = _numbers(value)
    if not numbers:
        return fallback
    return float(min(numbers))


def _title_from_id(preset_id: str) -> str:
    overrides = {
        "i4_json_auto_best": "Ideogram JSON - Auto Best",
        "i4_json_art_style_detailed": "Ideogram JSON - Art / Style Detailed",
        "i4_json_text_poster_logo": "Ideogram JSON - Text / Poster / Logo",
        "txt_flux2_general": "FLUX.2 Text - General",
        "txt_flux2_style_lora_content_only": "FLUX.2 LoRA - Style Content Only",
        "txt_flux2_subject_person_lora": "FLUX.2 LoRA - Subject Person",
        "txt_flux2_character_lora": "FLUX.2 LoRA - Character",
        "txt_flux2_product_object_lora": "FLUX.2 LoRA - Product / Object",
    }
    if preset_id in overrides:
        return overrides[preset_id]
    title = preset_id
    for prefix, label in (
        ("i4_json_", "Ideogram JSON - "),
        ("txt_flux2_", "FLUX.2 Text - "),
        ("txt_", "Text - "),
        ("qc_", "QC - "),
    ):
        if title.startswith(prefix):
            title = label + title[len(prefix) :]
            break
    return title.replace("_", " ").replace(" lora", " LoRA").replace(" ocr", " OCR").title().replace("Json", "JSON")


def _system_prompt_for(preset_id: str, output_format: str, systems: dict[str, str]) -> str:
    if preset_id.startswith("qc_"):
        return systems.get("validation", "")
    if output_format == "json":
        return systems.get("json", "")
    return systems.get("visual", "")


def _extract_system_prompts(text: str) -> dict[str, str]:
    systems = {
        "visual": "You are a precise image captioning engine for machine learning datasets. Follow the requested output format exactly.",
        "json": "You are a precise image-to-JSON captioning engine. Return exactly one valid JSON object.",
        "validation": "You are a caption quality-control engine. Return only the requested output.",
    }
    visual_section = _section_text(text, "### 3.1 System prompt: strict visual captioner", "### 3.2 System prompt: strict JSON captioner")
    json_section = _section_text(text, "### 3.2 System prompt: strict JSON captioner", "### 3.3 System prompt: validation / repair")
    validation_section = _section_text(text, "### 3.3 System prompt: validation / repair", "---")
    systems["visual"] = _first_text_fence(visual_section) or systems["visual"]
    systems["json"] = _first_text_fence(json_section) or systems["json"]
    systems["validation"] = _first_text_fence(validation_section) or systems["validation"]
    return systems


def _fallback_presets() -> dict[str, QwenPreset]:
    prompt = "Describe this image in detail. Output only the caption."
    system = "You are a precise image captioning engine. Output only the requested caption."
    preset = QwenPreset(
        id="txt_flux2_general",
        label="FLUX.2 Text - General",
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


def _compose_prompt(preset_id: str, output_format: str, block_prompt: str, bases: dict[str, str]) -> str:
    if preset_id.startswith("i4_json_"):
        return "\n\n".join(part for part in [bases.get("i4", ""), block_prompt] if part).strip()
    if preset_id in SPECIAL_TEXT_PROMPTS:
        return block_prompt.strip()
    if preset_id.startswith("txt_"):
        return "\n\n".join(part for part in [bases.get("txt", ""), block_prompt] if part).strip()
    return block_prompt.strip()


def _preset_from_block(
    preset_id: str,
    block: str,
    systems: dict[str, str],
    bases: dict[str, str],
) -> QwenPreset | None:
    yaml_blocks = _all_yaml_fences(block)
    meta = _parse_yaml_scalars(yaml_blocks[0]) if yaml_blocks else {"id": preset_id}
    meta.setdefault("id", preset_id)
    output_format = meta.get("output_format") or ("json" if preset_id.startswith(("i4_", "qc_")) else "txt")
    app_side_only = meta.get("kind") == "file_system_check"
    extension = meta.get("extension") or (".json" if output_format == "json" else ".txt")
    prompt = _first_text_fence(block)
    if app_side_only and not prompt:
        checks = yaml_blocks[1] if len(yaml_blocks) > 1 else ""
        prompt = "App-side dataset sidecar audit checks:\n" + checks
    if not prompt and not app_side_only:
        return None
    user_prompt = _compose_prompt(preset_id, output_format, prompt, bases)
    max_new_tokens = _int_default(meta.get("max_new_tokens", ""), 4096 if output_format == "json" else 512, maximum=8192)
    temperature = _float_default(meta.get("temperature", ""), 0.1)
    image_long_edge = _int_default(meta.get("image_long_edge", ""), 1024 if output_format == "json" else 768, maximum=1536)
    return QwenPreset(
        id=preset_id,
        label=_title_from_id(preset_id),
        output_format=output_format,
        extension=extension,
        system_prompt=_system_prompt_for(preset_id, output_format, systems),
        user_prompt_template=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        image_long_edge=image_long_edge,
        raw_meta=meta,
        app_side_only=app_side_only,
    )


@lru_cache(maxsize=1)
def load_qwen_presets() -> dict[str, QwenPreset]:
    text = _read_readme()
    if not text:
        return _fallback_presets()

    systems = _extract_system_prompts(text)
    a2 = _section_text(text, "## A2. Ideogram JSON base user prompt", "---\n\n## A3.")
    b2 = _section_text(text, "## B2. Regular caption base prompt", "---\n\n## B3.")
    bases = {"i4": _first_text_fence(a2), "txt": _first_text_fence(b2)}
    presets: dict[str, QwenPreset] = {}

    preset_matches = list(re.finditer(r"^### Preset: `([^`]+)`", text, re.M))
    for index, match in enumerate(preset_matches):
        end = preset_matches[index + 1].start() if index + 1 < len(preset_matches) else text.find("\n# Part B", match.end())
        if end < 0:
            end = len(text)
        block = text[match.start() : end]
        preset = _preset_from_block(match.group(1), block, systems, bases)
        if preset:
            presets[preset.id] = preset

    qc_matches = list(re.finditer(r"^## C\d+\. Preset: `([^`]+)`", text, re.M))
    for index, match in enumerate(qc_matches):
        end = qc_matches[index + 1].start() if index + 1 < len(qc_matches) else text.find("\n# Part D", match.end())
        if end < 0:
            end = len(text)
        block = text[match.start() : end]
        preset = _preset_from_block(match.group(1), block, systems, bases)
        if preset:
            presets[preset.id] = preset

    return presets or _fallback_presets()


def qwen_preset_choices() -> list[tuple[str, str]]:
    return [(preset.label, preset.id) for preset in load_qwen_presets().values()]


def default_qwen_preset_id() -> str:
    preferred = "i4_json_training_balanced"
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
