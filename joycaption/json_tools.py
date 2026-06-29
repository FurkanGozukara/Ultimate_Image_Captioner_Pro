from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


ELEMENT_HEADERS = ["type", "y_min", "x_min", "y_max", "x_max", "caption", "box_title", "text"]
XYXY_ELEMENT_HEADERS = ["type", "x_min", "y_min", "x_max", "y_max", "caption", "box_title", "text"]
EMPTY_ELEMENT_ROW = ["obj", 80, 80, 360, 360, "", "", ""]

BOX_COLORS = [
    "#7C3AED",
    "#EF4444",
    "#14B8A6",
    "#F59E0B",
    "#22C55E",
    "#06B6D4",
    "#EC4899",
    "#A3E635",
]

VERTICAL_BBOX_TERMS = (
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "adult",
    "model",
    "figure",
    "athlete",
    "player",
    "baseball player",
    "tennis player",
    "pitcher",
    "batter",
    "catcher",
    "umpire",
    "line judge",
    "skier",
    "snowboarder",
    "surfer",
    "skateboarder",
    "paddleboarder",
    "standing",
    "full body",
    "full-body",
    "floor lamp",
    "lamp post",
    "potted plant",
    "plant",
    "tree",
    "bottle",
    "vase",
    "statue",
    "tower",
)
HORIZONTAL_BBOX_TERMS = (
    "car",
    "vehicle",
    "truck",
    "bus",
    "train",
    "boat",
    "ship",
    "swan",
    "goose",
    "duck",
    "airplane",
    "plane",
    "motorcycle",
    "bicycle",
    "sofa",
    "couch",
    "bed",
    "table",
    "desk",
    "bench",
    "banner",
    "sign",
    "poster",
    "screen",
    "monitor",
    "keyboard",
    "sedan",
    "coupe",
    "suv",
    "pickup",
    "mercedes",
    "mercedes-benz",
    "mercedes-amg",
    "bmw",
    "audi",
    "porsche",
    "ferrari",
    "lamborghini",
    "tesla",
)


def strip_markdown_fences(text: str) -> str:
    value = str(text or "").strip()
    fence = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", value, re.S)
    if fence:
        return fence.group(1).strip()
    value = re.sub(r"^```(?:json|JSON)?\s*", "", value.strip())
    value = re.sub(r"\s*```$", "", value.strip())
    return value.strip()


def extract_json_candidate(text: str) -> str:
    value = strip_markdown_fences(text)
    if value.startswith("{") and value.endswith("}"):
        return value
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        return value[start : end + 1].strip()
    return value


def parse_json_caption(text: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    candidate = extract_json_candidate(text)
    warnings: list[str] = []
    try:
        parsed = json.loads(candidate)
    except Exception as exc:
        return None, candidate, [f"JSON parse failed: {type(exc).__name__}: {exc}"]
    if not isinstance(parsed, dict):
        return None, candidate, ["JSON output must be a top-level object."]
    return parsed, json.dumps(parsed, ensure_ascii=False, indent=2), warnings


def validate_ideogram_json(data: dict[str, Any]) -> list[str]:
    return validate_official_v1_json(data)


def validate_official_v1_json(data: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    expected_keys = ["aspect_ratio", "high_level_description", "compositional_deconstruction"]
    actual_keys = list(data.keys())
    for key in expected_keys:
        if key not in data:
            warnings.append(f'Missing top-level key "{key}".')
    if "style_description" in data:
        warnings.append('Official v1 preset should not include "style_description".')
    extras = [key for key in actual_keys if key not in expected_keys]
    if extras:
        warnings.append("Official v1 preset has extra top-level key(s): " + ", ".join(f'"{key}"' for key in extras) + ".")
    if actual_keys[: len(expected_keys)] != expected_keys:
        warnings.append('Official v1 key order should be "aspect_ratio", "high_level_description", "compositional_deconstruction".')
    aspect_ratio = data.get("aspect_ratio")
    if not isinstance(aspect_ratio, str) or not re.fullmatch(r"\d+:\d+", aspect_ratio.strip()):
        warnings.append('"aspect_ratio" must be a concrete W:H string such as "1:1" or "16:9".')
    comp = data.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        warnings.append('"compositional_deconstruction" must be an object.')
        return warnings
    if list(comp.keys())[:2] != ["background", "elements"]:
        warnings.append('"compositional_deconstruction" key order should be "background" then "elements".')
    if "background" not in comp:
        warnings.append('Missing "compositional_deconstruction.background".')
    elements = comp.get("elements")
    if not isinstance(elements, list):
        warnings.append('"compositional_deconstruction.elements" must be an array.')
        return warnings
    for index, element in enumerate(elements, start=1):
        if not isinstance(element, dict):
            warnings.append(f"Element {index} is not an object.")
            continue
        element_type = element.get("type")
        if element_type not in {"obj", "text"}:
            warnings.append(f'Element {index} type should be "obj" or "text".')
        if element_type == "text" and "text" not in element:
            warnings.append(f'Element {index} text element is missing "text".')
        allowed_keys = {"type", "bbox", "desc", "text"} if element_type == "text" else {"type", "bbox", "desc"}
        extra_keys = [key for key in element.keys() if key not in allowed_keys]
        if extra_keys:
            warnings.append(f"Element {index} has non-official key(s): " + ", ".join(f'\"{key}\"' for key in extra_keys) + ".")
        bbox = element.get("bbox")
        if bbox is None:
            continue
        valid, message = validate_bbox(bbox)
        if not valid:
            warnings.append(f"Element {index} bbox: {message}")
    return warnings


def normalize_json_output(text: str, preset_id: str = "", compact: bool = False) -> tuple[str, dict[str, Any] | None, list[str]]:
    parsed, pretty, warnings = parse_json_caption(text)
    if parsed is None:
        return pretty, None, warnings
    if preset_id.startswith(("i4_official_v1", "i4_json_")):
        parsed = coerce_official_v1_payload(parsed, rows=None)
        parsed, repair_warnings = repair_official_v1_bbox_order(parsed)
        warnings.extend(repair_warnings)
        warnings.extend(validate_official_v1_json(parsed))
    normalized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) if compact else json.dumps(parsed, ensure_ascii=False, indent=2)
    return normalized, parsed, warnings


def validate_bbox(value: Any) -> tuple[bool, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False, "bbox must contain four numbers."
    try:
        min_1, min_2, max_1, max_2 = [int(round(float(item))) for item in value]
    except Exception:
        return False, "bbox values must be numeric."
    if not (0 <= min_1 < max_1 <= 1000 and 0 <= min_2 < max_2 <= 1000):
        return False, "expected normalized min/max pairs in the 0-1000 range."
    return True, ""


def clamp_bbox(min_1: Any, min_2: Any, max_1: Any, max_2: Any) -> list[int]:
    vals = []
    for value in (min_1, min_2, max_1, max_2):
        try:
            vals.append(max(0, min(1000, int(round(float(value))))))
        except Exception:
            vals.append(0)
    if vals[2] <= vals[0]:
        vals[2] = min(1000, vals[0] + 1)
    if vals[3] <= vals[1]:
        vals[3] = min(1000, vals[1] + 1)
    return vals


def _coerce_bbox_numbers(bbox: Any) -> list[int]:
    try:
        raw_values = list(bbox)
    except Exception:
        raw_values = []
    values = raw_values[:4] + [0] * max(0, 4 - len(raw_values))
    coerced: list[int] = []
    for value in values[:4]:
        try:
            coerced.append(max(0, min(1000, int(round(float(value))))))
        except Exception:
            coerced.append(0)
    return coerced


def _axis_pair(first: int, second: int) -> tuple[int, int]:
    lower, upper = sorted((first, second))
    if upper <= lower:
        if lower >= 1000:
            return 999, 1000
        return lower, lower + 1
    return lower, upper


def clean_bbox_order(value: Any) -> str:
    return "xyxy" if str(value or "").lower() == "xyxy" else "yxyx"


def headers_for_bbox_order(value: Any) -> list[str]:
    return XYXY_ELEMENT_HEADERS if clean_bbox_order(value) == "xyxy" else ELEMENT_HEADERS


def bbox_to_yxyx(bbox: Any, bbox_order: str = "yxyx") -> list[int]:
    first, second, third, fourth = _coerce_bbox_numbers(bbox)
    if clean_bbox_order(bbox_order) == "xyxy":
        x_min, y_min, x_max, y_max = first, second, third, fourth
        y_min, y_max = _axis_pair(y_min, y_max)
        x_min, x_max = _axis_pair(x_min, x_max)
        return [y_min, x_min, y_max, x_max]
    y_min, x_min, y_max, x_max = first, second, third, fourth
    y_min, y_max = _axis_pair(y_min, y_max)
    x_min, x_max = _axis_pair(x_min, x_max)
    return [y_min, x_min, y_max, x_max]


def row_bbox_to_yxyx(bbox: Any, bbox_order: str = "yxyx") -> list[int]:
    return bbox_to_yxyx(bbox, bbox_order=bbox_order)


def yxyx_bbox_to_row(bbox: Any, bbox_order: str = "yxyx") -> list[int]:
    y_min, x_min, y_max, x_max = bbox_to_yxyx(bbox, bbox_order="yxyx")
    if clean_bbox_order(bbox_order) == "xyxy":
        return [x_min, y_min, x_max, y_max]
    return [y_min, x_min, y_max, x_max]


def _word_hit(text: str, terms: tuple[str, ...]) -> bool:
    value = f" {str(text or '').lower()} "
    for term in terms:
        term = term.lower()
        if " " in term or "-" in term:
            if term in value:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", value):
            return True
    return False


def _first_word_hit_index(text: str, terms: tuple[str, ...]) -> int | None:
    value = str(text or "").lower()
    hits: list[int] = []
    for term in terms:
        term = term.lower()
        if " " in term or "-" in term:
            index = value.find(term)
        else:
            match = re.search(rf"\b{re.escape(term)}\b", value)
            index = match.start() if match else -1
        if index >= 0:
            hits.append(index)
    return min(hits) if hits else None


def _identity_text(text: str) -> str:
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    lead = re.split(r"[,.;]\s+", value, maxsplit=1)[0]
    words = lead.split()
    return " ".join(words[:18])


def _expectation_from_text(text: str) -> str | None:
    vertical_at = _first_word_hit_index(text, VERTICAL_BBOX_TERMS)
    horizontal_at = _first_word_hit_index(text, HORIZONTAL_BBOX_TERMS)
    if vertical_at is None and horizontal_at is None:
        return None
    if vertical_at is None:
        return "horizontal"
    if horizontal_at is None:
        return "vertical"
    return "vertical" if vertical_at <= horizontal_at else "horizontal"


def _element_aspect_expectation(element: dict[str, Any]) -> tuple[str | None, float]:
    text = " ".join(str(element.get(key) or "") for key in ("desc", "text", "type"))
    if str(element.get("type") or "").strip() == "text":
        return "horizontal", 1.25
    identity = _identity_text(text)
    expectation = _expectation_from_text(identity)
    if expectation is not None:
        return expectation, 2.0
    expectation = _expectation_from_text(text)
    if expectation is not None:
        return expectation, 1.0
    return None, 0.0


def _bbox_dimensions_for_order(bbox: Any, bbox_order: str) -> tuple[int, int]:
    y_min, x_min, y_max, x_max = bbox_to_yxyx(bbox, bbox_order=bbox_order)
    return max(1, x_max - x_min), max(1, y_max - y_min)


def _aspect_fit_score(width: int, height: int, expectation: str | None) -> float:
    ratio = max(0.001, width / max(height, 1))
    score = 0.0
    if ratio < 0.08 or ratio > 12:
        score -= 1.0
    if expectation == "vertical":
        if ratio <= 0.80:
            score += 2.0
        elif ratio <= 1.10:
            score += 0.5
        elif ratio >= 1.75:
            score -= 3.0
        elif ratio >= 1.25:
            score -= 2.0
        else:
            score -= 0.5
    elif expectation == "horizontal":
        if ratio >= 1.25:
            score += 2.0
        elif ratio >= 0.90:
            score += 0.5
        elif ratio <= 0.55:
            score -= 3.0
        elif ratio <= 0.80:
            score -= 2.0
        else:
            score -= 0.5
    return score


def infer_official_v1_bbox_source_order(data: dict[str, Any] | None) -> tuple[str, float, int]:
    if not isinstance(data, dict):
        return "yxyx", 0.0, 0
    comp = data.get("compositional_deconstruction")
    elements = comp.get("elements") if isinstance(comp, dict) else None
    if not isinstance(elements, list):
        return "yxyx", 0.0, 0

    yxyx_score = 0.0
    xyxy_score = 0.0
    evidence = 0
    for element in elements:
        if not isinstance(element, dict):
            continue
        bbox = element.get("bbox")
        valid, _message = validate_bbox(bbox)
        if not valid:
            continue
        expectation, weight = _element_aspect_expectation(element)
        if expectation is None:
            continue
        y_width, y_height = _bbox_dimensions_for_order(bbox, "yxyx")
        x_width, x_height = _bbox_dimensions_for_order(bbox, "xyxy")
        y_score = _aspect_fit_score(y_width, y_height, expectation) * weight
        x_score = _aspect_fit_score(x_width, x_height, expectation) * weight
        yxyx_score += y_score
        xyxy_score += x_score
        if abs(x_score - y_score) >= 1.0:
            evidence += 1

    margin = xyxy_score - yxyx_score
    if evidence >= 2 and margin >= 4.0:
        return "xyxy", margin, evidence
    if evidence == 1 and margin >= 8.0:
        return "xyxy", margin, evidence
    return "yxyx", -margin, evidence


def repair_official_v1_bbox_order(data: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(data, dict):
        return data, []
    source_order, confidence, evidence = infer_official_v1_bbox_source_order(data)
    if source_order != "xyxy":
        return data, []

    repaired = deepcopy(data)
    comp = repaired.get("compositional_deconstruction")
    elements = comp.get("elements") if isinstance(comp, dict) else None
    if not isinstance(elements, list):
        return repaired, []
    repaired_count = 0
    for element in elements:
        if not isinstance(element, dict):
            continue
        bbox = element.get("bbox")
        valid, _message = validate_bbox(bbox)
        if not valid:
            continue
        element["bbox"] = bbox_to_yxyx(bbox, bbox_order="xyxy")
        repaired_count += 1
    if repaired_count == 0:
        return repaired, []
    return repaired, [
        "Detected likely x_min,y_min,x_max,y_max bbox output and converted "
        f"{repaired_count} box(es) to official y_min,x_min,y_max,x_max order "
        f"(confidence margin {confidence:.1f}, evidence {evidence})."
    ]


def _palette_to_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def _palette_from_text(value: Any) -> list[str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return [part.strip() for part in re.split(r"[,;\n]", text) if part.strip()]


def json_to_element_rows(data: dict[str, Any] | None, bbox_order: str = "yxyx") -> list[list[Any]]:
    if not isinstance(data, dict):
        return []
    comp = data.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        return []
    elements = comp.get("elements")
    if not isinstance(elements, list):
        return []
    rows: list[list[Any]] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        bbox = element.get("bbox") if isinstance(element.get("bbox"), list) else ["", "", "", ""]
        bbox = list(bbox)[:4] + [""] * max(0, 4 - len(list(bbox)))
        row_bbox = yxyx_bbox_to_row(bbox, bbox_order) if all(str(item).strip() for item in bbox) else bbox
        element_type = "text" if element.get("type") == "text" else "obj"
        caption = str(element.get("desc", "") or "")
        official_text = str(element.get("text", "") or "") if element_type == "text" else ""
        rows.append(
            [
                element_type,
                row_bbox[0],
                row_bbox[1],
                row_bbox[2],
                row_bbox[3],
                caption,
                str(element.get("box_title", "") or ""),
                official_text,
            ]
        )
    return rows


def rows_to_elements(rows: Any, bbox_order: str = "yxyx") -> list[dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    elements: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or not any(str(cell or "").strip() for cell in row):
            continue
        values = list(row) + [""] * max(0, len(ELEMENT_HEADERS) - len(row))
        element_type = str(values[0] or "obj").strip() or "obj"
        min_1, min_2, max_1, max_2 = values[1], values[2], values[3], values[4]
        caption_value = str(values[5] or "").strip()
        official_text_value = str(values[7] or "").strip()
        element_is_text = element_type == "text" or bool(official_text_value)
        element: dict[str, Any] = {"type": "text" if element_is_text else "obj"}
        if all(str(v).strip() for v in (min_1, min_2, max_1, max_2)):
            element["bbox"] = row_bbox_to_yxyx([min_1, min_2, max_1, max_2], bbox_order=bbox_order)
        if element["type"] == "text":
            text_value = official_text_value or caption_value
            if text_value:
                element["text"] = text_value
            if caption_value:
                element["desc"] = caption_value
            elif text_value:
                element["desc"] = text_value
        elif caption_value:
            element["desc"] = caption_value
        if element.get("desc") or element.get("text") or element.get("bbox"):
            elements.append(element)
    return elements


def _coerce_official_element(element: Any) -> dict[str, Any] | None:
    if not isinstance(element, dict):
        return None
    element_type = "text" if element.get("type") == "text" or str(element.get("text") or "").strip() else "obj"
    payload: dict[str, Any] = {"type": element_type}
    bbox = element.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        valid, _message = validate_bbox(bbox)
        if valid:
            payload["bbox"] = clamp_bbox(*bbox)
    if element_type == "text":
        text_value = str(element.get("text") or element.get("desc") or "").strip()
        desc_value = str(element.get("desc") or text_value).strip()
        if text_value:
            payload["text"] = text_value
        if desc_value:
            payload["desc"] = desc_value
    else:
        desc_value = str(element.get("desc") or element.get("caption") or "").strip()
        if desc_value:
            payload["desc"] = desc_value
    if payload.get("bbox") or payload.get("desc") or payload.get("text"):
        return payload
    return None


def _source_elements(data: dict[str, Any]) -> list[dict[str, Any]]:
    comp = data.get("compositional_deconstruction") if isinstance(data.get("compositional_deconstruction"), dict) else {}
    elements = comp.get("elements")
    if not isinstance(elements, list):
        return []
    normalized: list[dict[str, Any]] = []
    for element in elements:
        coerced = _coerce_official_element(element)
        if coerced is not None:
            normalized.append(coerced)
    return normalized


def coerce_official_v1_payload(data: dict[str, Any] | None, rows: Any = None, bbox_order: str = "yxyx", aspect_ratio: str | None = None) -> dict[str, Any]:
    source = data if isinstance(data, dict) else {}
    comp = source.get("compositional_deconstruction") if isinstance(source.get("compositional_deconstruction"), dict) else {}
    ratio = str(aspect_ratio or source.get("aspect_ratio") or "1:1").strip()
    if not re.fullmatch(r"\d+:\d+", ratio):
        ratio = "1:1"
    if rows is not None:
        elements = rows_to_elements(rows, bbox_order=bbox_order)
    else:
        elements = _source_elements(source)
    return {
        "aspect_ratio": ratio,
        "high_level_description": str(source.get("high_level_description", "") or "").strip(),
        "compositional_deconstruction": {
            "background": str(comp.get("background", "") or "").strip(),
            "elements": elements,
        },
    }


def apply_rows_to_json(json_text: str, rows: Any, bbox_order: str = "yxyx") -> tuple[str, dict[str, Any] | None, list[str]]:
    data, _pretty, warnings = parse_json_caption(json_text)
    data = coerce_official_v1_payload(data, rows=rows, bbox_order=bbox_order)
    normalized = json.dumps(data, ensure_ascii=False, indent=2)
    return normalized, data, warnings


def _file_url(path: str | Path) -> str:
    normalized = str(Path(path)).replace("\\", "/")
    return "/gradio_api/file=" + quote(normalized, safe="/:")


def _label_for_element(index: int, element: dict[str, Any]) -> str:
    label = str(element.get("text") or element.get("desc") or element.get("type") or "element")
    label = " ".join(label.split())
    if len(label) > 72:
        label = label[:69] + "..."
    return f"{index:02d} {element.get('type', 'obj')} - {label}"


def _normalize_row_values(rows: Any) -> list[list[Any]]:
    if rows is None:
        return []
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    normalized: list[list[Any]] = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or not any(str(cell or "").strip() for cell in row):
            continue
        values = list(row) + [""] * max(0, len(ELEMENT_HEADERS) - len(row))
        values = values[: len(ELEMENT_HEADERS)]
        if all(str(v).strip() for v in values[1:5]):
            values[1:5] = clamp_bbox(values[1], values[2], values[3], values[4])
        normalized.append(values)
    return normalized


def overlay_html(
    image_path: str | Path | None,
    rows: Any,
    aspect_ratio: str = "1:1",
    max_height: int = 768,
    empty_message: str = "No boxes to preview.",
    interactive: bool = False,
    bbox_order: str = "yxyx",
    visible_indices: Any = None,
    disable_auto_update: bool = False,
) -> str:
    bbox_order = clean_bbox_order(bbox_order)
    visible_set = None
    if visible_indices is not None:
        try:
            visible_set = {int(item) for item in visible_indices}
        except Exception:
            visible_set = set()
    row_values = _normalize_row_values(rows)
    boxes = []
    for row_index, values in enumerate(row_values):
        if visible_set is not None and row_index not in visible_set:
            continue
        if not all(str(v).strip() for v in values[1:5]):
            continue
        y_min, x_min, y_max, x_max = row_bbox_to_yxyx(values[1:5], bbox_order=bbox_order)
        valid, _message = validate_bbox([y_min, x_min, y_max, x_max])
        if not valid:
            continue
        color = BOX_COLORS[(row_index) % len(BOX_COLORS)]
        row_type = "text" if str(values[0] or "").strip() == "text" or str(values[7] or "").strip() else "obj"
        label_text = str(values[6] or values[5] or values[7] or row_type or "element")
        label_text = " ".join(label_text.split())
        if len(label_text) > 72:
            label_text = label_text[:69] + "..."
        label = html.escape(f"{row_index + 1:02d} {row_type} - {label_text}")
        row_json = html.escape(json.dumps(values, ensure_ascii=False), quote=True)
        handles = ""
        if interactive:
            handles = (
                '<i class="jc-box-handle jc-box-handle-nw" data-handle="nw"></i>'
                '<i class="jc-box-handle jc-box-handle-n" data-handle="n"></i>'
                '<i class="jc-box-handle jc-box-handle-ne" data-handle="ne"></i>'
                '<i class="jc-box-handle jc-box-handle-e" data-handle="e"></i>'
                '<i class="jc-box-handle jc-box-handle-se" data-handle="se"></i>'
                '<i class="jc-box-handle jc-box-handle-s" data-handle="s"></i>'
                '<i class="jc-box-handle jc-box-handle-sw" data-handle="sw"></i>'
                '<i class="jc-box-handle jc-box-handle-w" data-handle="w"></i>'
            )
        boxes.append(
            f'<div class="jc-box" data-row-index="{row_index}" data-row="{row_json}" '
            f'style="top:{y_min / 10:.3f}%;left:{x_min / 10:.3f}%;'
            f'width:{(x_max - x_min) / 10:.3f}%;height:{(y_max - y_min) / 10:.3f}%;'
            f'border-color:{color};background:{color}18;">'
            f'<span style="background:{color};">{label}</span>{handles}</div>'
        )

    if not boxes:
        box_markup = f'<div class="jc-overlay-empty">{html.escape(empty_message)}</div>'
    else:
        box_markup = "\n".join(boxes)

    shell_class = "jc-overlay-shell jc-overlay-interactive" if interactive else "jc-overlay-shell"
    rows_json = html.escape(json.dumps(row_values, ensure_ascii=False), quote=True)
    disable_attr = "1" if disable_auto_update else "0"
    if image_path:
        src = html.escape(_file_url(image_path), quote=True)
        return (
            f'<div class="{shell_class}">'
            f'<div class="jc-overlay-frame" data-rows="{rows_json}" data-bbox-order="{bbox_order}" '
            f'data-disable-auto-update="{disable_attr}" style="max-height:{int(max_height)}px;">'
            f'<img class="jc-overlay-image" src="{src}" alt="overlay source" />'
            f"{box_markup}</div></div>"
        )

    ratio = aspect_ratio if re.fullmatch(r"\d+(\.\d+)?:\d+(\.\d+)?", str(aspect_ratio or "")) else "1:1"
    width, height = ratio.split(":", 1)
    return (
        f'<div class="{shell_class}">'
        f'<div class="jc-overlay-frame jc-overlay-blank" data-rows="{rows_json}" data-bbox-order="{bbox_order}" '
        f'data-disable-auto-update="{disable_attr}" style="aspect-ratio:{float(width)}/{float(height)};max-height:{int(max_height)}px;">'
        f"{box_markup}</div></div>"
    )


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        return (124, 58, 237)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except Exception:
        return (124, 58, 237)


def _label_font(image_width: int, image_height: int) -> ImageFont.ImageFont:
    size = max(14, min(34, int(min(image_width, image_height) * 0.018)))
    for font_name in ("arial.ttf", "consola.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def _fit_label_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    value = " ".join(str(text or "").split())
    if not value:
        return "box"
    if _text_size(draw, value, font)[0] <= max_width:
        return value
    suffix = "..."
    while len(value) > 4:
        value = value[:-1].rstrip()
        if _text_size(draw, value + suffix, font)[0] <= max_width:
            return value + suffix
    return value[:1] + suffix


def render_boxed_image(image_path: str | Path, rows: Any, bbox_order: str = "yxyx") -> Image.Image | None:
    row_values = _normalize_row_values(rows)
    valid_rows: list[tuple[int, list[Any], list[int]]] = []
    for row_index, values in enumerate(row_values):
        if not all(str(v).strip() for v in values[1:5]):
            continue
        y_min, x_min, y_max, x_max = row_bbox_to_yxyx(values[1:5], bbox_order=bbox_order)
        valid, _message = validate_bbox([y_min, x_min, y_max, x_max])
        if valid:
            valid_rows.append((row_index, values, [y_min, x_min, y_max, x_max]))
    if not valid_rows:
        return None

    with Image.open(image_path) as source:
        base = ImageOps.exif_transpose(source).convert("RGBA")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    line_width = max(3, int(round(min(width, height) * 0.003)))

    for row_index, _values, bbox in valid_rows:
        y_min, x_min, y_max, x_max = bbox
        color = _hex_to_rgb(BOX_COLORS[row_index % len(BOX_COLORS)])
        left = int(round((x_min / 1000) * width))
        top = int(round((y_min / 1000) * height))
        right = int(round((x_max / 1000) * width))
        bottom = int(round((y_max / 1000) * height))
        overlay_draw.rectangle((left, top, right, bottom), fill=(*color, 30), outline=(*color, 255), width=line_width)

    boxed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(boxed)
    font = _label_font(width, height)
    pad_x = max(6, line_width * 2)
    pad_y = max(4, line_width)

    for row_index, values, bbox in valid_rows:
        y_min, x_min, _y_max, _x_max = bbox
        color = _hex_to_rgb(BOX_COLORS[row_index % len(BOX_COLORS)])
        row_type = "text" if str(values[0] or "").strip() == "text" or str(values[7] or "").strip() else "obj"
        label_text = str(values[6] or values[5] or values[7] or row_type or "element")
        label = f"{row_index + 1:02d} {row_type} - {label_text}"
        left = int(round((x_min / 1000) * width))
        top = int(round((y_min / 1000) * height))
        max_label_width = max(80, min(int(width * 0.52), width - 8))
        fitted = _fit_label_text(draw, label, font, max_label_width - (pad_x * 2))
        text_width, text_height = _text_size(draw, fitted, font)
        label_width = text_width + pad_x * 2
        label_height = text_height + pad_y * 2
        label_left = max(0, min(left, width - label_width))
        label_top = top - label_height - max(1, line_width // 2)
        if label_top < 0:
            label_top = min(height - label_height, top + max(1, line_width // 2))
        draw.rectangle(
            (label_left, label_top, label_left + label_width, label_top + label_height),
            fill=(*color, 235),
        )
        draw.text((label_left + pad_x, label_top + pad_y), fitted, fill=(255, 255, 255, 255), font=font)

    return boxed.convert("RGB")


def save_boxed_image(image_path: str | Path, rows: Any, output_path: str | Path, bbox_order: str = "yxyx") -> Path | None:
    boxed = render_boxed_image(image_path, rows, bbox_order=bbox_order)
    if boxed is None:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    boxed.save(path, format="PNG")
    return path


def boxed_image_png_bytes(image_path: str | Path, rows: Any, bbox_order: str = "yxyx") -> bytes | None:
    boxed = render_boxed_image(image_path, rows, bbox_order=bbox_order)
    if boxed is None:
        return None
    buffer = BytesIO()
    boxed.save(buffer, format="PNG")
    return buffer.getvalue()


def build_ideogram_json(
    aspect_ratio: str,
    high_level_description: str,
    style_mode: str,
    aesthetics: str,
    lighting: str,
    photo: str,
    medium: str,
    art_style: str,
    color_palette: str,
    background: str,
    rows: Any,
    bbox_order: str = "yxyx",
    compact: bool = True,
) -> str:
    ratio = str(aspect_ratio or "1:1").strip()
    if not re.fullmatch(r"\d+:\d+", ratio):
        ratio = "1:1"
    style_notes: list[str] = []
    if str(style_mode).lower().startswith("photo"):
        style_notes.extend(part for part in [str(photo or "").strip(), "photograph"] if part)
    else:
        style_notes.extend(part for part in [str(medium or "illustration").strip(), str(art_style or "").strip()] if part)
    style_notes.extend(part for part in [str(aesthetics or "").strip(), str(lighting or "").strip(), _palette_to_text(_palette_from_text(color_palette) or [])] if part)
    high_level = str(high_level_description or "").strip()
    if style_notes:
        style_text = ", ".join(dict.fromkeys(style_notes))
        high_level = f"{high_level.rstrip('.')} in {style_text}." if high_level else style_text
    payload = {
        "aspect_ratio": ratio,
        "high_level_description": high_level,
        "compositional_deconstruction": {
            "background": str(background or "").strip(),
            "elements": rows_to_elements(rows, bbox_order=bbox_order),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if compact else json.dumps(payload, ensure_ascii=False, indent=2)
