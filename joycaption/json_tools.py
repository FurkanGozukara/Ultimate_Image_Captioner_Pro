from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote


ELEMENT_HEADERS = ["type", "y_min", "x_min", "y_max", "x_max", "text", "desc", "color_palette"]
XYXY_ELEMENT_HEADERS = ["type", "x_min", "y_min", "x_max", "y_max", "text", "desc", "color_palette"]
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
    warnings: list[str] = []
    for key in ("high_level_description", "style_description", "compositional_deconstruction"):
        if key not in data:
            warnings.append(f'Missing top-level key "{key}".')
    comp = data.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        warnings.append('"compositional_deconstruction" must be an object.')
        return warnings
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
        bbox = element.get("bbox")
        if bbox is None:
            continue
        valid, message = validate_bbox(bbox)
        if not valid:
            warnings.append(f"Element {index} bbox: {message}")
    return warnings


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
    if preset_id.startswith("i4_official_v1"):
        warnings.extend(validate_official_v1_json(parsed))
    elif preset_id.startswith("i4_json_"):
        warnings.extend(validate_ideogram_json(parsed))
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


def clean_bbox_order(value: Any) -> str:
    return "xyxy" if str(value or "").lower() == "xyxy" else "yxyx"


def headers_for_bbox_order(value: Any) -> list[str]:
    return XYXY_ELEMENT_HEADERS if clean_bbox_order(value) == "xyxy" else ELEMENT_HEADERS


def bbox_to_yxyx(bbox: Any, bbox_order: str = "yxyx") -> list[int]:
    first, second, third, fourth = clamp_bbox(*bbox)
    if clean_bbox_order(bbox_order) == "xyxy":
        x_min, y_min, x_max, y_max = first, second, third, fourth
        return [y_min, x_min, y_max, x_max]
    return [first, second, third, fourth]


def _palette_to_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def _palette_from_text(value: Any) -> list[str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return [part.strip() for part in re.split(r"[,;\n]", text) if part.strip()]


def json_to_element_rows(data: dict[str, Any] | None) -> list[list[Any]]:
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
        rows.append(
            [
                element.get("type", "obj"),
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                element.get("text", ""),
                element.get("desc", ""),
                _palette_to_text(element.get("color_palette")),
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
        text_value = str(values[5] or "").strip()
        desc_value = str(values[6] or "").strip()
        palette = _palette_from_text(values[7])
        element: dict[str, Any] = {"type": "text" if element_type == "text" else "obj"}
        if all(str(v).strip() for v in (min_1, min_2, max_1, max_2)):
            element["bbox"] = clamp_bbox(min_1, min_2, max_1, max_2)
        if element["type"] == "text":
            element["text"] = text_value
        if desc_value:
            element["desc"] = desc_value
        elif element["type"] == "obj" and text_value:
            element["desc"] = text_value
        if palette:
            element["color_palette"] = palette
        if element.get("desc") or element.get("text") or element.get("bbox"):
            elements.append(element)
    return elements


def apply_rows_to_json(json_text: str, rows: Any, bbox_order: str = "yxyx") -> tuple[str, dict[str, Any] | None, list[str]]:
    data, _pretty, warnings = parse_json_caption(json_text)
    if data is None:
        data = {
            "high_level_description": "",
            "style_description": {},
            "compositional_deconstruction": {"background": "", "elements": []},
        }
    comp = data.setdefault("compositional_deconstruction", {})
    if not isinstance(comp, dict):
        comp = {}
        data["compositional_deconstruction"] = comp
    comp.setdefault("background", "")
    comp["elements"] = rows_to_elements(rows, bbox_order=bbox_order)
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
        element = rows_to_elements([values], bbox_order=bbox_order)
        if not element:
            continue
        item = element[0]
        bbox = item.get("bbox")
        valid, _message = validate_bbox(bbox)
        if not valid:
            continue
        y_min, x_min, y_max, x_max = bbox_to_yxyx(bbox, bbox_order=bbox_order)
        color = BOX_COLORS[(row_index) % len(BOX_COLORS)]
        label = html.escape(_label_for_element(row_index + 1, item))
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


def build_ideogram_json(
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
    compact: bool = False,
) -> str:
    palette = _palette_from_text(color_palette) or []
    style: dict[str, Any] = {
        "aesthetics": str(aesthetics or "").strip(),
        "lighting": str(lighting or "").strip(),
    }
    if str(style_mode).lower().startswith("photo"):
        style["photo"] = str(photo or "").strip()
        style["medium"] = "photograph"
    else:
        style["medium"] = str(medium or "illustration").strip() or "illustration"
        style["art_style"] = str(art_style or "").strip()
    style["color_palette"] = palette
    payload = {
        "high_level_description": str(high_level_description or "").strip(),
        "style_description": style,
        "compositional_deconstruction": {
            "background": str(background or "").strip(),
            "elements": rows_to_elements(rows),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if compact else json.dumps(payload, ensure_ascii=False, indent=2)
