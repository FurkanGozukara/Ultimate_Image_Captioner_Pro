from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote


ELEMENT_HEADERS = ["type", "y_min", "x_min", "y_max", "x_max", "text", "desc", "color_palette"]
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


def normalize_json_output(text: str, preset_id: str = "", compact: bool = False) -> tuple[str, dict[str, Any] | None, list[str]]:
    parsed, pretty, warnings = parse_json_caption(text)
    if parsed is None:
        return pretty, None, warnings
    if preset_id.startswith("i4_json_"):
        warnings.extend(validate_ideogram_json(parsed))
    normalized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) if compact else json.dumps(parsed, ensure_ascii=False, indent=2)
    return normalized, parsed, warnings


def validate_bbox(value: Any) -> tuple[bool, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False, "bbox must contain four numbers."
    try:
        y_min, x_min, y_max, x_max = [int(round(float(item))) for item in value]
    except Exception:
        return False, "bbox values must be numeric."
    if not (0 <= y_min < y_max <= 1000 and 0 <= x_min < x_max <= 1000):
        return False, "expected 0 <= y_min < y_max <= 1000 and 0 <= x_min < x_max <= 1000."
    return True, ""


def clamp_bbox(y_min: Any, x_min: Any, y_max: Any, x_max: Any) -> list[int]:
    vals = []
    for value in (y_min, x_min, y_max, x_max):
        try:
            vals.append(max(0, min(1000, int(round(float(value))))))
        except Exception:
            vals.append(0)
    if vals[2] <= vals[0]:
        vals[2] = min(1000, vals[0] + 1)
    if vals[3] <= vals[1]:
        vals[3] = min(1000, vals[1] + 1)
    return vals


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


def rows_to_elements(rows: Any) -> list[dict[str, Any]]:
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
        y_min, x_min, y_max, x_max = values[1], values[2], values[3], values[4]
        text_value = str(values[5] or "").strip()
        desc_value = str(values[6] or "").strip()
        palette = _palette_from_text(values[7])
        element: dict[str, Any] = {"type": "text" if element_type == "text" else "obj"}
        if all(str(v).strip() for v in (y_min, x_min, y_max, x_max)):
            element["bbox"] = clamp_bbox(y_min, x_min, y_max, x_max)
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


def apply_rows_to_json(json_text: str, rows: Any) -> tuple[str, dict[str, Any] | None, list[str]]:
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
    comp["elements"] = rows_to_elements(rows)
    normalized = json.dumps(data, ensure_ascii=False, indent=2)
    return normalized, data, warnings


def _file_url(path: str | Path) -> str:
    normalized = str(Path(path)).replace("\\", "/")
    return "/file=" + quote(normalized, safe="/:")


def _label_for_element(index: int, element: dict[str, Any]) -> str:
    label = str(element.get("text") or element.get("desc") or element.get("type") or "element")
    label = " ".join(label.split())
    if len(label) > 72:
        label = label[:69] + "..."
    return f"{index:02d} {element.get('type', 'obj')} - {label}"


def overlay_html(
    image_path: str | Path | None,
    rows: Any,
    aspect_ratio: str = "1:1",
    max_height: int = 768,
    empty_message: str = "No boxes to preview.",
) -> str:
    elements = rows_to_elements(rows)
    boxes = []
    for index, element in enumerate(elements, start=1):
        bbox = element.get("bbox")
        valid, _message = validate_bbox(bbox)
        if not valid:
            continue
        y_min, x_min, y_max, x_max = bbox
        color = BOX_COLORS[(index - 1) % len(BOX_COLORS)]
        label = html.escape(_label_for_element(index, element))
        boxes.append(
            f'<div class="jc-box" style="top:{y_min / 10:.3f}%;left:{x_min / 10:.3f}%;'
            f'width:{(x_max - x_min) / 10:.3f}%;height:{(y_max - y_min) / 10:.3f}%;'
            f'border-color:{color};background:{color}18;">'
            f'<span style="background:{color};">{label}</span></div>'
        )

    if not boxes:
        box_markup = f'<div class="jc-overlay-empty">{html.escape(empty_message)}</div>'
    else:
        box_markup = "\n".join(boxes)

    if image_path:
        src = html.escape(_file_url(image_path), quote=True)
        return (
            '<div class="jc-overlay-shell">'
            f'<div class="jc-overlay-frame" style="max-height:{int(max_height)}px;">'
            f'<img class="jc-overlay-image" src="{src}" alt="overlay source" />'
            f"{box_markup}</div></div>"
        )

    ratio = aspect_ratio if re.fullmatch(r"\d+(\.\d+)?:\d+(\.\d+)?", str(aspect_ratio or "")) else "1:1"
    width, height = ratio.split(":", 1)
    return (
        '<div class="jc-overlay-shell">'
        f'<div class="jc-overlay-frame jc-overlay-blank" style="aspect-ratio:{float(width)}/{float(height)};max-height:{int(max_height)}px;">'
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
