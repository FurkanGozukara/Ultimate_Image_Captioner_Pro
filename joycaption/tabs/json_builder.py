from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageOps

from ..common import IMAGE_EXTENSIONS, OUTPUTS_DIR, html_message, natural_sort_key
from ..json_tools import (
    EMPTY_ELEMENT_ROW,
    apply_rows_to_json,
    build_ideogram_json,
    clean_bbox_order,
    headers_for_bbox_order,
    json_to_element_rows,
    overlay_html,
    parse_json_caption,
)
from ..overlay_js import OVERLAY_EDIT_JS
from .shared import TabUI


DEFAULT_BBOX_ORDER = "xyxy"
BBOX_ORDER_CHOICES = [
    ("x_min, y_min, x_max, y_max", "xyxy"),
    ("y_min, x_min, y_max, x_max", "yxyx"),
]
ASPECT_RATIO_CHOICES = ["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "Custom"]
DEFAULT_BUILDER_ROW = ["obj", 80, 80, 360, 360, "", "box 1", ""]


TABLE_EDIT_JS = r"""
if (!element.dataset.jcJsonTableBound) {
  element.dataset.jcJsonTableBound = "1";

  const parseRows = (root) => {
    try {
      const rows = JSON.parse(root.dataset.rows || "[]");
      return Array.isArray(rows) ? rows : [];
    } catch {
      return [];
    }
  };

  const syncSnapshot = (rows) => {
    const target = document.querySelector("#jc-json-rows-snapshot textarea, #jc-json-rows-snapshot input");
    if (!target) return;
    const value = JSON.stringify(Array.isArray(rows) ? rows : []);
    if (target.value === value) return;
    const proto = target.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (setter) setter.call(target, value);
    else target.value = value;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    target.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const readRows = (root) => {
    const rows = parseRows(root);
    root.querySelectorAll("[data-row-index][data-col-index]").forEach((input) => {
      const rowIndex = Number(input.dataset.rowIndex);
      const colIndex = Number(input.dataset.colIndex);
      if (!Number.isFinite(rowIndex) || !Number.isFinite(colIndex) || !rows[rowIndex]) return;
      rows[rowIndex][colIndex] = input.value;
    });
    return rows;
  };

  const commit = (target) => {
    const root = target.closest(".jc-json-table-editor");
    if (!root) return;
    const rows = readRows(root);
    syncSnapshot(rows);
    trigger("click", { action: "table-edit", rows });
  };

  element.addEventListener("focusin", (event) => {
    const target = event.target.closest(".jc-json-table-editor [data-row-index][data-col-index]");
    if (!target) return;
    const root = target.closest(".jc-json-table-editor");
    if (root) syncSnapshot(readRows(root));
  });

  element.addEventListener("change", (event) => {
    if (event.target.closest(".jc-json-table-editor [data-row-index][data-col-index]")) {
      commit(event.target);
    }
  });

  element.addEventListener("input", (event) => {
    const target = event.target.closest(".jc-json-table-editor [data-row-index][data-col-index]");
    if (!target) return;
    const root = target.closest(".jc-json-table-editor");
    syncSnapshot(readRows(root));
    clearTimeout(root._jcCommitTimer);
    root._jcCommitTimer = setTimeout(() => commit(target), 250);
  });

  element.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.closest(".jc-json-table-editor [data-row-index][data-col-index]")) {
      event.preventDefault();
      event.target.blur();
    }
  });

  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("button")) return;
    element.querySelectorAll(".jc-json-table-editor").forEach((root) => syncSnapshot(readRows(root)));
  }, true);

  queueMicrotask(() => {
    element.querySelectorAll(".jc-json-table-editor").forEach((root) => syncSnapshot(readRows(root)));
  });
}
"""


def _ratio(aspect_ratio: str, width: float | int | None, height: float | int | None) -> str:
    if str(aspect_ratio) == "Custom":
        try:
            w = max(1, float(width or 1))
            h = max(1, float(height or 1))
            return f"{w:g}:{h:g}"
        except Exception:
            return "1:1"
    return str(aspect_ratio or "1:1")


def _image_size(image_path: str | Path | None) -> tuple[int, int] | None:
    if not image_path:
        return None
    try:
        with Image.open(image_path) as image:
            oriented = ImageOps.exif_transpose(image)
            return oriented.size
    except Exception:
        return None


def _ratio_from_size(width: int, height: int) -> str:
    divisor = math.gcd(max(1, int(width)), max(1, int(height)))
    return f"{int(width) // divisor}:{int(height) // divisor}"


def _aspect_controls_for_image(image_path: str | Path | None) -> tuple[Any, Any, Any]:
    size = _image_size(image_path)
    if not size:
        return gr.update(), gr.update(), gr.update()
    width, height = size
    ratio = _ratio_from_size(width, height)
    ratio_value = ratio if ratio in ASPECT_RATIO_CHOICES else "Custom"
    return gr.update(value=ratio_value), gr.update(value=width), gr.update(value=height)


def _outputs_image_choices() -> list[tuple[str, str]]:
    if not OUTPUTS_DIR.exists():
        return []
    images = [path for path in OUTPUTS_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    images = sorted(images, key=natural_sort_key)
    choices: list[tuple[str, str]] = []
    for path in images:
        try:
            label = str(path.relative_to(OUTPUTS_DIR))
        except Exception:
            label = path.name
        choices.append((label.replace("\\", "/"), str(path)))
    return choices


def _row_data(rows: Any) -> list[list[Any]]:
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    return [list(row) for row in (rows or []) if isinstance(row, (list, tuple))]


def _rows_snapshot_value(rows: Any) -> str:
    return json.dumps(_row_data(rows), ensure_ascii=False)


def _rows_from_snapshot(snapshot: Any, fallback: Any) -> list[list[Any]]:
    if isinstance(snapshot, str):
        text = snapshot.strip()
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return _row_data(data)
            except Exception:
                pass
    return _row_data(fallback)


def _df_value(rows: Any, bbox_order: str = DEFAULT_BBOX_ORDER) -> Any:
    import pandas as pd

    return pd.DataFrame(_row_data(rows), columns=headers_for_bbox_order(bbox_order))


def _df_update(rows: Any, bbox_order: str = DEFAULT_BBOX_ORDER):
    return _df_value(rows, bbox_order)


def _choice_indices(selected: Any) -> list[int]:
    if selected is None:
        return []
    if isinstance(selected, str):
        selected = [selected]
    indices: list[int] = []
    for value in selected or []:
        match = str(value).strip()[:2]
        try:
            indices.append(int(match) - 1)
        except Exception:
            continue
    return [index for index in indices if index >= 0]


def _visible_rows(rows: Any, selected: Any) -> list[list[Any]]:
    row_data = _row_data(rows)
    return [row_data[index] for index in _choice_indices(selected) if index < len(row_data)]


def _visible_df_value(rows: Any, selected: Any, bbox_order: str = DEFAULT_BBOX_ORDER):
    return _df_update(_visible_rows(rows, selected), bbox_order)


def _table_editor_html(rows: Any, selected: Any, bbox_order: str = DEFAULT_BBOX_ORDER) -> str:
    row_data = _row_data(rows)
    indices = _choice_indices(selected)
    if not indices and row_data:
        indices = list(range(len(row_data)))
    headers = headers_for_bbox_order(bbox_order)
    rows_json = html.escape(json.dumps(row_data, ensure_ascii=False), quote=True)
    parts = [
        f'<div class="jc-json-table-editor" data-rows="{rows_json}">',
        "<table>",
        "<thead><tr>",
    ]
    for header in headers:
        parts.append(f"<th>{html.escape(str(header))}</th>")
    parts.append("</tr></thead><tbody>")
    rendered = 0
    for row_index in indices:
        if row_index >= len(row_data):
            continue
        values = row_data[row_index] + [""] * max(0, len(headers) - len(row_data[row_index]))
        if not any(str(cell or "").strip() for cell in values):
            continue
        rendered += 1
        parts.append("<tr>")
        for col_index, value in enumerate(values[: len(headers)]):
            input_type = "number" if 1 <= col_index <= 4 else "text"
            escaped_value = html.escape(str(value or ""), quote=True)
            parts.append(
                f'<td><input type="{input_type}" value="{escaped_value}" '
                f'data-row-index="{row_index}" data-col-index="{col_index}" /></td>'
            )
        parts.append("</tr>")
    if rendered == 0:
        parts.append(f'<tr><td colspan="{len(headers)}" class="jc-json-table-empty">No visible boxes.</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _merge_visible_rows(all_rows: Any, selected: Any, visible_rows: Any) -> list[list[Any]]:
    merged = _row_data(all_rows)
    visible_data = _row_data(visible_rows)
    if not merged and visible_data:
        return visible_data
    for target_index, row in zip(_choice_indices(selected), visible_data):
        if 0 <= target_index < len(merged):
            merged[target_index] = row
    return merged


def _new_element_row(existing_rows: Any) -> list[Any]:
    row = EMPTY_ELEMENT_ROW.copy()
    index = len(_row_data(existing_rows)) + 1
    offset = min(640, 80 + ((index - 1) * 48))
    row[1] = offset
    row[2] = offset
    row[3] = min(1000, offset + 280)
    row[4] = min(1000, offset + 280)
    row[6] = f"box {index}"
    return row


def _box_choices(rows: Any) -> list[str]:
    choices: list[str] = []
    for index, row in enumerate(_row_data(rows), start=1):
        values = row + [""] * max(0, 8 - len(row))
        if not any(str(cell or "").strip() for cell in values):
            continue
        label = str(values[5] or values[6] or values[0] or "box")
        label = " ".join(label.split())
        if len(label) > 58:
            label = label[:55] + "..."
        choices.append(f"{index:02d} {values[0] or 'obj'} - {label}")
    return choices


def _preserve_visible(rows: Any, selected: Any, default_all: bool = False) -> tuple[list[str], list[str]]:
    choices = _box_choices(rows)
    selected_indices = _choice_indices(selected)
    values = [choices[index] for index in selected_indices if index < len(choices)]
    if default_all and not values:
        values = choices
    return choices, values


def _sidecar_json_for_image(image_path: str | Path | None) -> Path | None:
    if not image_path:
        return None
    candidate = Path(image_path).with_suffix(".json")
    return candidate if candidate.exists() else None


def _fields_from_json(parsed: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str, str]:
    style = parsed.get("style_description") if isinstance(parsed.get("style_description"), dict) else {}
    comp = parsed.get("compositional_deconstruction") if isinstance(parsed.get("compositional_deconstruction"), dict) else {}
    style_mode = "Photograph" if "photo" in style else "Art / Design"
    palette = style.get("color_palette")
    if isinstance(palette, list):
        palette_value = ", ".join(str(item) for item in palette)
    else:
        palette_value = str(palette or "")
    return (
        str(parsed.get("high_level_description", "") or ""),
        style_mode,
        str(style.get("aesthetics", "") or ""),
        str(style.get("lighting", "") or ""),
        str(style.get("photo", "") or ""),
        str(style.get("medium", "") or ""),
        str(style.get("art_style", "") or ""),
        palette_value,
        str(comp.get("background", "") or ""),
    )


def build_tab() -> TabUI:
    all_element_rows = gr.State([DEFAULT_BUILDER_ROW.copy()])
    rows_snapshot = gr.Textbox(
        value=_rows_snapshot_value([DEFAULT_BUILDER_ROW.copy()]),
        label="Rows Snapshot",
        elem_id="jc-json-rows-snapshot",
        elem_classes=["jc-hidden-sync"],
        interactive=True,
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes=["jc-compact"]):
            image = gr.Image(type="filepath", label="Optional Preview Image", height=390)
            with gr.Accordion("Load / Continue From Outputs", open=True):
                output_image = gr.Dropdown(
                    choices=_outputs_image_choices(),
                    label="Output Image",
                    allow_custom_value=False,
                )
                with gr.Row():
                    refresh_outputs_btn = gr.Button("Refresh", elem_classes=["btn-refresh"])
                    load_output_btn = gr.Button("Load Selected", elem_classes=["btn-load-preset"])
            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    choices=ASPECT_RATIO_CHOICES,
                    value="1:1",
                    label="Aspect Ratio",
                )
                canvas_width = gr.Number(label="Width", value=1000, precision=0)
                canvas_height = gr.Number(label="Height", value=1000, precision=0)
            overlay = gr.HTML("", elem_classes=["jc-qwen-overlay"], js_on_load=OVERLAY_EDIT_JS)

        with gr.Column(scale=10, elem_classes=["jc-compact"]):
            high_level = gr.Textbox(label="High Level Description", lines=4)
            with gr.Accordion("Elements", open=True):
                with gr.Row(elem_classes=["jc-qwen-box-toolbar"]):
                    bbox_order = gr.Radio(
                        choices=BBOX_ORDER_CHOICES,
                        value=DEFAULT_BBOX_ORDER,
                        label="BBox Order",
                        scale=4,
                    )
                    disable_auto_update = gr.Checkbox(
                        label="Disable Auto Update Coordinates",
                        value=True,
                        scale=2,
                    )
                    check_all_boxes_btn = gr.Button("Check All", elem_classes=["btn-load-preset"], scale=1)
                    uncheck_all_boxes_btn = gr.Button("Uncheck All", elem_classes=["btn-reset-preset"], scale=1)
                box_visibility = gr.CheckboxGroup(
                    choices=_box_choices([DEFAULT_BUILDER_ROW.copy()]),
                    value=_box_choices([DEFAULT_BUILDER_ROW.copy()]),
                    label="Visible Boxes",
                    elem_classes=["jc-qwen-box-filter"],
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate JSON", elem_classes=["btn-json-build"])
                    preview_btn = gr.Button("Preview Boxes", elem_classes=["btn-qwen-render"])
                    apply_box_btn = gr.Button("Apply Box Edits", elem_classes=["btn-qwen-apply"])
                    add_row_btn = gr.Button("Add Box", elem_classes=["btn-json-add"])
                    clear_rows_btn = gr.Button("Clear Boxes", elem_classes=["btn-reset-preset"])
                    import_btn = gr.Button("Import JSON Rows", elem_classes=["btn-load-preset"])
                json_output = gr.Textbox(
                    label="Generated Ideogram JSON",
                    lines=10,
                    interactive=True,
                    elem_classes=["jc-output", "jc-codeish"],
                )
                status = gr.HTML("")
            element_rows = gr.HTML(
                _table_editor_html(
                    [DEFAULT_BUILDER_ROW.copy()],
                    _box_choices([DEFAULT_BUILDER_ROW.copy()]),
                    DEFAULT_BBOX_ORDER,
                ),
                label="Boxes",
                elem_classes=["jc-qwen-elements-wide", "jc-json-builder-boxes-large"],
                js_on_load=TABLE_EDIT_JS,
            )
            with gr.Row():
                style_mode = gr.Radio(choices=["Photograph", "Art / Design"], value="Photograph", label="Style Branch")
                medium = gr.Textbox(label="Medium", value="illustration")
            aesthetics = gr.Textbox(label="Aesthetics", lines=2)
            lighting = gr.Textbox(label="Lighting", lines=2)
            photo = gr.Textbox(label="Photo", lines=2, placeholder="eye-level close-up, shallow depth of field")
            art_style = gr.Textbox(label="Art Style", lines=2, placeholder="flat vector icon, detailed digital illustration")
            color_palette = gr.Textbox(label="Color Palette", placeholder="#FFFFFF, #1E88E5, #FDD835")
            background = gr.Textbox(label="Background", lines=3)

    def _overlay(image_path, ratio_value, width_value, height_value, rows, bbox_order_value, visible_choices, disable_auto_update_value):
        return overlay_html(
            image_path,
            rows,
            aspect_ratio=_ratio(ratio_value, width_value, height_value),
            interactive=True,
            bbox_order=clean_bbox_order(bbox_order_value),
            visible_indices=_choice_indices(visible_choices),
            disable_auto_update=bool(disable_auto_update_value),
        )

    def _build_json(
        high_level_value,
        style_mode_value,
        aesthetics_value,
        lighting_value,
        photo_value,
        medium_value,
        art_style_value,
        palette_value,
        background_value,
        rows,
        bbox_order_value,
    ):
        return build_ideogram_json(
            high_level_value,
            style_mode_value,
            aesthetics_value,
            lighting_value,
            photo_value,
            medium_value,
            art_style_value,
            palette_value,
            background_value,
            rows,
            bbox_order=clean_bbox_order(bbox_order_value),
        )

    def generate_json(
        image_path,
        ratio_value,
        width_value,
        height_value,
        high_level_value,
        style_mode_value,
        aesthetics_value,
        lighting_value,
        photo_value,
        medium_value,
        art_style_value,
        palette_value,
        background_value,
        all_rows,
        snapshot_value,
        bbox_order_value,
        visible_choices,
        disable_auto_update_value,
    ):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        merged_rows = _rows_from_snapshot(snapshot_value, all_rows)
        choices, visible = _preserve_visible(merged_rows, visible_choices, default_all=True)
        json_text = _build_json(
            high_level_value,
            style_mode_value,
            aesthetics_value,
            lighting_value,
            photo_value,
            medium_value,
            art_style_value,
            palette_value,
            background_value,
            merged_rows,
            bbox_order_value,
        )
        preview = _overlay(image_path, ratio_value, width_value, height_value, merged_rows, bbox_order_value, visible, disable_auto_update_value)
        return json_text, merged_rows, _rows_snapshot_value(merged_rows), _table_editor_html(merged_rows, visible, bbox_order_value), gr.update(choices=choices, value=visible), preview, ""

    def preview_boxes(image_path, ratio_value, width_value, height_value, all_rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        merged_rows = _rows_from_snapshot(snapshot_value, all_rows)
        choices, visible = _preserve_visible(merged_rows, visible_choices, default_all=True)
        return (
            merged_rows,
            _rows_snapshot_value(merged_rows),
            _table_editor_html(merged_rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            _overlay(image_path, ratio_value, width_value, height_value, merged_rows, bbox_order_value, visible, disable_auto_update_value),
        )

    def apply_box_edits(image_path, ratio_value, width_value, height_value, json_text, all_rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        merged_rows = _rows_from_snapshot(snapshot_value, all_rows)
        final, parsed, warnings = apply_rows_to_json(json_text, merged_rows, bbox_order=bbox_order_value)
        display_rows = _row_data(merged_rows)
        choices, visible = _preserve_visible(display_rows, visible_choices, default_all=True)
        status_html = html_message("success", "JSON box edits applied.")
        if warnings:
            status_html = html_message("info", "Applied box edits after JSON repair fallback:<br><pre>" + "\n".join(warnings) + "</pre>")
        return (
            final,
            display_rows,
            _rows_snapshot_value(display_rows),
            _table_editor_html(display_rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            _overlay(image_path, ratio_value, width_value, height_value, display_rows, bbox_order_value, visible, disable_auto_update_value),
            status_html,
        )

    def add_row(image_path, ratio_value, width_value, height_value, all_rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        current_rows = _rows_from_snapshot(snapshot_value, all_rows)
        next_rows = [*current_rows, _new_element_row(current_rows)]
        choices = _box_choices(next_rows)
        selected = [choice for choice in choices if choice in set(visible_choices or [])]
        if choices:
            selected.append(choices[-1])
        return (
            next_rows,
            _rows_snapshot_value(next_rows),
            _table_editor_html(next_rows, selected, bbox_order_value),
            gr.update(choices=choices, value=selected),
            _overlay(image_path, ratio_value, width_value, height_value, next_rows, bbox_order_value, selected, disable_auto_update_value),
        )

    def clear_rows(bbox_order_value):
        return [], _rows_snapshot_value([]), _table_editor_html([], [], bbox_order_value), gr.update(choices=[], value=[])

    def import_json(json_text, image_path, ratio_value, width_value, height_value, bbox_order_value, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        parsed, pretty, warnings = parse_json_caption(json_text)
        if parsed is None:
            empty_overlay = _overlay(image_path, ratio_value, width_value, height_value, [], bbox_order_value, [], disable_auto_update_value)
            return (
                [],
                _rows_snapshot_value([]),
                _table_editor_html([], [], bbox_order_value),
                gr.update(choices=[], value=[]),
                "",
                "",
                "Photograph",
                "",
                "",
                "",
                "illustration",
                "",
                "",
                "",
                empty_overlay,
                html_message("error", "Could not parse JSON.<br><pre>" + "\n".join(warnings) + "</pre>"),
            )
        rows = json_to_element_rows(parsed)
        choices, visible = _preserve_visible(rows, [], default_all=True)
        fields = _fields_from_json(parsed)
        status_html = html_message("success", "JSON imported.")
        if warnings:
            status_html = html_message("info", "JSON imported with warnings:<br><pre>" + "\n".join(warnings) + "</pre>")
        return (
            rows,
            _rows_snapshot_value(rows),
            _table_editor_html(rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            pretty,
            *fields,
            _overlay(image_path, ratio_value, width_value, height_value, rows, bbox_order_value, visible, disable_auto_update_value),
            status_html,
        )

    def load_output_choice(selected_image, bbox_order_value, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        if not selected_image:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "",
                [],
                _rows_snapshot_value([]),
                _table_editor_html([], [], bbox_order_value),
                gr.update(choices=[], value=[]),
                "",
                "Photograph",
                "",
                "",
                "",
                "illustration",
                "",
                "",
                "",
                gr.update(),
                html_message("error", "Select an output image first."),
            )
        image_path = Path(selected_image)
        ratio_update, width_update, height_update = _aspect_controls_for_image(image_path)
        ratio_value = getattr(ratio_update, "get", lambda _key, _default=None: _default)("value", None) if isinstance(ratio_update, dict) else None
        width_value = getattr(width_update, "get", lambda _key, _default=None: _default)("value", None) if isinstance(width_update, dict) else None
        height_value = getattr(height_update, "get", lambda _key, _default=None: _default)("value", None) if isinstance(height_update, dict) else None
        ratio_for_overlay = ratio_value or "1:1"
        width_for_overlay = width_value or 1000
        height_for_overlay = height_value or 1000
        sidecar = _sidecar_json_for_image(image_path)
        if sidecar is None:
            empty_overlay = _overlay(image_path, ratio_for_overlay, width_for_overlay, height_for_overlay, [], bbox_order_value, [], disable_auto_update_value)
            return (
                str(image_path),
                ratio_update,
                width_update,
                height_update,
                "",
                [],
                _rows_snapshot_value([]),
                _table_editor_html([], [], bbox_order_value),
                gr.update(choices=[], value=[]),
                "",
                "Photograph",
                "",
                "",
                "",
                "illustration",
                "",
                "",
                "",
                empty_overlay,
                html_message("info", f"Loaded image, but no same-name JSON was found: {image_path.with_suffix('.json')}"),
            )
        text = sidecar.read_text(encoding="utf-8")
        parsed, pretty, warnings = parse_json_caption(text)
        if parsed is None:
            empty_overlay = _overlay(image_path, ratio_for_overlay, width_for_overlay, height_for_overlay, [], bbox_order_value, [], disable_auto_update_value)
            return (
                str(image_path),
                ratio_update,
                width_update,
                height_update,
                text,
                [],
                _rows_snapshot_value([]),
                _table_editor_html([], [], bbox_order_value),
                gr.update(choices=[], value=[]),
                "",
                "Photograph",
                "",
                "",
                "",
                "illustration",
                "",
                "",
                "",
                empty_overlay,
                html_message("error", "Loaded image, but same-name JSON could not be parsed.<br><pre>" + "\n".join(warnings) + "</pre>"),
            )
        rows = json_to_element_rows(parsed)
        choices, visible = _preserve_visible(rows, [], default_all=True)
        fields = _fields_from_json(parsed)
        status_html = html_message("success", f"Loaded {image_path.name} and {sidecar.name}.")
        return (
            str(image_path),
            ratio_update,
            width_update,
            height_update,
            pretty,
            rows,
            _rows_snapshot_value(rows),
            _table_editor_html(rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            *fields,
            _overlay(image_path, ratio_for_overlay, width_for_overlay, height_for_overlay, rows, bbox_order_value, visible, disable_auto_update_value),
            status_html,
        )

    def update_box_visibility(image_path, ratio_value, width_value, height_value, rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        current_rows = _rows_from_snapshot(snapshot_value, rows)
        return (
            _table_editor_html(current_rows, visible_choices, bbox_order_value),
            _overlay(image_path, ratio_value, width_value, height_value, current_rows, bbox_order_value, visible_choices, disable_auto_update_value),
        )

    def update_auto_update_setting(image_path, ratio_value, width_value, height_value, rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        current_rows = _rows_from_snapshot(snapshot_value, rows)
        return _overlay(image_path, ratio_value, width_value, height_value, current_rows, bbox_order_value, visible_choices, disable_auto_update_value)

    def apply_overlay_edit(image_path, ratio_value, width_value, height_value, json_text, bbox_order_value, visible_choices, disable_auto_update_value, evt: gr.EventData):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        payload = getattr(evt, "_data", {}) or {}
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        final, parsed, warnings = apply_rows_to_json(json_text, rows, bbox_order=bbox_order_value)
        display_rows = _row_data(rows)
        choices, visible = _preserve_visible(display_rows, visible_choices, default_all=True)
        status_html = "" if not warnings else html_message("info", "JSON repaired while applying overlay edit:<br><pre>" + "\n".join(warnings) + "</pre>")
        return (
            final,
            display_rows,
            _rows_snapshot_value(display_rows),
            _table_editor_html(display_rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            _overlay(image_path, ratio_value, width_value, height_value, display_rows, bbox_order_value, visible, disable_auto_update_value),
            status_html,
        )

    def apply_table_edit(image_path, ratio_value, width_value, height_value, bbox_order_value, visible_choices, disable_auto_update_value, evt: gr.EventData):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        payload = getattr(evt, "_data", {}) or {}
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        normalized_rows = _row_data(rows)
        choices, visible = _preserve_visible(normalized_rows, visible_choices, default_all=True)
        return (
            normalized_rows,
            _rows_snapshot_value(normalized_rows),
            _table_editor_html(normalized_rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            _overlay(image_path, ratio_value, width_value, height_value, normalized_rows, bbox_order_value, visible, disable_auto_update_value),
        )

    def check_all_boxes(rows, snapshot_value):
        choices = _box_choices(_rows_from_snapshot(snapshot_value, rows))
        return gr.update(choices=choices, value=choices)

    def uncheck_all_boxes(rows, snapshot_value):
        choices = _box_choices(_rows_from_snapshot(snapshot_value, rows))
        return gr.update(choices=choices, value=[])

    def refresh_output_choices():
        return gr.update(choices=_outputs_image_choices())

    generate_inputs = [
        image,
        aspect_ratio,
        canvas_width,
        canvas_height,
        high_level,
        style_mode,
        aesthetics,
        lighting,
        photo,
        medium,
        art_style,
        color_palette,
        background,
        all_element_rows,
        rows_snapshot,
        bbox_order,
        box_visibility,
        disable_auto_update,
    ]
    generate_btn.click(
        generate_json,
        inputs=generate_inputs,
        outputs=[json_output, all_element_rows, rows_snapshot, element_rows, box_visibility, overlay, status],
    )
    preview_btn.click(
        preview_boxes,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[all_element_rows, rows_snapshot, element_rows, box_visibility, overlay],
        queue=False,
    )
    apply_box_btn.click(
        apply_box_edits,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, json_output, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[json_output, all_element_rows, rows_snapshot, element_rows, box_visibility, overlay, status],
    )
    add_row_btn.click(
        add_row,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[all_element_rows, rows_snapshot, element_rows, box_visibility, overlay],
        queue=False,
    )
    clear_rows_btn.click(clear_rows, inputs=[bbox_order], outputs=[all_element_rows, rows_snapshot, element_rows, box_visibility], queue=False).then(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[element_rows, overlay],
        queue=False,
    )
    import_btn.click(
        import_json,
        inputs=[json_output, image, aspect_ratio, canvas_width, canvas_height, bbox_order, disable_auto_update],
        outputs=[
            all_element_rows,
            rows_snapshot,
            element_rows,
            box_visibility,
            json_output,
            high_level,
            style_mode,
            aesthetics,
            lighting,
            photo,
            medium,
            art_style,
            color_palette,
            background,
            overlay,
            status,
        ],
        queue=False,
    )
    box_visibility.change(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[element_rows, overlay],
        queue=False,
    )
    bbox_order.change(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[element_rows, overlay],
        queue=False,
    )
    disable_auto_update.change(
        update_auto_update_setting,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=overlay,
        queue=False,
    )
    json_overlay_inputs = [image, aspect_ratio, canvas_width, canvas_height, json_output, bbox_order, box_visibility, disable_auto_update]
    overlay.click(
        apply_overlay_edit,
        inputs=json_overlay_inputs,
        outputs=[json_output, all_element_rows, rows_snapshot, element_rows, box_visibility, overlay, status],
        queue=False,
    )
    element_rows.click(
        apply_table_edit,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, bbox_order, box_visibility, disable_auto_update],
        outputs=[all_element_rows, rows_snapshot, element_rows, box_visibility, overlay],
        queue=False,
    )
    check_all_boxes_btn.click(check_all_boxes, inputs=[all_element_rows, rows_snapshot], outputs=box_visibility, queue=False).then(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[element_rows, overlay],
        queue=False,
    )
    uncheck_all_boxes_btn.click(uncheck_all_boxes, inputs=[all_element_rows, rows_snapshot], outputs=box_visibility, queue=False).then(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[element_rows, overlay],
        queue=False,
    )
    refresh_outputs_btn.click(refresh_output_choices, outputs=output_image, queue=False)
    load_outputs = [
        image,
        aspect_ratio,
        canvas_width,
        canvas_height,
        json_output,
        all_element_rows,
        rows_snapshot,
        element_rows,
        box_visibility,
        high_level,
        style_mode,
        aesthetics,
        lighting,
        photo,
        medium,
        art_style,
        color_palette,
        background,
        overlay,
        status,
    ]
    output_image.change(load_output_choice, inputs=[output_image, bbox_order, disable_auto_update], outputs=load_outputs, queue=False)
    load_output_btn.click(load_output_choice, inputs=[output_image, bbox_order, disable_auto_update], outputs=load_outputs, queue=False)

    return TabUI(key="json_builder", order=[], defaults={}, inputs=[])
