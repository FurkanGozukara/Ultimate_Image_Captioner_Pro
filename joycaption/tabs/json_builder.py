from __future__ import annotations

import html
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageOps

from ..common import (
    IMAGE_EXTENSIONS,
    OUTPUTS_DIR,
    copy_image_if_needed,
    html_message,
    natural_sort_key,
    next_numbered_output_dir,
    write_generation_metadata,
)
from ..json_tools import (
    EMPTY_ELEMENT_ROW,
    _file_url,
    apply_rows_to_json,
    build_ideogram_json,
    clean_bbox_order,
    headers_for_bbox_order,
    json_to_element_rows,
    normalize_json_output,
    overlay_html,
    save_boxed_image,
)
from ..overlay_js import OVERLAY_EDIT_JS
from .shared import TabUI


DEFAULT_BBOX_ORDER = "yxyx"
BBOX_ORDER_CHOICES = [
    ("y_min, x_min, y_max, x_max", "yxyx"),
    ("x_min, y_min, x_max, y_max", "xyxy"),
]
TABLE_COLUMN_CLASSES = [
    "type",
    "bbox",
    "bbox",
    "bbox",
    "bbox",
    "caption",
    "box-title",
    "text",
]
WRAPPING_TABLE_COLUMNS = {5, 7}
ASPECT_RATIO_CHOICES = ["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "Custom"]
DEFAULT_BUILDER_ROW = ["obj", 80, 80, 360, 360, "", "box 1", ""]
BUILDER_PRESET_ID = "i4_official_v1_app_compare"
OUTPUT_NOT_SELECTED_LABEL = "Not-Selected"
OUTPUT_NOT_SELECTED_VALUE = ""


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

  const resizeTextarea = (textarea) => {
    textarea.style.height = "auto";
    const maxHeight = Number.parseFloat(getComputedStyle(textarea).maxHeight) || 160;
    const nextHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  };

  const resizeTextareas = (root = element) => {
    root.querySelectorAll(".jc-json-table-editor textarea").forEach(resizeTextarea);
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
    if (target.tagName === "TEXTAREA") resizeTextarea(target);
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

  new MutationObserver(() => queueMicrotask(() => resizeTextareas())).observe(element, {
    childList: true,
    subtree: true,
  });

  queueMicrotask(() => {
    element.querySelectorAll(".jc-json-table-editor").forEach((root) => {
      resizeTextareas(root);
      syncSnapshot(readRows(root));
    });
  });
}
"""


UPLOAD_METADATA_JS = r"""
(imageValue, previousMetadata, loadedJsonPath, outputImage, bboxOrder, disableAutoUpdate) => {
  let metadata = "";
  try {
    const root = document.querySelector("#jc-json-preview-image");
    const input = root?.querySelector('input[type="file"]');
    const file = input?.files?.[0];
    if (file) {
      metadata = JSON.stringify({
        name: file.name,
        size: file.size,
        last_modified: file.lastModified,
      });
    }
  } catch {
    metadata = "";
  }
  const uploadStore = document.documentElement?.dataset || {};
  if (!metadata) metadata = uploadStore.jcJsonUploadMetadata || "";
  if (!imageValue) {
    metadata = "";
    uploadStore.jcJsonUploadMetadata = "";
  }
  return [imageValue, metadata, loadedJsonPath, outputImage, bboxOrder, disableAutoUpdate];
}
"""


UPLOAD_METADATA_CAPTURE_JS = r"""
const jcUploadStore = document.documentElement.dataset;
if (jcUploadStore.jcJsonUploadMetadataCaptureBound !== "1") {
  jcUploadStore.jcJsonUploadMetadataCaptureBound = "1";
  jcUploadStore.jcJsonUploadMetadata = "";
  document.addEventListener("change", (event) => {
    const target = event.target;
    if (!target?.matches?.('#jc-json-preview-image input[type="file"]')) return;
    const file = target.files?.[0];
    if (!file) {
      jcUploadStore.jcJsonUploadMetadata = "";
      return;
    }
    jcUploadStore.jcJsonUploadMetadata = JSON.stringify({
      name: file.name,
      size: file.size,
      last_modified: file.lastModified,
    });
  }, true);
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


def _pick_image_file_path(current_path: str | Path | None = None) -> tuple[str, str]:
    try:
        from tkinter import Tk, filedialog
    except Exception as exc:
        return str(current_path or ""), f"File picker is unavailable: {type(exc).__name__}: {exc}"

    current_value = str(current_path or "").strip()
    initial_dir = str(OUTPUTS_DIR if OUTPUTS_DIR.exists() else Path.cwd())

    root = None
    try:
        root = Tk()
        root.withdraw()
        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass
        file_path = filedialog.askopenfilename(
            title="Select preview image",
            initialdir=initial_dir,
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.jfif *.webp *.bmp *.gif *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
    except Exception as exc:
        return current_value, f"Failed to open file picker: {type(exc).__name__}: {exc}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    return str(Path(file_path)) if file_path else current_value, ""


def _outputs_image_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = [(OUTPUT_NOT_SELECTED_LABEL, OUTPUT_NOT_SELECTED_VALUE)]
    if not OUTPUTS_DIR.exists():
        return choices
    images = [path for path in OUTPUTS_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    images = sorted(images, key=natural_sort_key)
    for path in images:
        if path.stem.endswith("_boxed"):
            continue
        try:
            label = str(path.relative_to(OUTPUTS_DIR))
        except Exception:
            label = path.name
        choices.append((label.replace("\\", "/"), str(path)))
    return choices


def _output_dropdown_update(image_path: str | Path | None = None) -> Any:
    choices = _outputs_image_choices()
    if not image_path:
        return gr.update(choices=choices, value=OUTPUT_NOT_SELECTED_VALUE)
    path = Path(image_path)
    if path.stem.endswith("_boxed") or not _is_relative_to(path, OUTPUTS_DIR):
        return gr.update(choices=choices, value=OUTPUT_NOT_SELECTED_VALUE)
    return gr.update(choices=choices, value=str(path))


def _resolve_output_image_selection(selected_image: str | Path | None) -> Path | None:
    if not selected_image:
        return None
    value = str(selected_image).strip()
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path

    normalized = value.replace("\\", "/").lstrip("/")
    candidate = OUTPUTS_DIR / Path(normalized)
    if _is_relative_to(candidate, OUTPUTS_DIR) and candidate.exists():
        return candidate
    return path


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


def _preview_rows_snapshot_value(rows: Any, edited_indices: Any = None) -> str:
    indices: list[int] = []
    for value in edited_indices or []:
        try:
            index = int(value)
        except Exception:
            continue
        if index >= 0 and index not in indices:
            indices.append(index)
    return json.dumps({"rows": _row_data(rows), "edited_indices": indices}, ensure_ascii=False)


def _preview_rows_from_snapshot(snapshot: Any) -> tuple[list[list[Any]] | None, list[int]]:
    if not isinstance(snapshot, str) or not snapshot.strip():
        return None, []
    try:
        payload = json.loads(snapshot)
    except Exception:
        return None, []
    if isinstance(payload, dict):
        rows = payload.get("rows")
        edited_values = payload.get("edited_indices")
    else:
        rows = payload
        edited_values = range(len(rows)) if isinstance(rows, list) else []
    if not isinstance(rows, list):
        return None, []
    edited_indices: list[int] = []
    for value in edited_values or []:
        try:
            index = int(value)
        except Exception:
            continue
        if index >= 0 and index not in edited_indices:
            edited_indices.append(index)
    return _row_data(rows), edited_indices


def _merge_preview_bbox_rows(rows: Any, preview_snapshot: Any) -> list[list[Any]]:
    merged = _row_data(rows)
    preview_rows, edited_indices = _preview_rows_from_snapshot(preview_snapshot)
    if preview_rows is None or not edited_indices:
        return merged
    for index in edited_indices:
        if index >= len(merged) or index >= len(preview_rows):
            continue
        preview_row = list(preview_rows[index]) + [""] * max(0, 8 - len(preview_rows[index]))
        if not all(str(value).strip() for value in preview_row[1:5]):
            continue
        merged_row = list(merged[index]) + [""] * max(0, 8 - len(merged[index]))
        merged_row[1:5] = preview_row[1:5]
        merged[index] = merged_row[: len(merged[index]) if len(merged[index]) >= 8 else 8]
    return merged


def _cell_text(value: Any) -> str:
    return "" if value is None else str(value)


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
        "<colgroup>",
    ]
    for column_class in TABLE_COLUMN_CLASSES[: len(headers)]:
        parts.append(f'<col class="jc-json-col-{column_class}">')
    parts.extend(
        [
            "</colgroup>",
            "<thead><tr>",
        ]
    )
    for col_index, header in enumerate(headers):
        column_class = TABLE_COLUMN_CLASSES[col_index] if col_index < len(TABLE_COLUMN_CLASSES) else "default"
        parts.append(f'<th class="jc-json-cell-{column_class}">{html.escape(str(header))}</th>')
    parts.append("</tr></thead><tbody>")
    rendered = 0
    for row_index in indices:
        if row_index >= len(row_data):
            continue
        values = row_data[row_index] + [""] * max(0, len(headers) - len(row_data[row_index]))
        if not any(_cell_text(cell).strip() for cell in values):
            continue
        rendered += 1
        parts.append("<tr>")
        for col_index, value in enumerate(values[: len(headers)]):
            input_type = "number" if 1 <= col_index <= 4 else "text"
            column_class = TABLE_COLUMN_CLASSES[col_index] if col_index < len(TABLE_COLUMN_CLASSES) else "default"
            escaped_value = html.escape(_cell_text(value), quote=True)
            parts.append(f'<td class="jc-json-cell-{column_class}">')
            if col_index in WRAPPING_TABLE_COLUMNS:
                parts.append(
                    f'<textarea rows="1" data-row-index="{row_index}" '
                    f'data-col-index="{col_index}">{escaped_value}</textarea>'
                )
            else:
                parts.append(
                    f'<input type="{input_type}" value="{escaped_value}" '
                    f'data-row-index="{row_index}" data-col-index="{col_index}" />'
                )
            parts.append("</td>")
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
        if not any(_cell_text(cell).strip() for cell in values):
            continue
        label = str(values[6] or values[5] or values[7] or values[0] or "box")
        label = " ".join(label.split())
        if len(label) > 58:
            label = label[:55] + "..."
        row_type = "text" if str(values[0] or "").strip() == "text" or str(values[7] or "").strip() else (values[0] or "obj")
        choices.append(f"{index:02d} {row_type} - {label}")
    return choices


def _preserve_visible(rows: Any, selected: Any, default_all: bool = False) -> tuple[list[str], list[str]]:
    choices = _box_choices(rows)
    selected_indices = _choice_indices(selected)
    values = [choices[index] for index in selected_indices if index < len(choices)]
    if default_all and not values:
        values = choices
    return choices, values


def _default_rows() -> list[list[Any]]:
    return [DEFAULT_BUILDER_ROW.copy()]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _is_gradio_temp_path(path: str | Path | None) -> bool:
    if not path:
        return False
    return _is_relative_to(Path(path), Path(tempfile.gettempdir()) / "gradio")


def _same_file_bytes(left: str | Path | None, right: str | Path | None) -> bool:
    if not left or not right:
        return False
    left_path = Path(left)
    right_path = Path(right)
    try:
        if left_path.resolve() == right_path.resolve():
            return True
        left_stat = left_path.stat()
        right_stat = right_path.stat()
        if left_stat.st_size != right_stat.st_size:
            return False
        with left_path.open("rb") as left_handle, right_path.open("rb") as right_handle:
            while True:
                left_chunk = left_handle.read(1024 * 1024)
                right_chunk = right_handle.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError:
        return False


def _sidecar_json_for_image(image_path: str | Path | None) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    if _is_gradio_temp_path(path):
        return None
    candidate = path.with_suffix(".json")
    if candidate.exists():
        return candidate
    if path.stem.endswith("_boxed"):
        boxed_source_candidate = path.with_name(path.stem[: -len("_boxed")] + ".json")
        if boxed_source_candidate.exists():
            return boxed_source_candidate
    return None


def _json_save_path_for_image(image_path: str | Path | None, loaded_json_path: str | Path | None = None) -> Path | None:
    loaded_value = str(loaded_json_path or "").strip()
    if loaded_value:
        return Path(loaded_value)
    if not image_path:
        return None
    path = Path(image_path)
    if _is_gradio_temp_path(path):
        return None
    direct_sidecar = path.with_suffix(".json")
    if direct_sidecar.exists():
        return direct_sidecar
    if path.stem.endswith("_boxed"):
        boxed_source_sidecar = path.with_name(path.stem[: -len("_boxed")] + ".json")
        return boxed_source_sidecar
    return direct_sidecar


def _boxed_source_for_image(image_path: str | Path | None) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    if not path.stem.endswith("_boxed"):
        return path
    source_stem = path.stem[: -len("_boxed")]
    for suffix in sorted(IMAGE_EXTENSIONS):
        candidate = path.with_name(source_stem + suffix)
        if candidate.exists():
            return candidate
    return path


def _boxed_save_path_for_json(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}_boxed.png")


def _relative_output_label(path: str | Path | None) -> str:
    if not path:
        return ""
    path_value = Path(path)
    try:
        return str(path_value.relative_to(OUTPUTS_DIR)).replace("\\", "/")
    except (OSError, ValueError):
        return str(path_value)


def _output_folder_label(path: str | Path | None) -> str:
    if not path:
        return ""
    path_value = Path(path)
    folder = path_value.parent
    try:
        return str(folder.relative_to(OUTPUTS_DIR)).replace("\\", "/")
    except (OSError, ValueError):
        return str(folder)


def _loaded_output_label(image_path: str | Path | None, json_path: str | Path | None) -> str:
    folder_label = _output_folder_label(json_path or image_path)
    image_name = Path(image_path).name if image_path else ""
    json_name = Path(json_path).name if json_path else ""
    if folder_label and image_name and json_name:
        return f"Folder: {folder_label} | Image: {image_name} | JSON: {json_name}"
    if image_name:
        return f"Image: {image_name}"
    return ""


def _loaded_output_label_for_json(json_path: str | Path | None) -> str:
    if not json_path:
        return ""
    path = Path(json_path)
    return _loaded_output_label(_image_for_json_path(path), path)


def _boxed_preview_html(boxed_path: str | Path | None, json_path: str | Path | None = None) -> str:
    if not boxed_path:
        return ""
    path = Path(boxed_path)
    if not path.exists():
        return ""
    src = html.escape(_file_url(path), quote=True)
    label = html.escape(path.name)
    json_label = html.escape(_relative_output_label(json_path), quote=False) if json_path else ""
    png_label = html.escape(_relative_output_label(path), quote=False)
    source_line = f'<div class="jc-boxed-preview-source">JSON: {json_label}<br>PNG: {png_label}</div>' if json_label else f'<div class="jc-boxed-preview-source">PNG: {png_label}</div>'
    return (
        '<div class="jc-boxed-preview">'
        '<div class="jc-boxed-preview-title">Rendered Boxed PNG</div>'
        f"{source_line}"
        f'<img src="{src}" alt="{label}" />'
        "</div>"
    )


def _boxed_preview_for_json_path(json_path: str | Path | None) -> str:
    if not json_path:
        return ""
    path = Path(json_path)
    return _boxed_preview_html(_boxed_save_path_for_json(path), path)


def _render_boxed_preview_from_json(json_path: str | Path | None, rows: Any, bbox_order: str) -> tuple[str, str]:
    if not json_path:
        return "", ""
    path = Path(json_path)
    source_path = _image_for_json_path(path)
    if source_path is None:
        return "", f"Boxed PNG was not rendered because no matching image exists for {path.name}."
    try:
        boxed_path = save_boxed_image(source_path, rows, _boxed_save_path_for_json(path), bbox_order=bbox_order)
    except Exception as exc:
        return "", f"Boxed PNG render failed: {type(exc).__name__}: {exc}"
    if boxed_path is None:
        return "", "Boxed PNG was not rendered because the matching JSON has no valid boxes."
    return _boxed_preview_html(boxed_path, path), f"Boxed PNG rendered from matching JSON: {_relative_output_label(path)}."


def _is_loaded_output_image_echo(selected_image: str | Path | None, loaded_json_path: str | Path | None) -> bool:
    selected_value = str(selected_image or "").strip()
    loaded_value = str(loaded_json_path or "").strip()
    if not selected_value or not loaded_value:
        return False
    selected_path = Path(selected_value)
    loaded_path = Path(loaded_value)
    if selected_path.stem != loaded_path.stem:
        return False
    if not _is_relative_to(loaded_path, OUTPUTS_DIR):
        return False
    if _is_relative_to(selected_path, OUTPUTS_DIR):
        return False
    return _is_gradio_temp_path(selected_path)


def _upload_metadata_payload(upload_metadata: Any) -> dict[str, Any]:
    if isinstance(upload_metadata, dict):
        payload = upload_metadata
    elif isinstance(upload_metadata, str) and upload_metadata.strip():
        try:
            parsed = json.loads(upload_metadata)
        except json.JSONDecodeError:
            return {}
        payload = parsed if isinstance(parsed, dict) else {}
    else:
        return {}
    normalized = dict(payload)
    if "lastModified" in normalized and "last_modified" not in normalized:
        normalized["last_modified"] = normalized.get("lastModified")
    return normalized


def _coerce_upload_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _uploaded_output_path_from_metadata(selected_image: str | Path | None, upload_metadata: Any) -> Path | None:
    selected_value = str(selected_image or "").strip()
    if not selected_value:
        return None
    selected_path = Path(selected_value)
    if not _is_gradio_temp_path(selected_path):
        return None
    metadata = _upload_metadata_payload(upload_metadata)
    upload_name = str(metadata.get("name") or selected_path.name or "").strip()
    if not upload_name:
        return None
    upload_size = _coerce_upload_number(metadata.get("size"))
    upload_mtime_ms = _coerce_upload_number(metadata.get("last_modified"))
    matches: dict[Path, Path] = {}
    for candidate_image in OUTPUTS_DIR.rglob(upload_name):
        if not candidate_image.is_file() or candidate_image.stem.endswith("_boxed"):
            continue
        if candidate_image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not candidate_image.with_suffix(".json").exists():
            continue
        try:
            candidate_stat = candidate_image.stat()
        except OSError:
            continue
        if upload_size is not None and candidate_stat.st_size != int(upload_size):
            continue
        if not _same_file_bytes(selected_path, candidate_image):
            continue
        matches[candidate_image.resolve()] = candidate_image

    if not matches:
        return None

    if upload_mtime_ms is not None:
        timed_matches: list[tuple[float, Path]] = []
        for candidate_image in matches.values():
            try:
                delta_ms = abs((candidate_image.stat().st_mtime * 1000) - upload_mtime_ms)
            except OSError:
                continue
            if delta_ms <= 2500:
                timed_matches.append((delta_ms, candidate_image))
        if len(timed_matches) == 1:
            return timed_matches[0][1]

    if len(matches) == 1:
        return next(iter(matches.values()))
    return None


def _gradio_temp_output_echo_path(
    selected_image: str | Path | None,
    loaded_json_path: str | Path | None,
    selected_output_image: str | Path | None,
    upload_metadata: Any = None,
) -> Path | None:
    selected_value = str(selected_image or "").strip()
    if not selected_value:
        return None
    selected_path = Path(selected_value)
    if not _is_gradio_temp_path(selected_path):
        return None
    loaded_value = str(loaded_json_path or "").strip()
    if loaded_value:
        loaded_path = Path(loaded_value)
        if _is_relative_to(loaded_path, OUTPUTS_DIR):
            loaded_image_path = _image_for_json_path(loaded_path)
            if (
                loaded_image_path is not None
                and loaded_image_path.name == selected_path.name
                and _same_file_bytes(selected_path, loaded_image_path)
            ):
                return loaded_image_path
    output_path = _resolve_output_image_selection(selected_output_image)
    if output_path is not None and output_path.name == selected_path.name and _same_file_bytes(selected_path, output_path):
        return output_path
    uploaded_output_path = _uploaded_output_path_from_metadata(selected_path, upload_metadata)
    if uploaded_output_path is not None:
        return uploaded_output_path
    temp_sidecar = selected_path.with_suffix(".json")
    matches: list[Path] = []
    if temp_sidecar.exists():
        try:
            temp_json = temp_sidecar.read_text(encoding="utf-8")
        except OSError:
            temp_json = ""
        if temp_json:
            for candidate_json in OUTPUTS_DIR.rglob(f"{selected_path.stem}.json"):
                if not candidate_json.is_file():
                    continue
                try:
                    if candidate_json.read_text(encoding="utf-8") != temp_json:
                        continue
                except OSError:
                    continue
                candidate_image = _image_for_json_path(candidate_json)
                if (
                    candidate_image is not None
                    and candidate_image.name == selected_path.name
                    and _same_file_bytes(selected_path, candidate_image)
                ):
                    matches.append(candidate_image)
            unique_json_matches = {path.resolve(): path for path in matches}
            if len(unique_json_matches) == 1:
                return next(iter(unique_json_matches.values()))

    matches = []
    for candidate_image in OUTPUTS_DIR.rglob(selected_path.name):
        if not candidate_image.is_file() or candidate_image.stem.endswith("_boxed"):
            continue
        if candidate_image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if candidate_image.with_suffix(".json").exists() and _same_file_bytes(selected_path, candidate_image):
            matches.append(candidate_image)
    unique_matches = {path.resolve(): path for path in matches}
    if len(unique_matches) == 1:
        return next(iter(unique_matches.values()))
    return None


def _image_for_json_path(json_path: Path) -> Path | None:
    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = json_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def _builder_image_filename(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return "image.png"
    return image_path.name or f"image{suffix}"


def _builder_save_paths(
    image_path: str | Path | None,
    loaded_json_path: str | Path | None = None,
) -> tuple[Path, Path | None, Path | None]:
    loaded_value = str(loaded_json_path or "").strip()
    if loaded_value:
        json_path = Path(loaded_value)
        return json_path, _image_for_json_path(json_path), None

    if image_path:
        path = Path(image_path)
        if _is_relative_to(path, OUTPUTS_DIR):
            json_path = _json_save_path_for_image(path)
            if json_path is not None:
                return json_path, path if path.exists() else _image_for_json_path(json_path), None

        sidecar = _sidecar_json_for_image(path)
        if sidecar is not None:
            return sidecar, path if path.exists() else _image_for_json_path(sidecar), None

        run_dir = next_numbered_output_dir(OUTPUTS_DIR)
        output_image_path: Path | None = None
        if path.exists():
            output_image_path = run_dir / _builder_image_filename(path)
            copy_image_if_needed(path, output_image_path, True)
        json_path = output_image_path.with_suffix(".json") if output_image_path is not None else run_dir / "image.json"
        return json_path, output_image_path, output_image_path

    run_dir = next_numbered_output_dir(OUTPUTS_DIR)
    return run_dir / "image.json", None, None


def _update_output_metadata_for_save(
    json_path: Path,
    caption_text: str,
    boxed_image_path: Path | None,
    output_image_path: Path | None,
) -> str:
    metadata_path = json_path.parent / "metadata.json"
    if not metadata_path.exists() and not _is_relative_to(json_path, OUTPUTS_DIR):
        return ""
    try:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {
                "generation_type": "json_prompt_builder",
                "engine": "json_prompt_builder",
                "source_image_path": str(output_image_path) if output_image_path else None,
            }
        if not isinstance(metadata, dict):
            return "Metadata was not updated because metadata.json is not an object."
        metadata.update(
            {
                "caption_final": caption_text,
                "box_edits_autosaved": True,
                "box_edits_autosaved_at": datetime.now(timezone.utc).isoformat(),
                "caption_path": str(json_path),
                "output_run_dir": str(json_path.parent),
            }
        )
        if output_image_path is not None:
            metadata["output_image_path"] = str(output_image_path)
        if boxed_image_path is not None:
            metadata["boxed_image_path"] = str(boxed_image_path)
        write_generation_metadata(metadata_path, metadata)
    except Exception as exc:
        return f"Metadata update skipped: {type(exc).__name__}: {exc}"
    return ""


def _save_json_builder_edits(
    image_path: str | Path | None,
    loaded_json_path: str | Path | None,
    caption_text: str,
    rows: Any,
    bbox_order: str,
) -> tuple[str, str]:
    try:
        json.loads(caption_text)
    except Exception as exc:
        return (
            html_message(
                "error",
                "JSON box edits applied in the UI, but save was blocked because the edited JSON does not parse: "
                + html.escape(f"{type(exc).__name__}: {exc}"),
            ),
            str(loaded_json_path or ""),
        )

    boxed_note = ""
    metadata_note = ""
    boxed_image_path: Path | None = None
    try:
        json_path, output_image_path, copied_image_path = _builder_save_paths(image_path, loaded_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(caption_text, encoding="utf-8")
        source_path = output_image_path if output_image_path and output_image_path.exists() else _boxed_source_for_image(image_path)
        if source_path is None:
            boxed_note = "Boxed PNG was not updated because the source image path is unavailable."
        else:
            boxed_image_path = save_boxed_image(source_path, rows, _boxed_save_path_for_json(json_path), bbox_order=bbox_order)
            if boxed_image_path is None:
                boxed_note = "Boxed PNG was not updated because there are no valid boxes to render."
            else:
                boxed_note = f"Boxed PNG saved to {boxed_image_path}."
        metadata_note = _update_output_metadata_for_save(
            json_path,
            caption_text,
            boxed_image_path,
            copied_image_path or output_image_path,
        )
    except Exception as exc:
        return (
            html_message(
                "error",
                "JSON box edits applied in the UI, but save failed: " + html.escape(f"{type(exc).__name__}: {exc}"),
            ),
            str(loaded_json_path or ""),
        )

    message = "JSON box edits applied and saved to " + html.escape(str(json_path)) + "."
    if boxed_note:
        message += "<br>" + html.escape(boxed_note)
    if metadata_note:
        message += "<br>" + html.escape(metadata_note)
    return html_message("success", message), str(json_path)


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
    preview_rows_snapshot = gr.Textbox(
        value=_preview_rows_snapshot_value([DEFAULT_BUILDER_ROW.copy()]),
        label="Preview Rows Snapshot",
        elem_id="jc-json-preview-rows-snapshot",
        elem_classes=["jc-hidden-sync"],
        interactive=True,
    )
    loaded_json_path = gr.State("")
    upload_metadata = gr.Textbox(
        value="",
        label="Upload Metadata",
        elem_id="jc-json-upload-metadata",
        elem_classes=["jc-hidden-sync"],
        interactive=True,
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes=["jc-compact"]):
            gr.HTML('<span data-jc-upload-metadata-capture="1"></span>', elem_classes=["jc-hidden-sync"])
            image = gr.Image(type="filepath", label="Optional Preview Image", height=390, elem_id="jc-json-preview-image")
            pick_image_btn = gr.Button("Browse File", elem_classes=["btn-refresh"])
            with gr.Accordion("Load / Continue From Outputs", open=True):
                output_image = gr.Dropdown(
                    choices=_outputs_image_choices(),
                    value=OUTPUT_NOT_SELECTED_VALUE,
                    label="Output Image",
                    allow_custom_value=False,
                )
                loaded_output_label = gr.Textbox(
                    label="Loaded Output",
                    value="",
                    interactive=False,
                    elem_classes=["jc-loaded-output"],
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
            boxed_preview = gr.HTML("", elem_classes=["jc-json-boxed-preview"])

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
                    apply_box_btn = gr.Button("Apply Box Edits & Save", elem_classes=["btn-qwen-apply"])
                    add_row_btn = gr.Button("Add Box", elem_classes=["btn-json-add"])
                    clear_rows_btn = gr.Button("Clear Boxes", elem_classes=["btn-reset-preset"])
                    import_btn = gr.Button("Import JSON Rows", elem_classes=["btn-load-preset"])
                json_output = gr.Textbox(
                    label="Generated Ideogram JSON",
                    lines=18,
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
        rows,
        bbox_order_value,
    ):
        return build_ideogram_json(
            _ratio(ratio_value, width_value, height_value),
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
            compact=False,
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
            merged_rows,
            bbox_order_value,
        )
        preview = _overlay(image_path, ratio_value, width_value, height_value, merged_rows, bbox_order_value, visible, disable_auto_update_value)
        return (
            json_text,
            merged_rows,
            _rows_snapshot_value(merged_rows),
            _preview_rows_snapshot_value(merged_rows),
            _table_editor_html(merged_rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            preview,
            "",
            "",
        )

    def preview_boxes(image_path, ratio_value, width_value, height_value, all_rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        merged_rows = _rows_from_snapshot(snapshot_value, all_rows)
        choices, visible = _preserve_visible(merged_rows, visible_choices, default_all=True)
        return (
            merged_rows,
            _rows_snapshot_value(merged_rows),
            _preview_rows_snapshot_value(merged_rows),
            _table_editor_html(merged_rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            _overlay(image_path, ratio_value, width_value, height_value, merged_rows, bbox_order_value, visible, disable_auto_update_value),
        )

    def apply_box_edits(
        image_path,
        ratio_value,
        width_value,
        height_value,
        json_text,
        all_rows,
        snapshot_value,
        preview_snapshot,
        loaded_json_path_value,
        bbox_order_value,
        visible_choices,
        disable_auto_update_value,
    ):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        merged_rows = _rows_from_snapshot(snapshot_value, all_rows)
        merged_rows = _merge_preview_bbox_rows(merged_rows, preview_snapshot)
        source_json = str(json_text or "").strip()
        if not source_json:
            source_json = json.dumps(
                {
                    "aspect_ratio": _ratio(ratio_value, width_value, height_value),
                    "high_level_description": "",
                    "compositional_deconstruction": {"background": "", "elements": []},
                },
                ensure_ascii=False,
                indent=2,
            )
        final, parsed, warnings = apply_rows_to_json(source_json, merged_rows, bbox_order=bbox_order_value)
        display_rows = _row_data(merged_rows)
        choices, visible = _preserve_visible(display_rows, visible_choices, default_all=True)
        save_status, next_loaded_json_path = _save_json_builder_edits(
            image_path,
            loaded_json_path_value,
            final,
            display_rows,
            bbox_order_value,
        )
        status_html = save_status
        if warnings:
            status_html = (
                html_message("info", "Applied box edits after JSON repair fallback:<br><pre>" + html.escape("\n".join(warnings)) + "</pre>")
                + save_status
            )
        return (
            final,
            display_rows,
            _rows_snapshot_value(display_rows),
            _preview_rows_snapshot_value(display_rows),
            _table_editor_html(display_rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            _overlay(image_path, ratio_value, width_value, height_value, display_rows, bbox_order_value, visible, disable_auto_update_value),
            _boxed_preview_for_json_path(next_loaded_json_path),
            next_loaded_json_path,
            _loaded_output_label_for_json(next_loaded_json_path),
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
            _preview_rows_snapshot_value(next_rows),
            _table_editor_html(next_rows, selected, bbox_order_value),
            gr.update(choices=choices, value=selected),
            _overlay(image_path, ratio_value, width_value, height_value, next_rows, bbox_order_value, selected, disable_auto_update_value),
        )

    def clear_rows(bbox_order_value):
        return [], _rows_snapshot_value([]), _preview_rows_snapshot_value([]), _table_editor_html([], [], bbox_order_value), gr.update(choices=[], value=[])

    def import_json(json_text, image_path, ratio_value, width_value, height_value, bbox_order_value, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        pretty, parsed, warnings = normalize_json_output(json_text, preset_id=BUILDER_PRESET_ID, compact=False)
        if parsed is None:
            empty_overlay = _overlay(image_path, ratio_value, width_value, height_value, [], bbox_order_value, [], disable_auto_update_value)
            return (
                [],
                _rows_snapshot_value([]),
                _preview_rows_snapshot_value([]),
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
        rows = json_to_element_rows(parsed, bbox_order=bbox_order_value)
        choices, visible = _preserve_visible(rows, [], default_all=True)
        fields = _fields_from_json(parsed)
        status_html = html_message("success", "JSON imported.")
        if warnings:
            status_html = html_message("info", "JSON imported with warnings:<br><pre>" + "\n".join(warnings) + "</pre>")
        return (
            rows,
            _rows_snapshot_value(rows),
            _preview_rows_snapshot_value(rows),
            _table_editor_html(rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            pretty,
            *fields,
            _overlay(image_path, ratio_value, width_value, height_value, rows, bbox_order_value, visible, disable_auto_update_value),
            "",
            status_html,
        )

    def fresh_empty_output_state(bbox_order_value, message):
        rows = _default_rows()
        choices, visible = _preserve_visible(rows, [], default_all=True)
        return (
            None,
            _output_dropdown_update(),
            "",
            "",
            gr.update(value="1:1"),
            gr.update(value=1000),
            gr.update(value=1000),
            "",
            rows,
            _rows_snapshot_value(rows),
            _preview_rows_snapshot_value(rows),
            _table_editor_html(rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            "",
            "Photograph",
            "",
            "",
            "",
            "illustration",
            "",
            "",
            "",
            "",
            "",
            message,
        )

    def load_output_choice(selected_image, bbox_order_value, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        if not selected_image:
            return fresh_empty_output_state(bbox_order_value, html_message("info", "No output selected."))
        image_path = _resolve_output_image_selection(selected_image)
        if image_path is None:
            return fresh_empty_output_state(bbox_order_value, html_message("info", "No output selected."))
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
                _output_dropdown_update(image_path),
                str(_json_save_path_for_image(image_path) or ""),
                _loaded_output_label(image_path, _json_save_path_for_image(image_path)),
                ratio_update,
                width_update,
                height_update,
                "",
                [],
                _rows_snapshot_value([]),
                _preview_rows_snapshot_value([]),
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
                "",
                html_message("info", f"Loaded image, but no same-name JSON was found: {image_path.with_suffix('.json')}"),
            )
        text = sidecar.read_text(encoding="utf-8")
        pretty, parsed, warnings = normalize_json_output(text, preset_id=BUILDER_PRESET_ID, compact=False)
        if parsed is None:
            empty_overlay = _overlay(image_path, ratio_for_overlay, width_for_overlay, height_for_overlay, [], bbox_order_value, [], disable_auto_update_value)
            return (
                str(image_path),
                _output_dropdown_update(image_path),
                str(sidecar),
                _loaded_output_label(image_path, sidecar),
                ratio_update,
                width_update,
                height_update,
                text,
                [],
                _rows_snapshot_value([]),
                _preview_rows_snapshot_value([]),
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
                "",
                html_message("error", "Loaded image, but same-name JSON could not be parsed.<br><pre>" + "\n".join(warnings) + "</pre>"),
            )
        rows = json_to_element_rows(parsed, bbox_order=bbox_order_value)
        choices, visible = _preserve_visible(rows, [], default_all=True)
        fields = _fields_from_json(parsed)
        boxed_html, boxed_note = _render_boxed_preview_from_json(sidecar, rows, bbox_order_value)
        status_message = (
            "Loaded output "
            + html.escape(_loaded_output_label(image_path, sidecar))
            + "."
        )
        if boxed_note:
            status_message += "<br>" + html.escape(boxed_note)
        status_html = html_message("success", status_message)
        return (
            str(image_path),
            _output_dropdown_update(image_path),
            str(sidecar),
            _loaded_output_label(image_path, sidecar),
            ratio_update,
            width_update,
            height_update,
            pretty,
            rows,
            _rows_snapshot_value(rows),
            _preview_rows_snapshot_value(rows),
            _table_editor_html(rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            *fields,
            _overlay(image_path, ratio_for_overlay, width_for_overlay, height_for_overlay, rows, bbox_order_value, visible, disable_auto_update_value),
            boxed_html,
            status_html,
        )

    def fresh_preview_state(selected_image, bbox_order_value, disable_auto_update_value, message):
        rows = _default_rows()
        choices, visible = _preserve_visible(rows, [], default_all=True)
        if selected_image:
            ratio_update, width_update, height_update = _aspect_controls_for_image(selected_image)
            ratio_value = ratio_update.get("value", "1:1") if isinstance(ratio_update, dict) else "1:1"
            width_value = width_update.get("value", 1000) if isinstance(width_update, dict) else 1000
            height_value = height_update.get("value", 1000) if isinstance(height_update, dict) else 1000
            overlay_value = _overlay(
                selected_image,
                ratio_value,
                width_value,
                height_value,
                rows,
                bbox_order_value,
                visible,
                disable_auto_update_value,
            )
        else:
            ratio_update = gr.update(value="1:1")
            width_update = gr.update(value=1000)
            height_update = gr.update(value=1000)
            overlay_value = ""
        return (
            _output_dropdown_update(selected_image),
            "",
            f"Unsaved preview image: {Path(selected_image).name}" if selected_image else "",
            ratio_update,
            width_update,
            height_update,
            "",
            rows,
            _rows_snapshot_value(rows),
            _preview_rows_snapshot_value(rows),
            _table_editor_html(rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            "",
            "Photograph",
            "",
            "",
            "",
            "illustration",
            "",
            "",
            "",
            overlay_value,
            "",
            message,
        )

    def load_preview_image_choice(selected_image, upload_metadata_value, current_loaded_json_path, current_output_image, bbox_order_value, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        if not selected_image:
            return fresh_preview_state(
                None,
                bbox_order_value,
                disable_auto_update_value,
                html_message("info", "Preview image cleared."),
            )
        sidecar = _sidecar_json_for_image(selected_image)
        loaded_json_value = str(current_loaded_json_path or "").strip()
        echo_output_path = _gradio_temp_output_echo_path(selected_image, loaded_json_value, current_output_image, upload_metadata_value)
        if echo_output_path is not None:
            return load_output_choice(echo_output_path, bbox_order_value, disable_auto_update_value)[1:]
        if _is_loaded_output_image_echo(selected_image, loaded_json_value):
            return tuple(gr.update() for _component in load_outputs[1:])
        if sidecar is None and loaded_json_value and _is_relative_to(Path(loaded_json_value), OUTPUTS_DIR):
            selected_text = str(selected_image or "")
            loaded_stem = Path(loaded_json_value).stem
            selected_stem = Path(selected_text).stem
            if loaded_stem and (selected_stem == loaded_stem or loaded_stem in selected_text):
                return tuple(gr.update() for _component in load_outputs[1:])
        if sidecar is None:
            return fresh_preview_state(
                selected_image,
                bbox_order_value,
                disable_auto_update_value,
                html_message("info", "Loaded preview image. Apply Box Edits & Save will create a new output folder."),
            )
        return load_output_choice(selected_image, bbox_order_value, disable_auto_update_value)[1:]

    def browse_preview_image(current_image, bbox_order_value, disable_auto_update_value):
        picked_path, error = _pick_image_file_path(current_image)
        if error:
            return tuple(gr.update() for _component in load_outputs[:-1]) + (html_message("error", error),)
        if not picked_path:
            return tuple(gr.update() for _component in load_outputs[:-1]) + (html_message("info", "File pick cancelled."),)

        picked = Path(picked_path)
        if picked.stem.endswith("_boxed"):
            source = _boxed_source_for_image(picked)
            if source is not None and source.exists():
                picked = source
        if not picked.exists() or picked.suffix.lower() not in IMAGE_EXTENSIONS:
            return tuple(gr.update() for _component in load_outputs[:-1]) + (
                html_message("error", f"Select a valid image file: {html.escape(str(picked))}"),
            )

        sidecar = _sidecar_json_for_image(picked)
        if sidecar is None:
            return (
                str(picked),
                *fresh_preview_state(
                    str(picked),
                    clean_bbox_order(bbox_order_value),
                    disable_auto_update_value,
                    html_message("info", "Picked preview image. Apply Box Edits & Save will create a new output folder."),
                ),
            )
        return load_output_choice(str(picked), bbox_order_value, disable_auto_update_value)

    def update_box_visibility(image_path, ratio_value, width_value, height_value, rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        current_rows = _rows_from_snapshot(snapshot_value, rows)
        return (
            _preview_rows_snapshot_value(current_rows),
            _table_editor_html(current_rows, visible_choices, bbox_order_value),
            _overlay(image_path, ratio_value, width_value, height_value, current_rows, bbox_order_value, visible_choices, disable_auto_update_value),
        )

    def update_auto_update_setting(image_path, ratio_value, width_value, height_value, rows, snapshot_value, bbox_order_value, visible_choices, disable_auto_update_value):
        current_rows = _rows_from_snapshot(snapshot_value, rows)
        return (
            _preview_rows_snapshot_value(current_rows),
            _overlay(image_path, ratio_value, width_value, height_value, current_rows, bbox_order_value, visible_choices, disable_auto_update_value),
        )

    def apply_overlay_edit(
        image_path,
        ratio_value,
        width_value,
        height_value,
        json_text,
        all_rows,
        snapshot_value,
        bbox_order_value,
        visible_choices,
        disable_auto_update_value,
        evt: gr.EventData,
    ):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        if bool(disable_auto_update_value):
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        payload = getattr(evt, "_data", {}) or {}
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        edited_index = payload.get("index") if isinstance(payload, dict) else None
        merged_rows = _rows_from_snapshot(snapshot_value, all_rows)
        merged_rows = _merge_preview_bbox_rows(merged_rows, _preview_rows_snapshot_value(rows, [edited_index]))
        final, parsed, warnings = apply_rows_to_json(json_text, merged_rows, bbox_order=bbox_order_value)
        display_rows = _row_data(merged_rows)
        choices, visible = _preserve_visible(display_rows, visible_choices, default_all=True)
        status_html = "" if not warnings else html_message("info", "JSON repaired while applying overlay edit:<br><pre>" + html.escape("\n".join(warnings)) + "</pre>")
        return (
            final,
            display_rows,
            _rows_snapshot_value(display_rows),
            _preview_rows_snapshot_value(display_rows),
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
            _preview_rows_snapshot_value(normalized_rows),
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
        outputs=[json_output, all_element_rows, rows_snapshot, preview_rows_snapshot, element_rows, box_visibility, overlay, boxed_preview, status],
    )
    preview_btn.click(
        preview_boxes,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[all_element_rows, rows_snapshot, preview_rows_snapshot, element_rows, box_visibility, overlay],
        queue=False,
    )
    apply_box_btn.click(
        apply_box_edits,
        inputs=[
            image,
            aspect_ratio,
            canvas_width,
            canvas_height,
            json_output,
            all_element_rows,
            rows_snapshot,
            preview_rows_snapshot,
            loaded_json_path,
            bbox_order,
            box_visibility,
            disable_auto_update,
        ],
        outputs=[
            json_output,
            all_element_rows,
            rows_snapshot,
            preview_rows_snapshot,
            element_rows,
            box_visibility,
            overlay,
            boxed_preview,
            loaded_json_path,
            loaded_output_label,
            status,
        ],
    )
    add_row_btn.click(
        add_row,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[all_element_rows, rows_snapshot, preview_rows_snapshot, element_rows, box_visibility, overlay],
        queue=False,
    )
    clear_rows_btn.click(clear_rows, inputs=[bbox_order], outputs=[all_element_rows, rows_snapshot, preview_rows_snapshot, element_rows, box_visibility], queue=False).then(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, overlay],
        queue=False,
    )
    import_btn.click(
        import_json,
        inputs=[json_output, image, aspect_ratio, canvas_width, canvas_height, bbox_order, disable_auto_update],
        outputs=[
            all_element_rows,
            rows_snapshot,
            preview_rows_snapshot,
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
            boxed_preview,
            status,
        ],
        queue=False,
    )
    box_visibility.change(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, overlay],
        queue=False,
    )
    bbox_order.change(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, overlay],
        queue=False,
    )
    disable_auto_update.change(
        update_auto_update_setting,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, overlay],
        queue=False,
    )
    json_overlay_inputs = [
        image,
        aspect_ratio,
        canvas_width,
        canvas_height,
        json_output,
        all_element_rows,
        rows_snapshot,
        bbox_order,
        box_visibility,
        disable_auto_update,
    ]
    overlay.click(
        apply_overlay_edit,
        inputs=json_overlay_inputs,
        outputs=[json_output, all_element_rows, rows_snapshot, preview_rows_snapshot, element_rows, box_visibility, overlay, status],
        queue=False,
    )
    element_rows.click(
        apply_table_edit,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, bbox_order, box_visibility, disable_auto_update],
        outputs=[all_element_rows, rows_snapshot, preview_rows_snapshot, element_rows, box_visibility, overlay],
        queue=False,
    )
    check_all_boxes_btn.click(check_all_boxes, inputs=[all_element_rows, rows_snapshot], outputs=box_visibility, queue=False).then(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, overlay],
        queue=False,
    )
    uncheck_all_boxes_btn.click(uncheck_all_boxes, inputs=[all_element_rows, rows_snapshot], outputs=box_visibility, queue=False).then(
        update_box_visibility,
        inputs=[image, aspect_ratio, canvas_width, canvas_height, all_element_rows, rows_snapshot, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, overlay],
        queue=False,
    )
    refresh_outputs_btn.click(refresh_output_choices, outputs=output_image, queue=False)
    load_outputs = [
        image,
        output_image,
        loaded_json_path,
        loaded_output_label,
        aspect_ratio,
        canvas_width,
        canvas_height,
        json_output,
        all_element_rows,
        rows_snapshot,
        preview_rows_snapshot,
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
        boxed_preview,
        status,
    ]
    pick_image_btn.click(
        browse_preview_image,
        inputs=[image, bbox_order, disable_auto_update],
        outputs=load_outputs,
        queue=False,
    )
    output_image.change(load_output_choice, inputs=[output_image, bbox_order, disable_auto_update], outputs=load_outputs, queue=False)
    load_output_btn.click(load_output_choice, inputs=[output_image, bbox_order, disable_auto_update], outputs=load_outputs, queue=False)
    image.change(
        load_preview_image_choice,
        inputs=[image, upload_metadata, loaded_json_path, output_image, bbox_order, disable_auto_update],
        outputs=load_outputs[1:],
        queue=False,
        js=UPLOAD_METADATA_JS,
    )

    return TabUI(key="json_builder", order=[], defaults={}, inputs=[])
