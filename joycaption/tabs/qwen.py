from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from ..attention import ATTENTION_BACKEND_CHOICES, DEFAULT_QWEN_ATTENTION
from ..common import IMAGE_EXTENSIONS, OUTPUTS_DIR, html_message
from ..json_tools import (
    EMPTY_ELEMENT_ROW,
    apply_rows_to_json,
    clean_bbox_order,
    headers_for_bbox_order,
    json_to_element_rows,
    normalize_json_output,
    overlay_html,
    save_boxed_image,
)
from ..qwen_presets import default_qwen_preset_id, preset_payload, qwen_preset_choices
from ..model_catalog import DEFAULT_QWEN_MODEL_KEY, get_model_spec, qwen_model_choices
from ..torch_compile import DEFAULT_COMPILE_SETTINGS
from ..vram import VRAM_PRESET_CHOICES, default_vram_preset, qwen_vram_settings
from .shared import (
    TabUI,
    build_replace_pair_controls,
    build_torch_compile_controls,
    run_open_folder,
    run_open_outputs,
    settings_from_values,
)


DEFAULT_PRESET_ID = default_qwen_preset_id()
DEFAULT_VRAM = default_vram_preset()
DEFAULT_PAYLOAD = preset_payload(DEFAULT_PRESET_ID)
DEFAULT_VRAM_SETTINGS = qwen_vram_settings(DEFAULT_VRAM)
DEFAULT_BBOX_ORDER = "yxyx"
BBOX_ORDER_CHOICES = [
    ("y_min, x_min, y_max, x_max", "yxyx"),
    ("x_min, y_min, x_max, y_max", "xyxy"),
]
ELEMENT_COLUMN_WIDTHS = [
    "68px",
    "70px",
    "70px",
    "70px",
    "70px",
    "480px",
    "108px",
    "300px",
]

ORDER = [
    "model_key",
    "vram_preset",
    "preset_id",
    "output_format",
    "extension",
    "beautify_saved_json",
    "system_prompt",
    "prompt",
    "trigger_phrase",
    "output_language",
    "caption_length",
    "dataset_goal",
    "known_subject_class",
    "brand_policy",
    "text_policy",
    "bbox_policy",
    "existing_caption",
    "existing_json_caption",
    "validation_error",
    "invalid_json_text",
    "ideogram_json_caption",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "image_long_edge",
    "model_quantization",
    "unload_model",
    "save_image",
    "use_subprocess",
    "allow_tf32",
    "clear_cuda_cache",
    "low_cpu_mem_usage",
    "attention_backend",
    "enable_thinking",
    "torch_compile",
    "compile_backend",
    "compile_mode",
    "compile_dynamic",
    "compile_fullgraph",
    "compile_cache_size_limit",
    "compile_threads",
    "compact_json",
    "json_retries",
    "remove_newlines",
    "auto_save_boxed_image",
    "caption_prefix",
    "caption_suffix",
    "replace_pairs",
    "replace_case_sensitive",
    "replace_single_word",
    "device_id",
    "file_batch_size",
    "folder_input",
    "folder_output",
    "overwrite_caption",
    "append_caption",
    "process_subfolders",
    "dont_save_boxed_images",
    "folder_batch_size",
    "app_side_only",
]

DEFAULTS: dict[str, Any] = {
    "model_key": DEFAULT_QWEN_MODEL_KEY,
    "vram_preset": DEFAULT_VRAM,
    "preset_id": DEFAULT_PRESET_ID,
    "output_format": DEFAULT_PAYLOAD["output_format"],
    "extension": DEFAULT_PAYLOAD["extension"],
    "beautify_saved_json": True,
    "system_prompt": DEFAULT_PAYLOAD["system_prompt"],
    "prompt": DEFAULT_PAYLOAD["prompt"],
    "trigger_phrase": "",
    "output_language": "English",
    "caption_length": "",
    "dataset_goal": "",
    "known_subject_class": "",
    "brand_policy": "Name only if visually certain; never guess.",
    "text_policy": "Copy exact readable text only; mark unclear text unreadable.",
    "bbox_policy": "Major localizable objects and readable text only.",
    "existing_caption": "",
    "existing_json_caption": "",
    "validation_error": "",
    "invalid_json_text": "",
    "ideogram_json_caption": "",
    "temperature": DEFAULT_PAYLOAD["temperature"],
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "max_new_tokens": DEFAULT_PAYLOAD["max_new_tokens"],
    "image_long_edge": min(DEFAULT_PAYLOAD["image_long_edge"], DEFAULT_VRAM_SETTINGS["image_long_edge"]),
    "model_quantization": DEFAULT_VRAM_SETTINGS["model_quantization"],
    "unload_model": False,
    "save_image": True,
    "use_subprocess": False,
    "allow_tf32": True,
    "clear_cuda_cache": True,
    "low_cpu_mem_usage": True,
    "attention_backend": DEFAULT_VRAM_SETTINGS.get("attention_backend", DEFAULT_QWEN_ATTENTION),
    "enable_thinking": False,
    **DEFAULT_COMPILE_SETTINGS,
    "compact_json": True,
    "json_retries": 1,
    "remove_newlines": False,
    "auto_save_boxed_image": True,
    "caption_prefix": "",
    "caption_suffix": "",
    "replace_pairs": [],
    "replace_case_sensitive": False,
    "replace_single_word": False,
    "device_id": "0",
    "file_batch_size": DEFAULT_VRAM_SETTINGS["file_batch_size"],
    "folder_input": "",
    "folder_output": "",
    "overwrite_caption": False,
    "append_caption": False,
    "process_subfolders": False,
    "dont_save_boxed_images": False,
    "folder_batch_size": DEFAULT_VRAM_SETTINGS["folder_batch_size"],
    "app_side_only": DEFAULT_PAYLOAD["app_side_only"],
}


OVERLAY_EDIT_JS = r"""
if (!element.dataset.jcQwenOverlayBound) {
  element.dataset.jcQwenOverlayBound = "1";

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  const rowsForFrame = (frame) => {
    try {
      const rows = JSON.parse(frame.dataset.rows || "[]");
      return Array.isArray(rows) ? rows : [];
    } catch {
      return [];
    }
  };

  const editedIndicesForFrame = (frame) => {
    try {
      const indices = JSON.parse(frame.dataset.previewEditedIndices || "[]");
      return Array.isArray(indices) ? indices.map(Number).filter(Number.isFinite) : [];
    } catch {
      return [];
    }
  };

  const syncPreviewSnapshot = (frame) => {
    const target = document.querySelector("#jc-qwen-preview-rows-snapshot textarea, #jc-qwen-preview-rows-snapshot input");
    if (!target) return;
    const value = JSON.stringify({
      rows: rowsForFrame(frame),
      edited_indices: editedIndicesForFrame(frame),
    });
    if (target.value === value) return;
    const proto = target.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (setter) setter.call(target, value);
    else target.value = value;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    target.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const frameRect = (frame) => {
    const img = frame.querySelector(".jc-overlay-image");
    const rect = (img && img.naturalWidth > 0) ? img.getBoundingClientRect() : frame.getBoundingClientRect();
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
  };

  const selectBox = (box) => {
    const frame = box.closest(".jc-overlay-frame");
    if (!frame) return;
    frame.querySelectorAll(".jc-box.is-selected").forEach((item) => item.classList.remove("is-selected"));
    box.classList.add("is-selected");
  };

  const applyBoxRect = (box, rect, surface) => {
    box.style.left = `${(rect.left / surface.width) * 100}%`;
    box.style.top = `${(rect.top / surface.height) * 100}%`;
    box.style.width = `${(rect.width / surface.width) * 100}%`;
    box.style.height = `${(rect.height / surface.height) * 100}%`;
  };

  const boxRect = (box, surface) => {
    const rect = box.getBoundingClientRect();
    return {
      left: rect.left - surface.left,
      top: rect.top - surface.top,
      width: rect.width,
      height: rect.height,
    };
  };

  const commitFrame = (frame, activeIndex) => {
    const surface = frameRect(frame);
    if (!surface.width || !surface.height) return;
    const rows = rowsForFrame(frame);
    const activeBox = frame.querySelector(`.jc-box[data-row-index="${activeIndex}"]`);
    if (!activeBox || !rows[activeIndex]) return;
    const rect = boxRect(activeBox, surface);
    const xMin = clamp(Math.round((rect.left / surface.width) * 1000), 0, 999);
    const yMin = clamp(Math.round((rect.top / surface.height) * 1000), 0, 999);
    const xMax = clamp(Math.round(((rect.left + rect.width) / surface.width) * 1000), xMin + 1, 1000);
    const yMax = clamp(Math.round(((rect.top + rect.height) / surface.height) * 1000), yMin + 1, 1000);
    const bboxOrder = (frame.dataset.bboxOrder || "yxyx").toLowerCase();
    if (bboxOrder === "xyxy") {
      rows[activeIndex][1] = xMin;
      rows[activeIndex][2] = yMin;
      rows[activeIndex][3] = xMax;
      rows[activeIndex][4] = yMax;
    } else {
      rows[activeIndex][1] = yMin;
      rows[activeIndex][2] = xMin;
      rows[activeIndex][3] = yMax;
      rows[activeIndex][4] = xMax;
    }
    frame.dataset.rows = JSON.stringify(rows);
    const edited = new Set(editedIndicesForFrame(frame));
    edited.add(activeIndex);
    frame.dataset.previewEditedIndices = JSON.stringify([...edited]);
    syncPreviewSnapshot(frame);
    if (frame.dataset.disableAutoUpdate !== "1") {
      trigger("click", { action: "box-edit", rows, index: activeIndex });
    }
  };

  const bindFrame = (frame) => {
    if (frame.dataset.jcQwenFrameBound || !frame.closest(".jc-overlay-interactive")) return;
    frame.dataset.jcQwenFrameBound = "1";

    frame.addEventListener("pointerdown", (event) => {
      const box = event.target.closest(".jc-box");
      if (!box || !frame.contains(box)) return;
      event.preventDefault();
      event.stopPropagation();
      selectBox(box);

      const surface = frameRect(frame);
      if (!surface.width || !surface.height) return;
      const start = boxRect(box, surface);
      const handle = event.target.dataset.handle || "move";
      const startPointer = { x: event.clientX, y: event.clientY };
      const minSize = Math.max(12, Math.min(surface.width, surface.height) * 0.015);
      const activeIndex = Number(box.dataset.rowIndex);
      let changed = false;

      box.setPointerCapture?.(event.pointerId);
      box.classList.add("is-editing");

      const move = (moveEvent) => {
        const dx = moveEvent.clientX - startPointer.x;
        const dy = moveEvent.clientY - startPointer.y;
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) changed = true;

        let left = start.left;
        let top = start.top;
        let width = start.width;
        let height = start.height;

        if (handle === "move") {
          left = clamp(start.left + dx, 0, surface.width - width);
          top = clamp(start.top + dy, 0, surface.height - height);
        } else {
          if (handle.includes("e")) width = clamp(start.width + dx, minSize, surface.width - start.left);
          if (handle.includes("s")) height = clamp(start.height + dy, minSize, surface.height - start.top);
          if (handle.includes("w")) {
            const nextLeft = clamp(start.left + dx, 0, start.left + start.width - minSize);
            width = start.left + start.width - nextLeft;
            left = nextLeft;
          }
          if (handle.includes("n")) {
            const nextTop = clamp(start.top + dy, 0, start.top + start.height - minSize);
            height = start.top + start.height - nextTop;
            top = nextTop;
          }
        }

        applyBoxRect(box, { left, top, width, height }, surface);
      };

      const up = () => {
        box.classList.remove("is-editing");
        box.releasePointerCapture?.(event.pointerId);
        window.removeEventListener("pointermove", move);
        window.removeEventListener("mousemove", move);
        window.removeEventListener("pointerup", up);
        window.removeEventListener("mouseup", up);
        if (changed) {
          commitFrame(frame, activeIndex);
        }
      };

      window.addEventListener("pointermove", move);
      window.addEventListener("mousemove", move);
      window.addEventListener("pointerup", up, { once: true });
      window.addEventListener("mouseup", up, { once: true });
    });
  };

  const install = () => {
    element.querySelectorAll(".jc-overlay-frame").forEach((frame) => {
      bindFrame(frame);
      syncPreviewSnapshot(frame);
    });
  };

  watch("value", () => queueMicrotask(install));
  queueMicrotask(install);
}
"""


COPY_JSON_HTML = """
<button type="button" class="jc-copy-json-button" title="Copy JSON" aria-label="Copy JSON">
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="9" y="9" width="11" height="11" rx="2"></rect>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
  </svg>
  <span class="jc-copy-json-label">Copy JSON</span>
</button>
"""


COPY_JSON_JS = r"""
if (!element.dataset.jcCopyJsonBound) {
  element.dataset.jcCopyJsonBound = "1";

  const button = element.querySelector(".jc-copy-json-button");
  const label = button?.querySelector(".jc-copy-json-label");
  const textareaSelector = "#jc-qwen-generated-json textarea";

  const setState = (state) => {
    if (!button) return;
    button.classList.remove("is-copied", "is-failed");
    if (state) button.classList.add(state);
    const message = state === "is-copied" ? "Copied" : state === "is-failed" ? "Copy failed" : "Copy JSON";
    if (label) label.textContent = message;
    button.title = message;
    button.setAttribute("aria-label", message);
    window.clearTimeout(button._jcCopyTimer);
    if (state) {
      button._jcCopyTimer = window.setTimeout(() => setState(""), 1400);
    }
  };

  const fallbackCopy = (text) => {
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.style.position = "fixed";
    scratch.style.left = "-9999px";
    scratch.style.top = "0";
    document.body.appendChild(scratch);
    scratch.select();
    try {
      return document.execCommand("copy");
    } finally {
      scratch.remove();
    }
  };

  button?.addEventListener("click", async () => {
    const textarea = document.querySelector(textareaSelector);
    const text = textarea?.value || "";
    if (!text.trim()) {
      setState("is-failed");
      return;
    }
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else if (!fallbackCopy(text)) {
        throw new Error("fallback copy failed");
      }
      setState("is-copied");
    } catch {
      try {
        if (fallbackCopy(text)) setState("is-copied");
        else setState("is-failed");
      } catch {
        setState("is-failed");
      }
    }
  });
}
"""


def _variables(
    trigger_phrase,
    output_language,
    caption_length,
    dataset_goal,
    known_subject_class,
    brand_policy,
    text_policy,
    bbox_policy,
    existing_caption,
    existing_json_caption,
    validation_error,
    invalid_json_text,
    ideogram_json_caption,
) -> dict[str, Any]:
    return {
        "TRIGGER_PHRASE": trigger_phrase,
        "OUTPUT_LANGUAGE": output_language or "English",
        "CAPTION_LENGTH": caption_length,
        "DATASET_GOAL": dataset_goal,
        "KNOWN_SUBJECT_CLASS": known_subject_class,
        "BRAND_POLICY": brand_policy,
        "TEXT_POLICY": text_policy,
        "BBOX_POLICY": bbox_policy,
        "EXISTING_CAPTION": existing_caption,
        "EXISTING_JSON_CAPTION": existing_json_caption,
        "VALIDATION_ERROR": validation_error,
        "INVALID_JSON_TEXT": invalid_json_text,
        "IDEOGRAM_JSON_CAPTION": ideogram_json_caption,
    }


def _row_data(rows: Any) -> list[list[Any]]:
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    return [list(row) for row in (rows or []) if isinstance(row, (list, tuple))]


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


def _df_value(rows: Any, bbox_order: str = DEFAULT_BBOX_ORDER) -> dict[str, Any]:
    return {"headers": headers_for_bbox_order(bbox_order), "data": _row_data(rows)}


def _visible_rows(rows: Any, selected: Any) -> list[list[Any]]:
    row_data = _row_data(rows)
    return [row_data[index] for index in _choice_indices(selected) if index < len(row_data)]


def _visible_df_value(rows: Any, selected: Any, bbox_order: str = DEFAULT_BBOX_ORDER) -> dict[str, Any]:
    return _df_value(_visible_rows(rows, selected), bbox_order)


def _merge_visible_rows(all_rows: Any, selected: Any, visible_rows: Any) -> list[list[Any]]:
    merged = _row_data(all_rows)
    visible_data = _row_data(visible_rows)
    for target_index, row in zip(_choice_indices(selected), visible_data):
        if 0 <= target_index < len(merged):
            merged[target_index] = row
    return merged


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


def _box_choices(rows: Any) -> list[str]:
    choices: list[str] = []
    for index, row in enumerate(_row_data(rows), start=1):
        values = row + [""] * max(0, 8 - len(row))
        if not any(str(cell or "").strip() for cell in values):
            continue
        label = str(values[6] or values[5] or values[7] or values[0] or "box")
        label = " ".join(label.split())
        if len(label) > 58:
            label = label[:55] + "..."
        row_type = "text" if str(values[0] or "").strip() == "text" or str(values[7] or "").strip() else (values[0] or "obj")
        choices.append(f"{index:02d} {row_type} - {label}")
    return choices


def _choice_indices(selected: Any) -> list[int]:
    if selected is None:
        return []
    if isinstance(selected, str):
        selected = [selected]
    indices: list[int] = []
    seen: set[int] = set()
    for value in selected or []:
        match = str(value).strip().split(maxsplit=1)[0]
        try:
            index = int(match) - 1
        except Exception:
            continue
        if index >= 0 and index not in seen:
            seen.add(index)
            indices.append(index)
    return [index for index in indices if index >= 0]


def _preserve_visible(rows: Any, selected: Any, default_all: bool = False) -> tuple[list[str], list[str]]:
    choices = _box_choices(rows)
    choice_by_index = {index: choice for choice in choices if (index := next(iter(_choice_indices(choice)), None)) is not None}
    values = [choice_by_index[index] for index in _choice_indices(selected) if index in choice_by_index]
    if default_all and not values:
        values = choices
    return choices, values


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _autosave_target_paths(target: Any) -> tuple[Path | None, Path | None, Path | None, str | None]:
    if not isinstance(target, dict):
        return None, None, None, "Autosave skipped because no generated output folder is active."
    caption_value = str(target.get("caption_path") or "").strip()
    if not caption_value:
        return None, None, None, "Autosave skipped because no generated output folder is active."

    caption_path = Path(caption_value)
    run_dir = Path(str(target.get("output_run_dir") or caption_path.parent))
    metadata_value = str(target.get("metadata_path") or "").strip()
    metadata_path = Path(metadata_value) if metadata_value else None

    if caption_path.suffix.lower() != ".json":
        return None, None, None, f"Autosave skipped because the active output is {caption_path.name}, not a .json file."
    if not _is_relative_to(caption_path, OUTPUTS_DIR) or not _is_relative_to(run_dir, OUTPUTS_DIR):
        return None, None, None, "Autosave blocked because the stored target is outside the outputs folder."
    if not _same_path(caption_path.parent, run_dir):
        return None, None, None, "Autosave blocked because the stored folder id does not match the caption path."
    if metadata_path is not None and (not _is_relative_to(metadata_path, OUTPUTS_DIR) or not _same_path(metadata_path.parent, run_dir)):
        return None, None, None, "Autosave blocked because the stored metadata path does not match the output folder."
    if not caption_path.exists():
        return None, None, None, f"Autosave skipped because the generated JSON file no longer exists: {caption_path}"
    return caption_path, run_dir, metadata_path, None


def _relative_output_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(OUTPUTS_DIR.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return path.name


def _boxed_output_path(target: Any, caption_path: Path, run_dir: Path) -> Path:
    boxed_value = str(target.get("boxed_image_path") or "").strip() if isinstance(target, dict) else ""
    if boxed_value:
        boxed_path = Path(boxed_value)
        if _is_relative_to(boxed_path, OUTPUTS_DIR) and _same_path(boxed_path.parent, run_dir):
            return boxed_path
    return run_dir / f"{caption_path.stem}_boxed.png"


def _boxed_source_path(image: Any, target: Any, run_dir: Path) -> Path | None:
    output_value = str(target.get("output_image_path") or "").strip() if isinstance(target, dict) else ""
    if output_value:
        output_path = Path(output_value)
        if output_path.exists() and _is_relative_to(output_path, OUTPUTS_DIR) and _same_path(output_path.parent, run_dir):
            return output_path
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if image_path.exists():
            return image_path
    return None


def _save_autosave_boxed_image(
    image: Any,
    rows: Any,
    bbox_order: str,
    target: Any,
    caption_path: Path,
    run_dir: Path,
) -> tuple[Path | None, str]:
    if rows is None:
        return None, ""
    source_path = _boxed_source_path(image, target, run_dir)
    if source_path is None:
        return None, "Boxed PNG was not updated because the source image path is unavailable."
    output_path = _boxed_output_path(target, caption_path, run_dir)
    try:
        saved = save_boxed_image(source_path, rows, output_path, bbox_order=bbox_order)
    except Exception as exc:
        return None, f"Boxed PNG update failed: {type(exc).__name__}: {exc}"
    if saved is None:
        return None, "Boxed PNG was not updated because there are no valid boxes to render."
    return saved, f"Boxed PNG updated: outputs/{_relative_output_label(saved)}."


def _update_autosave_metadata(
    metadata_path: Path | None,
    caption_path: Path,
    run_dir: Path,
    caption_text: str,
    boxed_image_path: Path | None = None,
) -> str:
    if metadata_path is None or not metadata_path.exists():
        return ""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return "Metadata was not updated because metadata.json is not an object."
        metadata.update(
            {
                "caption_final": caption_text,
                "box_edits_autosaved": True,
                "box_edits_autosaved_at": datetime.now(timezone.utc).isoformat(),
                "caption_path": str(caption_path),
                "output_run_dir": str(run_dir),
            }
        )
        if boxed_image_path is not None:
            metadata["boxed_image_path"] = str(boxed_image_path)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        return f"Metadata update skipped: {type(exc).__name__}: {exc}"
    return ""


def _autosave_json_caption(
    caption_text: str,
    target: Any,
    image: Any = None,
    rows: Any = None,
    bbox_order: str = DEFAULT_BBOX_ORDER,
) -> str:
    caption_path, run_dir, metadata_path, target_error = _autosave_target_paths(target)
    if target_error:
        return html_message("info", "JSON box edits applied. " + html.escape(target_error))
    if caption_path is None or run_dir is None:
        return html_message("info", "JSON box edits applied. Autosave skipped because no generated output folder is active.")

    try:
        json.loads(caption_text)
    except Exception as exc:
        return html_message(
            "error",
            "JSON box edits applied in the UI, but autosave was blocked because the edited JSON does not parse: "
            + html.escape(f"{type(exc).__name__}: {exc}"),
        )

    try:
        caption_path.write_text(caption_text, encoding="utf-8")
        boxed_image_path, boxed_note = _save_autosave_boxed_image(image, rows, bbox_order, target, caption_path, run_dir)
        metadata_note = _update_autosave_metadata(metadata_path, caption_path, run_dir, caption_text, boxed_image_path)
    except Exception as exc:
        return html_message(
            "error",
            "JSON box edits applied in the UI, but autosave failed: "
            + html.escape(f"{type(exc).__name__}: {exc}"),
        )

    folder_id = run_dir.name
    location = f"outputs/{folder_id}/{caption_path.name}"
    message = "JSON box edits applied and autosaved to " + html.escape(location) + "."
    if boxed_note:
        message += "<br>" + html.escape(boxed_note)
    if metadata_note:
        message += "<br>" + html.escape(metadata_note)
    return html_message("success", message)


def _json_caption_for_save(caption_text: str, beautify_saved_json: Any, compact_json: Any) -> str:
    parsed = json.loads(caption_text)
    if bool(beautify_saved_json):
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    if bool(compact_json):
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return caption_text


def _box_edit_status(warnings: list[str], autosave_status: str) -> str:
    if not warnings:
        return autosave_status
    warning_text = html.escape("\n".join(warnings))
    return html_message("info", "Applied box edits after JSON repair fallback:<br><pre>" + warning_text + "</pre>") + autosave_status


def build_tab(engine: Any) -> TabUI:
    components: dict[str, gr.components.Component] = {}
    global_error = gr.HTML(visible=False)
    all_element_rows = gr.State([])
    autosave_target = gr.State({})
    preview_rows_snapshot = gr.Textbox(
        value=_preview_rows_snapshot_value([]),
        label="Preview Rows Snapshot",
        elem_id="jc-qwen-preview-rows-snapshot",
        elem_classes=["jc-hidden-sync"],
        interactive=True,
    )

    with gr.Row(equal_height=False, elem_classes=["jc-qwen-grid"]):
        with gr.Column(scale=9, elem_classes=["jc-qwen-workspace"]):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes=["jc-compact"]):
                    input_image = gr.Image(type="filepath", label="Input Image", height=430)
                    with gr.Row():
                        single_btn = gr.Button("Caption Image", elem_classes=["btn-qwen-caption"])
                        single_cancel_btn = gr.Button("Cancel", elem_classes=["btn-cancel"])
                        open_outputs_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])

                with gr.Column(scale=5, elem_classes=["jc-compact"]):
                    with gr.Row(elem_classes=["jc-generated-json-header"]):
                        gr.Markdown("**Generated Caption / JSON**", elem_classes=["jc-generated-json-title"])
                        gr.HTML(COPY_JSON_HTML, elem_classes=["jc-copy-json-control"], js_on_load=COPY_JSON_JS)
                    output_caption = gr.Textbox(
                        label="",
                        lines=18,
                        interactive=True,
                        elem_id="jc-qwen-generated-json",
                        elem_classes=["jc-output", "jc-codeish"],
                    )
                    with gr.Row():
                        render_json_btn = gr.Button("Render JSON Boxes", elem_classes=["btn-qwen-render"])
                        apply_box_btn = gr.Button("Apply Box Edits & Save", elem_classes=["btn-qwen-apply"])
                        add_box_btn = gr.Button("Add Box", elem_classes=["btn-json-add"])
                        clear_box_btn = gr.Button("Clear Boxes", elem_classes=["btn-reset-preset"])

            element_rows = gr.Dataframe(
                headers=headers_for_bbox_order(DEFAULT_BBOX_ORDER),
                value=_df_value([], DEFAULT_BBOX_ORDER),
                type="array",
                interactive=True,
                label="JSON Elements",
                max_height=320,
                datatype=["str", "number", "number", "number", "number", "str", "str", "str"],
                wrap=True,
                line_breaks=True,
                column_widths=ELEMENT_COLUMN_WIDTHS,
                elem_classes=["jc-qwen-elements-wide"],
            )

            with gr.Accordion("JSON Box Preview", open=True, elem_classes=["jc-qwen-preview-panel"]):
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
                    choices=[],
                    value=[],
                    label="Visible Boxes",
                    elem_classes=["jc-qwen-box-filter"],
                )
                json_overlay = gr.HTML("", elem_classes=["jc-qwen-overlay"], js_on_load=OVERLAY_EDIT_JS)

            single_status = gr.HTML("", elem_classes=["jc-qwen-status-scroll", "jc-qwen-status-bottom"])

        with gr.Column(scale=4, elem_classes=["jc-compact", "jc-qwen-settings-rail"]):
            with gr.Accordion("Preset & Prompt", open=True):
                components["preset_id"] = gr.Dropdown(
                    choices=qwen_preset_choices(),
                    value=DEFAULTS["preset_id"],
                    label="Preset",
                    allow_custom_value=False,
                )
                with gr.Row():
                    components["output_format"] = gr.Dropdown(
                        choices=["json", "txt", "tags", "qc"],
                        value=DEFAULTS["output_format"],
                        label="Format",
                        allow_custom_value=True,
                    )
                    components["extension"] = gr.Textbox(label="Extension", value=DEFAULTS["extension"], scale=1)
                    components["beautify_saved_json"] = gr.Checkbox(
                        label="Beautify saved JSON",
                        value=DEFAULTS["beautify_saved_json"],
                        scale=2,
                    )
                components["system_prompt"] = gr.Textbox(label="System Prompt", lines=3, value=DEFAULTS["system_prompt"])
                components["prompt"] = gr.Textbox(label="Prompt", lines=10, value=DEFAULTS["prompt"], elem_classes=["jc-codeish"])

            with gr.Accordion("Template Variables", open=False):
                with gr.Row():
                    components["trigger_phrase"] = gr.Textbox(label="Trigger Phrase", value=DEFAULTS["trigger_phrase"])
                    components["output_language"] = gr.Textbox(label="Language", value=DEFAULTS["output_language"])
                with gr.Row():
                    components["caption_length"] = gr.Textbox(label="Caption Length", value=DEFAULTS["caption_length"])
                    components["dataset_goal"] = gr.Textbox(label="Dataset Goal", value=DEFAULTS["dataset_goal"])
                components["known_subject_class"] = gr.Textbox(label="Known Subject Class", value=DEFAULTS["known_subject_class"])
                components["brand_policy"] = gr.Textbox(label="Brand Policy", value=DEFAULTS["brand_policy"])
                components["text_policy"] = gr.Textbox(label="Text Policy", value=DEFAULTS["text_policy"])
                components["bbox_policy"] = gr.Textbox(label="BBox Policy", value=DEFAULTS["bbox_policy"])
                components["existing_caption"] = gr.Textbox(label="Existing Caption", lines=3, value=DEFAULTS["existing_caption"])
                components["existing_json_caption"] = gr.Textbox(label="Existing JSON Caption", lines=3, value=DEFAULTS["existing_json_caption"])
                components["validation_error"] = gr.Textbox(label="Validation Error", lines=2, value=DEFAULTS["validation_error"])
                components["invalid_json_text"] = gr.Textbox(label="Invalid JSON Text", lines=3, value=DEFAULTS["invalid_json_text"])
                components["ideogram_json_caption"] = gr.Textbox(label="Ideogram JSON Caption", lines=3, value=DEFAULTS["ideogram_json_caption"])

            with gr.Accordion("Generation & Model", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        components["model_key"] = gr.Dropdown(
                            choices=qwen_model_choices(),
                            value=DEFAULTS["model_key"],
                            label="Caption Model",
                            allow_custom_value=False,
                        )
                        components["vram_preset"] = gr.Dropdown(
                            choices=VRAM_PRESET_CHOICES,
                            value=DEFAULTS["vram_preset"],
                            label="VRAM Preset",
                            allow_custom_value=False,
                        )
                    with gr.Column(scale=1):
                        components["use_subprocess"] = gr.Checkbox(
                            label="Run single and batch in subprocess, then terminate it",
                            value=DEFAULTS["use_subprocess"],
                        )
                with gr.Row():
                    components["model_quantization"] = gr.Radio(
                        choices=["bf16", "fp16", "int8", "nf4"],
                        value=DEFAULTS["model_quantization"],
                        label="Quantization",
                    )
                    components["device_id"] = gr.Textbox(
                        label="Device ID",
                        value=DEFAULTS["device_id"],
                        info="Use 0,1 for dual-GPU folder/ZIP batches; images are split evenly across both GPUs. Single image uses the first ID.",
                    )
                with gr.Row():
                    components["temperature"] = gr.Slider(0.0, 2.0, value=DEFAULTS["temperature"], step=0.01, label="Temperature")
                    components["top_p"] = gr.Slider(0.0, 1.0, value=DEFAULTS["top_p"], step=0.01, label="Top-p")
                with gr.Row():
                    components["top_k"] = gr.Slider(1, 100, value=DEFAULTS["top_k"], step=1, label="Top-k")
                    components["repetition_penalty"] = gr.Slider(0.8, 1.5, value=DEFAULTS["repetition_penalty"], step=0.01, label="Repetition Penalty")
                with gr.Row():
                    components["max_new_tokens"] = gr.Slider(64, 8192, value=DEFAULTS["max_new_tokens"], step=1, label="Max New Tokens")
                    components["image_long_edge"] = gr.Slider(256, 1536, value=DEFAULTS["image_long_edge"], step=1, label="Image Long Edge")
                with gr.Row():
                    components["save_image"] = gr.Checkbox(label="Save image copy", value=DEFAULTS["save_image"])
                    components["unload_model"] = gr.Checkbox(label="Unload after run", value=DEFAULTS["unload_model"])
                with gr.Row():
                    components["allow_tf32"] = gr.Checkbox(label="Allow TF32", value=DEFAULTS["allow_tf32"])
                    components["clear_cuda_cache"] = gr.Checkbox(label="Clear CUDA cache", value=DEFAULTS["clear_cuda_cache"])
                with gr.Row():
                    components["low_cpu_mem_usage"] = gr.Checkbox(label="Low CPU memory loading", value=DEFAULTS["low_cpu_mem_usage"])
                    components["attention_backend"] = gr.Dropdown(
                        choices=ATTENTION_BACKEND_CHOICES,
                        value=DEFAULTS["attention_backend"],
                        label="Attention Backend",
                        allow_custom_value=False,
                    )
                    components["enable_thinking"] = gr.Checkbox(
                        label="Enable Thinking",
                        value=DEFAULTS["enable_thinking"],
                        interactive=get_model_spec(DEFAULT_QWEN_MODEL_KEY).supports_thinking,
                    )
                build_torch_compile_controls(components, DEFAULTS)
                with gr.Row():
                    components["compact_json"] = gr.Checkbox(label="Compact JSON", value=DEFAULTS["compact_json"])
                    components["json_retries"] = gr.Slider(0, 3, value=DEFAULTS["json_retries"], step=1, label="JSON Repair Retries")
                with gr.Row():
                    components["remove_newlines"] = gr.Checkbox(label="Remove newlines for text", value=DEFAULTS["remove_newlines"])
                    components["auto_save_boxed_image"] = gr.Checkbox(label="Auto Save Boxed Image", value=DEFAULTS["auto_save_boxed_image"])
                with gr.Row():
                    components["caption_prefix"] = gr.Textbox(label="Text Prefix", value=DEFAULTS["caption_prefix"])
                    components["caption_suffix"] = gr.Textbox(label="Text Suffix", value=DEFAULTS["caption_suffix"])
                build_replace_pair_controls(components, DEFAULTS)
                components["app_side_only"] = gr.Checkbox(value=DEFAULTS["app_side_only"], visible=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes=["jc-compact"]):
            with gr.Accordion("Uploaded Files to ZIP", open=False):
                input_files = gr.File(
                    file_count="multiple",
                    file_types=sorted(IMAGE_EXTENSIONS),
                    label="Image Files",
                )
                components["file_batch_size"] = gr.Slider(1, 16, value=DEFAULTS["file_batch_size"], step=1, label="Batch Size")
                with gr.Row():
                    zip_btn = gr.Button("Create Caption ZIP", elem_classes=["btn-qwen-zip"])
                    zip_cancel_btn = gr.Button("Cancel Batch", elem_classes=["btn-cancel"])
                    zip_open_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])
                zip_status = gr.HTML("")
                zip_output = gr.File(label="Caption ZIP")

        with gr.Column(scale=1, elem_classes=["jc-compact"]):
            with gr.Accordion("Folder Batch", open=True):
                with gr.Row():
                    components["folder_input"] = gr.Textbox(label="Input Folder", value=DEFAULTS["folder_input"])
                    components["folder_output"] = gr.Textbox(label="Output Folder", value=DEFAULTS["folder_output"])
                with gr.Row():
                    components["process_subfolders"] = gr.Checkbox(label="Process subfolders", value=DEFAULTS["process_subfolders"])
                    components["dont_save_boxed_images"] = gr.Checkbox(label="Don't save boxed images", value=DEFAULTS["dont_save_boxed_images"])
                with gr.Row():
                    components["overwrite_caption"] = gr.Checkbox(label="Overwrite captions", value=DEFAULTS["overwrite_caption"])
                    components["append_caption"] = gr.Checkbox(label="Append captions", value=DEFAULTS["append_caption"])
                    components["folder_batch_size"] = gr.Slider(1, 16, value=DEFAULTS["folder_batch_size"], step=1, label="Batch Size")
                with gr.Row():
                    folder_btn = gr.Button("Start Qwen Folder Batch", elem_classes=["btn-qwen-folder"])
                    folder_cancel_btn = gr.Button("Cancel Batch", elem_classes=["btn-cancel"])
                    folder_open_btn = gr.Button("Open Batch Output", elem_classes=["btn-open-folder"])
                folder_status = gr.HTML("")

    ordered_components = [components[key] for key in ORDER]

    variable_inputs = [
        components["trigger_phrase"],
        components["output_language"],
        components["caption_length"],
        components["dataset_goal"],
        components["known_subject_class"],
        components["brand_policy"],
        components["text_policy"],
        components["bbox_policy"],
        components["existing_caption"],
        components["existing_json_caption"],
        components["validation_error"],
        components["invalid_json_text"],
        components["ideogram_json_caption"],
    ]

    def apply_preset(preset_id, vram_preset, model_key, *vars_values):
        variables = _variables(*vars_values)
        payload = preset_payload(preset_id, variables)
        vram_settings = qwen_vram_settings(vram_preset, model_key)
        return (
            payload["system_prompt"],
            payload["prompt"],
            payload["output_format"],
            payload["extension"],
            payload["temperature"],
            min(int(payload["max_new_tokens"]), int(vram_settings["max_new_tokens"])),
            min(int(payload["image_long_edge"]), int(vram_settings["image_long_edge"])),
            payload["app_side_only"],
        )

    def apply_vram_preset(vram_preset, model_key):
        settings = qwen_vram_settings(vram_preset, model_key)
        return (
            settings["model_quantization"],
            settings["image_long_edge"],
            settings["attention_backend"],
            settings["file_batch_size"],
            settings["folder_batch_size"],
            settings["max_new_tokens"],
        )

    def apply_model_defaults(vram_preset, model_key):
        settings = qwen_vram_settings(vram_preset, model_key)
        spec = get_model_spec(model_key)
        return (
            settings["model_quantization"],
            settings["image_long_edge"],
            settings["attention_backend"],
            settings["file_batch_size"],
            settings["folder_batch_size"],
            settings["max_new_tokens"],
            gr.update(value=False, interactive=spec.supports_thinking),
        )

    def run_single(image, *values):
        settings = settings_from_values(ORDER, values)
        yield from engine.caption_single(image, settings)

    def run_zip(files, *values):
        settings = settings_from_values(ORDER, values)
        yield from engine.process_batch_files_to_zip(files, settings)

    def run_folder(*values):
        settings = settings_from_values(ORDER, values)
        yield from engine.run_batch_folder_processing(settings)

    def render_json_boxes(image, caption_text, preset_id_value, bbox_order_value, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        final, parsed, warnings = normalize_json_output(caption_text, preset_id=str(preset_id_value or ""))
        rows = json_to_element_rows(parsed, bbox_order=bbox_order_value)
        choices, visible = _preserve_visible(rows, [], default_all=True)
        overlay = overlay_html(
            image,
            rows,
            interactive=True,
            bbox_order=bbox_order_value,
            visible_indices=_choice_indices(visible),
            disable_auto_update=bool(disable_auto_update_value),
        )
        status = ""
        if warnings:
            status = html_message("info", "JSON rendered with warnings:<br><pre>" + "\n".join(warnings) + "</pre>")
        return final, rows, _preview_rows_snapshot_value(rows), _visible_df_value(rows, visible, bbox_order_value), gr.update(choices=choices, value=visible), overlay, status

    def apply_box_edits(
        image,
        caption_text,
        all_rows,
        visible_rows,
        preview_snapshot,
        bbox_order_value,
        visible_choices,
        disable_auto_update_value,
        autosave_target_value,
        beautify_saved_json_value,
        compact_json_value,
    ):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        merged_rows = _merge_visible_rows(all_rows, visible_choices, visible_rows)
        merged_rows = _merge_preview_bbox_rows(merged_rows, preview_snapshot)
        final, parsed, warnings = apply_rows_to_json(caption_text, merged_rows, bbox_order=bbox_order_value)
        saved_final = _json_caption_for_save(final, beautify_saved_json_value, compact_json_value)
        display_rows = _row_data(merged_rows)
        choices, visible = _preserve_visible(display_rows, visible_choices, default_all=False)
        overlay = overlay_html(
            image,
            display_rows,
            interactive=True,
            bbox_order=bbox_order_value,
            visible_indices=_choice_indices(visible),
            disable_auto_update=bool(disable_auto_update_value),
        )
        status = _box_edit_status(
            warnings,
            _autosave_json_caption(saved_final, autosave_target_value, image, display_rows, bbox_order_value),
        )
        return final, display_rows, _preview_rows_snapshot_value(display_rows), _visible_df_value(display_rows, visible, bbox_order_value), gr.update(choices=choices, value=visible), overlay, status

    def apply_overlay_edit(
        image,
        caption_text,
        all_rows,
        visible_rows,
        bbox_order_value,
        visible_choices,
        disable_auto_update_value,
        autosave_target_value,
        beautify_saved_json_value,
        compact_json_value,
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
        merged_rows = _merge_visible_rows(all_rows, visible_choices, visible_rows)
        merged_rows = _merge_preview_bbox_rows(merged_rows, _preview_rows_snapshot_value(rows, [edited_index]))
        final, parsed, _warnings = apply_rows_to_json(caption_text, merged_rows, bbox_order=bbox_order_value)
        saved_final = _json_caption_for_save(final, beautify_saved_json_value, compact_json_value)
        display_rows = _row_data(merged_rows)
        choices, visible = _preserve_visible(display_rows, visible_choices, default_all=False)
        overlay = overlay_html(
            image,
            display_rows,
            interactive=True,
            bbox_order=bbox_order_value,
            visible_indices=_choice_indices(visible),
            disable_auto_update=bool(disable_auto_update_value),
        )
        status = _autosave_json_caption(saved_final, autosave_target_value, image, display_rows, bbox_order_value)
        return final, display_rows, _preview_rows_snapshot_value(display_rows), _visible_df_value(display_rows, visible, bbox_order_value), gr.update(choices=choices, value=visible), overlay, status

    def add_box(all_rows, bbox_order_value, visible_choices):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        row_data = _row_data(all_rows)
        next_rows = [*row_data, EMPTY_ELEMENT_ROW.copy()]
        choices = _box_choices(next_rows)
        selected = [choice for choice in choices if choice in set(visible_choices or [])]
        if choices:
            selected.append(choices[-1])
        return next_rows, _preview_rows_snapshot_value(next_rows), _visible_df_value(next_rows, selected, bbox_order_value), gr.update(choices=choices, value=selected)

    def clear_boxes(bbox_order_value):
        return [], _preview_rows_snapshot_value([]), _df_value([], bbox_order_value), gr.update(choices=[], value=[])

    def update_box_visibility(image, rows, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        return (
            _preview_rows_snapshot_value(rows),
            _visible_df_value(rows, visible_choices, bbox_order_value),
            overlay_html(
                image,
                rows,
                interactive=True,
                bbox_order=bbox_order_value,
                visible_indices=_choice_indices(visible_choices),
                disable_auto_update=bool(disable_auto_update_value),
            ),
        )

    def check_all_boxes(rows):
        choices = _box_choices(rows)
        return gr.update(choices=choices, value=choices)

    def uncheck_all_boxes(rows):
        choices = _box_choices(rows)
        return gr.update(choices=choices, value=[])

    def update_bbox_order(image, rows, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        return (
            _preview_rows_snapshot_value(rows),
            _visible_df_value(rows, visible_choices, bbox_order_value),
            overlay_html(
                image,
                rows,
                interactive=True,
                bbox_order=bbox_order_value,
                visible_indices=_choice_indices(visible_choices),
                disable_auto_update=bool(disable_auto_update_value),
            ),
        )

    def update_auto_update_setting(image, rows, bbox_order_value, visible_choices, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        return (
            _preview_rows_snapshot_value(rows),
            overlay_html(
                image,
                rows,
                interactive=True,
                bbox_order=bbox_order_value,
                visible_indices=_choice_indices(visible_choices),
                disable_auto_update=bool(disable_auto_update_value),
            ),
        )

    def sync_generated_rows(image, rows, bbox_order_value, disable_auto_update_value):
        bbox_order_value = clean_bbox_order(bbox_order_value)
        choices, visible = _preserve_visible(rows, [], default_all=True)
        return (
            _preview_rows_snapshot_value(rows),
            _visible_df_value(rows, visible, bbox_order_value),
            gr.update(choices=choices, value=visible),
            overlay_html(
                image,
                rows,
                interactive=True,
                bbox_order=bbox_order_value,
                visible_indices=_choice_indices(visible),
                disable_auto_update=bool(disable_auto_update_value),
            ),
        )

    def update_generated_visibility(rows):
        choices, visible = _preserve_visible(rows, [], default_all=True)
        return gr.update(choices=choices, value=visible)

    preset_outputs = [
        components["system_prompt"],
        components["prompt"],
        components["output_format"],
        components["extension"],
        components["temperature"],
        components["max_new_tokens"],
        components["image_long_edge"],
        components["app_side_only"],
    ]
    preset_inputs = [components["preset_id"], components["vram_preset"], components["model_key"]] + variable_inputs
    components["preset_id"].change(
        apply_preset,
        inputs=preset_inputs,
        outputs=preset_outputs,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    for variable_component in variable_inputs:
        variable_component.change(
            apply_preset,
            inputs=preset_inputs,
            outputs=preset_outputs,
            queue=False,
            show_progress="hidden",
            show_progress_on=[],
        )
    components["vram_preset"].change(
        apply_vram_preset,
        inputs=[components["vram_preset"], components["model_key"]],
        outputs=[
            components["model_quantization"],
            components["image_long_edge"],
            components["attention_backend"],
            components["file_batch_size"],
            components["folder_batch_size"],
            components["max_new_tokens"],
        ],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    components["model_key"].change(
        apply_model_defaults,
        inputs=[components["vram_preset"], components["model_key"]],
        outputs=[
            components["model_quantization"],
            components["image_long_edge"],
            components["attention_backend"],
            components["file_batch_size"],
            components["folder_batch_size"],
            components["max_new_tokens"],
            components["enable_thinking"],
        ],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )

    single_event = single_btn.click(
        run_single,
        inputs=[input_image] + ordered_components,
        outputs=[single_status, output_caption, json_overlay, all_element_rows, global_error, autosave_target],
    )
    single_event.then(
        sync_generated_rows,
        inputs=[input_image, all_element_rows, bbox_order, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, box_visibility, json_overlay],
        queue=False,
    )
    input_image.change(lambda _image: {}, inputs=[input_image], outputs=autosave_target, queue=False)
    single_cancel_btn.click(engine.cancel_single, outputs=single_status, queue=False)
    open_outputs_btn.click(run_open_outputs, outputs=single_status, queue=False)
    render_json_btn.click(
        render_json_boxes,
        inputs=[input_image, output_caption, components["preset_id"], bbox_order, disable_auto_update],
        outputs=[output_caption, all_element_rows, preview_rows_snapshot, element_rows, box_visibility, json_overlay, single_status],
    )
    apply_box_btn.click(
        apply_box_edits,
        inputs=[
            input_image,
            output_caption,
            all_element_rows,
            element_rows,
            preview_rows_snapshot,
            bbox_order,
            box_visibility,
            disable_auto_update,
            autosave_target,
            components["beautify_saved_json"],
            components["compact_json"],
        ],
        outputs=[output_caption, all_element_rows, preview_rows_snapshot, element_rows, box_visibility, json_overlay, single_status],
    )
    json_overlay.click(
        apply_overlay_edit,
        inputs=[
            input_image,
            output_caption,
            all_element_rows,
            element_rows,
            bbox_order,
            box_visibility,
            disable_auto_update,
            autosave_target,
            components["beautify_saved_json"],
            components["compact_json"],
        ],
        outputs=[output_caption, all_element_rows, preview_rows_snapshot, element_rows, box_visibility, json_overlay, single_status],
        queue=False,
    )
    add_event = add_box_btn.click(
        add_box,
        inputs=[all_element_rows, bbox_order, box_visibility],
        outputs=[all_element_rows, preview_rows_snapshot, element_rows, box_visibility],
        queue=False,
    )
    add_event.then(
        update_box_visibility,
        inputs=[input_image, all_element_rows, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, json_overlay],
        queue=False,
    )
    clear_event = clear_box_btn.click(
        clear_boxes,
        inputs=[bbox_order],
        outputs=[all_element_rows, preview_rows_snapshot, element_rows, box_visibility],
        queue=False,
    )
    clear_event.then(
        update_box_visibility,
        inputs=[input_image, all_element_rows, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, json_overlay],
        queue=False,
    )
    box_visibility.change(
        update_box_visibility,
        inputs=[input_image, all_element_rows, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, json_overlay],
        queue=False,
    )
    bbox_order.change(
        update_bbox_order,
        inputs=[input_image, all_element_rows, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, json_overlay],
        queue=False,
    )
    disable_auto_update.change(
        update_auto_update_setting,
        inputs=[input_image, all_element_rows, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, json_overlay],
        queue=False,
    )
    check_all_boxes_btn.click(check_all_boxes, inputs=[all_element_rows], outputs=box_visibility, queue=False).then(
        update_box_visibility,
        inputs=[input_image, all_element_rows, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, json_overlay],
        queue=False,
    )
    uncheck_all_boxes_btn.click(uncheck_all_boxes, inputs=[all_element_rows], outputs=box_visibility, queue=False).then(
        update_box_visibility,
        inputs=[input_image, all_element_rows, bbox_order, box_visibility, disable_auto_update],
        outputs=[preview_rows_snapshot, element_rows, json_overlay],
        queue=False,
    )

    zip_btn.click(
        run_zip,
        inputs=[input_files] + ordered_components,
        outputs=[zip_status, zip_output, global_error],
    )
    zip_cancel_btn.click(engine.cancel_batch, outputs=zip_status, queue=False)
    zip_open_btn.click(run_open_outputs, outputs=zip_status, queue=False)

    folder_btn.click(run_folder, inputs=ordered_components, outputs=[folder_status, global_error])
    folder_cancel_btn.click(engine.cancel_batch, outputs=folder_status, queue=False)
    folder_open_btn.click(run_open_folder, inputs=[components["folder_output"], components["folder_input"]], outputs=folder_status, queue=False)

    return TabUI(key="qwen3_vl_8b_instruct", order=ORDER, defaults=DEFAULTS, inputs=ordered_components)
