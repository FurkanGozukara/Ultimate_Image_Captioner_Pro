from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gradio as gr

from ..common import IMAGE_EXTENSIONS, OUTPUTS_DIR, ensure_runtime_dirs, html_message, natural_sort_key, open_folder
from ..json_tools import json_to_element_rows, normalize_json_output, overlay_html
from .shared import TabUI


TABLE_HEADERS = ["Day", "Folder", "Caption", "Image", "Type", "Modified"]


def _day_for_folder(folder: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(folder.stat().st_ctime))
    except Exception:
        return ""


def _time_for(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
    except Exception:
        return ""


def _find_image_for_caption(caption_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = caption_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    images = sorted(
        [path for path in caption_path.parent.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key,
    )
    return images[0] if images else None


def _scan_records() -> list[dict[str, Any]]:
    ensure_runtime_dirs()
    if not OUTPUTS_DIR.exists():
        return []
    records: list[dict[str, Any]] = []
    folders = sorted([path for path in OUTPUTS_DIR.iterdir() if path.is_dir()], key=natural_sort_key, reverse=True)
    for folder in folders:
        day = _day_for_folder(folder)
        captions = sorted(
            [
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in {".txt", ".json"} and path.name.lower() != "metadata.json"
            ],
            key=natural_sort_key,
        )
        for caption_path in captions:
            image_path = _find_image_for_caption(caption_path)
            records.append(
                {
                    "day": day,
                    "folder": str(folder),
                    "caption_path": str(caption_path),
                    "caption_name": caption_path.name,
                    "image_path": str(image_path) if image_path else "",
                    "image_name": image_path.name if image_path else "",
                    "type": caption_path.suffix.lower(),
                    "modified": _time_for(caption_path),
                }
            )
    return records


def _filter_records(records: list[dict[str, Any]], day: str, search: str) -> list[dict[str, Any]]:
    query = str(search or "").strip().lower()
    filtered: list[dict[str, Any]] = []
    for record in records:
        if day and day != "All" and record["day"] != day:
            continue
        haystack = " ".join(
            [
                record.get("folder", ""),
                record.get("caption_name", ""),
                record.get("image_name", ""),
                record.get("caption_path", ""),
            ]
        ).lower()
        if query and query not in haystack:
            continue
        filtered.append(record)
    return filtered


def _table(records: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            record["day"],
            Path(record["folder"]).name,
            record["caption_name"],
            record["image_name"],
            record["type"],
            record["modified"],
        ]
        for record in records
    ]


def _choices(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (
            f'{record["day"]} / {Path(record["folder"]).name} / {record["caption_name"]}',
            record["caption_path"],
        )
        for record in records
    ]


def _day_choices(records: list[dict[str, Any]]) -> list[str]:
    return ["All"] + sorted({record["day"] for record in records if record["day"]}, reverse=True)


def _load_caption(caption_path: str | None) -> tuple[str | None, str, str, str, str]:
    if not caption_path:
        return None, "", "", "", html_message("info", "No output selected.")
    path = Path(caption_path)
    if not path.exists():
        return None, "", "", "", html_message("error", f"Caption file not found: {path}")
    text = path.read_text(encoding="utf-8")
    image_path = _find_image_for_caption(path)
    overlay = ""
    if path.suffix.lower() == ".json":
        _normalized, parsed, _warnings = normalize_json_output(text)
        overlay = overlay_html(image_path, json_to_element_rows(parsed)) if parsed else ""
    return (
        str(image_path) if image_path else None,
        text,
        str(path),
        overlay,
        html_message("success", f"Loaded {path.name}."),
    )


def build_tab() -> TabUI:
    initial_records = _scan_records()
    filtered = _filter_records(initial_records, "All", "")

    records_state = gr.State(initial_records)
    filtered_state = gr.State(filtered)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            with gr.Row():
                day_filter = gr.Dropdown(choices=_day_choices(initial_records), value="All", label="Created Day")
                search_box = gr.Textbox(label="Search Name", placeholder="image name, caption name, folder")
                refresh_btn = gr.Button("Refresh", elem_classes=["btn-refresh"])
                audit_btn = gr.Button("Audit Sidecars", elem_classes=["btn-qwen-render"])
            table = gr.Dataframe(
                headers=TABLE_HEADERS,
                value=_table(filtered),
                type="array",
                interactive=False,
                label="Saved Outputs",
                max_height=360,
                show_search="search",
                wrap=True,
            )
            selected_dropdown = gr.Dropdown(choices=_choices(filtered), label="Selected Caption", allow_custom_value=False)
            status = gr.HTML("")

        with gr.Column(scale=4, elem_classes=["jc-compact"]):
            image_preview = gr.Image(type="filepath", label="Image", height=430)
            overlay_preview = gr.HTML("")

        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            selected_path = gr.Textbox(label="Caption Path", interactive=False)
            editor = gr.Textbox(label="Caption / JSON Editor", lines=24, interactive=True, elem_classes=["jc-output", "jc-codeish"])
            with gr.Row():
                save_btn = gr.Button("Save Edit", elem_classes=["btn-save-preset"])
                open_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])

    def refresh(day, search):
        records = _scan_records()
        visible = _filter_records(records, day or "All", search)
        day_choices = _day_choices(records)
        selected_day = day if day in day_choices else "All"
        if selected_day != day:
            visible = _filter_records(records, selected_day, search)
        return (
            records,
            visible,
            gr.update(choices=day_choices, value=selected_day),
            _table(visible),
            gr.update(choices=_choices(visible), value=None),
            html_message("success", f"Found {len(visible)} caption file(s)."),
        )

    def filter_existing(records, day, search):
        visible = _filter_records(records or [], day or "All", search)
        return visible, _table(visible), gr.update(choices=_choices(visible), value=None), html_message("info", f"Showing {len(visible)} file(s).")

    def select_from_table(evt: gr.SelectData, visible_records):
        row = 0
        if isinstance(evt.index, (list, tuple)) and evt.index:
            row = int(evt.index[0])
        elif isinstance(evt.index, int):
            row = evt.index
        records = visible_records or []
        if row < 0 or row >= len(records):
            return gr.update(), None, "", "", "", html_message("error", "Selected row is out of range.")
        path = records[row]["caption_path"]
        image, text, selected, overlay, message = _load_caption(path)
        return gr.update(value=path), image, text, selected, overlay, message

    def load_dropdown(path):
        image, text, selected, overlay, message = _load_caption(path)
        return image, text, selected, overlay, message

    def save_edit(path_text, content):
        if not path_text:
            return html_message("error", "No caption file selected.")
        path = Path(path_text)
        if not path.exists():
            return html_message("error", f"Caption file not found: {path}")
        text = str(content or "")
        if path.suffix.lower() == ".json":
            normalized, parsed, warnings = normalize_json_output(text)
            if parsed is None:
                return html_message("error", "JSON was not saved because it does not parse:<br><pre>" + "\n".join(warnings) + "</pre>")
            text = normalized
        path.write_text(text, encoding="utf-8")
        return html_message("success", f"Saved {path.name}.")

    def audit_sidecars():
        records = _scan_records()
        lines: list[str] = []
        folders = sorted([path for path in OUTPUTS_DIR.iterdir() if path.is_dir()], key=natural_sort_key) if OUTPUTS_DIR.exists() else []
        for folder in folders:
            images = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            captions = [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".txt", ".json"} and path.name.lower() != "metadata.json"
            ]
            caption_stems = {path.stem for path in captions}
            for image in images:
                if image.stem not in caption_stems:
                    lines.append(f"Missing sidecar for image: {image}")
            for caption in captions:
                if _find_image_for_caption(caption) is None:
                    lines.append(f"No matching image for sidecar: {caption}")
                if caption.suffix.lower() == ".json":
                    text = caption.read_text(encoding="utf-8")
                    _normalized, parsed, warnings = normalize_json_output(text, preset_id="i4_json_auto_best")
                    if parsed is None or warnings:
                        lines.append(f"JSON issue in {caption}: {'; '.join(warnings)}")
        if not lines:
            return html_message("success", f"Audit passed for {len(records)} caption file(s).")
        return html_message("info", "Audit findings:<br><pre>" + "\n".join(lines[:300]) + "</pre>")

    refresh_btn.click(
        refresh,
        inputs=[day_filter, search_box],
        outputs=[records_state, filtered_state, day_filter, table, selected_dropdown, status],
        queue=False,
    )
    day_filter.change(
        filter_existing,
        inputs=[records_state, day_filter, search_box],
        outputs=[filtered_state, table, selected_dropdown, status],
        queue=False,
    )
    search_box.change(
        filter_existing,
        inputs=[records_state, day_filter, search_box],
        outputs=[filtered_state, table, selected_dropdown, status],
        queue=False,
    )
    table.select(
        select_from_table,
        inputs=[filtered_state],
        outputs=[selected_dropdown, image_preview, editor, selected_path, overlay_preview, status],
        queue=False,
    )
    selected_dropdown.change(load_dropdown, inputs=[selected_dropdown], outputs=[image_preview, editor, selected_path, overlay_preview, status], queue=False)
    save_btn.click(save_edit, inputs=[selected_path, editor], outputs=status, queue=False)
    open_btn.click(lambda: open_folder(OUTPUTS_DIR), outputs=status, queue=False)
    audit_btn.click(audit_sidecars, outputs=status, queue=False)

    return TabUI(key="output_browser", order=[], defaults={}, inputs=[])
