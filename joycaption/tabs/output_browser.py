from __future__ import annotations

import time
from fnmatch import fnmatchcase
from math import ceil
from pathlib import Path
from typing import Any

import gradio as gr

from ..common import IMAGE_EXTENSIONS, OUTPUTS_DIR, ensure_runtime_dirs, html_message, natural_sort_key, open_folder
from ..json_tools import json_to_element_rows, normalize_json_output, overlay_html
from .shared import TabUI


TABLE_HEADERS = ["Day", "Folder", "Caption", "Image", "Type", "Modified"]
PAGE_SIZE_CHOICES = [10, 25, 50, 100]
DEFAULT_PAGE_SIZE = 50
WILDCARD_CHARS = "*?["


def _day_for_folder(folder: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(folder.stat().st_mtime))
    except Exception:
        return ""


def _mtime_for(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _time_for(path: Path) -> str:
    try:
        mtime = _mtime_for(path)
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)) if mtime else ""
    except Exception:
        return ""


def _folder_display_sort_key(path: Path) -> tuple[Any, ...]:
    if path.name.isdigit():
        return (1, int(path.name), natural_sort_key(path))
    return (0, _mtime_for(path), natural_sort_key(path))


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


def _scan_folder_index() -> list[dict[str, Any]]:
    ensure_runtime_dirs()
    if not OUTPUTS_DIR.exists():
        return []
    folders = sorted(
        [path for path in OUTPUTS_DIR.iterdir() if path.is_dir()],
        key=_folder_display_sort_key,
        reverse=True,
    )
    return [
        {
            "day": _day_for_folder(folder),
            "folder": str(folder),
            "folder_name": folder.name,
            "modified": _time_for(folder),
        }
        for folder in folders
    ]


def _has_search(search: str) -> bool:
    return bool(str(search or "").strip())


def _has_wildcard(search: str) -> bool:
    return any(char in str(search or "") for char in WILDCARD_CHARS)


def _read_caption_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _matches_search(record: dict[str, Any], caption_text: str, search: str) -> bool:
    query = str(search or "").strip().lower()
    if not query:
        return True

    fields = [
        record.get("folder", ""),
        Path(str(record.get("folder", ""))).name,
        record.get("caption_path", ""),
        record.get("caption_name", ""),
        Path(str(record.get("caption_name", ""))).stem,
        record.get("image_path", ""),
        record.get("image_name", ""),
        Path(str(record.get("image_name", ""))).stem,
        caption_text,
    ]
    lowered_fields = [str(field or "").lower() for field in fields]
    if _has_wildcard(query):
        for field in lowered_fields:
            if fnmatchcase(field, query):
                return True
        return any(fnmatchcase(line.strip().lower(), query) for line in str(caption_text or "").splitlines())

    return any(query in field for field in lowered_fields)


def _scan_records_for_folders(folders: list[Path], search: str = "") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    has_search = _has_search(search)
    for folder in folders:
        day = _day_for_folder(folder)
        captions = sorted(
            [
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in {".txt", ".json"} and path.name.lower() != "metadata.json"
            ],
            key=natural_sort_key,
            reverse=True,
        )
        for caption_path in captions:
            image_path = _find_image_for_caption(caption_path)
            record = {
                "day": day,
                "folder": str(folder),
                "caption_path": str(caption_path),
                "caption_name": caption_path.name,
                "image_path": str(image_path) if image_path else "",
                "image_name": image_path.name if image_path else "",
                "type": caption_path.suffix.lower(),
                "modified": _time_for(caption_path),
            }
            if has_search:
                caption_text = _read_caption_text(caption_path)
                if not _matches_search(record, caption_text, search):
                    continue
            records.append(record)
    return records


def _filter_folders(folders: list[dict[str, Any]], day: str) -> list[dict[str, Any]]:
    if not day or day == "All":
        return folders
    return [folder for folder in folders if folder.get("day") == day]


def _page_size(value: Any) -> int:
    try:
        size = int(float(value))
    except (TypeError, ValueError):
        size = DEFAULT_PAGE_SIZE
    return max(1, size)


def _page_number(value: Any) -> int:
    try:
        page = int(float(value))
    except (TypeError, ValueError):
        page = 1
    return max(1, page)


def _page_count(total_items: int, page_size: int) -> int:
    return max(1, ceil(max(0, total_items) / max(1, page_size)))


def _page_folder_paths(folders: list[dict[str, Any]], page: int, page_size: int) -> list[Path]:
    start = (page - 1) * page_size
    return [Path(item["folder"]) for item in folders[start : start + page_size]]


def _page_records(records: list[dict[str, Any]], page: int, page_size: int) -> list[dict[str, Any]]:
    start = (page - 1) * page_size
    return records[start : start + page_size]


def _day_choices_from_folders(folders: list[dict[str, Any]]) -> list[str]:
    return ["All"] + sorted({str(folder.get("day") or "") for folder in folders if folder.get("day")}, reverse=True)


def _page_payload(
    folders: list[dict[str, Any]],
    day: str,
    search: str,
    page: Any,
    page_size: Any,
) -> tuple[list[dict[str, Any]], list[list[str]], gr.update, gr.update, str, str]:
    selected_day = day or "All"
    size = _page_size(page_size)
    matching_folders = _filter_folders(folders, selected_day)

    if _has_search(search):
        searched_folders = [Path(item["folder"]) for item in matching_folders]
        matching_records = _scan_records_for_folders(searched_folders, search)
        count = _page_count(len(matching_records), size)
        selected_page = min(_page_number(page), count)
        records = _page_records(matching_records, selected_page, size)
        page_text = f"Search Page {selected_page} / {count} - {len(matching_records)} matching caption file(s)"
        if selected_day != "All":
            page_text += f" on {selected_day}"
        mode = "wildcard" if _has_wildcard(search) else "text"
        message = (
            f"Showing {len(records)} matching caption file(s). "
            f"Searched {len(searched_folders)} folder(s) using {mode} search over file names and caption text."
        )
    else:
        count = _page_count(len(matching_folders), size)
        selected_page = min(_page_number(page), count)
        page_folders = _page_folder_paths(matching_folders, selected_page, size)
        records = _scan_records_for_folders(page_folders)
        page_text = f"Page {selected_page} / {count} - {len(matching_folders)} folder(s)"
        if selected_day != "All":
            page_text += f" on {selected_day}"
        message = (
            f"Showing {len(records)} caption file(s) from {len(page_folders)} scanned folder(s). "
            f"{len(matching_folders)} folder(s) match the day filter."
        )
    return (
        records,
        _table(records),
        gr.update(choices=_choices(records), value=None),
        gr.update(value=selected_page),
        page_text,
        html_message("info", message),
    )


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


def build_tab(saved_outputs_tab=None) -> TabUI:
    initial_folders = _scan_folder_index()
    initial_records, initial_table, _initial_dropdown, _initial_page, initial_page_text, initial_status = _page_payload(
        initial_folders,
        "All",
        "",
        1,
        DEFAULT_PAGE_SIZE,
    )

    folders_state = gr.State(initial_folders)
    filtered_state = gr.State(initial_records)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            with gr.Row():
                day_filter = gr.Dropdown(choices=_day_choices_from_folders(initial_folders), value="All", label="Output Day")
                search_box = gr.Textbox(
                    label="Search Names/Captions",
                    placeholder="wildcards supported: *.json, *portrait*, image_??",
                )
                refresh_btn = gr.Button("Refresh", elem_classes=["btn-refresh"])
                audit_btn = gr.Button("Audit Sidecars", elem_classes=["btn-qwen-render"])
            with gr.Row():
                prev_btn = gr.Button("Previous Page")
                page_number = gr.Number(label="Page", value=1, precision=0)
                next_btn = gr.Button("Next Page")
                page_size = gr.Dropdown(choices=PAGE_SIZE_CHOICES, value=DEFAULT_PAGE_SIZE, label="Per Page")
            page_info = gr.HTML(initial_page_text)
            table = gr.Dataframe(
                headers=TABLE_HEADERS,
                value=initial_table,
                type="array",
                interactive=False,
                label="Saved Outputs",
                max_height=360,
                show_search="search",
                wrap=True,
            )
            selected_dropdown = gr.Dropdown(
                choices=_choices(initial_records),
                value=None,
                label="Selected Caption",
                allow_custom_value=False,
            )
            status = gr.HTML(initial_status)

        with gr.Column(scale=4, elem_classes=["jc-compact"]):
            image_preview = gr.Image(type="filepath", label="Image", height=430)
            overlay_preview = gr.HTML("")

        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            selected_path = gr.Textbox(label="Caption Path", interactive=False)
            editor = gr.Textbox(label="Caption / JSON Editor", lines=24, interactive=True, elem_classes=["jc-output", "jc-codeish"])
            with gr.Row():
                save_btn = gr.Button("Save Edit", elem_classes=["btn-save-preset"])
                open_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])

    def refresh(day, search, current_page_size):
        folders = _scan_folder_index()
        day_choices = _day_choices_from_folders(folders)
        selected_day = day if day in day_choices else "All"
        records, table_value, dropdown, page_update, page_text, message = _page_payload(
            folders,
            selected_day,
            search,
            1,
            current_page_size,
        )
        return (
            folders,
            records,
            gr.update(choices=day_choices, value=selected_day),
            page_update,
            table_value,
            dropdown,
            page_text,
            message,
        )

    def render_page(folders, day, search, page, current_page_size):
        records, table_value, dropdown, page_update, page_text, message = _page_payload(
            folders or [],
            day or "All",
            search,
            page,
            current_page_size,
        )
        return records, page_update, table_value, dropdown, page_text, message

    def first_page(folders, day, search, current_page_size):
        return render_page(folders, day, search, 1, current_page_size)

    def previous_page(folders, day, search, page, current_page_size):
        return render_page(folders, day, search, _page_number(page) - 1, current_page_size)

    def next_page(folders, day, search, page, current_page_size):
        return render_page(folders, day, search, _page_number(page) + 1, current_page_size)

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
        folders = _scan_folder_index()
        records = _scan_records_for_folders([Path(folder["folder"]) for folder in folders])
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
        inputs=[day_filter, search_box, page_size],
        outputs=[folders_state, filtered_state, day_filter, page_number, table, selected_dropdown, page_info, status],
        queue=False,
    )
    if saved_outputs_tab is not None:
        saved_outputs_tab.select(
            refresh,
            inputs=[day_filter, search_box, page_size],
            outputs=[folders_state, filtered_state, day_filter, page_number, table, selected_dropdown, page_info, status],
            queue=False,
        )
    day_filter.change(
        first_page,
        inputs=[folders_state, day_filter, search_box, page_size],
        outputs=[filtered_state, page_number, table, selected_dropdown, page_info, status],
        queue=False,
    )
    search_box.change(
        first_page,
        inputs=[folders_state, day_filter, search_box, page_size],
        outputs=[filtered_state, page_number, table, selected_dropdown, page_info, status],
        queue=False,
    )
    page_size.change(
        first_page,
        inputs=[folders_state, day_filter, search_box, page_size],
        outputs=[filtered_state, page_number, table, selected_dropdown, page_info, status],
        queue=False,
    )
    page_number.change(
        render_page,
        inputs=[folders_state, day_filter, search_box, page_number, page_size],
        outputs=[filtered_state, page_number, table, selected_dropdown, page_info, status],
        queue=False,
    )
    prev_btn.click(
        previous_page,
        inputs=[folders_state, day_filter, search_box, page_number, page_size],
        outputs=[filtered_state, page_number, table, selected_dropdown, page_info, status],
        queue=False,
    )
    next_btn.click(
        next_page,
        inputs=[folders_state, day_filter, search_box, page_number, page_size],
        outputs=[filtered_state, page_number, table, selected_dropdown, page_info, status],
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
