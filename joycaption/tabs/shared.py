from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Sequence

import gradio as gr

from ..common import format_exception, normalize_replace_pairs, open_folder, ordered_values, values_from_components


REPLACE_PAIR_LIST_JS = r"""
if (!element.dataset.jcReplacePairsBound) {
  element.dataset.jcReplacePairsBound = "1";
  element.addEventListener("click", (event) => {
    const button = event.target.closest(".jc-replace-remove");
    if (!button || !element.contains(button)) return;
    event.preventDefault();
    event.stopPropagation();
    trigger("click", { action: "remove", index: Number(button.dataset.index) });
  });
}
"""


@dataclass
class TabUI:
    key: str
    order: list[str]
    defaults: dict[str, Any]
    inputs: list[gr.components.Component]


def settings_from_values(order: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    return values_from_components(order, values)


def values_for_settings(order: Sequence[str], defaults: dict[str, Any], data: dict[str, Any] | None) -> list[Any]:
    return ordered_values(order, data or {}, defaults)


def run_open_folder(output_folder: str, input_folder: str = "") -> str:
    return open_folder(output_folder or input_folder or None)


def run_open_outputs() -> str:
    return open_folder(None)


def error_pair(exc: BaseException) -> tuple[str, str]:
    return "Failed", f"Error: {format_exception(exc)}"


def error_triple(exc: BaseException) -> tuple[str, str, str]:
    return "Error", "Failed", f"Error: {format_exception(exc)}"


def render_replace_pairs(pairs: Any) -> str:
    normalized = normalize_replace_pairs(pairs)
    if not normalized:
        return '<div class="jc-replace-empty">No replace word pairs added.</div>'
    items = []
    for index, (find_text, replace_text) in enumerate(normalized):
        items.append(
            '<div class="jc-replace-chip">'
            f'<span class="jc-replace-find">{html.escape(find_text)}</span>'
            '<span class="jc-replace-arrow">-></span>'
            f'<span class="jc-replace-to">{html.escape(replace_text)}</span>'
            f'<button type="button" class="jc-replace-remove" data-index="{index}" aria-label="Remove replace pair">X</button>'
            "</div>"
        )
    return '<div class="jc-replace-list">' + "".join(items) + "</div>"


def add_replace_pair(find_text: str, replace_text: str, pairs: Any):
    normalized = normalize_replace_pairs(pairs)
    find_value = (find_text or "").strip()
    if not find_value:
        return normalized, render_replace_pairs(normalized), gr.update(), gr.update()
    normalized.append([find_value, str(replace_text or "")])
    return normalized, render_replace_pairs(normalized), "", ""


def remove_replace_pair(pairs: Any, evt: gr.EventData):
    normalized = normalize_replace_pairs(pairs)
    payload = getattr(evt, "_data", {}) or {}
    if isinstance(payload, dict) and payload.get("action") == "remove":
        try:
            index = int(payload.get("index"))
        except Exception:
            index = -1
        if 0 <= index < len(normalized):
            normalized.pop(index)
    return normalized, render_replace_pairs(normalized)


def refresh_replace_pair_list(pairs: Any) -> str:
    return render_replace_pairs(pairs)


def build_replace_pair_controls(
    components: dict[str, gr.components.Component],
    defaults: dict[str, Any],
) -> gr.HTML:
    with gr.Row():
        replace_find = gr.Textbox(label="Replace Word", placeholder="man", scale=2)
        replace_with = gr.Textbox(label="With", placeholder="ohwx", scale=2)
        add_replace_btn = gr.Button("Add", elem_classes=["btn-add-replace-pair"], scale=1)
    with gr.Row():
        components["replace_case_sensitive"] = gr.Checkbox(
            label="Case sensitive",
            value=defaults.get("replace_case_sensitive", False),
        )
        components["replace_single_word"] = gr.Checkbox(
            label="Single word sensitive",
            value=defaults.get("replace_single_word", False),
        )
    components["replace_pairs"] = gr.State(defaults.get("replace_pairs", []))
    pair_list = gr.HTML(
        render_replace_pairs(defaults.get("replace_pairs", [])),
        elem_classes=["jc-replace-pairs"],
        js_on_load=REPLACE_PAIR_LIST_JS,
    )
    add_replace_btn.click(
        add_replace_pair,
        inputs=[replace_find, replace_with, components["replace_pairs"]],
        outputs=[components["replace_pairs"], pair_list, replace_find, replace_with],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    pair_list.click(
        remove_replace_pair,
        inputs=[components["replace_pairs"]],
        outputs=[components["replace_pairs"], pair_list],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    components["replace_pairs"].change(
        refresh_replace_pair_list,
        inputs=[components["replace_pairs"]],
        outputs=pair_list,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    return pair_list
