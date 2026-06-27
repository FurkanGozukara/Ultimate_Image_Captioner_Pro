from __future__ import annotations

from typing import Any

import gradio as gr

from ..json_tools import (
    ELEMENT_HEADERS,
    EMPTY_ELEMENT_ROW,
    build_ideogram_json,
    json_to_element_rows,
    overlay_html,
    parse_json_caption,
)
from .shared import TabUI


def _ratio(aspect_ratio: str, width: float | int | None, height: float | int | None) -> str:
    if str(aspect_ratio) == "Custom":
        try:
            w = max(1, float(width or 1))
            h = max(1, float(height or 1))
            return f"{w:g}:{h:g}"
        except Exception:
            return "1:1"
    return str(aspect_ratio or "1:1")


def build_tab() -> TabUI:
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, elem_classes=["jc-compact"]):
            image = gr.Image(type="filepath", label="Optional Preview Image", height=390)
            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    choices=["1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "Custom"],
                    value="1:1",
                    label="Aspect Ratio",
                )
                canvas_width = gr.Number(label="Width", value=1000, precision=0)
                canvas_height = gr.Number(label="Height", value=1000, precision=0)
            overlay = gr.HTML("")

        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            high_level = gr.Textbox(label="High Level Description", lines=4)
            with gr.Row():
                style_mode = gr.Radio(choices=["Photograph", "Art / Design"], value="Photograph", label="Style Branch")
                medium = gr.Textbox(label="Medium", value="illustration")
            aesthetics = gr.Textbox(label="Aesthetics", lines=2)
            lighting = gr.Textbox(label="Lighting", lines=2)
            photo = gr.Textbox(label="Photo", lines=2, placeholder="eye-level close-up, shallow depth of field")
            art_style = gr.Textbox(label="Art Style", lines=2, placeholder="flat vector icon, detailed digital illustration")
            color_palette = gr.Textbox(label="Color Palette", placeholder="#FFFFFF, #1E88E5, #FDD835")
            background = gr.Textbox(label="Background", lines=3)

        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            json_output = gr.Textbox(label="Generated Ideogram JSON", lines=23, interactive=True, elem_classes=["jc-output", "jc-codeish"])

    with gr.Accordion("Elements", open=True):
        element_rows = gr.Dataframe(
            headers=ELEMENT_HEADERS,
            value=[EMPTY_ELEMENT_ROW],
            type="array",
            interactive=True,
            label="Boxes",
            datatype=["str", "number", "number", "number", "number", "str", "str", "str"],
            row_count=1,
            wrap=True,
        )
        with gr.Row():
            generate_btn = gr.Button("Generate JSON", elem_classes=["btn-json-build"])
            preview_btn = gr.Button("Preview Boxes", elem_classes=["btn-qwen-render"])
            add_row_btn = gr.Button("Add Box", elem_classes=["btn-json-add"])
            clear_rows_btn = gr.Button("Clear Boxes", elem_classes=["btn-reset-preset"])
            import_btn = gr.Button("Import JSON Rows", elem_classes=["btn-load-preset"])
        status = gr.HTML("")

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
        rows,
    ):
        json_text = build_ideogram_json(
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
        )
        preview = overlay_html(image_path, rows, aspect_ratio=_ratio(ratio_value, width_value, height_value))
        return json_text, preview, ""

    def preview_boxes(image_path, ratio_value, width_value, height_value, rows):
        return overlay_html(image_path, rows, aspect_ratio=_ratio(ratio_value, width_value, height_value))

    def add_row(rows):
        rows = rows or []
        if isinstance(rows, dict) and "data" in rows:
            rows = rows["data"]
        return [*rows, EMPTY_ELEMENT_ROW.copy()]

    def clear_rows():
        return []

    def import_json(json_text):
        parsed, _pretty, warnings = parse_json_caption(json_text)
        if parsed is None:
            return [], "", "Photograph", "", "", "", "", "", "", "", '<div class="jc-error">Could not parse JSON.</div>'
        style = parsed.get("style_description") if isinstance(parsed.get("style_description"), dict) else {}
        comp = parsed.get("compositional_deconstruction") if isinstance(parsed.get("compositional_deconstruction"), dict) else {}
        style_mode = "Photograph" if "photo" in style else "Art / Design"
        rows = json_to_element_rows(parsed)
        status_html = ""
        if warnings:
            status_html = '<div class="jc-info"><pre>' + "\n".join(warnings) + "</pre></div>"
        return (
            rows,
            parsed.get("high_level_description", ""),
            style_mode,
            style.get("aesthetics", ""),
            style.get("lighting", ""),
            style.get("photo", ""),
            style.get("medium", ""),
            style.get("art_style", ""),
            ", ".join(style.get("color_palette", []) if isinstance(style.get("color_palette"), list) else []),
            comp.get("background", ""),
            status_html,
        )

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
        element_rows,
    ]
    generate_btn.click(generate_json, inputs=generate_inputs, outputs=[json_output, overlay, status])
    preview_btn.click(preview_boxes, inputs=[image, aspect_ratio, canvas_width, canvas_height, element_rows], outputs=overlay)
    add_row_btn.click(add_row, inputs=[element_rows], outputs=element_rows, queue=False)
    clear_rows_btn.click(clear_rows, outputs=element_rows, queue=False)
    import_btn.click(
        import_json,
        inputs=[json_output],
        outputs=[
            element_rows,
            high_level,
            style_mode,
            aesthetics,
            lighting,
            photo,
            medium,
            art_style,
            color_palette,
            background,
            status,
        ],
        queue=False,
    )

    return TabUI(key="json_builder", order=[], defaults={}, inputs=[])
