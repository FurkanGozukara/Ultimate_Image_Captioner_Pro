from __future__ import annotations

from typing import Any

import gradio as gr

from ..common import IMAGE_EXTENSIONS, html_message
from ..json_tools import ELEMENT_HEADERS, EMPTY_ELEMENT_ROW, apply_rows_to_json, json_to_element_rows, normalize_json_output, overlay_html
from ..qwen_presets import default_qwen_preset_id, preset_payload, qwen_preset_choices
from ..vram import VRAM_PRESET_CHOICES, default_vram_preset, qwen_vram_settings
from .shared import TabUI, run_open_folder, run_open_outputs, settings_from_values


DEFAULT_PRESET_ID = default_qwen_preset_id()
DEFAULT_VRAM = default_vram_preset()
DEFAULT_PAYLOAD = preset_payload(DEFAULT_PRESET_ID)
DEFAULT_VRAM_SETTINGS = qwen_vram_settings(DEFAULT_VRAM)

ORDER = [
    "vram_preset",
    "preset_id",
    "output_format",
    "extension",
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
    "use_sdpa_attention",
    "compact_json",
    "json_retries",
    "remove_newlines",
    "caption_prefix",
    "caption_suffix",
    "device_id",
    "file_batch_size",
    "folder_input",
    "folder_output",
    "skip_exists",
    "overwrite_caption",
    "append_caption",
    "process_subfolders",
    "folder_batch_size",
    "app_side_only",
]

DEFAULTS: dict[str, Any] = {
    "vram_preset": DEFAULT_VRAM,
    "preset_id": DEFAULT_PRESET_ID,
    "output_format": DEFAULT_PAYLOAD["output_format"],
    "extension": DEFAULT_PAYLOAD["extension"],
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
    "top_p": 0.9,
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
    "use_sdpa_attention": False,
    "compact_json": False,
    "json_retries": 1,
    "remove_newlines": False,
    "caption_prefix": "",
    "caption_suffix": "",
    "device_id": "0",
    "file_batch_size": DEFAULT_VRAM_SETTINGS["file_batch_size"],
    "folder_input": "",
    "folder_output": "",
    "skip_exists": True,
    "overwrite_caption": False,
    "append_caption": False,
    "process_subfolders": False,
    "folder_batch_size": DEFAULT_VRAM_SETTINGS["folder_batch_size"],
    "app_side_only": DEFAULT_PAYLOAD["app_side_only"],
}


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


def build_tab(engine: Any) -> TabUI:
    components: dict[str, gr.components.Component] = {}
    global_error = gr.HTML(visible=False)

    with gr.Row(equal_height=False, elem_classes=["jc-qwen-grid"]):
        with gr.Column(scale=9, elem_classes=["jc-qwen-workspace"]):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes=["jc-compact"]):
                    input_image = gr.Image(type="filepath", label="Input Image", height=430)
                    with gr.Row():
                        single_btn = gr.Button("Caption Image", elem_classes=["btn-qwen-caption"])
                        single_cancel_btn = gr.Button("Cancel", elem_classes=["btn-cancel"])
                        open_outputs_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])
                    single_status = gr.HTML("", elem_classes=["jc-qwen-status-scroll"])

                with gr.Column(scale=5, elem_classes=["jc-compact"]):
                    output_caption = gr.Textbox(
                        label="Generated Caption / JSON",
                        lines=18,
                        interactive=True,
                        elem_classes=["jc-output", "jc-codeish"],
                    )
                    with gr.Row():
                        render_json_btn = gr.Button("Render JSON Boxes", elem_classes=["btn-qwen-render"])
                        apply_box_btn = gr.Button("Apply Box Edits", elem_classes=["btn-qwen-apply"])
                        add_box_btn = gr.Button("Add Box", elem_classes=["btn-json-add"])
                        clear_box_btn = gr.Button("Clear Boxes", elem_classes=["btn-reset-preset"])
                    element_rows = gr.Dataframe(
                        headers=ELEMENT_HEADERS,
                        value=[],
                        type="array",
                        interactive=True,
                        label="JSON Elements",
                        max_height=210,
                        datatype=["str", "number", "number", "number", "number", "str", "str", "str"],
                        wrap=True,
                    )

            with gr.Accordion("JSON Box Preview", open=True, elem_classes=["jc-qwen-preview-panel"]):
                json_overlay = gr.HTML("", elem_classes=["jc-qwen-overlay"])

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
                components["vram_preset"] = gr.Dropdown(
                    choices=VRAM_PRESET_CHOICES,
                    value=DEFAULTS["vram_preset"],
                    label="VRAM Preset",
                    allow_custom_value=False,
                )
                with gr.Row():
                    components["model_quantization"] = gr.Radio(
                        choices=["bf16", "fp16", "int8", "nf4"],
                        value=DEFAULTS["model_quantization"],
                        label="Quantization",
                    )
                    components["device_id"] = gr.Textbox(label="Device ID", value=DEFAULTS["device_id"])
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
                components["use_subprocess"] = gr.Checkbox(
                    label="Run single and batch in subprocess, then terminate it",
                    value=DEFAULTS["use_subprocess"],
                )
                with gr.Row():
                    components["allow_tf32"] = gr.Checkbox(label="Allow TF32", value=DEFAULTS["allow_tf32"])
                    components["clear_cuda_cache"] = gr.Checkbox(label="Clear CUDA cache", value=DEFAULTS["clear_cuda_cache"])
                with gr.Row():
                    components["low_cpu_mem_usage"] = gr.Checkbox(label="Low CPU memory loading", value=DEFAULTS["low_cpu_mem_usage"])
                    components["use_sdpa_attention"] = gr.Checkbox(label="Use SDPA attention", value=DEFAULTS["use_sdpa_attention"])
                with gr.Row():
                    components["compact_json"] = gr.Checkbox(label="Compact JSON", value=DEFAULTS["compact_json"])
                    components["json_retries"] = gr.Slider(0, 3, value=DEFAULTS["json_retries"], step=1, label="JSON Repair Retries")
                with gr.Row():
                    components["remove_newlines"] = gr.Checkbox(label="Remove newlines for text", value=DEFAULTS["remove_newlines"])
                    components["caption_prefix"] = gr.Textbox(label="Text Prefix", value=DEFAULTS["caption_prefix"])
                    components["caption_suffix"] = gr.Textbox(label="Text Suffix", value=DEFAULTS["caption_suffix"])
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
            with gr.Accordion("Folder Batch", open=False):
                with gr.Row():
                    components["folder_input"] = gr.Textbox(label="Input Folder", value=DEFAULTS["folder_input"])
                    components["folder_output"] = gr.Textbox(label="Output Folder", value=DEFAULTS["folder_output"])
                with gr.Row():
                    components["skip_exists"] = gr.Checkbox(label="Skip existing outputs", value=DEFAULTS["skip_exists"])
                    components["process_subfolders"] = gr.Checkbox(label="Process subfolders", value=DEFAULTS["process_subfolders"])
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

    def apply_preset(preset_id, vram_preset, *vars_values):
        variables = _variables(*vars_values)
        payload = preset_payload(preset_id, variables)
        vram_settings = qwen_vram_settings(vram_preset)
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

    def apply_vram_preset(vram_preset):
        settings = qwen_vram_settings(vram_preset)
        return (
            settings["model_quantization"],
            settings["image_long_edge"],
            settings["file_batch_size"],
            settings["folder_batch_size"],
            settings["max_new_tokens"],
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

    def render_json_boxes(image, caption_text):
        final, parsed, warnings = normalize_json_output(caption_text)
        rows = json_to_element_rows(parsed)
        overlay = overlay_html(image, rows)
        status = ""
        if warnings:
            status = html_message("info", "JSON rendered with warnings:<br><pre>" + "\n".join(warnings) + "</pre>")
        return final, rows, overlay, status

    def apply_box_edits(image, caption_text, rows):
        final, parsed, warnings = apply_rows_to_json(caption_text, rows)
        overlay = overlay_html(image, json_to_element_rows(parsed))
        status = html_message("success", "JSON box edits applied.")
        if warnings:
            status = html_message("info", "Applied box edits after JSON repair fallback:<br><pre>" + "\n".join(warnings) + "</pre>")
        return final, overlay, status

    def add_box(rows):
        rows = rows or []
        if isinstance(rows, dict) and "data" in rows:
            rows = rows["data"]
        return [*rows, EMPTY_ELEMENT_ROW.copy()]

    def clear_boxes():
        return []

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
    preset_inputs = [components["preset_id"], components["vram_preset"]] + variable_inputs
    components["preset_id"].change(apply_preset, inputs=preset_inputs, outputs=preset_outputs, queue=False)
    for variable_component in variable_inputs:
        variable_component.change(apply_preset, inputs=preset_inputs, outputs=preset_outputs, queue=False)
    components["vram_preset"].change(
        apply_vram_preset,
        inputs=[components["vram_preset"]],
        outputs=[
            components["model_quantization"],
            components["image_long_edge"],
            components["file_batch_size"],
            components["folder_batch_size"],
            components["max_new_tokens"],
        ],
        queue=False,
    )

    single_btn.click(
        run_single,
        inputs=[input_image] + ordered_components,
        outputs=[single_status, output_caption, json_overlay, element_rows, global_error],
    )
    single_cancel_btn.click(engine.cancel_single, outputs=single_status, queue=False)
    open_outputs_btn.click(run_open_outputs, outputs=single_status, queue=False)
    render_json_btn.click(render_json_boxes, inputs=[input_image, output_caption], outputs=[output_caption, element_rows, json_overlay, single_status])
    apply_box_btn.click(apply_box_edits, inputs=[input_image, output_caption, element_rows], outputs=[output_caption, json_overlay, single_status])
    add_box_btn.click(add_box, inputs=[element_rows], outputs=element_rows, queue=False)
    clear_box_btn.click(clear_boxes, outputs=element_rows, queue=False)

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
