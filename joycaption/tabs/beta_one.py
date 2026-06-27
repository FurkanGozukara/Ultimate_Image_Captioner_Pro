from __future__ import annotations

import os
from typing import Any

import gradio as gr

from ..common import NAME_OPTION, get_all_extra_options, html_message, save_custom_extra_option
from ..prompt_options import BETA_CAPTION_TYPE_MAP as CAPTION_TYPE_MAP
from ..vram import VRAM_PRESET_CHOICES, beta_vram_settings, default_vram_preset
from .shared import TabUI, run_open_folder, run_open_outputs


ORDER = [
    "vram_preset",
    "model_quantization",
    "unload_model",
    "save_image",
    "use_subprocess",
    "allow_tf32",
    "clear_cuda_cache",
    "low_cpu_mem_usage",
    "use_sdpa_attention",
    "use_liger_kernel",
    "device_id",
    "caption_type",
    "caption_length",
    "temperature",
    "top_p",
    "max_tokens",
    "custom_prompt",
    "single_prompt",
    "extra_options",
    "name_input",
    "zip_num_workers",
    "zip_batch_size",
    "folder_input",
    "folder_output",
    "skip_exists",
    "overwrite_caption",
    "append_caption",
    "remove_newlines",
    "discard_repeats",
    "process_subfolders",
    "downscale_max_res",
    "caption_prefix",
    "caption_suffix",
    "folder_num_workers",
    "folder_batch_size",
]

DEFAULTS: dict[str, Any] = {
    "vram_preset": default_vram_preset(),
    "model_quantization": "bf16",
    "unload_model": False,
    "save_image": True,
    "use_subprocess": False,
    "allow_tf32": True,
    "clear_cuda_cache": True,
    "low_cpu_mem_usage": True,
    "use_sdpa_attention": False,
    "use_liger_kernel": True,
    "device_id": "0",
    "caption_type": "Descriptive",
    "caption_length": "long",
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 512,
    "custom_prompt": "",
    "single_prompt": "Write a detailed description for this image.",
    "extra_options": [],
    "name_input": "",
    "zip_num_workers": min(4, os.cpu_count() or 4),
    "zip_batch_size": 4,
    "folder_input": "",
    "folder_output": "",
    "skip_exists": True,
    "overwrite_caption": False,
    "append_caption": False,
    "remove_newlines": True,
    "discard_repeats": True,
    "process_subfolders": False,
    "downscale_max_res": "1536",
    "caption_prefix": "",
    "caption_suffix": "",
    "folder_num_workers": min(4, os.cpu_count() or 4),
    "folder_batch_size": 4,
}


def build_tab(engine: Any) -> TabUI:
    components: dict[str, gr.components.Component] = {}
    global_error = gr.HTML(visible=False)

    with gr.Tabs():
        with gr.Tab("Single Image", render_children=True):
            single_status = gr.HTML("")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    input_image = gr.Image(type="filepath", label="Upload Image", height=440)
                with gr.Column(scale=1):
                    components["single_prompt"] = gr.Textbox(
                        label="Prompt",
                        lines=5,
                        value=DEFAULTS["single_prompt"],
                        interactive=True,
                    )
                    with gr.Row():
                        single_btn = gr.Button("Caption Image", elem_classes=["btn-beta-caption"])
                        single_cancel_btn = gr.Button("Cancel", elem_classes=["btn-cancel"])
                        single_open_outputs_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])
                    output_caption = gr.Textbox(label="Generated Caption", lines=9, interactive=True, elem_classes=["jc-output"])

        with gr.Tab("Files to ZIP", render_children=True):
            batch_status = gr.HTML("")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    input_files = gr.File(
                        file_count="multiple",
                        file_types=sorted({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif", ".heic", ".heif"}),
                        label="Image Files",
                    )
                with gr.Column(scale=2):
                    components["zip_num_workers"] = gr.Slider(
                        0,
                        os.cpu_count() or 4,
                        value=DEFAULTS["zip_num_workers"],
                        step=1,
                        label="DataLoader Workers",
                    )
                    components["zip_batch_size"] = gr.Slider(1, 32, value=DEFAULTS["zip_batch_size"], step=1, label="Batch Size")
                    with gr.Row():
                        zip_btn = gr.Button("Create Caption ZIP", elem_classes=["btn-beta-zip"])
                        zip_cancel_btn = gr.Button("Cancel Batch", elem_classes=["btn-cancel"])
                        zip_open_outputs_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])
            zip_output = gr.File(label="Caption ZIP")

        with gr.Tab("Folder Batch", render_children=True):
            folder_status = gr.HTML("")
            with gr.Row():
                components["folder_input"] = gr.Textbox(label="Input Folder", value=DEFAULTS["folder_input"])
                components["folder_output"] = gr.Textbox(label="Output Folder", value=DEFAULTS["folder_output"])
            with gr.Accordion("Folder Options", open=True):
                with gr.Row():
                    components["skip_exists"] = gr.Checkbox(label="Skip existing outputs", value=DEFAULTS["skip_exists"])
                    components["process_subfolders"] = gr.Checkbox(label="Process subfolders", value=DEFAULTS["process_subfolders"])
                with gr.Row():
                    components["overwrite_caption"] = gr.Checkbox(label="Overwrite captions", value=DEFAULTS["overwrite_caption"])
                    components["append_caption"] = gr.Checkbox(label="Append captions", value=DEFAULTS["append_caption"])
                with gr.Row():
                    components["remove_newlines"] = gr.Checkbox(label="Remove newlines", value=DEFAULTS["remove_newlines"])
                    components["discard_repeats"] = gr.Checkbox(label="Discard repeating sentences", value=DEFAULTS["discard_repeats"])
                components["downscale_max_res"] = gr.Textbox(label="Downscale Max Resolution", value=DEFAULTS["downscale_max_res"])
                with gr.Row():
                    components["caption_prefix"] = gr.Textbox(label="Caption Prefix", value=DEFAULTS["caption_prefix"])
                    components["caption_suffix"] = gr.Textbox(label="Caption Suffix", value=DEFAULTS["caption_suffix"])
                with gr.Row():
                    components["folder_num_workers"] = gr.Slider(
                        0,
                        os.cpu_count() or 4,
                        value=DEFAULTS["folder_num_workers"],
                        step=1,
                        label="DataLoader Workers",
                    )
                    components["folder_batch_size"] = gr.Slider(1, 32, value=DEFAULTS["folder_batch_size"], step=1, label="Batch Size")
            with gr.Row():
                folder_btn = gr.Button("Start Folder Batch", elem_classes=["btn-beta-folder"])
                folder_stop_btn = gr.Button("Cancel Batch", elem_classes=["btn-alpha2-stop"])
                folder_open_btn = gr.Button("Open Batch Output", elem_classes=["btn-open-folder"])

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Accordion("Model", open=True):
                components["vram_preset"] = gr.Dropdown(
                    choices=VRAM_PRESET_CHOICES,
                    value=DEFAULTS["vram_preset"],
                    label="VRAM Preset",
                    allow_custom_value=False,
                )
                components["model_quantization"] = gr.Radio(
                    choices=["bf16", "int8", "nf4"],
                    value=DEFAULTS["model_quantization"],
                    label="Model Quantization",
                )
                components["unload_model"] = gr.Checkbox(label="Unload model after single caption", value=DEFAULTS["unload_model"])
                components["save_image"] = gr.Checkbox(label="Save image copy", value=DEFAULTS["save_image"])
                components["use_subprocess"] = gr.Checkbox(
                    label="Run single and batch in subprocess, then terminate it",
                    value=DEFAULTS["use_subprocess"],
                )
                components["device_id"] = gr.Textbox(label="Device ID", value=DEFAULTS["device_id"])
            with gr.Accordion("Optimizations", open=False):
                with gr.Row():
                    components["allow_tf32"] = gr.Checkbox(label="Allow TF32 on CUDA", value=DEFAULTS["allow_tf32"])
                    components["clear_cuda_cache"] = gr.Checkbox(label="Clear CUDA cache before/after run", value=DEFAULTS["clear_cuda_cache"])
                with gr.Row():
                    components["low_cpu_mem_usage"] = gr.Checkbox(label="Low CPU memory model loading", value=DEFAULTS["low_cpu_mem_usage"])
                    components["use_sdpa_attention"] = gr.Checkbox(label="Use SDPA attention load hint", value=DEFAULTS["use_sdpa_attention"])
                components["use_liger_kernel"] = gr.Checkbox(label="Apply Liger kernel when available", value=DEFAULTS["use_liger_kernel"])
        with gr.Column(scale=2):
            with gr.Accordion("Prompt", open=True):
                with gr.Row():
                    components["caption_type"] = gr.Dropdown(
                        choices=list(CAPTION_TYPE_MAP.keys()),
                        value=DEFAULTS["caption_type"],
                        label="Caption Type",
                    )
                    components["caption_length"] = gr.Dropdown(
                        choices=["any", "very short", "short", "medium-length", "long", "very long"]
                        + [str(i) for i in range(20, 261, 10)],
                        value=DEFAULTS["caption_length"],
                        label="Caption Length",
                    )
                with gr.Row():
                    components["temperature"] = gr.Slider(0.0, 2.0, value=DEFAULTS["temperature"], step=0.05, label="Temperature")
                    components["top_p"] = gr.Slider(0.0, 1.0, value=DEFAULTS["top_p"], step=0.01, label="Top-p")
                    components["max_tokens"] = gr.Slider(1, 2048, value=DEFAULTS["max_tokens"], step=1, label="Max New Tokens")
                components["custom_prompt"] = gr.Textbox(label="Custom Prompt Override", lines=3, value=DEFAULTS["custom_prompt"])

    with gr.Accordion("Extra Options", open=True):
        with gr.Row():
            new_extra_option = gr.Textbox(label="New Extra Option", lines=2)
            add_option_btn = gr.Button("Add Option", elem_classes=["btn-beta-option"])
            refresh_option_btn = gr.Button("Refresh Options", elem_classes=["btn-refresh"])
        option_status = gr.HTML("")
        components["extra_options"] = gr.CheckboxGroup(
            choices=get_all_extra_options(),
            value=DEFAULTS["extra_options"],
            label="Selected Extra Options",
        )
        components["name_input"] = gr.Textbox(label="Person / Character Name", value=DEFAULTS["name_input"], visible=False)

    ordered_components = [components[key] for key in ORDER]

    def update_prompt(caption_type, caption_length, extra_options, name_input, custom_prompt):
        return engine.build_prompt(caption_type, caption_length, extra_options or [], name_input or "", custom_prompt or "")

    def toggle_name(selected):
        return gr.update(visible=NAME_OPTION in (selected or []))

    def add_option(option_text):
        if save_custom_extra_option(option_text):
            return html_message("success", "Extra option saved."), gr.update(choices=get_all_extra_options()), ""
        return html_message("error", "Enter option text before saving."), gr.update(), option_text

    def refresh_options():
        return html_message("success", "Extra options refreshed."), gr.update(choices=get_all_extra_options())

    def apply_vram_preset(vram_preset):
        settings = beta_vram_settings(vram_preset)
        return (
            settings["model_quantization"],
            settings["downscale_max_res"],
            settings["zip_batch_size"],
            settings["folder_batch_size"],
        )

    prompt_inputs = [
        components["caption_type"],
        components["caption_length"],
        components["extra_options"],
        components["name_input"],
        components["custom_prompt"],
    ]
    for prompt_component in prompt_inputs:
        prompt_component.change(update_prompt, inputs=prompt_inputs, outputs=components["single_prompt"])
    components["extra_options"].change(toggle_name, inputs=[components["extra_options"]], outputs=[components["name_input"]])

    add_option_btn.click(add_option, inputs=[new_extra_option], outputs=[option_status, components["extra_options"], new_extra_option])
    refresh_option_btn.click(refresh_options, outputs=[option_status, components["extra_options"]])
    components["vram_preset"].change(
        apply_vram_preset,
        inputs=[components["vram_preset"]],
        outputs=[
            components["model_quantization"],
            components["downscale_max_res"],
            components["zip_batch_size"],
            components["folder_batch_size"],
        ],
        queue=False,
    )

    single_btn.click(
        engine.caption_single,
        inputs=[
            input_image,
            components["single_prompt"],
            components["temperature"],
            components["top_p"],
            components["max_tokens"],
            components["model_quantization"],
            components["device_id"],
            components["unload_model"],
            components["save_image"],
            components["use_subprocess"],
            components["allow_tf32"],
            components["clear_cuda_cache"],
            components["low_cpu_mem_usage"],
            components["use_sdpa_attention"],
            components["use_liger_kernel"],
        ],
        outputs=[single_status, output_caption, global_error],
    )
    single_cancel_btn.click(engine.cancel_single, outputs=single_status, queue=False)
    single_open_outputs_btn.click(run_open_outputs, outputs=single_status, queue=False)

    zip_btn.click(
        engine.process_batch_files_to_zip,
        inputs=[
            input_files,
            components["caption_type"],
            components["caption_length"],
            components["extra_options"],
            components["name_input"],
            components["custom_prompt"],
            components["temperature"],
            components["top_p"],
            components["max_tokens"],
            components["zip_num_workers"],
            components["zip_batch_size"],
            components["model_quantization"],
            components["device_id"],
            components["use_subprocess"],
            components["allow_tf32"],
            components["clear_cuda_cache"],
            components["low_cpu_mem_usage"],
            components["use_sdpa_attention"],
            components["use_liger_kernel"],
        ],
        outputs=[batch_status, zip_output, global_error],
    )
    zip_cancel_btn.click(engine.cancel_batch, outputs=batch_status, queue=False)
    zip_open_outputs_btn.click(run_open_outputs, outputs=batch_status, queue=False)

    folder_btn.click(
        engine.run_batch_folder_processing,
        inputs=[
            components["folder_input"],
            components["folder_output"],
            components["save_image"],
            components["skip_exists"],
            components["overwrite_caption"],
            components["append_caption"],
            components["remove_newlines"],
            components["discard_repeats"],
            components["process_subfolders"],
            components["downscale_max_res"],
            components["caption_prefix"],
            components["caption_suffix"],
            components["caption_type"],
            components["caption_length"],
            components["extra_options"],
            components["name_input"],
            components["custom_prompt"],
            components["temperature"],
            components["top_p"],
            components["max_tokens"],
            components["folder_num_workers"],
            components["folder_batch_size"],
            components["model_quantization"],
            components["device_id"],
            components["use_subprocess"],
            components["allow_tf32"],
            components["clear_cuda_cache"],
            components["low_cpu_mem_usage"],
            components["use_sdpa_attention"],
            components["use_liger_kernel"],
        ],
        outputs=[folder_status, global_error],
    )
    folder_stop_btn.click(engine.cancel_batch, outputs=folder_status, queue=False)
    folder_open_btn.click(run_open_folder, inputs=[components["folder_output"], components["folder_input"]], outputs=folder_status, queue=False)

    return TabUI(key="beta_one", order=ORDER, defaults=DEFAULTS, inputs=ordered_components)
