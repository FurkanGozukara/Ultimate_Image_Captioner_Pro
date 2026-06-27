from __future__ import annotations

from typing import Any

import gradio as gr

from .shared import TabUI, error_pair, run_open_folder, run_open_outputs, settings_from_values


ORDER = [
    "overwrite",
    "append",
    "remove_newlines",
    "cut_off_sentence",
    "discard_repeating_sentences",
    "save_image",
    "max_resolution",
    "use_fp16",
    "use_4bit",
    "use_subprocess",
    "allow_tf32",
    "clear_cuda_cache",
    "low_cpu_mem_usage",
    "use_sdpa_attention",
    "caption_type",
    "caption_tone",
    "caption_length",
    "max_new_tokens",
    "prefix",
    "suffix",
    "custom_prompt",
    "device_id",
    "input_folder",
    "output_folder",
    "process_subfolders",
    "skip_existing",
    "gpu_ids",
    "batch_size",
]

DEFAULTS: dict[str, Any] = {
    "overwrite": False,
    "append": False,
    "remove_newlines": True,
    "cut_off_sentence": True,
    "discard_repeating_sentences": True,
    "save_image": True,
    "max_resolution": 1536,
    "use_fp16": False,
    "use_4bit": False,
    "use_subprocess": False,
    "allow_tf32": True,
    "clear_cuda_cache": True,
    "low_cpu_mem_usage": True,
    "use_sdpa_attention": False,
    "caption_type": "descriptive",
    "caption_tone": "formal",
    "caption_length": "any",
    "max_new_tokens": 300,
    "prefix": "",
    "suffix": "",
    "custom_prompt": "",
    "device_id": "0",
    "input_folder": "",
    "output_folder": "",
    "process_subfolders": True,
    "skip_existing": True,
    "gpu_ids": "0",
    "batch_size": 1,
}


def build_tab(engine: Any) -> TabUI:
    components: dict[str, gr.components.Component] = {}

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            input_image = gr.Image(type="filepath", label="Input Image", height=440)
            with gr.Row():
                caption_btn = gr.Button("Caption Image", elem_classes=["btn-alpha1-caption"])
                cancel_single_btn = gr.Button("Cancel", elem_classes=["btn-cancel"])
                open_outputs_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])
            output_caption = gr.Textbox(label="Caption", lines=8, elem_classes=["jc-output"])
            save_info = gr.Textbox(label="Save Information", lines=4, elem_classes=["jc-status"])

        with gr.Column(scale=4, elem_classes=["jc-compact"]):
            with gr.Accordion("Caption Prompt", open=True):
                components["caption_type"] = gr.Dropdown(
                    choices=["descriptive", "training_prompt", "rng-tags"],
                    value=DEFAULTS["caption_type"],
                    label="Caption Type",
                )
                components["caption_tone"] = gr.Dropdown(
                    choices=["formal", "informal"],
                    value=DEFAULTS["caption_tone"],
                    label="Caption Tone",
                )
                components["caption_length"] = gr.Dropdown(
                    choices=["any", "very short", "short", "medium-length", "long", "very long"]
                    + [str(i) for i in range(20, 261, 10)],
                    value=DEFAULTS["caption_length"],
                    label="Caption Length",
                )
                components["max_new_tokens"] = gr.Slider(1, 1000, value=DEFAULTS["max_new_tokens"], step=1, label="Max New Tokens")
                components["custom_prompt"] = gr.Textbox(label="Custom Prompt", lines=3, value=DEFAULTS["custom_prompt"])

            with gr.Accordion("Output", open=True):
                with gr.Row():
                    components["overwrite"] = gr.Checkbox(label="Overwrite captions", value=DEFAULTS["overwrite"])
                    components["append"] = gr.Checkbox(label="Append captions", value=DEFAULTS["append"])
                with gr.Row():
                    components["remove_newlines"] = gr.Checkbox(label="Remove newlines", value=DEFAULTS["remove_newlines"])
                    components["cut_off_sentence"] = gr.Checkbox(label="Cut at last complete sentence", value=DEFAULTS["cut_off_sentence"])
                with gr.Row():
                    components["discard_repeating_sentences"] = gr.Checkbox(label="Discard repeating sentences", value=DEFAULTS["discard_repeating_sentences"])
                    components["save_image"] = gr.Checkbox(label="Save image copy", value=DEFAULTS["save_image"])
                components["max_resolution"] = gr.Slider(256, 8192, value=DEFAULTS["max_resolution"], step=1, label="Resize Input Max Resolution")
                with gr.Row():
                    components["use_fp16"] = gr.Checkbox(label="Use FP16", value=DEFAULTS["use_fp16"])
                    components["use_4bit"] = gr.Checkbox(label="Use 4-bit Quantization", value=DEFAULTS["use_4bit"])
                components["use_subprocess"] = gr.Checkbox(
                    label="Run single and batch in subprocess, then terminate it",
                    value=DEFAULTS["use_subprocess"],
                )
                with gr.Row():
                    components["prefix"] = gr.Textbox(label="Caption Prefix", value=DEFAULTS["prefix"])
                    components["suffix"] = gr.Textbox(label="Caption Suffix", value=DEFAULTS["suffix"])
                components["device_id"] = gr.Textbox(label="Single Image Device ID", value=DEFAULTS["device_id"])

            with gr.Accordion("Optimizations", open=False):
                with gr.Row():
                    components["allow_tf32"] = gr.Checkbox(label="Allow TF32 on CUDA", value=DEFAULTS["allow_tf32"])
                    components["clear_cuda_cache"] = gr.Checkbox(label="Clear CUDA cache before/after run", value=DEFAULTS["clear_cuda_cache"])
                with gr.Row():
                    components["low_cpu_mem_usage"] = gr.Checkbox(label="Low CPU memory model loading", value=DEFAULTS["low_cpu_mem_usage"])
                    components["use_sdpa_attention"] = gr.Checkbox(label="Use SDPA attention load hint", value=DEFAULTS["use_sdpa_attention"])

    with gr.Accordion("Universal Folder Batch", open=True):
        with gr.Row():
            components["input_folder"] = gr.Textbox(label="Input Folder", value=DEFAULTS["input_folder"])
            components["output_folder"] = gr.Textbox(label="Output Folder", value=DEFAULTS["output_folder"])
        with gr.Row():
            components["process_subfolders"] = gr.Checkbox(label="Process subfolders", value=DEFAULTS["process_subfolders"])
            components["skip_existing"] = gr.Checkbox(label="Skip existing captions", value=DEFAULTS["skip_existing"])
            components["gpu_ids"] = gr.Textbox(label="GPU IDs", value=DEFAULTS["gpu_ids"])
            components["batch_size"] = gr.Slider(1, 32, value=DEFAULTS["batch_size"], step=1, label="Batch Size")
        with gr.Row():
            batch_btn = gr.Button("Start Alpha 1 Folder Batch", elem_classes=["btn-alpha1-batch"])
            stop_btn = gr.Button("Cancel Batch", elem_classes=["btn-alpha1-stop"])
            open_btn = gr.Button("Open Batch Output", elem_classes=["btn-open-folder"])
        batch_progress = gr.Textbox(label="Batch Progress", lines=16, elem_classes=["jc-status"])

    ordered_components = [components[key] for key in ORDER]

    def run_single(image, *values):
        settings = settings_from_values(ORDER, values)
        try:
            result = engine.caption_single(image, settings)
            return result.caption_with_status
        except Exception as exc:
            return error_pair(exc)

    def run_batch(*values):
        settings = settings_from_values(ORDER, values)
        yield from engine.batch_folder(settings)

    caption_btn.click(run_single, inputs=[input_image] + ordered_components, outputs=[output_caption, save_info])
    cancel_single_btn.click(engine.cancel_single, outputs=save_info, queue=False)
    open_outputs_btn.click(run_open_outputs, outputs=save_info, queue=False)
    batch_btn.click(run_batch, inputs=ordered_components, outputs=batch_progress)
    stop_btn.click(engine.cancel_batch, outputs=batch_progress, queue=False)
    open_btn.click(run_open_folder, inputs=[components["output_folder"], components["input_folder"]], outputs=batch_progress)

    return TabUI(key="alpha_one", order=ORDER, defaults=DEFAULTS, inputs=ordered_components)
