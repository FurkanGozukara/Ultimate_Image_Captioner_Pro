from __future__ import annotations

from typing import Any

import gradio as gr

from ..attention import ATTENTION_BACKEND_CHOICES, DEFAULT_JOY_ATTENTION
from ..common import get_all_extra_options
from ..prompt_options import ALPHA_TWO_CAPTION_TYPE_MAP
from ..vram import VRAM_PRESET_CHOICES, default_vram_preset, legacy_vram_settings
from ..torch_compile import DEFAULT_COMPILE_SETTINGS
from .shared import TabUI, build_replace_pair_controls, build_torch_compile_controls, error_triple, run_open_folder, run_open_outputs, settings_from_values


ORDER = [
    "vram_preset",
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
    "attention_backend",
    "torch_compile",
    "compile_backend",
    "compile_mode",
    "compile_dynamic",
    "compile_fullgraph",
    "compile_cache_size_limit",
    "compile_threads",
    "caption_type",
    "caption_length",
    "extra_options",
    "name_input",
    "max_new_tokens",
    "prefix",
    "suffix",
    "replace_pairs",
    "replace_case_sensitive",
    "replace_single_word",
    "custom_prompt",
    "device_id",
    "input_folder",
    "output_folder",
    "process_subfolders",
    "gpu_ids",
    "batch_size",
]

DEFAULTS: dict[str, Any] = {
    "vram_preset": default_vram_preset(),
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
    "attention_backend": DEFAULT_JOY_ATTENTION,
    **DEFAULT_COMPILE_SETTINGS,
    "caption_type": "Descriptive",
    "caption_length": "long",
    "extra_options": [],
    "name_input": "",
    "max_new_tokens": 300,
    "prefix": "",
    "suffix": "",
    "replace_pairs": [],
    "replace_case_sensitive": False,
    "replace_single_word": False,
    "custom_prompt": "",
    "device_id": "0",
    "input_folder": "",
    "output_folder": "",
    "process_subfolders": True,
    "gpu_ids": "0",
    "batch_size": 1,
}


def build_tab(engine: Any) -> TabUI:
    components: dict[str, gr.components.Component] = {}

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["jc-compact"]):
            input_image = gr.Image(type="filepath", label="Input Image", height=440)
            with gr.Row():
                caption_btn = gr.Button("Caption Image", elem_classes=["btn-alpha2-caption"])
                cancel_single_btn = gr.Button("Cancel", elem_classes=["btn-cancel"])
                open_outputs_btn = gr.Button("Open Outputs", elem_classes=["btn-open-folder"])
            output_prompt = gr.Textbox(label="Prompt Used", lines=4, elem_classes=["jc-output"])
            output_caption = gr.Textbox(label="Caption", lines=8, elem_classes=["jc-output"])

        with gr.Column(scale=4, elem_classes=["jc-compact"]):
            with gr.Accordion("Caption Prompt", open=True):
                components["caption_type"] = gr.Dropdown(
                    choices=list(ALPHA_TWO_CAPTION_TYPE_MAP.keys()),
                    value=DEFAULTS["caption_type"],
                    label="Caption Type",
                )
                components["caption_length"] = gr.Dropdown(
                    choices=["any", "very short", "short", "medium-length", "long", "very long"]
                    + [str(i) for i in range(20, 261, 10)],
                    value=DEFAULTS["caption_length"],
                    label="Caption Length",
                )
                components["extra_options"] = gr.CheckboxGroup(
                    choices=get_all_extra_options(),
                    value=DEFAULTS["extra_options"],
                    label="Extra Options",
                )
                components["name_input"] = gr.Textbox(label="Person / Character Name", value=DEFAULTS["name_input"])
                components["max_new_tokens"] = gr.Slider(1, 1000, value=DEFAULTS["max_new_tokens"], step=1, label="Max New Tokens")
                components["custom_prompt"] = gr.Textbox(label="Custom Prompt", lines=3, value=DEFAULTS["custom_prompt"])

            with gr.Accordion("Output", open=True):
                with gr.Row():
                    components["vram_preset"] = gr.Dropdown(
                        choices=VRAM_PRESET_CHOICES,
                        value=DEFAULTS["vram_preset"],
                        label="VRAM Preset",
                        allow_custom_value=False,
                    )
                    components["use_subprocess"] = gr.Checkbox(
                        label="Run single and batch in subprocess, then terminate it",
                        value=DEFAULTS["use_subprocess"],
                    )
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
                with gr.Row():
                    components["prefix"] = gr.Textbox(label="Text Prefix", value=DEFAULTS["prefix"])
                    components["suffix"] = gr.Textbox(label="Text Suffix", value=DEFAULTS["suffix"])
                build_replace_pair_controls(components, DEFAULTS)
                components["device_id"] = gr.Textbox(
                    label="Single Image Device ID",
                    value=DEFAULTS["device_id"],
                    info="Single image uses one device. For folder batch dual GPU, enter 0,1 in GPU IDs below to split images evenly across both GPUs.",
                )

            with gr.Accordion("Optimizations", open=False):
                with gr.Row():
                    components["allow_tf32"] = gr.Checkbox(label="Allow TF32 on CUDA", value=DEFAULTS["allow_tf32"])
                    components["clear_cuda_cache"] = gr.Checkbox(label="Clear CUDA cache before/after run", value=DEFAULTS["clear_cuda_cache"])
                with gr.Row():
                    components["low_cpu_mem_usage"] = gr.Checkbox(label="Low CPU memory model loading", value=DEFAULTS["low_cpu_mem_usage"])
                    components["attention_backend"] = gr.Dropdown(
                        choices=ATTENTION_BACKEND_CHOICES,
                        value=DEFAULTS["attention_backend"],
                        label="Attention Backend",
                        allow_custom_value=False,
                    )
                build_torch_compile_controls(components, DEFAULTS)

    save_info = gr.Textbox(label="Save Information", lines=4, elem_classes=["jc-status"])

    with gr.Accordion("Universal Folder Batch", open=True):
        with gr.Row():
            components["input_folder"] = gr.Textbox(label="Input Folder", value=DEFAULTS["input_folder"])
            components["output_folder"] = gr.Textbox(label="Output Folder", value=DEFAULTS["output_folder"])
        with gr.Row():
            components["overwrite"] = gr.Checkbox(label="Overwrite batch captions", value=DEFAULTS["overwrite"])
            components["append"] = gr.Checkbox(label="Append batch captions", value=DEFAULTS["append"])
            components["process_subfolders"] = gr.Checkbox(label="Process subfolders", value=DEFAULTS["process_subfolders"])
            components["gpu_ids"] = gr.Textbox(
                label="GPU IDs",
                value=DEFAULTS["gpu_ids"],
                info="Use 0,1 to run folder batch on dual GPU with equal distribution across devices.",
            )
            components["batch_size"] = gr.Slider(1, 32, value=DEFAULTS["batch_size"], step=1, label="Batch Size")
        with gr.Row():
            batch_btn = gr.Button("Start Alpha 2 Folder Batch", elem_classes=["btn-alpha2-batch"])
            stop_btn = gr.Button("Cancel Batch", elem_classes=["btn-alpha2-stop"])
            open_btn = gr.Button("Open Batch Output", elem_classes=["btn-open-folder"])
        batch_progress = gr.Textbox(label="Batch Progress", lines=16, elem_classes=["jc-status"])

    ordered_components = [components[key] for key in ORDER]

    def run_single(image, *values):
        settings = settings_from_values(ORDER, values)
        settings["overwrite"] = False
        settings["append"] = False
        try:
            if image is None:
                raise ValueError("No input image selected.")
            if engine.download_needed():
                yield "", "", f"{engine.model_spec.label} is not installed. Downloading it now with the resumable model downloader..."
            label, downloaded = engine.prepare_model()
            if downloaded:
                yield "", "", f"{label} download verified. Loading the model..."
            result = engine.caption_single(image, settings)
            caption, info = result.caption_with_status
            yield result.prompt, caption, info
        except Exception as exc:
            yield error_triple(exc)

    def run_batch(*values):
        settings = settings_from_values(ORDER, values)
        yield from engine.batch_folder(settings)

    def apply_vram_preset(vram_preset):
        settings = legacy_vram_settings(vram_preset)
        return (
            settings["use_fp16"],
            settings["use_4bit"],
            settings["max_resolution"],
            settings["attention_backend"],
            settings["batch_size"],
        )

    caption_btn.click(run_single, inputs=[input_image] + ordered_components, outputs=[output_prompt, output_caption, save_info])
    cancel_single_btn.click(engine.cancel_single, outputs=save_info, queue=False)
    open_outputs_btn.click(run_open_outputs, outputs=save_info, queue=False)
    batch_btn.click(run_batch, inputs=ordered_components, outputs=batch_progress)
    stop_btn.click(engine.cancel_batch, outputs=batch_progress, queue=False)
    open_btn.click(run_open_folder, inputs=[components["output_folder"], components["input_folder"]], outputs=batch_progress)
    components["vram_preset"].change(
        apply_vram_preset,
        inputs=[components["vram_preset"]],
        outputs=[
            components["use_fp16"],
            components["use_4bit"],
            components["max_resolution"],
            components["attention_backend"],
            components["batch_size"],
        ],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )

    return TabUI(key="alpha_two", order=ORDER, defaults=DEFAULTS, inputs=ordered_components)
