from __future__ import annotations

import gc
import tempfile
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama

    LIGER_AVAILABLE = True
except Exception:  # pragma: no cover - optional import
    apply_liger_kernel_to_llama = None  # type: ignore[assignment]
    LIGER_AVAILABLE = False

from ..attention import attention_load_kwargs, attention_runtime_context, normalize_attention_backend
from ..common import (
    BatchStopFlag,
    IMAGE_EXTENSIONS,
    NAME_OPTION,
    OUTPUTS_DIR,
    apply_torch_optimizations,
    clean_legacy_caption,
    coerce_image_path,
    copy_image_if_needed,
    discover_images,
    finalize_caption_text,
    format_exception,
    get_all_extra_options,
    load_rgb_image,
    log_event,
    natural_sort_key,
    parse_device_ids,
    remove_repeating_sentences,
    resolve_output_paths,
    reset_vram_peak_stats,
    save_caption_file,
    save_numbered_generation,
    optimization_status_text,
    vram_usage_text,
)
from ..subprocess_runner import run_worker


CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what is absent, resolution, or unobservable details. Vary sentence structure and avoid starting with 'This image is'.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what is absent, resolution, or unobservable details. Vary sentence structure and avoid starting with 'This image is'.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what is absent, resolution, or unobservable details. Vary sentence structure and avoid starting with 'This image is'.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags using lowercase underscores. Strict order: artist, copyright, character, meta, then general tags. Include counts, appearance, clothing, accessories, pose, expression, actions, and background. No extra text.",
        "Generate only comma-separated Danbooru tags using lowercase underscores. Strict order: artist, copyright, character, meta, then general tags. Include counts, appearance, clothing, accessories, pose, expression, actions, and background. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags using lowercase underscores. Strict order: artist, copyright, character, meta, then general tags. Include counts, appearance, clothing, accessories, pose, expression, actions, and background. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with artist, copyright, character, species, meta, and lore tags when present, then general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with artist, copyright, character, species, meta, and lore tags when present, then general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with artist, copyright, character, species, meta, and lore tags when present, then general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with artist, copyright, character, and meta tags when present, then general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with artist, copyright, character, and meta tags when present, then general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with artist, copyright, character, and meta tags when present, then general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about composition, style, symbolism, color, light, and any artistic movement it might belong to.",
        "Analyze this image like an art critic would with information about composition, style, symbolism, color, light, and any artistic movement it might belong to. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about composition, style, symbolism, color, light, and any artistic movement it might belong to. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

SYSTEM_PROMPT = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."


def html_message(kind: str, message: str) -> str:
    cls = {"error": "jc-error", "success": "jc-success", "info": "jc-info"}.get(kind, "jc-info")
    return f'<div class="{cls}">{message}</div>'


@dataclass
class BetaModelState:
    processor: Any | None = None
    models: dict[int | str, LlavaForConditionalGeneration] | None = None
    quant: str | None = None
    devices: list[int | str] | None = None
    optimization_key: tuple[bool, str, bool] | None = None
    attention_settings: dict[str, Any] | None = None


@dataclass
class BetaGenerationStats:
    generated_tokens: int = 0
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0


def _generation_stats_text(stats: BetaGenerationStats) -> str:
    return (
        f"{stats.generated_tokens} token(s) in {stats.elapsed_seconds:.2f}s "
        f"({stats.tokens_per_second:.2f} tok/s)"
    )


def _synchronize_if_cuda(device: Any) -> None:
    if not torch.cuda.is_available():
        return
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _token_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        token_id = int(value)
    except (TypeError, ValueError):
        return None
    return token_id if token_id >= 0 else None


def _pad_token_id(tokenizer: Any) -> int | None:
    return _token_id(getattr(tokenizer, "pad_token_id", None)) or _token_id(getattr(tokenizer, "eos_token_id", None))


def _normalize_generation_config(model: Any, tokenizer: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    pad_id = _pad_token_id(tokenizer)
    if pad_id is not None:
        generation_config.pad_token_id = pad_id
    if hasattr(generation_config, "max_length"):
        generation_config.max_length = None


class BetaEngine:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.state = BetaModelState(models={})
        self._lock = threading.RLock()
        self.stop_flag = BatchStopFlag()
        self.last_generation_stats = BetaGenerationStats()

    def clear_models(self) -> None:
        with self._lock:
            if self.state.models:
                for model in self.state.models.values():
                    del model
                self.state.models.clear()
            self.state.quant = None
            self.state.devices = None
            self.state.optimization_key = None
            self.state.attention_settings = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop_batch(self) -> str:
        self.stop_flag.stop()
        return html_message("info", "Stopping batch after the current image finishes.")

    def build_prompt(
        self,
        caption_type: str,
        caption_length: str | int,
        extra_options: Sequence[str] | None,
        name_input: str,
        custom_prompt_text: str,
    ) -> str:
        if custom_prompt_text and custom_prompt_text.strip():
            return custom_prompt_text.strip()
        if caption_type not in CAPTION_TYPE_MAP:
            return "Error: Invalid caption type selected."
        if caption_length == "any":
            idx = 0
        elif isinstance(caption_length, str) and caption_length.isdigit():
            idx = 1
        else:
            idx = 2
        prompt = CAPTION_TYPE_MAP[caption_type][idx]
        selected = list(extra_options or [])
        if (not name_input or not name_input.strip()) and NAME_OPTION in selected:
            selected.remove(NAME_OPTION)
        if selected:
            prompt += " " + " ".join(opt for opt in selected if opt)
        return (
            prompt.replace("{name}", name_input or "{NAME}")
            .replace("{length}", str(caption_length))
            .replace("{word_count}", str(caption_length))
            .strip()
        )

    def _load_processor(self) -> Any:
        if self.state.processor is None:
            log_event(f"Loading processor from {self.model_path}.", "Joy Caption Beta 1")
            processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, backend="pil")
            if getattr(processor.tokenizer, "pad_token", None) is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            self.state.processor = processor
            log_event("Processor ready.", "Joy Caption Beta 1")
        return self.state.processor

    def _quant_config(self, quant: str) -> BitsAndBytesConfig | None:
        params = {"llm_int8_skip_modules": ["vision_tower", "multi_modal_projector"]}
        if quant == "bf16":
            return None
        if quant == "int8":
            return BitsAndBytesConfig(load_in_8bit=True, **params)
        if quant == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                **params,
            )
        raise ValueError(f"Unknown quantization type: {quant}")

    def load_model(self, quant: str, device_id_str: str, optimizations: dict[str, Any] | None = None) -> str:
        optimizations = optimizations or {}
        log_event(f"Model load requested | quant={quant} | devices={device_id_str}.", "Joy Caption Beta 1")
        apply_torch_optimizations(optimizations, "before")
        with self._lock:
            processor = self._load_processor()
            devices = parse_device_ids(device_id_str, allow_cpu=True)
            optimization_key = (
                bool(optimizations.get("low_cpu_mem_usage", False)),
                normalize_attention_backend(optimizations),
                bool(optimizations.get("use_liger_kernel", True)),
            )
            if (
                self.state.models
                and self.state.quant == quant
                and self.state.devices == devices
                and self.state.optimization_key == optimization_key
            ):
                log_event("Using cached model.", "Joy Caption Beta 1")
                return "Model ready."
            self.clear_models()
            self.state.processor = processor
            models: dict[int | str, LlavaForConditionalGeneration] = {}
            for device_id in devices:
                if device_id == "cpu":
                    device_map: str = "cpu"
                    if quant != "bf16":
                        raise RuntimeError("int8 and nf4 quantization require CUDA in this app.")
                else:
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is not available. Use device 'cpu' for CPU loading.")
                    if int(device_id) >= torch.cuda.device_count():
                        raise RuntimeError(f"Invalid CUDA device ID {device_id}. Available count: {torch.cuda.device_count()}.")
                    torch.cuda.set_device(int(device_id))
                    device_map = f"cuda:{int(device_id)}"
                qconfig = self._quant_config(quant)
                load_kwargs: dict[str, Any] = {}
                if optimizations.get("low_cpu_mem_usage", False):
                    load_kwargs["low_cpu_mem_usage"] = True
                load_kwargs.update(attention_load_kwargs(optimizations, quant=quant))
                log_event(f"Loading model on {device_map} with quant={quant}.", "Joy Caption Beta 1")
                if quant == "bf16":
                    model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_path,
                        dtype=torch.bfloat16,
                        device_map=device_map,
                        trust_remote_code=True,
                        **load_kwargs,
                    )
                    if (
                        optimizations.get("use_liger_kernel", True)
                        and LIGER_AVAILABLE
                        and apply_liger_kernel_to_llama is not None
                        and device_id != "cpu"
                    ):
                        try:
                            log_event("Applying Liger kernel.", "Joy Caption Beta 1")
                            with torch.cuda.device(int(device_id)):
                                apply_liger_kernel_to_llama(model=model.language_model)
                        except Exception:
                            log_event("Liger kernel was requested but could not be applied; continuing.", "Joy Caption Beta 1")
                else:
                    model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_path,
                        dtype="auto",
                        device_map=device_map,
                        quantization_config=qconfig,
                        trust_remote_code=True,
                        **load_kwargs,
                    )
                _normalize_generation_config(model, processor.tokenizer)
                model.eval()
                models[device_id] = model
                log_event(f"Model ready on {device_map}.", "Joy Caption Beta 1")
            self.state.models = models
            self.state.quant = quant
            self.state.devices = devices
            self.state.optimization_key = optimization_key
            self.state.attention_settings = dict(optimizations)
            log_event(f"Model load complete on {', '.join(str(d) for d in devices)}.", "Joy Caption Beta 1")
            return f"Model ready on {', '.join(str(d) for d in devices)}."

    def _first_model(self) -> tuple[Any, LlavaForConditionalGeneration, int | str]:
        processor = self.state.processor
        models = self.state.models or {}
        if processor is None or not models:
            raise RuntimeError("Model is not loaded.")
        device_id = next(iter(models))
        return processor, models[device_id], device_id

    @torch.inference_mode()
    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        processor, model, _device_id = self._first_model()
        log_event(f"Preparing Beta prompt and image tensors (max_new_tokens={int(max_new_tokens)}).", "Joy Caption Beta 1")
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ]
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(model.device)
        if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "dtype"):
            inputs["pixel_values"] = inputs["pixel_values"].to(model.vision_tower.dtype)
        do_sample = bool(float(temperature) > 0)
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        pad_id = _pad_token_id(processor.tokenizer)
        if pad_id is not None:
            generation_kwargs["pad_token_id"] = pad_id
        if do_sample:
            generation_kwargs["temperature"] = max(float(temperature), 1e-5)
            generation_kwargs["top_p"] = float(top_p)
        log_event(f"Generating caption | do_sample={do_sample}.", "Joy Caption Beta 1")
        self.last_generation_stats = BetaGenerationStats()
        _synchronize_if_cuda(model.device)
        generation_started = time.time()
        with attention_runtime_context(self.state.attention_settings or {}):
            generate_ids = model.generate(**generation_kwargs)
        _synchronize_if_cuda(model.device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        preds = generate_ids[:, inputs["input_ids"].shape[1] :]
        generated_token_count = max(0, int(preds.shape[-1]))
        self.last_generation_stats = BetaGenerationStats(
            generated_tokens=generated_token_count,
            elapsed_seconds=generation_elapsed,
            tokens_per_second=generated_token_count / generation_elapsed,
        )
        caption = processor.tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        log_event(f"Generation complete. Token speed: {_generation_stats_text(self.last_generation_stats)}.", "Joy Caption Beta 1")
        return caption.strip()

    def caption_single(
        self,
        input_image: Any | None,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        quant: str,
        device_id: str,
        unload_after_caption: bool,
        save_image: bool = True,
        use_subprocess: bool = False,
        allow_tf32: bool = True,
        clear_cuda_cache: bool = True,
        low_cpu_mem_usage: bool = True,
        attention_backend: str = "sdpa",
        use_liger_kernel: bool = True,
        remove_newlines: bool = True,
        discard_repeats: bool = True,
        caption_prefix: str = "",
        caption_suffix: str = "",
        replace_pairs: Any | None = None,
        replace_case_sensitive: bool = False,
        replace_single_word: bool = False,
    ) -> Generator[tuple[str, str, str], None, None]:
        log_event("Single image caption requested.", "Joy Caption Beta 1")
        optimizations = {
            "allow_tf32": allow_tf32,
            "clear_cuda_cache": clear_cuda_cache,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "attention_backend": attention_backend,
            "use_liger_kernel": use_liger_kernel,
        }
        image_path = coerce_image_path(input_image, OUTPUTS_DIR / "temp")
        if image_path is None:
            yield html_message("error", "No image selected."), "", ""
            return
        delete_after = isinstance(input_image, Image.Image)
        if use_subprocess:
            try:
                yield html_message("info", "Starting subprocess caption run..."), "", ""
                data = run_worker(
                    "beta_single",
                    {
                        "image_path": str(image_path),
                        "settings": {
                            "prompt": prompt,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                            "quant": quant,
                            "device_id": device_id,
                            "save_image": save_image,
                            "remove_newlines": remove_newlines,
                            "discard_repeats": discard_repeats,
                            "caption_prefix": caption_prefix,
                            "caption_suffix": caption_suffix,
                            "replace_pairs": replace_pairs,
                            "replace_case_sensitive": replace_case_sensitive,
                            "replace_single_word": replace_single_word,
                            **optimizations,
                        },
                    },
                )
                detail_lines = [
                    f"Caption saved to: {data.get('caption_path')}" if data.get("caption_path") else "Caption was generated but not saved.",
                    f"Image output: {data.get('image_path')}" if data.get("image_path") else "Image copy disabled.",
                    f"Metadata saved to: {data.get('metadata_path')}" if data.get("metadata_path") else "",
                    str(data.get("details") or ""),
                ]
                detail = "<br>".join(line for line in detail_lines if line).replace("\n", "<br>")
                yield html_message("success", f"Subprocess captioning complete. Child process exited.<br>{detail}"), str(data.get("caption") or ""), ""
                return
            except Exception as exc:
                traceback.print_exc()
                yield html_message("error", format_exception(exc)), "", html_message("error", "Subprocess generation failed.")
                return
            finally:
                if delete_after:
                    try:
                        image_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        try:
            start = time.time()
            yield html_message("info", "Loading model..."), "", ""
            reset_vram_peak_stats(parse_device_ids(device_id, allow_cpu=True))
            before_vram = vram_usage_text()
            status = self.load_model(quant, device_id, optimizations)
            yield html_message("info", f"{status} Generating caption..."), "", ""
            log_event(f"Loading image: {image_path}", "Joy Caption Beta 1")
            image = load_rgb_image(image_path)
            raw_caption = self.generate_caption(image, prompt, temperature, top_p, max_new_tokens)
            generation_stats = self.last_generation_stats
            apply_torch_optimizations(optimizations, "after")
            after_vram = vram_usage_text()
            caption = raw_caption
            if discard_repeats:
                caption = remove_repeating_sentences(caption)
            caption = clean_legacy_caption(caption)
            final_caption = finalize_caption_text(
                caption,
                remove_newlines=remove_newlines,
                prefix=caption_prefix,
                suffix=caption_suffix,
                replace_pairs=replace_pairs,
                replace_case_sensitive=replace_case_sensitive,
                replace_single_word=replace_single_word,
            )
            metadata = {
                "generation_type": "single_image",
                "engine": "beta_one",
                "model_path": str(self.model_path),
                "source_image_path": str(image_path),
                "prompt": prompt,
                "caption_raw": raw_caption,
                "caption_final": final_caption,
                "settings": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "quant": quant,
                    "device_id": device_id,
                    "save_image": save_image,
                    "remove_newlines": remove_newlines,
                    "discard_repeats": discard_repeats,
                    "caption_prefix": caption_prefix,
                    "caption_suffix": caption_suffix,
                    "replace_pairs": replace_pairs,
                    "replace_case_sensitive": replace_case_sensitive,
                    "replace_single_word": replace_single_word,
                    **optimizations,
                },
                "elapsed_seconds": time.time() - start,
                "generated_tokens": generation_stats.generated_tokens,
                "generation_elapsed_seconds": generation_stats.elapsed_seconds,
                "tokens_per_second": generation_stats.tokens_per_second,
                "vram_before": before_vram,
                "vram_after": after_vram,
                "optimizations": optimization_status_text(optimizations),
            }
            output_image_path, caption_path, metadata_path, _run_dir = save_numbered_generation(
                image_path,
                final_caption,
                metadata,
                OUTPUTS_DIR,
                copy_image=bool(save_image),
            )
            log_event(f"Single image caption saved: {caption_path}", "Joy Caption Beta 1")
            detail = (
                f"Caption saved to: {caption_path}<br>"
                f"Image output: {output_image_path}<br>"
                f"Metadata saved to: {metadata_path}<br>"
                f"Token speed: {_generation_stats_text(generation_stats)}<br>"
                f"{optimization_status_text(optimizations)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"
            )
            yield html_message("success", f"Captioning complete.<br>{detail}"), final_caption, ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), "", html_message("error", "Generation failed. Check the terminal for details.")
        finally:
            if delete_after:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if unload_after_caption:
                self.clear_models()

    def _file_path(self, item: Any) -> Path:
        if isinstance(item, (str, Path)):
            return Path(item)
        name = getattr(item, "name", None)
        if name:
            return Path(name)
        raise ValueError(f"Unsupported file object: {item!r}")

    @torch.inference_mode()
    def process_batch_files_to_zip(
        self,
        files_list: Sequence[Any] | None,
        caption_type: str,
        caption_length: str | int,
        extra_options: Sequence[str] | None,
        name_input: str,
        custom_prompt_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        _num_workers: int,
        batch_size: int,
        quant: str,
        device_id: str,
        use_subprocess: bool = False,
        allow_tf32: bool = True,
        clear_cuda_cache: bool = True,
        low_cpu_mem_usage: bool = True,
        attention_backend: str = "sdpa",
        use_liger_kernel: bool = True,
        remove_newlines: bool = True,
        discard_repeats: bool = True,
        caption_prefix: str = "",
        caption_suffix: str = "",
        replace_pairs: Any | None = None,
        replace_case_sensitive: bool = False,
        replace_single_word: bool = False,
    ) -> Generator[tuple[str, str | None, str], None, None]:
        log_event("Files-to-ZIP batch requested.", "Joy Caption Beta 1")
        optimizations = {
            "allow_tf32": allow_tf32,
            "clear_cuda_cache": clear_cuda_cache,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "attention_backend": attention_backend,
            "use_liger_kernel": use_liger_kernel,
        }
        if not files_list:
            yield html_message("error", "No files selected."), None, ""
            return
        if use_subprocess:
            try:
                paths = [str(self._file_path(item)) for item in files_list]
                yield html_message("info", "Starting subprocess ZIP batch..."), None, ""
                data = run_worker(
                    "beta_zip",
                    {
                        "files": paths,
                        "settings": {
                            "caption_type": caption_type,
                            "caption_length": caption_length,
                            "extra_options": list(extra_options or []),
                            "name_input": name_input,
                            "custom_prompt": custom_prompt_text,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                            "num_workers": _num_workers,
                            "batch_size": batch_size,
                            "quant": quant,
                            "device_id": device_id,
                            "remove_newlines": remove_newlines,
                            "discard_repeats": discard_repeats,
                            "caption_prefix": caption_prefix,
                            "caption_suffix": caption_suffix,
                            "replace_pairs": replace_pairs,
                            "replace_case_sensitive": replace_case_sensitive,
                            "replace_single_word": replace_single_word,
                            **optimizations,
                        },
                    },
                )
                status = str(data.get("status") or html_message("success", "Subprocess ZIP batch complete. Child process exited."))
                yield status, data.get("zip_path"), str(data.get("error") or "")
                return
            except Exception as exc:
                traceback.print_exc()
                yield html_message("error", format_exception(exc)), None, html_message("error", "Subprocess ZIP batch failed.")
                return
        self.stop_flag.reset()
        try:
            yield html_message("info", "Loading model..."), None, ""
            reset_vram_peak_stats(parse_device_ids(device_id, allow_cpu=True))
            before_vram = vram_usage_text()
            self.load_model(quant, device_id, optimizations)
            paths = sorted([self._file_path(item) for item in files_list], key=natural_sort_key)
            prompt = self.build_prompt(caption_type, caption_length, extra_options, name_input, custom_prompt_text)
            captions: dict[str, str] = {}
            total = len(paths)
            total_generated_tokens = 0
            total_generation_seconds = 0.0
            log_event(f"Files-to-ZIP batch started: {total} image(s), batch_size={batch_size}.", "Joy Caption Beta 1")
            for offset in range(0, total, max(1, int(batch_size))):
                if self.stop_flag.value:
                    yield html_message("info", f"ZIP batch cancelled. Processed {len(captions)}/{total} images."), None, ""
                    return
                batch = paths[offset : offset + max(1, int(batch_size))]
                for path in batch:
                    if self.stop_flag.value:
                        break
                    log_event(f"ZIP batch captioning {path.name}.", "Joy Caption Beta 1")
                    image = load_rgb_image(path)
                    caption = self.generate_caption(image, prompt, temperature, top_p, max_new_tokens)
                    stats = self.last_generation_stats
                    total_generated_tokens += stats.generated_tokens
                    total_generation_seconds += stats.elapsed_seconds
                    if discard_repeats:
                        caption = remove_repeating_sentences(caption)
                    caption = clean_legacy_caption(caption)
                    captions[path.with_suffix(".txt").name] = finalize_caption_text(
                        caption,
                        remove_newlines=remove_newlines,
                        prefix=caption_prefix,
                        suffix=caption_suffix,
                        replace_pairs=replace_pairs,
                        replace_case_sensitive=replace_case_sensitive,
                        replace_single_word=replace_single_word,
                    )
                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                yield html_message(
                    "info",
                    f"Processed {min(offset + len(batch), total)}/{total} images. Token speed {token_speed:.2f} tok/s.",
                ), None, ""
                if self.stop_flag.value:
                    yield html_message("info", f"ZIP batch cancelled. Processed {len(captions)}/{total} images."), None, ""
                    return
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as archive:
                for filename, text in captions.items():
                    archive.writestr(filename, text)
            apply_torch_optimizations(optimizations, "after")
            after_vram = vram_usage_text()
            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
            token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
            log_event(f"Files-to-ZIP batch complete: {tmp.name}", "Joy Caption Beta 1")
            yield html_message("success", f"ZIP batch complete. Processed {len(captions)}/{total} images.<br>Token speed: {token_detail}<br>{optimization_status_text(optimizations)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), tmp.name, ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), None, html_message("error", "ZIP batch failed. Check the terminal for details.")

    @torch.inference_mode()
    def run_batch_folder_processing(
        self,
        input_folder_str: str,
        output_folder_str: str,
        copy_images_cb: bool,
        skip_exists_cb: bool,
        overwrite_caption_cb: bool,
        append_caption_cb: bool,
        remove_newlines_cb: bool,
        discard_repeats_cb: bool,
        process_subfolders_cb: bool,
        downscale_max_res_str: str,
        caption_prefix: str,
        caption_suffix: str,
        caption_type: str,
        caption_length: str | int,
        extra_options: Sequence[str] | None,
        name_input: str,
        custom_prompt_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        _num_workers: int,
        batch_size: int,
        quant: str,
        device_id: str,
        use_subprocess: bool = False,
        allow_tf32: bool = True,
        clear_cuda_cache: bool = True,
        low_cpu_mem_usage: bool = True,
        attention_backend: str = "sdpa",
        use_liger_kernel: bool = True,
        replace_pairs: Any | None = None,
        replace_case_sensitive: bool = False,
        replace_single_word: bool = False,
    ) -> Generator[tuple[str, str], None, None]:
        log_event("Folder batch requested.", "Joy Caption Beta 1")
        optimizations = {
            "allow_tf32": allow_tf32,
            "clear_cuda_cache": clear_cuda_cache,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "attention_backend": attention_backend,
            "use_liger_kernel": use_liger_kernel,
        }
        if use_subprocess:
            try:
                yield html_message("info", "Starting subprocess folder batch..."), ""
                data = run_worker(
                    "beta_folder",
                    {
                        "settings": {
                            "input_folder": input_folder_str,
                            "output_folder": output_folder_str,
                            "copy_images": copy_images_cb,
                            "skip_exists": skip_exists_cb,
                            "overwrite_caption": overwrite_caption_cb,
                            "append_caption": append_caption_cb,
                            "remove_newlines": remove_newlines_cb,
                            "discard_repeats": discard_repeats_cb,
                            "process_subfolders": process_subfolders_cb,
                            "downscale_max_res": downscale_max_res_str,
                            "caption_prefix": caption_prefix,
                            "caption_suffix": caption_suffix,
                            "replace_pairs": replace_pairs,
                            "replace_case_sensitive": replace_case_sensitive,
                            "replace_single_word": replace_single_word,
                            "caption_type": caption_type,
                            "caption_length": caption_length,
                            "extra_options": list(extra_options or []),
                            "name_input": name_input,
                            "custom_prompt": custom_prompt_text,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                            "num_workers": _num_workers,
                            "batch_size": batch_size,
                            "quant": quant,
                            "device_id": device_id,
                            **optimizations,
                        },
                    },
                )
                yield str(data.get("status") or html_message("success", "Subprocess folder batch complete. Child process exited.")), str(data.get("error") or "")
                return
            except Exception as exc:
                traceback.print_exc()
                yield html_message("error", format_exception(exc)), html_message("error", "Subprocess folder batch failed.")
                return

        input_text = (input_folder_str or "").strip()
        if not input_text:
            yield html_message("error", "Input folder path is required."), ""
            return
        input_dir = Path(input_text)
        if not input_dir.is_dir():
            yield html_message("error", f"Input folder not found: {input_dir}"), ""
            return
        output_dir = Path(output_folder_str.strip()) if output_folder_str and output_folder_str.strip() else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            downscale = int(downscale_max_res_str) if str(downscale_max_res_str).strip().isdigit() else 0
        except Exception:
            downscale = 0

        try:
            self.stop_flag.reset()
            log_event(f"Scanning folder: {input_dir}", "Joy Caption Beta 1")
            all_paths = discover_images(input_dir, include_subfolders=process_subfolders_cb)
            paths: list[Path] = []
            skipped = 0
            for path in all_paths:
                _, caption_path = resolve_output_paths(
                    path,
                    input_dir,
                    output_dir,
                    preserve_subfolders=process_subfolders_cb,
                )
                if caption_path.exists() and not overwrite_caption_cb:
                    skipped += 1
                    continue
                paths.append(path)
            if not paths:
                yield html_message("info", f"No images to process. Found {len(all_paths)} image files, skipped {skipped} existing captions."), ""
                return

            prompt = self.build_prompt(caption_type, caption_length, extra_options, name_input, custom_prompt_text)
            yield html_message("info", "Loading model..."), ""
            reset_vram_peak_stats(parse_device_ids(device_id, allow_cpu=True))
            before_vram = vram_usage_text()
            self.load_model(quant, device_id, optimizations)
            total = len(paths)
            processed = 0
            failed = 0
            started = time.time()
            total_generated_tokens = 0
            total_generation_seconds = 0.0
            log_event(f"Folder batch started: {total} queued, {skipped} skipped before run.", "Joy Caption Beta 1")
            for offset in range(0, total, max(1, int(batch_size))):
                if self.stop_flag.value:
                    yield html_message("info", f"Stopped. Processed {processed}/{total} queued images, {skipped} skipped."), ""
                    return
                for path in paths[offset : offset + max(1, int(batch_size))]:
                    if self.stop_flag.value:
                        break
                    try:
                        output_image_path, output_caption_path = resolve_output_paths(
                            path,
                            input_dir,
                            output_dir,
                            preserve_subfolders=process_subfolders_cb,
                        )
                        if output_caption_path.exists() and not overwrite_caption_cb:
                            skipped += 1
                            continue
                        log_event(f"Folder batch captioning {path.name}.", "Joy Caption Beta 1")
                        image = load_rgb_image(path, downscale if downscale > 0 else None)
                        caption = self.generate_caption(image, prompt, temperature, top_p, max_new_tokens)
                        stats = self.last_generation_stats
                        total_generated_tokens += stats.generated_tokens
                        total_generation_seconds += stats.elapsed_seconds
                        if discard_repeats_cb:
                            caption = remove_repeating_sentences(caption)
                        caption = clean_legacy_caption(caption)
                        actual_caption = save_caption_file(
                            output_caption_path,
                            caption,
                            overwrite=overwrite_caption_cb,
                            append=append_caption_cb,
                            remove_newlines=remove_newlines_cb,
                            prefix=caption_prefix,
                            suffix=caption_suffix,
                            replace_pairs=replace_pairs,
                            replace_case_sensitive=replace_case_sensitive,
                            replace_single_word=replace_single_word,
                        )
                        copy_image_if_needed(path, output_image_path, copy_images_cb)
                        if actual_caption:
                            processed += 1
                            log_event(f"Folder batch saved: {actual_caption}", "Joy Caption Beta 1")
                        else:
                            skipped += 1
                    except Exception as exc:
                        failed += 1
                        print(f"Failed {path}: {format_exception(exc)}")
                elapsed = max(0.01, time.time() - started)
                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                yield html_message(
                    "info",
                    f"Processed {processed}/{total} queued images, {skipped} skipped, {failed} failed. Speed {processed / elapsed:.2f} img/s, {token_speed:.2f} tok/s.",
                ), ""
            apply_torch_optimizations(optimizations, "after")
            after_vram = vram_usage_text()
            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
            token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
            log_event(f"Folder batch complete: processed={processed}, skipped={skipped}, failed={failed}.", "Joy Caption Beta 1")
            yield html_message("success", f"Folder batch complete. Processed {processed}/{len(all_paths)} images, {skipped} skipped, {failed} failed.<br>Token speed: {token_detail}<br>{optimization_status_text(optimizations)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), html_message("error", "Folder batch failed. Check the terminal for details.")


def extra_options_choices() -> list[str]:
    return get_all_extra_options()
