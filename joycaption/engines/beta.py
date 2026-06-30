from __future__ import annotations

import gc
import multiprocessing as mp
import queue
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
    batch_progress_line,
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
    split_round_robin,
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
            if hasattr(processor.tokenizer, "padding_side"):
                processor.tokenizer.padding_side = "left"
            self.state.processor = processor
            log_event("Processor ready.", "Joy Caption Beta 1")
        return self.state.processor

    def _quant_config(self, quant: str) -> BitsAndBytesConfig | None:
        params = {"llm_int8_skip_modules": ["vision_tower", "multi_modal_projector"]}
        if quant in {"bf16", "fp16"}:
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
                    if quant in {"int8", "nf4"}:
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
                if quant in {"bf16", "fp16"}:
                    dtype = torch.float32 if device_id == "cpu" else (torch.float16 if quant == "fp16" else torch.bfloat16)
                    model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_path,
                        dtype=dtype,
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

    def _model_for_device(self, device_id: int | str | None = None) -> tuple[Any, LlavaForConditionalGeneration, int | str]:
        processor = self.state.processor
        models = self.state.models or {}
        if processor is None or not models:
            raise RuntimeError("Model is not loaded.")
        if device_id is None:
            return self._first_model()
        key: int | str = "cpu" if str(device_id).lower() == "cpu" else int(device_id)
        if key not in models:
            raise RuntimeError(f"Model is not loaded on device {device_id}.")
        return processor, models[key], key

    @torch.inference_mode()
    def generate_captions(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        device_id: int | str | None = None,
    ) -> list[str]:
        if not images:
            return []
        processor, model, _device_id = self._model_for_device(device_id)
        batch_count = len(images)
        log_event(
            f"Preparing Beta prompt and image tensors (batch={batch_count}, max_new_tokens={int(max_new_tokens)}).",
            "Joy Caption Beta 1",
        )
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ]
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[convo_string] * batch_count, images=list(images), return_tensors="pt", padding=True).to(model.device)
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
        log_event(f"Generating {batch_count} Beta caption(s) in one batch | do_sample={do_sample}.", "Joy Caption Beta 1")
        self.last_generation_stats = BetaGenerationStats()
        _synchronize_if_cuda(model.device)
        generation_started = time.time()
        with attention_runtime_context(self.state.attention_settings or {}):
            generate_ids = model.generate(**generation_kwargs)
        _synchronize_if_cuda(model.device)
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        preds = generate_ids[:, inputs["input_ids"].shape[1] :]
        generated_token_count = max(0, int(preds.numel()))
        self.last_generation_stats = BetaGenerationStats(
            generated_tokens=generated_token_count,
            elapsed_seconds=generation_elapsed,
            tokens_per_second=generated_token_count / generation_elapsed,
        )
        captions = processor.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        log_event(f"Batch generation complete. Token speed: {_generation_stats_text(self.last_generation_stats)}.", "Joy Caption Beta 1")
        return [str(caption).strip() for caption in captions]

    @torch.inference_mode()
    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        captions = self.generate_captions([image], prompt, temperature, top_p, max_new_tokens)
        return captions[0] if captions else ""

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
            yield html_message("info", "Loading model(s)..."), None, ""
            devices = parse_device_ids(device_id, allow_cpu=True)
            reset_vram_peak_stats(devices)
            before_vram = vram_usage_text()
            paths = sorted([self._file_path(item) for item in files_list], key=natural_sort_key)
            prompt = self.build_prompt(caption_type, caption_length, extra_options, name_input, custom_prompt_text)
            captions: dict[str, str] = {}
            total = len(paths)
            batch_size_value = max(1, int(batch_size))
            total_generated_tokens = 0
            total_generation_seconds = 0.0
            aggregate = {"processed": 0, "failed": 0}
            aggregate_lock = threading.Lock()
            batch_queue: queue.Queue[tuple[str, str]] = queue.Queue()
            chunks = split_round_robin(paths, devices)
            started = time.time()
            log_event(
                f"Files-to-ZIP batch started: {total} image(s), batch_size={batch_size_value}, devices={devices}.",
                "Joy Caption Beta 1",
            )
            process_chunks = [
                (worker_device, chunk)
                for worker_device, chunk in zip(devices, chunks)
                if chunk
            ]

            if len(process_chunks) > 1:
                ctx = mp.get_context("spawn")
                process_queue: Any = ctx.Queue()
                processes: list[mp.Process] = []
                chunk_sizes: dict[str, int] = {}
                done_devices: set[str] = set()
                for worker_device, chunk in process_chunks:
                    chunk_sizes[str(worker_device)] = len(chunk)
                    process = ctx.Process(
                        target=_beta_zip_process_worker,
                        args=(
                            process_queue,
                            str(self.model_path),
                            [str(path) for path in chunk],
                            prompt,
                            float(temperature),
                            float(top_p),
                            int(max_new_tokens),
                            batch_size_value,
                            str(quant),
                            worker_device,
                            dict(optimizations),
                            bool(remove_newlines),
                            bool(discard_repeats),
                            str(caption_prefix),
                            str(caption_suffix),
                            replace_pairs,
                            bool(replace_case_sensitive),
                            bool(replace_single_word),
                        ),
                        daemon=False,
                    )
                    processes.append(process)
                    log_event(f"Starting Beta ZIP process worker for device {worker_device}: {len(chunk)} image(s).", "Joy Caption Beta 1")
                    process.start()

                try:
                    done_count = 0
                    while done_count < len(processes):
                        if self.stop_flag.value:
                            for process in processes:
                                if process.is_alive():
                                    process.terminate()
                            yield html_message("info", f"ZIP batch cancelled. Processed {aggregate['processed']}/{total} images."), None, ""
                            return
                        try:
                            event = process_queue.get(timeout=0.5)
                        except queue.Empty:
                            if not any(process.is_alive() for process in processes):
                                break
                            continue

                        kind = str(event.get("kind") or "")
                        worker_device = event.get("device_id")
                        if kind == "progress":
                            with aggregate_lock:
                                aggregate["processed"] += int(event.get("processed_delta", 0) or 0)
                                aggregate["failed"] += int(event.get("failed_delta", 0) or 0)
                                total_generated_tokens += int(event.get("generated_tokens", 0) or 0)
                                total_generation_seconds += float(event.get("generation_seconds", 0.0) or 0.0)
                                captions.update(event.get("captions") or {})
                                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                                line = batch_progress_line(
                                    processed=aggregate["processed"],
                                    total=total,
                                    failed=aggregate["failed"],
                                    started=started,
                                    last_batch_count=int(event.get("last_batch_count", 0) or 0),
                                    last_batch_seconds=float(event.get("last_batch_seconds", 0.0) or 0.0),
                                    token_speed=token_speed,
                                    device_id=worker_device,
                                    worker_processed=int(event.get("local_processed", 0) or 0),
                                    worker_total=int(event.get("worker_total", 0) or 0),
                                    worker_failed=int(event.get("local_failed", 0) or 0),
                                )
                            if event.get("message"):
                                line = f"{line} {event['message']}"
                            log_event(line, "Joy Caption Beta 1")
                            yield html_message("info", line), None, ""
                        elif kind == "done":
                            key = str(worker_device)
                            if key not in done_devices:
                                done_devices.add(key)
                                done_count += 1
                            log_event(
                                f"Device {worker_device}: ZIP process complete with "
                                f"{int(event.get('local_processed', 0) or 0)} processed, "
                                f"{int(event.get('local_failed', 0) or 0)} failed.",
                                "Joy Caption Beta 1",
                            )
                        elif kind == "fatal":
                            key = str(worker_device)
                            if key not in done_devices:
                                done_devices.add(key)
                                done_count += 1
                                with aggregate_lock:
                                    aggregate["failed"] += chunk_sizes.get(key, 0)
                            line = f"Device {worker_device}: ZIP process failed: {event.get('error') or 'unknown error'}"
                            log_event(line, "Joy Caption Beta 1")
                            yield html_message("error", line), None, ""

                    for process in processes:
                        process.join(timeout=5.0)
                    for worker_device, process in zip([device for device, _chunk in process_chunks], processes):
                        key = str(worker_device)
                        if process.exitcode not in (0, None) and key not in done_devices:
                            with aggregate_lock:
                                aggregate["failed"] += chunk_sizes.get(key, 0)
                            line = f"Device {worker_device}: ZIP process exited with code {process.exitcode}."
                            log_event(line, "Joy Caption Beta 1")
                            yield html_message("error", line), None, ""
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=2.0)

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as archive:
                    for filename, text in captions.items():
                        archive.writestr(filename, text)
                apply_torch_optimizations(optimizations, "after")
                after_vram = vram_usage_text()
                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
                log_event(f"Files-to-ZIP batch complete: {tmp.name}", "Joy Caption Beta 1")
                yield html_message("success", f"ZIP batch complete. Processed {len(captions)}/{total} images, {aggregate['failed']} failed.<br>Token speed: {token_detail}<br>{optimization_status_text(optimizations)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), tmp.name, ""
                return

            def worker(worker_device: int | str, chunk: list[Path]) -> None:
                nonlocal total_generated_tokens, total_generation_seconds
                worker_engine = self if len(devices) == 1 else BetaEngine(self.model_path)
                local_processed = 0
                local_failed = 0
                try:
                    worker_engine.load_model(quant, str(worker_device), optimizations)
                    for offset in range(0, len(chunk), batch_size_value):
                        if self.stop_flag.value:
                            break
                        batch_paths = chunk[offset : offset + batch_size_value]
                        try:
                            images = [load_rgb_image(path) for path in batch_paths]
                            log_event(
                                f"Device {worker_device}: ZIP true batch {offset + 1}-{offset + len(batch_paths)} of {len(chunk)}.",
                                "Joy Caption Beta 1",
                            )
                            batch_started = time.time()
                            raw_captions = worker_engine.generate_captions(
                                images,
                                prompt,
                                temperature,
                                top_p,
                                max_new_tokens,
                                device_id=worker_device,
                            )
                            batch_elapsed = max(time.time() - batch_started, 1e-9)
                            stats = worker_engine.last_generation_stats
                            with aggregate_lock:
                                total_generated_tokens += stats.generated_tokens
                                total_generation_seconds += stats.elapsed_seconds
                                for path, caption in zip(batch_paths, raw_captions):
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
                                    aggregate["processed"] += 1
                                    local_processed += 1
                                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                                line = batch_progress_line(
                                    processed=aggregate["processed"],
                                    total=total,
                                    failed=aggregate["failed"],
                                    started=started,
                                    last_batch_count=len(batch_paths),
                                    last_batch_seconds=batch_elapsed,
                                    token_speed=token_speed,
                                    device_id=worker_device if len(devices) > 1 else None,
                                    worker_processed=local_processed,
                                    worker_total=len(chunk),
                                    worker_failed=local_failed,
                                )
                            batch_queue.put(("progress", line))
                        except Exception as exc:
                            with aggregate_lock:
                                aggregate["failed"] += len(batch_paths)
                                local_failed += len(batch_paths)
                                line = batch_progress_line(
                                    processed=aggregate["processed"],
                                    total=total,
                                    failed=aggregate["failed"],
                                    started=started,
                                    last_batch_count=0,
                                    last_batch_seconds=0.0,
                                    token_speed=total_generated_tokens / max(total_generation_seconds, 1e-9),
                                    device_id=worker_device if len(devices) > 1 else None,
                                    worker_processed=local_processed,
                                    worker_total=len(chunk),
                                    worker_failed=local_failed,
                                )
                            batch_queue.put(("progress", f"{line} Failed batch: {format_exception(exc)}"))
                finally:
                    if worker_engine is not self:
                        worker_engine.clear_models()
                    batch_queue.put(("done", f"Device {worker_device}: done."))

            threads = [
                threading.Thread(target=worker, args=(worker_device, chunk), daemon=True)
                for worker_device, chunk in zip(devices, chunks)
                if chunk
            ]
            for thread in threads:
                thread.start()

            done_count = 0
            while done_count < len(threads):
                try:
                    kind, line = batch_queue.get(timeout=0.5)
                except queue.Empty:
                    if not any(thread.is_alive() for thread in threads):
                        break
                    continue
                if kind == "done":
                    done_count += 1
                    log_event(line, "Joy Caption Beta 1")
                else:
                    log_event(line, "Joy Caption Beta 1")
                    yield html_message("info", line), None, ""
                if self.stop_flag.value:
                    yield html_message("info", f"ZIP batch cancelled. Processed {aggregate['processed']}/{total} images."), None, ""
                    return

            for thread in threads:
                thread.join(timeout=1.0)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as archive:
                for filename, text in captions.items():
                    archive.writestr(filename, text)
            apply_torch_optimizations(optimizations, "after")
            after_vram = vram_usage_text()
            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
            token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
            log_event(f"Files-to-ZIP batch complete: {tmp.name}", "Joy Caption Beta 1")
            yield html_message("success", f"ZIP batch complete. Processed {len(captions)}/{total} images, {aggregate['failed']} failed.<br>Token speed: {token_detail}<br>{optimization_status_text(optimizations)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), tmp.name, ""
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
                if caption_path.exists() and not overwrite_caption_cb and not append_caption_cb:
                    skipped += 1
                    continue
                paths.append(path)
            if not paths:
                yield html_message("info", f"No images to process. Found {len(all_paths)} image files, skipped {skipped} existing captions."), ""
                return

            prompt = self.build_prompt(caption_type, caption_length, extra_options, name_input, custom_prompt_text)
            devices = parse_device_ids(device_id, allow_cpu=True)
            yield html_message(
                "info",
                f"Loading model(s). Queued {len(paths)} image(s), skipped {skipped}. Devices: {', '.join(str(device) for device in devices)}.",
            ), ""
            reset_vram_peak_stats(devices)
            before_vram = vram_usage_text()
            total = len(all_paths)
            batch_size_value = max(1, int(batch_size))
            aggregate = {"processed": 0, "skipped": skipped, "failed": 0}
            started = time.time()
            total_generated_tokens = 0
            total_generation_seconds = 0.0
            aggregate_lock = threading.Lock()
            batch_queue: queue.Queue[tuple[str, str]] = queue.Queue()
            chunks = split_round_robin(paths, devices)
            log_event(
                f"Folder batch started: {len(paths)} queued, {skipped} skipped before run, batch_size={batch_size_value}, devices={devices}.",
                "Joy Caption Beta 1",
            )
            process_chunks = [
                (worker_device, chunk)
                for worker_device, chunk in zip(devices, chunks)
                if chunk
            ]

            if len(process_chunks) > 1:
                ctx = mp.get_context("spawn")
                process_queue: Any = ctx.Queue()
                processes: list[mp.Process] = []
                chunk_sizes: dict[str, int] = {}
                done_devices: set[str] = set()
                for worker_device, chunk in process_chunks:
                    chunk_sizes[str(worker_device)] = len(chunk)
                    process = ctx.Process(
                        target=_beta_folder_process_worker,
                        args=(
                            process_queue,
                            str(self.model_path),
                            [str(path) for path in chunk],
                            str(input_dir),
                            str(output_dir),
                            bool(copy_images_cb),
                            bool(skip_exists_cb),
                            bool(overwrite_caption_cb),
                            bool(append_caption_cb),
                            bool(remove_newlines_cb),
                            bool(discard_repeats_cb),
                            bool(process_subfolders_cb),
                            int(downscale),
                            str(caption_prefix),
                            str(caption_suffix),
                            prompt,
                            float(temperature),
                            float(top_p),
                            int(max_new_tokens),
                            batch_size_value,
                            str(quant),
                            worker_device,
                            dict(optimizations),
                            replace_pairs,
                            bool(replace_case_sensitive),
                            bool(replace_single_word),
                        ),
                        daemon=False,
                    )
                    processes.append(process)
                    log_event(f"Starting Beta process worker for device {worker_device}: {len(chunk)} image(s).", "Joy Caption Beta 1")
                    process.start()

                try:
                    done_count = 0
                    while done_count < len(processes):
                        if self.stop_flag.value:
                            for process in processes:
                                if process.is_alive():
                                    process.terminate()
                            yield html_message("info", "Beta folder batch stopped. Child GPU processes were terminated."), ""
                            return
                        try:
                            event = process_queue.get(timeout=0.5)
                        except queue.Empty:
                            if not any(process.is_alive() for process in processes):
                                break
                            continue

                        kind = str(event.get("kind") or "")
                        worker_device = event.get("device_id")
                        if kind == "progress":
                            with aggregate_lock:
                                aggregate["processed"] += int(event.get("processed_delta", 0) or 0)
                                aggregate["skipped"] += int(event.get("skipped_delta", 0) or 0)
                                aggregate["failed"] += int(event.get("failed_delta", 0) or 0)
                                total_generated_tokens += int(event.get("generated_tokens", 0) or 0)
                                total_generation_seconds += float(event.get("generation_seconds", 0.0) or 0.0)
                                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                                line = batch_progress_line(
                                    processed=aggregate["processed"],
                                    total=total,
                                    skipped=aggregate["skipped"],
                                    failed=aggregate["failed"],
                                    started=started,
                                    last_batch_count=int(event.get("last_batch_count", 0) or 0),
                                    last_batch_seconds=float(event.get("last_batch_seconds", 0.0) or 0.0),
                                    token_speed=token_speed,
                                    device_id=worker_device,
                                    worker_processed=int(event.get("local_processed", 0) or 0),
                                    worker_total=int(event.get("worker_total", 0) or 0),
                                    worker_skipped=int(event.get("local_skipped", 0) or 0),
                                    worker_failed=int(event.get("local_failed", 0) or 0),
                                )
                            if event.get("message"):
                                line = f"{line} {event['message']}"
                            log_event(line, "Joy Caption Beta 1")
                            yield html_message("info", line), ""
                        elif kind == "done":
                            key = str(worker_device)
                            if key not in done_devices:
                                done_devices.add(key)
                                done_count += 1
                            line = (
                                f"Device {worker_device}: process complete with "
                                f"{int(event.get('local_processed', 0) or 0)} processed, "
                                f"{int(event.get('local_skipped', 0) or 0)} skipped, "
                                f"{int(event.get('local_failed', 0) or 0)} failed."
                            )
                            log_event(line, "Joy Caption Beta 1")
                            yield html_message("info", line), ""
                        elif kind == "fatal":
                            key = str(worker_device)
                            if key not in done_devices:
                                done_devices.add(key)
                                done_count += 1
                                with aggregate_lock:
                                    aggregate["failed"] += chunk_sizes.get(key, 0)
                            line = f"Device {worker_device}: process failed: {event.get('error') or 'unknown error'}"
                            log_event(line, "Joy Caption Beta 1")
                            yield html_message("error", line), ""

                    for process in processes:
                        process.join(timeout=5.0)
                    for worker_device, process in zip([device for device, _chunk in process_chunks], processes):
                        key = str(worker_device)
                        if process.exitcode not in (0, None) and key not in done_devices:
                            with aggregate_lock:
                                aggregate["failed"] += chunk_sizes.get(key, 0)
                            line = f"Device {worker_device}: process exited with code {process.exitcode}."
                            log_event(line, "Joy Caption Beta 1")
                            yield html_message("error", line), ""
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=2.0)

                apply_torch_optimizations(optimizations, "after")
                after_vram = vram_usage_text()
                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
                log_event(
                    f"Folder batch complete: processed={aggregate['processed']}, skipped={aggregate['skipped']}, failed={aggregate['failed']}.",
                    "Joy Caption Beta 1",
                )
                yield html_message("success", f"Folder batch complete. Processed {aggregate['processed']}/{total} images, {aggregate['skipped']} skipped, {aggregate['failed']} failed.<br>Token speed: {token_detail}<br>{optimization_status_text(optimizations)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), ""
                return

            def worker(worker_device: int | str, chunk: list[Path]) -> None:
                nonlocal total_generated_tokens, total_generation_seconds
                worker_engine = self if len(devices) == 1 else BetaEngine(self.model_path)
                local_processed = 0
                local_skipped = 0
                local_failed = 0
                try:
                    worker_engine.load_model(quant, str(worker_device), optimizations)
                    for offset in range(0, len(chunk), batch_size_value):
                        if self.stop_flag.value:
                            break
                        batch_paths = chunk[offset : offset + batch_size_value]
                        work_items: list[tuple[Path, Path, Path]] = []
                        with aggregate_lock:
                            for path in batch_paths:
                                output_image_path, output_caption_path = resolve_output_paths(
                                    path,
                                    input_dir,
                                    output_dir,
                                    preserve_subfolders=process_subfolders_cb,
                                )
                                if output_caption_path.exists() and not overwrite_caption_cb and not append_caption_cb:
                                    aggregate["skipped"] += 1
                                    local_skipped += 1
                                    continue
                                work_items.append((path, output_image_path, output_caption_path))
                        if not work_items:
                            with aggregate_lock:
                                line = batch_progress_line(
                                    processed=aggregate["processed"],
                                    total=total,
                                    skipped=aggregate["skipped"],
                                    failed=aggregate["failed"],
                                    started=started,
                                    last_batch_count=0,
                                    last_batch_seconds=0.0,
                                    token_speed=total_generated_tokens / max(total_generation_seconds, 1e-9),
                                    device_id=worker_device if len(devices) > 1 else None,
                                    worker_processed=local_processed,
                                    worker_total=len(chunk),
                                    worker_skipped=local_skipped,
                                    worker_failed=local_failed,
                                )
                            batch_queue.put(("progress", line))
                            continue
                        try:
                            images = [load_rgb_image(path, downscale if downscale > 0 else None) for path, _, _ in work_items]
                            log_event(
                                f"Device {worker_device}: folder true batch {offset + 1}-{offset + len(work_items)} of {len(chunk)}.",
                                "Joy Caption Beta 1",
                            )
                            batch_started = time.time()
                            raw_captions = worker_engine.generate_captions(
                                images,
                                prompt,
                                temperature,
                                top_p,
                                max_new_tokens,
                                device_id=worker_device,
                            )
                            batch_elapsed = max(time.time() - batch_started, 1e-9)
                            stats = worker_engine.last_generation_stats
                            with aggregate_lock:
                                total_generated_tokens += stats.generated_tokens
                                total_generation_seconds += stats.elapsed_seconds
                                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                                for (path, output_image_path, output_caption_path), caption in zip(work_items, raw_captions):
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
                                        aggregate["processed"] += 1
                                        local_processed += 1
                                        log_event(f"Folder batch saved: {actual_caption}", "Joy Caption Beta 1")
                                    else:
                                        aggregate["skipped"] += 1
                                        local_skipped += 1
                                line = batch_progress_line(
                                    processed=aggregate["processed"],
                                    total=total,
                                    skipped=aggregate["skipped"],
                                    failed=aggregate["failed"],
                                    started=started,
                                    last_batch_count=len(work_items),
                                    last_batch_seconds=batch_elapsed,
                                    token_speed=token_speed,
                                    device_id=worker_device if len(devices) > 1 else None,
                                    worker_processed=local_processed,
                                    worker_total=len(chunk),
                                    worker_skipped=local_skipped,
                                    worker_failed=local_failed,
                                )
                            batch_queue.put(("progress", line))
                        except Exception as exc:
                            with aggregate_lock:
                                aggregate["failed"] += len(work_items)
                                local_failed += len(work_items)
                                line = batch_progress_line(
                                    processed=aggregate["processed"],
                                    total=total,
                                    skipped=aggregate["skipped"],
                                    failed=aggregate["failed"],
                                    started=started,
                                    last_batch_count=0,
                                    last_batch_seconds=0.0,
                                    token_speed=total_generated_tokens / max(total_generation_seconds, 1e-9),
                                    device_id=worker_device if len(devices) > 1 else None,
                                    worker_processed=local_processed,
                                    worker_total=len(chunk),
                                    worker_skipped=local_skipped,
                                    worker_failed=local_failed,
                                )
                            batch_queue.put(("progress", f"{line} Failed batch: {format_exception(exc)}"))
                finally:
                    if worker_engine is not self:
                        worker_engine.clear_models()
                    batch_queue.put(("done", f"Device {worker_device}: done."))

            threads = [
                threading.Thread(target=worker, args=(worker_device, chunk), daemon=True)
                for worker_device, chunk in zip(devices, chunks)
                if chunk
            ]
            for thread in threads:
                thread.start()

            done_count = 0
            while done_count < len(threads):
                try:
                    kind, line = batch_queue.get(timeout=0.5)
                except queue.Empty:
                    if not any(thread.is_alive() for thread in threads):
                        break
                    continue
                if kind == "done":
                    done_count += 1
                    log_event(line, "Joy Caption Beta 1")
                else:
                    log_event(line, "Joy Caption Beta 1")
                    yield html_message("info", line), ""
                if self.stop_flag.value:
                    yield html_message("info", f"Stopped. Processed {aggregate['processed']}/{total} images, {aggregate['skipped']} skipped."), ""
                    return

            for thread in threads:
                thread.join(timeout=1.0)
            apply_torch_optimizations(optimizations, "after")
            after_vram = vram_usage_text()
            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
            token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
            log_event(
                f"Folder batch complete: processed={aggregate['processed']}, skipped={aggregate['skipped']}, failed={aggregate['failed']}.",
                "Joy Caption Beta 1",
            )
            yield html_message("success", f"Folder batch complete. Processed {aggregate['processed']}/{total} images, {aggregate['skipped']} skipped, {aggregate['failed']} failed.<br>Token speed: {token_detail}<br>{optimization_status_text(optimizations)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), html_message("error", "Folder batch failed. Check the terminal for details.")


def _beta_folder_process_worker(
    event_queue: Any,
    model_path_text: str,
    path_texts: list[str],
    input_dir_text: str,
    output_dir_text: str,
    copy_images_cb: bool,
    skip_exists_cb: bool,
    overwrite_caption_cb: bool,
    append_caption_cb: bool,
    remove_newlines_cb: bool,
    discard_repeats_cb: bool,
    process_subfolders_cb: bool,
    downscale: int,
    caption_prefix: str,
    caption_suffix: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size_value: int,
    quant: str,
    worker_device: int | str,
    optimizations: dict[str, Any],
    replace_pairs: Any | None,
    replace_case_sensitive: bool,
    replace_single_word: bool,
) -> None:
    engine = BetaEngine(Path(model_path_text))
    paths = [Path(path) for path in path_texts]
    input_dir = Path(input_dir_text)
    output_dir = Path(output_dir_text)
    local_processed = 0
    local_skipped = 0
    local_failed = 0
    try:
        log_event(f"Device {worker_device}: Beta process worker started with {len(paths)} image(s).", "Joy Caption Beta 1")
        engine.load_model(quant, str(worker_device), optimizations)
        for offset in range(0, len(paths), batch_size_value):
            batch_paths = paths[offset : offset + batch_size_value]
            work_items: list[tuple[Path, Path, Path]] = []
            for path in batch_paths:
                output_image_path, output_caption_path = resolve_output_paths(
                    path,
                    input_dir,
                    output_dir,
                    preserve_subfolders=process_subfolders_cb,
                )
                if output_caption_path.exists() and not overwrite_caption_cb and not append_caption_cb:
                    local_skipped += 1
                    continue
                work_items.append((path, output_image_path, output_caption_path))

            if not work_items:
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": worker_device,
                        "processed_delta": 0,
                        "skipped_delta": len(batch_paths),
                        "failed_delta": 0,
                        "local_processed": local_processed,
                        "local_skipped": local_skipped,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": 0,
                        "last_batch_seconds": 0.0,
                        "generated_tokens": 0,
                        "generation_seconds": 0.0,
                    }
                )
                continue

            try:
                log_event(
                    f"Device {worker_device}: Beta process true batch {offset + 1}-{offset + len(work_items)} of {len(paths)}.",
                    "Joy Caption Beta 1",
                )
                images = [load_rgb_image(path, downscale if downscale > 0 else None) for path, _, _ in work_items]
                batch_started = time.time()
                raw_captions = engine.generate_captions(
                    images,
                    prompt,
                    temperature,
                    top_p,
                    max_new_tokens,
                    device_id=worker_device,
                )
                batch_elapsed = max(time.time() - batch_started, 1e-9)
                stats = engine.last_generation_stats
                saved_count = 0
                skipped_count = 0
                for (path, output_image_path, output_caption_path), caption in zip(work_items, raw_captions):
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
                        saved_count += 1
                        local_processed += 1
                        log_event(f"Device {worker_device}: process saved {actual_caption}", "Joy Caption Beta 1")
                    else:
                        skipped_count += 1
                        local_skipped += 1
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": worker_device,
                        "processed_delta": saved_count,
                        "skipped_delta": skipped_count,
                        "failed_delta": 0,
                        "local_processed": local_processed,
                        "local_skipped": local_skipped,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": saved_count,
                        "last_batch_seconds": batch_elapsed,
                        "generated_tokens": stats.generated_tokens,
                        "generation_seconds": stats.elapsed_seconds,
                    }
                )
            except Exception as exc:
                failed_count = len(work_items)
                local_failed += failed_count
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": worker_device,
                        "processed_delta": 0,
                        "skipped_delta": 0,
                        "failed_delta": failed_count,
                        "local_processed": local_processed,
                        "local_skipped": local_skipped,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": 0,
                        "last_batch_seconds": 0.0,
                        "generated_tokens": 0,
                        "generation_seconds": 0.0,
                        "message": f"Failed batch: {format_exception(exc)}",
                    }
                )
        event_queue.put(
            {
                "kind": "done",
                "device_id": worker_device,
                "local_processed": local_processed,
                "local_skipped": local_skipped,
                "local_failed": local_failed,
            }
        )
    except Exception as exc:
        event_queue.put({"kind": "fatal", "device_id": worker_device, "error": format_exception(exc)})
    finally:
        try:
            engine.clear_models()
        except Exception:
            pass


def _beta_zip_process_worker(
    event_queue: Any,
    model_path_text: str,
    path_texts: list[str],
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size_value: int,
    quant: str,
    worker_device: int | str,
    optimizations: dict[str, Any],
    remove_newlines: bool,
    discard_repeats: bool,
    caption_prefix: str,
    caption_suffix: str,
    replace_pairs: Any | None,
    replace_case_sensitive: bool,
    replace_single_word: bool,
) -> None:
    engine = BetaEngine(Path(model_path_text))
    paths = [Path(path) for path in path_texts]
    local_processed = 0
    local_failed = 0
    try:
        log_event(f"Device {worker_device}: Beta ZIP process worker started with {len(paths)} image(s).", "Joy Caption Beta 1")
        engine.load_model(quant, str(worker_device), optimizations)
        for offset in range(0, len(paths), batch_size_value):
            batch_paths = paths[offset : offset + batch_size_value]
            try:
                images = [load_rgb_image(path) for path in batch_paths]
                log_event(
                    f"Device {worker_device}: Beta ZIP process true batch {offset + 1}-{offset + len(batch_paths)} of {len(paths)}.",
                    "Joy Caption Beta 1",
                )
                batch_started = time.time()
                raw_captions = engine.generate_captions(
                    images,
                    prompt,
                    temperature,
                    top_p,
                    max_new_tokens,
                    device_id=worker_device,
                )
                batch_elapsed = max(time.time() - batch_started, 1e-9)
                stats = engine.last_generation_stats
                caption_payload: dict[str, str] = {}
                for path, caption in zip(batch_paths, raw_captions):
                    if discard_repeats:
                        caption = remove_repeating_sentences(caption)
                    caption = clean_legacy_caption(caption)
                    caption_payload[path.with_suffix(".txt").name] = finalize_caption_text(
                        caption,
                        remove_newlines=remove_newlines,
                        prefix=caption_prefix,
                        suffix=caption_suffix,
                        replace_pairs=replace_pairs,
                        replace_case_sensitive=replace_case_sensitive,
                        replace_single_word=replace_single_word,
                    )
                local_processed += len(caption_payload)
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": worker_device,
                        "processed_delta": len(caption_payload),
                        "failed_delta": 0,
                        "local_processed": local_processed,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": len(caption_payload),
                        "last_batch_seconds": batch_elapsed,
                        "generated_tokens": stats.generated_tokens,
                        "generation_seconds": stats.elapsed_seconds,
                        "captions": caption_payload,
                    }
                )
            except Exception as exc:
                failed_count = len(batch_paths)
                local_failed += failed_count
                event_queue.put(
                    {
                        "kind": "progress",
                        "device_id": worker_device,
                        "processed_delta": 0,
                        "failed_delta": failed_count,
                        "local_processed": local_processed,
                        "local_failed": local_failed,
                        "worker_total": len(paths),
                        "last_batch_count": 0,
                        "last_batch_seconds": 0.0,
                        "generated_tokens": 0,
                        "generation_seconds": 0.0,
                        "message": f"Failed batch: {format_exception(exc)}",
                    }
                )
        event_queue.put(
            {
                "kind": "done",
                "device_id": worker_device,
                "local_processed": local_processed,
                "local_failed": local_failed,
            }
        )
    except Exception as exc:
        event_queue.put({"kind": "fatal", "device_id": worker_device, "error": format_exception(exc)})
    finally:
        try:
            engine.clear_models()
        except Exception:
            pass


def extra_options_choices() -> list[str]:
    return get_all_extra_options()
