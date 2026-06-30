from __future__ import annotations

import gc
import json
import math
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Sequence

import torch
from PIL import Image
from PIL import ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from transformers.utils import logging as hf_logging

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception as exc:  # pragma: no cover - import depends on user venv
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
    QWEN_IMPORT_ERROR = exc
else:
    QWEN_IMPORT_ERROR = None

hf_logging.disable_progress_bar()

from ..attention import attention_load_kwargs, attention_runtime_context, normalize_attention_backend
from ..common import (
    BatchStopFlag,
    IMAGE_EXTENSIONS,
    OUTPUTS_DIR,
    apply_torch_optimizations,
    coerce_image_path,
    copy_image_if_needed,
    discover_images,
    finalize_caption_text,
    format_exception,
    html_message,
    load_rgb_image,
    log_event,
    natural_sort_key,
    optimization_status_text,
    parse_device_ids,
    reset_vram_peak_stats,
    save_caption_file,
    save_numbered_generation,
    vram_usage_text,
    write_generation_metadata,
)
from ..json_tools import (
    boxed_image_png_bytes,
    json_to_element_rows,
    normalize_json_output,
    official_v1_warnings_require_retry,
    overlay_html,
    save_boxed_image,
)
from ..qwen_presets import OFFICIAL_V1_PRESET_ID


SCOPE = "Qwen3 VL 8B Instruct"


@dataclass
class QwenModelState:
    processor: Any | None = None
    model: Any | None = None
    quant: str | None = None
    device_id: int | str | None = None
    optimization_key: tuple[bool, str] | None = None


@dataclass
class QwenGenerationStats:
    generated_tokens: int = 0
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0


def _generation_stats_text(stats: QwenGenerationStats) -> str:
    return (
        f"{stats.generated_tokens} token(s) in {stats.elapsed_seconds:.2f}s "
        f"({stats.tokens_per_second:.2f} tok/s)"
    )


def _synchronize_if_cuda(device: Any) -> None:
    if not torch.cuda.is_available():
        return
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _resolve_device(device_id: int | str) -> str:
    if str(device_id).lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use device 'cpu' only if the selected model fits system RAM.")
    index = int(device_id)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid CUDA device ID {index}. Available count: {torch.cuda.device_count()}.")
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def _caption_extension(settings: dict[str, Any]) -> str:
    extension = str(settings.get("extension") or "").strip()
    if not extension:
        extension = ".json" if str(settings.get("output_format")) == "json" else ".txt"
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension.lower()


def _file_path(item: Any) -> Path:
    if isinstance(item, (str, Path)):
        return Path(item)
    name = getattr(item, "name", None)
    if name:
        return Path(name)
    raise ValueError(f"Unsupported file object: {item!r}")


def _normalize_text_output(text: str, settings: dict[str, Any]) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        from ..json_tools import strip_markdown_fences

        value = strip_markdown_fences(value)
    return finalize_caption_text(
        value,
        remove_newlines=bool(settings.get("remove_newlines", False)),
        prefix=str(settings.get("caption_prefix", "")),
        suffix=str(settings.get("caption_suffix", "")),
    )


def _image_for_overlay(source: Path, saved_image: Path | None) -> Path:
    if saved_image and saved_image.exists():
        return saved_image
    try:
        source.resolve().relative_to(OUTPUTS_DIR.parent.resolve())
        return source
    except Exception:
        temp_dir = OUTPUTS_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        target = temp_dir / f"preview_{int(time.time() * 1000)}{source.suffix or '.png'}"
        copy_image_if_needed(source, target, True)
        return target


def _oriented_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        oriented = ImageOps.exif_transpose(image)
        return oriented.size


def _aspect_ratio_text(width: int, height: int) -> str:
    width = max(1, int(width))
    height = max(1, int(height))
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _settings_for_image(settings: dict[str, Any], image_path: Path, image: Image.Image) -> dict[str, Any]:
    preset_id = str(settings.get("preset_id") or settings.get("id") or "")
    if not preset_id.startswith(("i4_official_v1", "i4_json_")):
        return settings
    try:
        width, height = _oriented_image_size(image_path)
    except Exception:
        width, height = image.size
    aspect_ratio = _aspect_ratio_text(width, height)
    prompt = str(settings.get("prompt") or "").strip()
    aspect_note = (
        "\n\nExact input image geometry for this run:\n"
        f"- Width: {int(width)} px\n"
        f"- Height: {int(height)} px\n"
        f"- Aspect ratio: {aspect_ratio}\n"
        f'Set the JSON "aspect_ratio" value exactly to "{aspect_ratio}".'
    )
    next_settings = dict(settings)
    next_settings["prompt"] = prompt + aspect_note
    next_settings["detected_aspect_ratio"] = aspect_ratio
    next_settings["detected_image_width"] = int(width)
    next_settings["detected_image_height"] = int(height)
    return next_settings


def _apply_detected_aspect_ratio(
    parsed: dict[str, Any] | None,
    warnings: list[str],
    settings: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    preset_id = str(settings.get("preset_id") or settings.get("id") or "")
    if not preset_id.startswith(("i4_official_v1", "i4_json_")) or parsed is None:
        return parsed, warnings
    aspect_ratio = str(settings.get("detected_aspect_ratio") or "").strip()
    if not aspect_ratio:
        return parsed, warnings
    ordered: dict[str, Any] = {
        "aspect_ratio": aspect_ratio,
        "high_level_description": parsed.get("high_level_description", ""),
        "compositional_deconstruction": parsed.get("compositional_deconstruction", {"background": "", "elements": []}),
    }
    filtered_warnings = [
        warning
        for warning in warnings
        if not warning.startswith('"aspect_ratio" must be')
        and warning != 'Missing top-level key "aspect_ratio".'
        and not warning.startswith("Official v1 key order should be")
    ]
    return ordered, filtered_warnings


def _caption_for_display(final_caption: str, parsed: dict[str, Any] | None) -> str:
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return final_caption


class ConsoleProgressStoppingCriteria(StoppingCriteria):
    def __init__(self, label: str, prompt_tokens: int, max_new_tokens: int) -> None:
        self.label = label
        self.prompt_tokens = max(0, int(prompt_tokens))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.last_percent = -1
        self.started = time.time()
        self._write(0, 0)

    def _write(self, generated: int, percent: int) -> None:
        elapsed = max(time.time() - self.started, 1e-9)
        tokens_per_second = max(0, int(generated)) / elapsed
        print(
            f"\r{self.label}: token generation {percent:3d}% "
            f"({generated}/{self.max_new_tokens}, {tokens_per_second:.2f} tok/s)",
            end="",
            flush=True,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        generated = max(0, int(input_ids.shape[-1]) - self.prompt_tokens)
        percent = min(99, int((generated / self.max_new_tokens) * 100))
        if percent != self.last_percent:
            self.last_percent = percent
            self._write(generated, percent)
        return False

    def finish(self, generated: int | None = None, failed: bool = False) -> None:
        if failed:
            print("", flush=True)
            return
        self._write(self.max_new_tokens if generated is None else generated, 100)
        print("", flush=True)


class QwenEngine:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.state = QwenModelState()
        self.stop_flag = BatchStopFlag()
        self.last_generation_stats = QwenGenerationStats()

    def clear_models(self) -> None:
        if self.state.model is not None:
            del self.state.model
        self.state = QwenModelState()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop_batch(self) -> str:
        self.stop_flag.stop()
        return html_message("info", "Stopping Qwen batch after the current image finishes.")

    def _load_processor(self) -> Any:
        if self.state.processor is None:
            log_event(f"Loading processor from {self.model_path}.", SCOPE)
            self.state.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        return self.state.processor

    def _quant_config(self, quant: str) -> BitsAndBytesConfig | None:
        if quant in {"bf16", "fp16"}:
            return None
        if quant == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        if quant == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        raise ValueError(f"Unknown Qwen quantization: {quant}")

    def load_model(self, settings: dict[str, Any]) -> str:
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError(
                "The installed transformers package cannot import Qwen3VLForConditionalGeneration. "
                f"Original error: {QWEN_IMPORT_ERROR}"
            )
        quant = str(settings.get("model_quantization") or "bf16")
        device_id = parse_device_ids(settings.get("device_id") or "0", allow_cpu=True)[0]
        optimization_key = (
            bool(settings.get("low_cpu_mem_usage", True)),
            normalize_attention_backend(settings),
        )
        if (
            self.state.model is not None
            and self.state.quant == quant
            and self.state.device_id == device_id
            and self.state.optimization_key == optimization_key
        ):
            return "Qwen model ready from cache."

        processor = self._load_processor()
        self.clear_models()
        self.state.processor = processor
        apply_torch_optimizations(settings, "before")
        device_map = _resolve_device(device_id)
        if device_map == "cpu" and quant in {"int8", "nf4"}:
            raise RuntimeError("int8 and nf4 quantization require CUDA in this app.")

        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if settings.get("low_cpu_mem_usage", True):
            load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs.update(attention_load_kwargs(settings, quant=quant))
        qconfig = self._quant_config(quant)
        if qconfig is not None:
            load_kwargs["quantization_config"] = qconfig
            load_kwargs["dtype"] = "auto"
        elif quant == "fp16":
            load_kwargs["dtype"] = torch.float16 if device_map != "cpu" else torch.float32
        else:
            load_kwargs["dtype"] = torch.bfloat16 if device_map != "cpu" else torch.float32

        log_event(f"Loading Qwen model on {device_map} with quant={quant}.", SCOPE)
        model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
        model.eval()
        self.state.model = model
        self.state.quant = quant
        self.state.device_id = device_id
        self.state.optimization_key = optimization_key
        return f"Qwen model ready on {device_map}."

    @torch.inference_mode()
    def generate_caption(self, image: Image.Image, settings: dict[str, Any]) -> str:
        if self.state.model is None or self.state.processor is None:
            raise RuntimeError("Qwen model is not loaded.")
        system_prompt = str(settings.get("system_prompt") or "").strip()
        prompt = str(settings.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        processor = self.state.processor
        model = self.state.model
        log_event(f"Preparing Qwen inputs (max_new_tokens={int(settings.get('max_new_tokens', 512))}).", SCOPE)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        temperature = float(settings.get("temperature", 0.1) or 0.0)
        do_sample = temperature > 0
        max_new_tokens = int(settings.get("max_new_tokens", 512) or 512)
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": float(settings.get("repetition_penalty", 1.0) or 1.0),
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
            generation_kwargs["top_p"] = float(settings.get("top_p", 0.8) or 0.8)
            generation_kwargs["top_k"] = int(settings.get("top_k", 20) or 20)
        log_event(f"Generating Qwen caption | do_sample={do_sample}.", SCOPE)
        progress = None
        if bool(settings.get("console_progress", True)):
            progress = ConsoleProgressStoppingCriteria(SCOPE, int(inputs.input_ids.shape[-1]), max_new_tokens)
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([progress])
        _synchronize_if_cuda(model.device)
        generation_started = time.time()
        try:
            with attention_runtime_context(settings):
                generated_ids = model.generate(**generation_kwargs)
        except Exception:
            if progress is not None:
                progress.finish(failed=True)
            raise
        _synchronize_if_cuda(model.device)
        generated_token_count = max(0, int(generated_ids.shape[-1]) - int(inputs.input_ids.shape[-1]))
        generation_elapsed = max(time.time() - generation_started, 1e-9)
        self.last_generation_stats = QwenGenerationStats(
            generated_tokens=generated_token_count,
            elapsed_seconds=generation_elapsed,
            tokens_per_second=generated_token_count / generation_elapsed,
        )
        if progress is not None:
            progress.finish(generated_token_count)
        log_event(f"Qwen generation speed: {_generation_stats_text(self.last_generation_stats)}.", SCOPE)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(output_text[0] if output_text else "").strip()

    def _finalize_output(
        self,
        image: Image.Image,
        raw_caption: str,
        settings: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None, list[str]]:
        output_format = str(settings.get("output_format") or "txt")
        if output_format == "json" or _caption_extension(settings) == ".json":
            final, parsed, warnings = normalize_json_output(
                raw_caption,
                preset_id=str(settings.get("preset_id") or ""),
                compact=bool(settings.get("compact_json", False)),
            )
            retries = int(settings.get("json_retries", 0) or 0)
            attempt = 0
            preset_id = str(settings.get("preset_id") or "")
            while (
                parsed is None
                or (preset_id.startswith(("i4_official_v1", "i4_json_")) and official_v1_warnings_require_retry(warnings))
            ) and attempt < retries:
                attempt += 1
                current_json = final
                if parsed is not None:
                    current_json = json.dumps(parsed, ensure_ascii=False, indent=2)
                repair_prompt = (
                    "The previous output did not satisfy the required Ideogram 4 / v1 JSON schema for this reason:\n"
                    + "\n".join(warnings)
                    + "\n\nRepair it using the image. Return one valid JSON object only. Do not add markdown, explanations, comments, labels, or extra text. "
                    + "Every element must include a tight normalized bbox in official [y_min,x_min,y_max,x_max] order. "
                    + "If a detail cannot be boxed, merge it into background or another element description instead of leaving bbox missing.\n\n"
                    + current_json
                )
                retry_settings = dict(settings)
                retry_settings["prompt"] = repair_prompt
                retry_settings["temperature"] = 0.0
                raw_caption = self.generate_caption(image, retry_settings)
                final, parsed, warnings = normalize_json_output(
                    raw_caption,
                    preset_id=str(settings.get("preset_id") or ""),
                    compact=bool(settings.get("compact_json", False)),
                )
            parsed, warnings = _apply_detected_aspect_ratio(parsed, warnings, settings)
            if parsed is not None:
                final = (
                    json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                    if bool(settings.get("compact_json", False))
                    else json.dumps(parsed, ensure_ascii=False, indent=2)
                )
            return final, parsed, warnings
        return _normalize_text_output(raw_caption, settings), None, []

    def caption_single(self, image_input: Any | None, settings: dict[str, Any]) -> Generator[tuple[str, str, str, list[list[Any]], str], None, None]:
        log_event("Qwen single image caption requested.", SCOPE)
        if settings.get("app_side_only", False):
            yield html_message("error", "This preset is an app-side utility and does not run image generation."), "", "", [], ""
            return
        image_path = coerce_image_path(image_input, OUTPUTS_DIR / "temp")
        if image_path is None:
            yield html_message("error", "No image selected."), "", "", [], ""
            return
        delete_after = isinstance(image_input, Image.Image)
        try:
            start = time.time()
            devices = parse_device_ids(settings.get("device_id") or "0", allow_cpu=True)
            reset_vram_peak_stats(devices)
            before_vram = vram_usage_text()
            yield html_message("info", "Loading Qwen model..."), "", "", [], ""
            status = self.load_model(settings)
            yield html_message("info", f"{status} Generating..."), "", "", [], ""
            image = load_rgb_image(image_path, int(settings.get("image_long_edge", 1024) or 1024))
            image_settings = _settings_for_image(settings, image_path, image)
            raw_caption = self.generate_caption(image, image_settings)
            final_caption, parsed_json, warnings = self._finalize_output(image, raw_caption, image_settings)
            display_caption = _caption_for_display(final_caption, parsed_json)
            generation_stats = self.last_generation_stats
            apply_torch_optimizations(settings, "after")
            after_vram = vram_usage_text()
            metadata = {
                "generation_type": "single_image",
                "engine": "qwen3_vl_8b_instruct",
                "model_path": str(self.model_path),
                "source_image_path": str(image_path),
                "preset_id": settings.get("preset_id"),
                "system_prompt": settings.get("system_prompt"),
                "prompt": image_settings.get("prompt"),
                "output_format": settings.get("output_format"),
                "caption_raw": raw_caption,
                "caption_final": final_caption,
                "json_warnings": warnings,
                "settings": dict(image_settings),
                "elapsed_seconds": time.time() - start,
                "generated_tokens": generation_stats.generated_tokens,
                "generation_elapsed_seconds": generation_stats.elapsed_seconds,
                "tokens_per_second": generation_stats.tokens_per_second,
                "vram_before": before_vram,
                "vram_after": after_vram,
                "optimizations": optimization_status_text(settings),
            }
            output_image_path, caption_path, metadata_path, _run_dir = save_numbered_generation(
                image_path,
                final_caption,
                metadata,
                OUTPUTS_DIR,
                copy_image=bool(settings.get("save_image", True)),
                caption_extension=_caption_extension(settings),
            )
            rows = json_to_element_rows(parsed_json, bbox_order="yxyx")
            boxed_image_path: Path | None = None
            if bool(settings.get("auto_save_boxed_image", True)) and rows:
                try:
                    boxed_source = output_image_path if output_image_path and output_image_path.exists() else image_path
                    boxed_image_path = save_boxed_image(
                        boxed_source,
                        rows,
                        _run_dir / f"{caption_path.stem}_boxed.png",
                        bbox_order="yxyx",
                    )
                    if boxed_image_path is not None:
                        enriched_metadata = dict(metadata)
                        enriched_metadata.update(
                            {
                                "output_run_dir": str(_run_dir),
                                "output_image_path": str(output_image_path) if output_image_path else None,
                                "caption_path": str(caption_path),
                                "metadata_path": str(metadata_path),
                                "boxed_image_path": str(boxed_image_path),
                            }
                        )
                        write_generation_metadata(metadata_path, enriched_metadata)
                except Exception as exc:
                    warnings.append(f"Boxed image save failed: {type(exc).__name__}: {exc}")
            overlay_source = _image_for_overlay(image_path, output_image_path)
            overlay = overlay_html(overlay_source, rows, interactive=True, bbox_order="yxyx")
            warning_html = ""
            if warnings:
                warning_html = "<br>Warnings:<br><pre>" + "\n".join(warnings) + "</pre>"
            detail = (
                f"Caption saved to: {caption_path}<br>"
                f"Image output: {output_image_path if output_image_path else 'Image copy disabled.'}<br>"
                f"Boxed image: {boxed_image_path if boxed_image_path else 'No boxed image saved.'}<br>"
                f"Metadata saved to: {metadata_path}<br>"
                f"Token speed: {_generation_stats_text(generation_stats)}<br>"
                f"{optimization_status_text(settings)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>{warning_html}"
            )
            yield html_message("success", f"Qwen generation complete.<br>{detail}"), display_caption, overlay, rows, ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), "", "", [], html_message("error", "Qwen generation failed. Check the terminal for details.")
        finally:
            if settings.get("unload_model", False):
                self.clear_models()
            if delete_after:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def process_batch_files_to_zip(
        self,
        files_list: Sequence[Any] | None,
        settings: dict[str, Any],
    ) -> Generator[tuple[str, str | None, str], None, None]:
        log_event("Qwen files-to-ZIP batch requested.", SCOPE)
        if not files_list:
            yield html_message("error", "No files selected."), None, ""
            return
        if settings.get("app_side_only", False):
            yield html_message("error", "This preset is an app-side utility and does not run image generation."), None, ""
            return
        try:
            paths = sorted([_file_path(item) for item in files_list], key=natural_sort_key)
            self.stop_flag.reset()
            yield html_message("info", "Loading Qwen model..."), None, ""
            devices = parse_device_ids(settings.get("device_id") or "0", allow_cpu=True)
            reset_vram_peak_stats(devices)
            before_vram = vram_usage_text()
            self.load_model(settings)
            captions: dict[str, str] = {}
            total = len(paths)
            extension = _caption_extension(settings)
            batch_size = max(1, int(settings.get("file_batch_size", 1) or 1))
            started = time.time()
            total_generated_tokens = 0
            total_generation_seconds = 0.0
            boxed_images: dict[str, bytes] = {}
            for offset in range(0, total, batch_size):
                if self.stop_flag.value:
                    yield html_message("info", f"ZIP batch cancelled. Processed {len(captions)}/{total} images."), None, ""
                    return
                for path in paths[offset : offset + batch_size]:
                    if self.stop_flag.value:
                        break
                    image = load_rgb_image(path, int(settings.get("image_long_edge", 1024) or 1024))
                    image_settings = _settings_for_image(settings, path, image)
                    raw = self.generate_caption(image, image_settings)
                    final, parsed, warnings = self._finalize_output(image, raw, image_settings)
                    stats = self.last_generation_stats
                    total_generated_tokens += stats.generated_tokens
                    total_generation_seconds += stats.elapsed_seconds
                    if warnings:
                        log_event(f"JSON warnings for {path.name}: {' | '.join(warnings)}", SCOPE)
                    captions[path.with_suffix(extension).name] = final
                    if bool(settings.get("auto_save_boxed_image", True)):
                        rows = json_to_element_rows(parsed, bbox_order="yxyx")
                        if rows:
                            try:
                                image_bytes = boxed_image_png_bytes(path, rows, bbox_order="yxyx")
                                if image_bytes:
                                    boxed_images[f"{path.stem}_boxed.png"] = image_bytes
                            except Exception as exc:
                                log_event(f"Boxed image save failed for {path.name}: {format_exception(exc)}", SCOPE)
                elapsed = max(time.time() - started, 0.01)
                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                yield html_message("info", f"Processed {min(offset + batch_size, total)}/{total} images. Speed {len(captions) / elapsed:.2f} img/s, {token_speed:.2f} tok/s."), None, ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as archive:
                for filename, text in captions.items():
                    archive.writestr(filename, text)
                for filename, image_bytes in boxed_images.items():
                    archive.writestr(filename, image_bytes)
            apply_torch_optimizations(settings, "after")
            after_vram = vram_usage_text()
            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
            token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
            yield html_message("success", f"ZIP batch complete. Processed {len(captions)}/{total} images.<br>Token speed: {token_detail}<br>{optimization_status_text(settings)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), tmp.name, ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), None, html_message("error", "Qwen ZIP batch failed. Check the terminal for details.")

    def run_batch_folder_processing(self, settings: dict[str, Any]) -> Generator[tuple[str, str], None, None]:
        log_event("Qwen folder batch requested.", SCOPE)
        if settings.get("app_side_only", False):
            yield html_message("error", "This preset is an app-side utility and does not run image generation."), ""
            return
        input_text = str(settings.get("folder_input") or "").strip()
        if not input_text:
            yield html_message("error", "Input folder path is required."), ""
            return
        input_dir = Path(input_text)
        if not input_dir.is_dir():
            yield html_message("error", f"Input folder not found: {input_dir}"), ""
            return
        output_text = str(settings.get("folder_output") or "").strip()
        output_dir = Path(output_text) if output_text else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        extension = _caption_extension(settings)
        process_subfolders = bool(settings.get("process_subfolders", False))
        overwrite = bool(settings.get("overwrite_caption", False))
        append = bool(settings.get("append_caption", False))
        skip_exists = bool(settings.get("skip_exists", True))
        try:
            all_paths = discover_images(input_dir, include_subfolders=process_subfolders)
            paths: list[Path] = []
            skipped = 0
            for path in all_paths:
                relative = path.relative_to(input_dir) if process_subfolders else Path(path.name)
                caption_path = (output_dir / relative).with_suffix(extension)
                if caption_path.exists() and skip_exists and not overwrite:
                    skipped += 1
                    continue
                paths.append(path)
            if not paths:
                yield html_message("info", f"No images to process. Found {len(all_paths)} image files, skipped {skipped} existing captions."), ""
                return
            self.stop_flag.reset()
            devices = parse_device_ids(settings.get("device_id") or "0", allow_cpu=True)
            reset_vram_peak_stats(devices)
            before_vram = vram_usage_text()
            yield html_message("info", f"Loading Qwen model. Queued {len(paths)} image(s), skipped {skipped}."), ""
            self.load_model(settings)
            batch_size = max(1, int(settings.get("folder_batch_size", 1) or 1))
            processed = 0
            failed = 0
            started = time.time()
            total_generated_tokens = 0
            total_generation_seconds = 0.0
            for offset in range(0, len(paths), batch_size):
                if self.stop_flag.value:
                    yield html_message("info", f"Stopped. Processed {processed}/{len(paths)} queued images, {skipped} skipped."), ""
                    return
                for path in paths[offset : offset + batch_size]:
                    if self.stop_flag.value:
                        break
                    try:
                        relative = path.relative_to(input_dir) if process_subfolders else Path(path.name)
                        output_image_path = output_dir / relative
                        output_caption_path = output_image_path.with_suffix(extension)
                        if output_caption_path.exists() and skip_exists and not overwrite:
                            skipped += 1
                            continue
                        image = load_rgb_image(path, int(settings.get("image_long_edge", 1024) or 1024))
                        image_settings = _settings_for_image(settings, path, image)
                        raw = self.generate_caption(image, image_settings)
                        final, parsed, warnings = self._finalize_output(image, raw, image_settings)
                        stats = self.last_generation_stats
                        total_generated_tokens += stats.generated_tokens
                        total_generation_seconds += stats.elapsed_seconds
                        if warnings:
                            log_event(f"JSON warnings for {path.name}: {' | '.join(warnings)}", SCOPE)
                        actual_caption = save_caption_file(
                            output_caption_path,
                            final,
                            overwrite=overwrite,
                            append=append,
                            remove_newlines=False if extension == ".json" else bool(settings.get("remove_newlines", False)),
                            prefix="" if extension == ".json" else str(settings.get("caption_prefix", "")),
                            suffix="" if extension == ".json" else str(settings.get("caption_suffix", "")),
                        )
                        copy_image_if_needed(path, output_image_path, bool(settings.get("save_image", True)))
                        if bool(settings.get("auto_save_boxed_image", True)):
                            rows = json_to_element_rows(parsed, bbox_order="yxyx")
                            if rows:
                                try:
                                    boxed_source = output_image_path if output_image_path.exists() else path
                                    save_boxed_image(
                                        boxed_source,
                                        rows,
                                        output_image_path.with_name(f"{output_image_path.stem}_boxed.png"),
                                        bbox_order="yxyx",
                                    )
                                except Exception as exc:
                                    log_event(f"Boxed image save failed for {path.name}: {format_exception(exc)}", SCOPE)
                        if actual_caption:
                            processed += 1
                    except Exception as exc:
                        failed += 1
                        log_event(f"Failed {path}: {format_exception(exc)}", SCOPE)
                elapsed = max(time.time() - started, 0.01)
                token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
                yield html_message("info", f"Processed {processed}/{len(paths)} queued images, {skipped} skipped, {failed} failed. Speed {processed / elapsed:.2f} img/s, {token_speed:.2f} tok/s."), ""
            apply_torch_optimizations(settings, "after")
            after_vram = vram_usage_text()
            token_speed = total_generated_tokens / max(total_generation_seconds, 1e-9)
            token_detail = f"{total_generated_tokens} token(s) in {total_generation_seconds:.2f}s ({token_speed:.2f} tok/s)"
            yield html_message("success", f"Qwen folder batch complete. Processed {processed}/{len(all_paths)} images, {skipped} skipped, {failed} failed.<br>Token speed: {token_detail}<br>{optimization_status_text(settings)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), html_message("error", "Qwen folder batch failed. Check the terminal for details.")
