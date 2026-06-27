from __future__ import annotations

import gc
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception as exc:  # pragma: no cover - import depends on user venv
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
    QWEN_IMPORT_ERROR = exc
else:
    QWEN_IMPORT_ERROR = None

hf_logging.disable_progress_bar()

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
)
from ..json_tools import json_to_element_rows, normalize_json_output, overlay_html


SCOPE = "Qwen3 VL 8B Instruct"


@dataclass
class QwenModelState:
    processor: Any | None = None
    model: Any | None = None
    quant: str | None = None
    device_id: int | str | None = None
    optimization_key: tuple[bool, bool] | None = None


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


class QwenEngine:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.state = QwenModelState()
        self.stop_flag = BatchStopFlag()

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
            bool(settings.get("use_sdpa_attention", False)),
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
        if settings.get("use_sdpa_attention", False):
            load_kwargs["attn_implementation"] = "sdpa"
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
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": int(settings.get("max_new_tokens", 512) or 512),
            "do_sample": do_sample,
            "repetition_penalty": float(settings.get("repetition_penalty", 1.0) or 1.0),
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
            generation_kwargs["top_p"] = float(settings.get("top_p", 0.9) or 0.9)
            generation_kwargs["top_k"] = int(settings.get("top_k", 20) or 20)
        log_event(f"Generating Qwen caption | do_sample={do_sample}.", SCOPE)
        generated_ids = model.generate(**generation_kwargs)
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
            while parsed is None and attempt < retries:
                attempt += 1
                repair_prompt = (
                    "The previous output was invalid for this reason:\n"
                    + "\n".join(warnings)
                    + "\n\nRepair it. Return one valid JSON object only. Do not add markdown, explanations, comments, or new image details.\n\n"
                    + final
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
            raw_caption = self.generate_caption(image, settings)
            final_caption, parsed_json, warnings = self._finalize_output(image, raw_caption, settings)
            apply_torch_optimizations(settings, "after")
            after_vram = vram_usage_text()
            metadata = {
                "generation_type": "single_image",
                "engine": "qwen3_vl_8b_instruct",
                "model_path": str(self.model_path),
                "source_image_path": str(image_path),
                "preset_id": settings.get("preset_id"),
                "system_prompt": settings.get("system_prompt"),
                "prompt": settings.get("prompt"),
                "output_format": settings.get("output_format"),
                "caption_raw": raw_caption,
                "caption_final": final_caption,
                "json_warnings": warnings,
                "settings": dict(settings),
                "elapsed_seconds": time.time() - start,
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
            rows = json_to_element_rows(parsed_json)
            overlay_source = _image_for_overlay(image_path, output_image_path)
            overlay = overlay_html(overlay_source, rows)
            warning_html = ""
            if warnings:
                warning_html = "<br>Warnings:<br><pre>" + "\n".join(warnings) + "</pre>"
            detail = (
                f"Caption saved to: {caption_path}<br>"
                f"Image output: {output_image_path if output_image_path else 'Image copy disabled.'}<br>"
                f"Metadata saved to: {metadata_path}<br>"
                f"{optimization_status_text(settings)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>{warning_html}"
            )
            yield html_message("success", f"Qwen generation complete.<br>{detail}"), final_caption, overlay, rows, ""
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
            for offset in range(0, total, batch_size):
                if self.stop_flag.value:
                    yield html_message("info", f"ZIP batch cancelled. Processed {len(captions)}/{total} images."), None, ""
                    return
                for path in paths[offset : offset + batch_size]:
                    if self.stop_flag.value:
                        break
                    image = load_rgb_image(path, int(settings.get("image_long_edge", 1024) or 1024))
                    raw = self.generate_caption(image, settings)
                    final, _parsed, warnings = self._finalize_output(image, raw, settings)
                    if warnings:
                        log_event(f"JSON warnings for {path.name}: {' | '.join(warnings)}", SCOPE)
                    captions[path.with_suffix(extension).name] = final
                elapsed = max(time.time() - started, 0.01)
                yield html_message("info", f"Processed {min(offset + batch_size, total)}/{total} images. Speed {len(captions) / elapsed:.2f} img/s."), None, ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as archive:
                for filename, text in captions.items():
                    archive.writestr(filename, text)
            apply_torch_optimizations(settings, "after")
            after_vram = vram_usage_text()
            yield html_message("success", f"ZIP batch complete. Processed {len(captions)}/{total} images.<br>{optimization_status_text(settings)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), tmp.name, ""
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
                        raw = self.generate_caption(image, settings)
                        final, _parsed, warnings = self._finalize_output(image, raw, settings)
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
                        if actual_caption:
                            processed += 1
                    except Exception as exc:
                        failed += 1
                        log_event(f"Failed {path}: {format_exception(exc)}", SCOPE)
                elapsed = max(time.time() - started, 0.01)
                yield html_message("info", f"Processed {processed}/{len(paths)} queued images, {skipped} skipped, {failed} failed. Speed {processed / elapsed:.2f} img/s."), ""
            apply_torch_optimizations(settings, "after")
            after_vram = vram_usage_text()
            yield html_message("success", f"Qwen folder batch complete. Processed {processed}/{len(all_paths)} images, {skipped} skipped, {failed} failed.<br>{optimization_status_text(settings)}<br><pre>Before {before_vram}\nAfter {after_vram}</pre>"), ""
        except Exception as exc:
            traceback.print_exc()
            yield html_message("error", format_exception(exc)), html_message("error", "Qwen folder batch failed. Check the terminal for details.")
