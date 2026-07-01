from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path
from typing import Any, Generator, Sequence

from .common import BASE_DIR, OUTPUTS_DIR, CaptionResult, coerce_image_path, format_exception, html_message
from .prompt_options import build_beta_prompt
from .subprocess_runner import cancel_active_workers, run_worker


def _clear_python_and_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _ModelSwitchRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_key: str | None = None
        self._owners: dict[str, weakref.ReferenceType[Any]] = {}

    def register(self, key: str, owner: Any) -> None:
        with self._lock:
            self._owners[key] = weakref.ref(owner)

    def activate(self, key: str) -> None:
        stale: list[str] = []
        owners_to_clear: list[Any] = []
        with self._lock:
            if self._active_key == key:
                return
            for owner_key, owner_ref in self._owners.items():
                owner = owner_ref()
                if owner is None:
                    stale.append(owner_key)
                elif owner_key != key:
                    owners_to_clear.append(owner)
            for owner_key in stale:
                self._owners.pop(owner_key, None)
            self._active_key = key
        for owner in owners_to_clear:
            clear = getattr(owner, "clear_models", None)
            if callable(clear):
                clear()


_MODEL_SWITCH_REGISTRY = _ModelSwitchRegistry()


class LazyLegacyEngine:
    def __init__(self, variant: str, base_dir: Path = BASE_DIR) -> None:
        self.variant = variant
        self.base_dir = Path(base_dir)
        self._engine: Any | None = None
        self._lock = threading.RLock()
        self._registry_key = f"legacy:{variant}"
        _MODEL_SWITCH_REGISTRY.register(self._registry_key, self)

    def _get_engine(self) -> Any:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        with self._lock:
            if self._engine is not None:
                return self._engine
            from .engines.legacy_siglip import (
                create_alpha_one_engine,
                create_alpha_two_engine,
                create_pre_alpha_engine,
            )

            if self.variant == "pre_alpha":
                self._engine = create_pre_alpha_engine(self.base_dir)
            elif self.variant == "alpha_one":
                self._engine = create_alpha_one_engine(self.base_dir)
            elif self.variant == "alpha_two":
                self._engine = create_alpha_two_engine(self.base_dir)
            else:
                raise ValueError(f"Unknown legacy variant: {self.variant}")
            return self._engine

    def clear_models(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            clear = getattr(engine, "clear_models", None)
            if callable(clear):
                clear()
            bundles = getattr(engine, "_bundles", None)
            if isinstance(bundles, dict):
                bundles.clear()
        _clear_python_and_cuda_cache()

    def stop_batch(self) -> str:
        if self._engine is None:
            return "No in-process batch is active."
        return str(self._engine.stop_batch())

    def cancel_single(self) -> str:
        count, message = cancel_active_workers(["legacy_single"])
        if count:
            return message
        return "No subprocess single-image worker is active. In-process single captioning cannot be cancelled."

    def cancel_batch(self) -> str:
        count, message = cancel_active_workers(["legacy_batch"])
        if self._engine is not None:
            stop_message = str(self._engine.stop_batch())
            return f"{message}\n{stop_message}" if count else stop_message
        return message if count else "No batch is active."

    def caption_single(self, image_input: Any, settings: dict[str, Any]) -> CaptionResult:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if settings.get("use_subprocess", False):
            image_path = coerce_image_path(image_input, OUTPUTS_DIR / "temp")
            if image_path is None:
                raise ValueError("No input image selected.")
            data = run_worker(
                "legacy_single",
                {
                    "variant": self.variant,
                    "image_path": str(image_path),
                    "settings": settings,
                },
            )
            return CaptionResult(
                prompt=str(data.get("prompt") or ""),
                caption=str(data.get("caption") or ""),
                caption_path=Path(data["caption_path"]) if data.get("caption_path") else None,
                image_path=Path(data["image_path"]) if data.get("image_path") else None,
                elapsed=float(data.get("elapsed") or 0.0),
                details=str(data.get("details") or ""),
                metadata_path=Path(data["metadata_path"]) if data.get("metadata_path") else None,
            )
        return self._get_engine().caption_single(image_input, settings)

    def batch_folder(self, settings: dict[str, Any]) -> Generator[str, None, None]:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if settings.get("use_subprocess", False):
            yield "Starting subprocess batch. The child process will exit when the run ends."
            try:
                data = run_worker(
                    "legacy_batch",
                    {
                        "variant": self.variant,
                        "settings": settings,
                    },
                )
                yield str(data.get("progress") or "Subprocess batch completed.")
            except Exception as exc:
                message = str(exc)
                if "cancelled" in message.lower():
                    yield "Subprocess batch cancelled. Child process was terminated."
                else:
                    yield f"Subprocess batch failed: {format_exception(exc)}"
            return
        yield from self._get_engine().batch_folder(settings)


class LazyBetaEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._engine: Any | None = None
        self._lock = threading.RLock()
        self._registry_key = f"beta:{self.model_path.resolve()}"
        _MODEL_SWITCH_REGISTRY.register(self._registry_key, self)

    def _get_engine(self) -> Any:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        with self._lock:
            if self._engine is None:
                from .engines.beta import BetaEngine

                self._engine = BetaEngine(self.model_path)
            return self._engine

    def clear_models(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            clear = getattr(engine, "clear_models", None)
            if callable(clear):
                clear()
        _clear_python_and_cuda_cache()

    def stop_batch(self) -> str:
        if self._engine is None:
            return html_message("info", "No in-process folder batch is active.")
        return str(self._engine.stop_batch())

    def cancel_single(self) -> str:
        count, message = cancel_active_workers(["beta_single"])
        if count:
            return html_message("info", message)
        return html_message("info", "No subprocess single-image worker is active. In-process single captioning cannot be cancelled.")

    def cancel_batch(self) -> str:
        count, message = cancel_active_workers(["beta_zip", "beta_folder"])
        if self._engine is not None:
            stop_message = str(self._engine.stop_batch())
            return html_message("info", f"{message}<br>{stop_message}") if count else stop_message
        return html_message("info", message if count else "No batch is active.")

    def build_prompt(
        self,
        caption_type: str,
        caption_length: str | int,
        extra_options: Sequence[str] | None,
        name_input: str,
        custom_prompt_text: str,
    ) -> str:
        return build_beta_prompt(caption_type, caption_length, extra_options, name_input, custom_prompt_text)

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
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if not use_subprocess:
            yield from self._get_engine().caption_single(
                input_image,
                prompt,
                temperature,
                top_p,
                max_new_tokens,
                quant,
                device_id,
                unload_after_caption,
                save_image,
                use_subprocess,
                allow_tf32,
                clear_cuda_cache,
                low_cpu_mem_usage,
                attention_backend,
                use_liger_kernel,
                remove_newlines,
                discard_repeats,
                caption_prefix,
                caption_suffix,
                replace_pairs,
                replace_case_sensitive,
                replace_single_word,
            )
            return

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
        except Exception as exc:
            yield html_message("error", format_exception(exc)), "", html_message("error", "Subprocess generation failed.")

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
        num_workers: int,
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
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if not use_subprocess:
            yield from self._get_engine().process_batch_files_to_zip(
                files_list,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt_text,
                temperature,
                top_p,
                max_new_tokens,
                num_workers,
                batch_size,
                quant,
                device_id,
                use_subprocess,
                allow_tf32,
                clear_cuda_cache,
                low_cpu_mem_usage,
                attention_backend,
                use_liger_kernel,
                remove_newlines,
                discard_repeats,
                caption_prefix,
                caption_suffix,
                replace_pairs,
                replace_case_sensitive,
                replace_single_word,
            )
            return
        if not files_list:
            yield html_message("error", "No files selected."), None, ""
            return
        try:
            yield html_message("info", "Starting subprocess ZIP batch..."), None, ""
            data = run_worker(
                "beta_zip",
                {
                    "files": [str(self._file_path(item)) for item in files_list],
                    "settings": {
                        "caption_type": caption_type,
                        "caption_length": caption_length,
                        "extra_options": list(extra_options or []),
                        "name_input": name_input,
                        "custom_prompt": custom_prompt_text,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "num_workers": num_workers,
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
                        "allow_tf32": allow_tf32,
                        "clear_cuda_cache": clear_cuda_cache,
                        "low_cpu_mem_usage": low_cpu_mem_usage,
                        "attention_backend": attention_backend,
                        "use_liger_kernel": use_liger_kernel,
                    },
                },
            )
            status = str(data.get("status") or html_message("success", "Subprocess ZIP batch complete. Child process exited."))
            yield status, data.get("zip_path"), str(data.get("error") or "")
        except Exception as exc:
            yield html_message("error", format_exception(exc)), None, html_message("error", "Subprocess ZIP batch failed.")

    def run_batch_folder_processing(
        self,
        input_folder_str: str,
        output_folder_str: str,
        copy_images_cb: bool,
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
        num_workers: int,
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
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if not use_subprocess:
            yield from self._get_engine().run_batch_folder_processing(
                input_folder_str,
                output_folder_str,
                copy_images_cb,
                overwrite_caption_cb,
                append_caption_cb,
                remove_newlines_cb,
                discard_repeats_cb,
                process_subfolders_cb,
                downscale_max_res_str,
                caption_prefix,
                caption_suffix,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt_text,
                temperature,
                top_p,
                max_new_tokens,
                num_workers,
                batch_size,
                quant,
                device_id,
                use_subprocess,
                allow_tf32,
                clear_cuda_cache,
                low_cpu_mem_usage,
                attention_backend,
                use_liger_kernel,
                replace_pairs,
                replace_case_sensitive,
                replace_single_word,
            )
            return
        try:
            yield html_message("info", "Starting subprocess folder batch..."), ""
            data = run_worker(
                "beta_folder",
                {
                    "settings": {
                        "input_folder": input_folder_str,
                        "output_folder": output_folder_str,
                        "copy_images": copy_images_cb,
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
                        "num_workers": num_workers,
                        "batch_size": batch_size,
                        "quant": quant,
                        "device_id": device_id,
                        "allow_tf32": allow_tf32,
                        "clear_cuda_cache": clear_cuda_cache,
                        "low_cpu_mem_usage": low_cpu_mem_usage,
                        "attention_backend": attention_backend,
                        "use_liger_kernel": use_liger_kernel,
                    },
                },
            )
            yield str(data.get("status") or html_message("success", "Subprocess folder batch complete. Child process exited.")), str(data.get("error") or "")
        except Exception as exc:
            yield html_message("error", format_exception(exc)), html_message("error", "Subprocess folder batch failed.")

    def _file_path(self, item: Any) -> Path:
        if isinstance(item, (str, Path)):
            return Path(item)
        name = getattr(item, "name", None)
        if name:
            return Path(name)
        raise ValueError(f"Unsupported file object: {item!r}")


class LazyQwenEngine:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._engine: Any | None = None
        self._lock = threading.RLock()
        self._registry_key = f"qwen:{self.model_path.resolve()}"
        _MODEL_SWITCH_REGISTRY.register(self._registry_key, self)

    def _get_engine(self) -> Any:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        with self._lock:
            if self._engine is None:
                from .engines.qwen import QwenEngine

                self._engine = QwenEngine(self.model_path)
            return self._engine

    def clear_models(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            clear = getattr(engine, "clear_models", None)
            if callable(clear):
                clear()
        _clear_python_and_cuda_cache()

    def cancel_single(self) -> str:
        count, message = cancel_active_workers(["qwen_single"])
        if count:
            return html_message("info", message)
        return html_message("info", "No subprocess single-image worker is active. In-process single captioning cannot be cancelled.")

    def cancel_batch(self) -> str:
        count, message = cancel_active_workers(["qwen_zip", "qwen_folder"])
        if self._engine is not None:
            stop_message = str(self._engine.stop_batch())
            return html_message("info", f"{message}<br>{stop_message}") if count else stop_message
        return html_message("info", message if count else "No Qwen batch is active.")

    def caption_single(self, input_image: Any | None, settings: dict[str, Any]) -> Generator[tuple[str, str, str, list[list[Any]], str, dict[str, Any]], None, None]:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if not settings.get("use_subprocess", False):
            yield from self._get_engine().caption_single(input_image, settings)
            return
        image_path = coerce_image_path(input_image, OUTPUTS_DIR / "temp")
        if image_path is None:
            yield html_message("error", "No image selected."), "", "", [], "", {}
            return
        try:
            yield html_message("info", "Starting Qwen subprocess caption run..."), "", "", [], "", {}
            data = run_worker(
                "qwen_single",
                {
                    "image_path": str(image_path),
                    "settings": {**settings, "use_subprocess": False},
                },
            )
            yield (
                str(data.get("status") or html_message("success", "Qwen subprocess captioning complete.")),
                str(data.get("caption") or ""),
                str(data.get("overlay") or ""),
                data.get("element_rows") or [],
                str(data.get("error") or ""),
                data.get("autosave_target") or {},
            )
        except Exception as exc:
            yield html_message("error", format_exception(exc)), "", "", [], html_message("error", "Qwen subprocess generation failed."), {}

    def process_batch_files_to_zip(
        self,
        files_list: Sequence[Any] | None,
        settings: dict[str, Any],
    ) -> Generator[tuple[str, str | None, str], None, None]:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if not settings.get("use_subprocess", False):
            yield from self._get_engine().process_batch_files_to_zip(files_list, settings)
            return
        if not files_list:
            yield html_message("error", "No files selected."), None, ""
            return
        try:
            yield html_message("info", "Starting Qwen subprocess ZIP batch..."), None, ""
            data = run_worker(
                "qwen_zip",
                {
                    "files": [str(self._file_path(item)) for item in files_list],
                    "settings": {**settings, "use_subprocess": False},
                },
            )
            yield str(data.get("status") or html_message("success", "Qwen ZIP batch complete.")), data.get("zip_path"), str(data.get("error") or "")
        except Exception as exc:
            yield html_message("error", format_exception(exc)), None, html_message("error", "Qwen subprocess ZIP batch failed.")

    def run_batch_folder_processing(self, settings: dict[str, Any]) -> Generator[tuple[str, str], None, None]:
        _MODEL_SWITCH_REGISTRY.activate(self._registry_key)
        if not settings.get("use_subprocess", False):
            yield from self._get_engine().run_batch_folder_processing(settings)
            return
        try:
            yield html_message("info", "Starting Qwen subprocess folder batch..."), ""
            data = run_worker(
                "qwen_folder",
                {"settings": {**settings, "use_subprocess": False}},
            )
            yield str(data.get("status") or html_message("success", "Qwen folder batch complete.")), str(data.get("error") or "")
        except Exception as exc:
            yield html_message("error", format_exception(exc)), html_message("error", "Qwen subprocess folder batch failed.")

    def _file_path(self, item: Any) -> Path:
        if isinstance(item, (str, Path)):
            return Path(item)
        name = getattr(item, "name", None)
        if name:
            return Path(name)
        raise ValueError(f"Unsupported file object: {item!r}")
