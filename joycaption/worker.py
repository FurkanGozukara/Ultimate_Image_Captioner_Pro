from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from .common import BASE_DIR, OUTPUTS_DIR, apply_torch_optimizations, log_event, optimization_status_text, parse_device_ids, reset_vram_peak_stats, save_numbered_generation, vram_usage_text
from .engines.beta import BetaEngine
from .engines.qwen import QwenEngine
from .engines.legacy_siglip import (
    create_alpha_one_engine,
    create_alpha_two_engine,
    create_pre_alpha_engine,
)


def _legacy_engine(variant: str):
    if variant == "pre_alpha":
        return create_pre_alpha_engine(BASE_DIR)
    if variant == "alpha_one":
        return create_alpha_one_engine(BASE_DIR)
    if variant == "alpha_two":
        return create_alpha_two_engine(BASE_DIR)
    raise ValueError(f"Unknown legacy variant: {variant}")


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def legacy_single(payload: dict[str, Any]) -> dict[str, Any]:
    log_event(f"Legacy single start: variant={payload['variant']}", "Worker")
    settings = dict(payload["settings"])
    settings["use_subprocess"] = False
    engine = _legacy_engine(str(payload["variant"]))
    result = engine.caption_single(payload["image_path"], settings)
    log_event(f"Legacy single complete: caption_path={result.caption_path}", "Worker")
    return {
        "ok": True,
        "prompt": result.prompt,
        "caption": result.caption,
        "caption_path": str(result.caption_path) if result.caption_path else None,
        "image_path": str(result.image_path) if result.image_path else None,
        "metadata_path": str(result.metadata_path) if result.metadata_path else None,
        "elapsed": result.elapsed,
        "details": result.details,
    }


def legacy_batch(payload: dict[str, Any]) -> dict[str, Any]:
    log_event(f"Legacy batch start: variant={payload['variant']}", "Worker")
    settings = dict(payload["settings"])
    settings["use_subprocess"] = False
    engine = _legacy_engine(str(payload["variant"]))
    last_progress = ""
    for progress in engine.batch_folder(settings):
        last_progress = progress
    log_event("Legacy batch complete.", "Worker")
    details = f"{optimization_status_text(settings)}\n{vram_usage_text()}"
    return {"ok": True, "progress": f"{last_progress or 'Batch completed.'}\n{details}"}


def beta_single(payload: dict[str, Any]) -> dict[str, Any]:
    log_event("Beta single start.", "Worker")
    settings = payload["settings"]
    engine = BetaEngine(BASE_DIR / "model_files_beta_one")
    start = time.time()
    image = Image.open(payload["image_path"]).convert("RGB")
    reset_vram_peak_stats(parse_device_ids(settings.get("device_id") or "0", allow_cpu=True))
    before_vram = vram_usage_text()
    engine.load_model(str(settings["quant"]), str(settings["device_id"]), settings)
    log_event("Beta model loaded. Generating caption.", "Worker")
    caption = engine.generate_caption(
        image,
        str(settings["prompt"]),
        float(settings["temperature"]),
        float(settings["top_p"]),
        int(settings["max_new_tokens"]),
    )
    apply_torch_optimizations(settings, "after")
    after_vram = vram_usage_text()
    metadata = {
        "generation_type": "single_image",
        "engine": "beta_one",
        "model_path": str(engine.model_path),
        "source_image_path": str(payload["image_path"]),
        "prompt": str(settings["prompt"]),
        "caption_final": caption,
        "settings": dict(settings),
        "elapsed_seconds": time.time() - start,
        "vram_before": before_vram,
        "vram_after": after_vram,
        "optimizations": optimization_status_text(settings),
    }
    output_image_path, caption_path, metadata_path, _run_dir = save_numbered_generation(
        payload["image_path"],
        caption,
        metadata,
        OUTPUTS_DIR,
        copy_image=bool(settings.get("save_image", True)),
    )
    engine.clear_models()
    log_event(f"Beta single complete: caption_path={caption_path}", "Worker")
    return {
        "ok": True,
        "caption": caption,
        "caption_path": str(caption_path),
        "image_path": str(output_image_path) if output_image_path else None,
        "metadata_path": str(metadata_path),
        "details": f"{optimization_status_text(settings)}\nBefore {before_vram}\nAfter {after_vram}",
    }


def beta_zip(payload: dict[str, Any]) -> dict[str, Any]:
    log_event("Beta files-to-ZIP batch start.", "Worker")
    settings = payload["settings"]
    engine = BetaEngine(BASE_DIR / "model_files_beta_one")
    last_status = ""
    zip_path = None
    error_html = ""
    for status, output, error in engine.process_batch_files_to_zip(
        payload["files"],
        settings["caption_type"],
        settings["caption_length"],
        settings.get("extra_options") or [],
        settings.get("name_input") or "",
        settings.get("custom_prompt") or "",
        float(settings["temperature"]),
        float(settings["top_p"]),
        int(settings["max_new_tokens"]),
        int(settings["num_workers"]),
        int(settings["batch_size"]),
        str(settings["quant"]),
        str(settings["device_id"]),
        False,
        bool(settings.get("allow_tf32", True)),
        bool(settings.get("clear_cuda_cache", True)),
        bool(settings.get("low_cpu_mem_usage", True)),
        str(settings.get("attention_backend") or ("sdpa" if settings.get("use_sdpa_attention", False) else "auto")),
        bool(settings.get("use_liger_kernel", True)),
    ):
        last_status = status
        zip_path = output or zip_path
        error_html = error or error_html
    engine.clear_models()
    log_event("Beta files-to-ZIP batch complete.", "Worker")
    return {"ok": True, "status": last_status, "zip_path": zip_path, "error": error_html}


def beta_folder(payload: dict[str, Any]) -> dict[str, Any]:
    log_event("Beta folder batch start.", "Worker")
    settings = payload["settings"]
    engine = BetaEngine(BASE_DIR / "model_files_beta_one")
    last_status = ""
    error_html = ""
    for status, error in engine.run_batch_folder_processing(
        settings["input_folder"],
        settings.get("output_folder") or "",
        bool(settings["copy_images"]),
        bool(settings["skip_exists"]),
        bool(settings["overwrite_caption"]),
        bool(settings["append_caption"]),
        bool(settings["remove_newlines"]),
        bool(settings["discard_repeats"]),
        bool(settings["process_subfolders"]),
        str(settings["downscale_max_res"]),
        settings.get("caption_prefix") or "",
        settings.get("caption_suffix") or "",
        settings["caption_type"],
        settings["caption_length"],
        settings.get("extra_options") or [],
        settings.get("name_input") or "",
        settings.get("custom_prompt") or "",
        float(settings["temperature"]),
        float(settings["top_p"]),
        int(settings["max_new_tokens"]),
        int(settings["num_workers"]),
        int(settings["batch_size"]),
        str(settings["quant"]),
        str(settings["device_id"]),
        False,
        bool(settings.get("allow_tf32", True)),
        bool(settings.get("clear_cuda_cache", True)),
        bool(settings.get("low_cpu_mem_usage", True)),
        str(settings.get("attention_backend") or ("sdpa" if settings.get("use_sdpa_attention", False) else "auto")),
        bool(settings.get("use_liger_kernel", True)),
    ):
        last_status = status
        error_html = error or error_html
    engine.clear_models()
    log_event("Beta folder batch complete.", "Worker")
    return {"ok": True, "status": last_status, "error": error_html}


def qwen_single(payload: dict[str, Any]) -> dict[str, Any]:
    log_event("Qwen single start.", "Worker")
    settings = dict(payload["settings"])
    settings["use_subprocess"] = False
    engine = QwenEngine(BASE_DIR / "model_files_qwen3_vl3_8b_instruct")
    last_status = ""
    caption = ""
    overlay = ""
    element_rows: list[list[Any]] = []
    error_html = ""
    for status, output, overlay_html, rows, error in engine.caption_single(payload["image_path"], settings):
        last_status = status
        caption = output or caption
        overlay = overlay_html or overlay
        element_rows = rows or element_rows
        error_html = error or error_html
    engine.clear_models()
    log_event("Qwen single complete.", "Worker")
    return {
        "ok": True,
        "status": last_status,
        "caption": caption,
        "overlay": overlay,
        "element_rows": element_rows,
        "error": error_html,
    }


def qwen_zip(payload: dict[str, Any]) -> dict[str, Any]:
    log_event("Qwen files-to-ZIP batch start.", "Worker")
    settings = dict(payload["settings"])
    settings["use_subprocess"] = False
    engine = QwenEngine(BASE_DIR / "model_files_qwen3_vl3_8b_instruct")
    last_status = ""
    zip_path = None
    error_html = ""
    for status, output, error in engine.process_batch_files_to_zip(payload["files"], settings):
        last_status = status
        zip_path = output or zip_path
        error_html = error or error_html
    engine.clear_models()
    log_event("Qwen files-to-ZIP batch complete.", "Worker")
    return {"ok": True, "status": last_status, "zip_path": zip_path, "error": error_html}


def qwen_folder(payload: dict[str, Any]) -> dict[str, Any]:
    log_event("Qwen folder batch start.", "Worker")
    settings = dict(payload["settings"])
    settings["use_subprocess"] = False
    engine = QwenEngine(BASE_DIR / "model_files_qwen3_vl3_8b_instruct")
    last_status = ""
    error_html = ""
    for status, error in engine.run_batch_folder_processing(settings):
        last_status = status
        error_html = error or error_html
    engine.clear_models()
    log_event("Qwen folder batch complete.", "Worker")
    return {"ok": True, "status": last_status, "error": error_html}


COMMANDS = {
    "legacy_single": legacy_single,
    "legacy_batch": legacy_batch,
    "beta_single": beta_single,
    "beta_zip": beta_zip,
    "beta_folder": beta_folder,
    "qwen_single": qwen_single,
    "qwen_zip": qwen_zip,
    "qwen_folder": qwen_folder,
}


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: python -m joycaption.worker <command> <payload_json> <result_json>", file=sys.stderr)
        return 2
    command = sys.argv[1]
    payload_path = Path(sys.argv[2])
    result_path = Path(sys.argv[3])
    try:
        if command not in COMMANDS:
            raise ValueError(f"Unknown worker command: {command}")
        log_event(f"Command received: {command}", "Worker")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result = COMMANDS[command](payload)
        _write_result(result_path, result)
        return 0
    except Exception as exc:
        _write_result(
            result_path,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
