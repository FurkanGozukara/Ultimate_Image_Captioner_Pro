from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joycaption.common import OUTPUTS_DIR
from joycaption.engines.qwen import QwenEngine
from joycaption.json_tools import parse_json_caption, validate_bbox
from joycaption.qwen_presets import OFFICIAL_V1_PRESET_ID, preset_payload


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp"}
DEFAULT_INPUT_DIR = ROOT / "outputs" / "bbox_real_batch_inputs"
DEFAULT_REPORT_DIR = ROOT / "outputs" / "bbox_qwen_real_batch"


def _numeric_output_dirs() -> set[Path]:
    if not OUTPUTS_DIR.exists():
        return set()
    return {path for path in OUTPUTS_DIR.iterdir() if path.is_dir() and path.name.isdigit()}


def _new_run_dir(before: set[Path]) -> Path | None:
    after = _numeric_output_dirs()
    created = sorted(after - before, key=lambda item: int(item.name))
    if created:
        return created[-1]
    numeric = sorted(after, key=lambda item: (item.stat().st_mtime, int(item.name)))
    return numeric[-1] if numeric else None


def _settings() -> dict[str, Any]:
    payload = preset_payload(OFFICIAL_V1_PRESET_ID)
    return {
        **payload,
        "preset_id": OFFICIAL_V1_PRESET_ID,
        "vram_preset": "32 GB",
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "max_new_tokens": 4096,
        "image_long_edge": 1024,
        "model_quantization": "bf16",
        "unload_model": False,
        "save_image": True,
        "use_subprocess": False,
        "allow_tf32": True,
        "clear_cuda_cache": True,
        "low_cpu_mem_usage": True,
        "attention_backend": "sdpa",
        "compact_json": True,
        "json_retries": 1,
        "remove_newlines": False,
        "auto_save_boxed_image": True,
        "caption_prefix": "",
        "caption_suffix": "",
        "device_id": "0",
        "file_batch_size": 1,
        "folder_input": "",
        "folder_output": "",
        "skip_exists": False,
        "overwrite_caption": True,
        "append_caption": False,
        "process_subfolders": False,
        "folder_batch_size": 1,
        "app_side_only": False,
        "console_progress": False,
    }


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", str(value or "")).strip()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).size


def _caption_stats(caption_path: Path | None) -> dict[str, Any]:
    if caption_path is None or not caption_path.exists():
        return {"box_count": 0, "valid_box_count": 0, "invalid_boxes": [], "aspect_ratio": None}
    parsed, _pretty, warnings = parse_json_caption(caption_path.read_text(encoding="utf-8"))
    if parsed is None:
        return {"box_count": 0, "valid_box_count": 0, "invalid_boxes": warnings, "aspect_ratio": None}
    elements = parsed.get("compositional_deconstruction", {}).get("elements", [])
    invalid: list[str] = []
    valid_count = 0
    box_count = 0
    if isinstance(elements, list):
        for index, element in enumerate(elements, start=1):
            if not isinstance(element, dict) or "bbox" not in element:
                continue
            box_count += 1
            valid, message = validate_bbox(element.get("bbox"))
            if valid:
                valid_count += 1
            else:
                invalid.append(f"element {index}: {message}")
    return {
        "box_count": box_count,
        "valid_box_count": valid_count,
        "invalid_boxes": invalid,
        "aspect_ratio": parsed.get("aspect_ratio"),
        "high_level_description": parsed.get("high_level_description"),
    }


def _load_metadata(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _make_montage(results: list[dict[str, Any]], output_path: Path) -> None:
    tiles = []
    for result in results:
        image_path = Path(result.get("boxed_image_path") or result.get("output_image_path") or result["source_path"])
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            thumb = ImageOps.exif_transpose(image).convert("RGB")
            thumb.thumbnail((300, 220))
            tile = Image.new("RGB", (320, 275), "white")
            tile.paste(thumb, ((320 - thumb.width) // 2, 8))
            draw = ImageDraw.Draw(tile)
            label = f"{result['source_name']} | boxes {result.get('box_count', 0)}"
            warning = "converted" if any("converted" in str(w) for w in result.get("json_warnings", [])) else ""
            draw.text((8, 232), label[:48], fill=(0, 0, 0))
            draw.text((8, 250), warning[:48], fill=(160, 80, 0))
            tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    montage = Image.new("RGB", (cols * 320, rows * 275), (235, 235, 235))
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % cols) * 320, (index // cols) * 275))
    montage.save(output_path, quality=92)


def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_DIR
    report_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REPORT_DIR
    paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.name != "contact_sheet.jpg")
    if not paths:
        raise SystemExit(f"No images found in {input_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    settings = _settings()
    engine = QwenEngine(ROOT / "model_files_qwen3_vl3_8b_instruct")
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        before = _numeric_output_dirs()
        final_status = ""
        final_caption = ""
        rows: list[list[Any]] = []
        error_html = ""
        image_started = time.time()
        try:
            for status, caption, _overlay, element_rows, error in engine.caption_single(path, settings):
                final_status = status
                final_caption = caption or final_caption
                rows = element_rows or rows
                error_html = error or error_html
        except Exception as exc:
            final_status = f"Exception: {type(exc).__name__}: {exc}"
            error_html = final_status
        run_dir = _new_run_dir(before)
        metadata = _load_metadata(run_dir)
        caption_path = Path(metadata["caption_path"]) if metadata.get("caption_path") else None
        caption_stats = _caption_stats(caption_path)
        width, height = _image_size(path)
        result = {
            "source_name": path.name,
            "source_path": str(path),
            "source_width": width,
            "source_height": height,
            "run_dir": str(run_dir) if run_dir else None,
            "status_text": _strip_html(final_status),
            "error_text": _strip_html(error_html),
            "caption_path": str(caption_path) if caption_path else None,
            "output_image_path": metadata.get("output_image_path"),
            "boxed_image_path": metadata.get("boxed_image_path"),
            "json_warnings": metadata.get("json_warnings", []),
            "generated_tokens": metadata.get("generated_tokens"),
            "generation_elapsed_seconds": metadata.get("generation_elapsed_seconds"),
            "elapsed_seconds": time.time() - image_started,
            "row_count": len(rows),
            "caption_length": len(final_caption),
            **caption_stats,
        }
        print(
            f"  boxes={result['box_count']} valid={result['valid_box_count']} "
            f"warnings={len(result['json_warnings'])} run={result['run_dir']}",
            flush=True,
        )
        results.append(result)
    report = {
        "input_dir": str(input_dir),
        "settings": {key: settings.get(key) for key in ("preset_id", "image_long_edge", "model_quantization", "device_id", "attention_backend")},
        "elapsed_seconds": time.time() - started,
        "image_count": len(results),
        "total_boxes": sum(int(item.get("box_count") or 0) for item in results),
        "total_valid_boxes": sum(int(item.get("valid_box_count") or 0) for item in results),
        "images_with_warnings": sum(1 for item in results if item.get("json_warnings")),
        "images_with_errors": sum(1 for item in results if item.get("error_text")),
        "results": results,
    }
    report_path = report_dir / "report.json"
    montage_path = report_dir / "boxed_montage.jpg"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_montage(results, montage_path)
    print(json.dumps({key: report[key] for key in ("image_count", "total_boxes", "total_valid_boxes", "images_with_warnings", "images_with_errors", "elapsed_seconds")}, ensure_ascii=False))
    print(f"report={report_path}")
    print(f"montage={montage_path}")


if __name__ == "__main__":
    main()
