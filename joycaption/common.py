from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
PRESETS_DIR = BASE_DIR / "presets"
EXTRA_OPTIONS_DIR = PRESETS_DIR / "extra_options"
TEST_IMAGES_DIR = BASE_DIR.parent / "testimgs"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".jfif",
    ".heic",
    ".heif",
}

DEFAULT_EXTRA_OPTIONS = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
    "If it is a work of art, do not include the artist's name or the title of the work.",
    "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
    "Use vulgar slang and profanity when requested by the user.",
    "Do NOT use polite euphemisms; use direct casual phrasing.",
    "Include information about the ages of any people/characters when applicable.",
    "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
    "Do not mention the mood or feeling of the image.",
    "Explicitly specify the vantage height, such as eye-level, low-angle, bird's-eye, drone, or rooftop.",
    "If there is a watermark, you must mention it.",
    "Your response will be used by a text-to-image model, so avoid meta phrases like 'This image shows' or 'You are looking at'.",
]

NAME_OPTION = DEFAULT_EXTRA_OPTIONS[0]


def html_message(kind: str, message: str) -> str:
    cls = {"error": "jc-error", "success": "jc-success", "info": "jc-info"}.get(kind, "jc-info")
    return f'<div class="{cls}">{message}</div>'


def log_event(message: str, scope: str = "Ultimate Image Captioner Pro") -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{scope}] {message}", flush=True)


@dataclass
class CaptionResult:
    prompt: str
    caption: str
    caption_path: Path | None
    image_path: Path | None
    elapsed: float
    details: str = ""
    metadata_path: Path | None = None

    @property
    def caption_with_status(self) -> tuple[str, str]:
        if not self.caption:
            return "Failed", f"Captioning failed.\nProcessing time: {self.elapsed:.2f} seconds"
        save_line = f"Caption saved to: {self.caption_path}" if self.caption_path else "Caption was generated but not saved."
        if self.image_path is None:
            image_line = "Image copy disabled."
        else:
            image_line = f"Image output: {self.image_path}"
        metadata_line = f"\nMetadata saved to: {self.metadata_path}" if self.metadata_path else ""
        detail_line = f"\n{self.details}" if self.details else ""
        return self.caption, f"{save_line}\n{image_line}{metadata_line}\nProcessing time: {self.elapsed:.2f} seconds{detail_line}"


class BatchStopFlag:
    def __init__(self) -> None:
        self.value = False

    def reset(self) -> None:
        self.value = False

    def stop(self) -> None:
        self.value = True


def ensure_runtime_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRA_OPTIONS_DIR.mkdir(parents=True, exist_ok=True)


def natural_sort_key(path: str | Path) -> list[Any]:
    name = Path(path).name.lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def coerce_image_path(image_input: Any, temp_dir: Path | None = None) -> Path | None:
    if image_input is None:
        return None
    if isinstance(image_input, dict):
        candidate = image_input.get("path") or image_input.get("name")
        return Path(candidate) if candidate else None
    if isinstance(image_input, (str, Path)):
        return Path(image_input)
    if isinstance(image_input, Image.Image):
        temp_root = temp_dir or OUTPUTS_DIR / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        path = temp_root / f"input_{int(time.time() * 1000)}.png"
        image_input.save(path)
        return path
    return None


def load_rgb_image(image_input: str | Path | Image.Image, max_resolution: int | None = None) -> Image.Image:
    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input)
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    if max_resolution and max_resolution > 0:
        image.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)
    return image


def discover_images(folder: str | Path, include_subfolders: bool = True) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    iterator: Iterable[Path] = root.rglob("*") if include_subfolders else root.glob("*")
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=natural_sort_key)


def resolve_output_paths(
    image_path: str | Path,
    input_root: str | Path | None,
    output_root: str | Path | None,
    preserve_subfolders: bool = True,
) -> tuple[Path, Path]:
    image_path = Path(image_path)
    output_dir = Path(output_root) if output_root else OUTPUTS_DIR
    if input_root and preserve_subfolders:
        try:
            relative = image_path.relative_to(Path(input_root))
        except ValueError:
            relative = Path(image_path.name)
    else:
        relative = Path(image_path.name)
    output_image_path = output_dir / relative
    caption_path = output_image_path.with_suffix(".txt")
    return output_image_path, caption_path


def next_numbered_output_dir(output_root: str | Path = OUTPUTS_DIR) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 100000):
        candidate = root / f"{idx:04d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a numbered output folder in {root}")


def finalize_caption_text(
    caption: str,
    remove_newlines: bool = True,
    prefix: str = "",
    suffix: str = "",
    replace_pairs: Any | None = None,
    replace_case_sensitive: bool = False,
    replace_single_word: bool = False,
) -> str:
    if remove_newlines:
        caption = " ".join(str(caption).split())
    caption = apply_replace_pairs(
        str(caption),
        replace_pairs,
        case_sensitive=replace_case_sensitive,
        single_word=replace_single_word,
    )
    return f"{prefix or ''}{caption}{suffix or ''}"


def normalize_replace_pairs(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    pairs: list[list[str]] = []
    if not isinstance(value, Iterable) or isinstance(value, (bytes, bytearray, dict)):
        return pairs
    for item in value:
        find_text = ""
        replace_text = ""
        if isinstance(item, dict):
            find_text = str(item.get("find") or item.get("from") or item.get("source") or "")
            replace_text = str(item.get("replace") or item.get("to") or item.get("target") or "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            find_text = str(item[0] or "")
            replace_text = str(item[1] or "")
        find_text = find_text.strip()
        if find_text:
            pairs.append([find_text, replace_text])
    return pairs


def apply_replace_pairs(
    text: str,
    replace_pairs: Any | None,
    case_sensitive: bool = False,
    single_word: bool = False,
) -> str:
    result = str(text or "")
    flags = 0 if case_sensitive else re.IGNORECASE
    for find_text, replace_text in normalize_replace_pairs(replace_pairs):
        pattern = re.escape(find_text)
        if single_word:
            pattern = rf"(?<!\w){pattern}(?!\w)"
        result = re.sub(pattern, str(replace_text), result, flags=flags)
    return result


def write_generation_metadata(metadata_path: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.setdefault("saved_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def save_numbered_generation(
    image_source: str | Path | Image.Image | None,
    caption: str,
    metadata: dict[str, Any],
    output_root: str | Path = OUTPUTS_DIR,
    copy_image: bool = True,
    caption_extension: str = ".txt",
) -> tuple[Path | None, Path, Path, Path]:
    run_dir = next_numbered_output_dir(output_root)

    output_image_path: Path | None = None
    caption_stem = "image"
    if image_source is not None:
        if isinstance(image_source, Image.Image):
            if copy_image:
                output_image_path = run_dir / "image.png"
                image_source.save(output_image_path)
        else:
            source_path = Path(image_source)
            source_name = source_path.name or f"image{source_path.suffix or '.png'}"
            caption_stem = source_path.stem or "image"
            if copy_image:
                output_image_path = run_dir / source_name
                copy_image_if_needed(source_path, output_image_path, True)

    extension = str(caption_extension or ".txt").strip()
    if not extension.startswith("."):
        extension = f".{extension}"
    caption_path = run_dir / f"{caption_stem}{extension}"
    caption_path.write_text(caption, encoding="utf-8")

    metadata_path = run_dir / "metadata.json"
    enriched = dict(metadata)
    enriched.update(
        {
            "output_run_dir": str(run_dir),
            "output_image_path": str(output_image_path) if output_image_path else None,
            "caption_path": str(caption_path),
            "metadata_path": str(metadata_path),
        }
    )
    write_generation_metadata(metadata_path, enriched)
    return output_image_path, caption_path, metadata_path, run_dir


def caption_exists_for(
    image_path: str | Path,
    input_root: str | Path,
    output_root: str | Path | None,
    preserve_subfolders: bool = True,
) -> bool:
    _, caption_path = resolve_output_paths(image_path, input_root, output_root, preserve_subfolders)
    return caption_path.exists()


def save_caption_file(
    caption_path: str | Path,
    caption: str,
    overwrite: bool = False,
    append: bool = False,
    remove_newlines: bool = True,
    prefix: str = "",
    suffix: str = "",
    replace_pairs: Any | None = None,
    replace_case_sensitive: bool = False,
    replace_single_word: bool = False,
) -> Path | None:
    caption_path = Path(caption_path)
    final_caption = finalize_caption_text(
        caption,
        remove_newlines=remove_newlines,
        prefix=prefix,
        suffix=suffix,
        replace_pairs=replace_pairs,
        replace_case_sensitive=replace_case_sensitive,
        replace_single_word=replace_single_word,
    )
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    if caption_path.exists():
        if overwrite:
            mode = "w"
        elif append:
            mode = "a"
        else:
            return None
    else:
        mode = "w"
    with caption_path.open(mode, encoding="utf-8") as handle:
        if mode == "a":
            handle.write("\n")
        handle.write(final_caption)
    return caption_path


def copy_image_if_needed(source: str | Path, target: str | Path, copy_image: bool) -> Path | None:
    if not copy_image:
        return None
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        return target_path
    except shutil.SameFileError:
        return target_path


def cut_off_last_sentence(caption: str) -> str:
    sentence_endings = re.findall(r"[.!?]", caption)
    if sentence_endings:
        last_ending_index = caption.rfind(sentence_endings[-1])
        return caption[: last_ending_index + 1].strip()
    return caption.strip()


def remove_repeating_sentences(caption: str) -> str:
    if not caption:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", caption)
    unique: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = re.sub(r"[^\w\s]", "", sentence).lower().strip()
        if not normalized or normalized not in seen:
            unique.append(sentence)
            if normalized:
                seen.add(normalized)
    return " ".join(s.strip() for s in unique if s.strip())


def clean_legacy_caption(caption: str, aggressive: bool = False) -> str:
    cleaned = caption.replace("|.", "").replace("<|end_of_text|>", "")
    cleaned = cleaned.replace("<|eot_id|>", "")
    if aggressive:
        cleaned = cleaned.replace("This photograph ", "a photograph ")
    cleaned = " ".join(cleaned.split()).strip()
    if cleaned.lower().startswith("this is "):
        cleaned = cleaned[8:].strip()
    if aggressive and cleaned.lower().startswith("this photograph captures "):
        cleaned = cleaned[25:].strip()
    return cleaned


def parse_device_ids(device_ids: str | int | None, allow_cpu: bool = True) -> list[int | str]:
    text = str(device_ids if device_ids is not None else "0").strip()
    if allow_cpu and text.lower() == "cpu":
        return ["cpu"]
    values: list[int | str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or [0]


def first_device(device_ids: str | int | None, allow_cpu: bool = True) -> int | str:
    return parse_device_ids(device_ids, allow_cpu=allow_cpu)[0]


def format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def open_folder(path: str | Path | None) -> str:
    target = Path(path or OUTPUTS_DIR)
    target.mkdir(parents=True, exist_ok=True)
    try:
        if platform.system() == "Windows":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return f"Opened: {target}"
    except Exception as exc:
        return f"Could not open folder {target}: {exc}"


def get_all_extra_options() -> list[str]:
    ensure_runtime_dirs()
    custom: list[str] = []
    for path in sorted(EXTRA_OPTIONS_DIR.glob("*.txt"), key=natural_sort_key):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                custom.append(text)
        except OSError:
            continue
    return custom + DEFAULT_EXTRA_OPTIONS


def save_custom_extra_option(option_text: str) -> bool:
    ensure_runtime_dirs()
    text = (option_text or "").strip()
    if not text:
        return False
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in text[:80])
    safe = "_".join(safe.split()) or f"option_{int(time.time())}"
    path = EXTRA_OPTIONS_DIR / f"{safe}.txt"
    path.write_text(text, encoding="utf-8")
    return True


def values_from_components(keys: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    return {key: value for key, value in zip(keys, values)}


def ordered_values(keys: Sequence[str], data: dict[str, Any], defaults: dict[str, Any]) -> list[Any]:
    return [data.get(key, defaults.get(key)) for key in keys]


def throttle_status(line: str, progress: str, max_chars: int = 12000) -> str:
    combined = f"{progress}{line}\n"
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def apply_torch_optimizations(settings: dict[str, Any], when: str = "before") -> list[str]:
    applied: list[str] = []
    try:
        import torch

        if settings.get("allow_tf32", False) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            applied.append("TF32 enabled")
        if settings.get("clear_cuda_cache", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            applied.append(f"CUDA cache cleared {when} run")
    except Exception as exc:
        applied.append(f"Optimization warning: {exc}")
    return applied


def vram_usage_text() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "VRAM: CUDA unavailable."
        lines = ["VRAM usage:"]
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(idx) / (1024**3)
            lines.append(
                f"GPU {idx}: allocated {allocated:.2f} GiB, reserved {reserved:.2f} GiB, peak {max_allocated:.2f} GiB, total {total:.2f} GiB"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"VRAM: unavailable ({exc})"


def reset_vram_peak_stats(device_ids: Sequence[int | str] | None = None) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        targets = list(device_ids) if device_ids else list(range(torch.cuda.device_count()))
        for device_id in targets:
            if str(device_id).lower() == "cpu":
                continue
            index = int(device_id)
            if 0 <= index < torch.cuda.device_count():
                torch.cuda.reset_peak_memory_stats(index)
    except Exception:
        return


def optimization_status_text(settings: dict[str, Any]) -> str:
    labels = []
    if settings.get("allow_tf32", False):
        labels.append("TF32")
    if settings.get("clear_cuda_cache", False):
        labels.append("clear CUDA cache")
    if settings.get("low_cpu_mem_usage", False):
        labels.append("low CPU memory loading")
    if settings.get("use_sdpa_attention", False):
        labels.append("SDPA attention")
    if settings.get("use_liger_kernel", False):
        labels.append("Liger kernel")
    try:
        from .attention import attention_status_text

        labels.append(attention_status_text(settings))
    except Exception:
        pass
    return "Optimizations: " + (", ".join(labels) if labels else "none")


ProgressCallback = Callable[[str], None]
