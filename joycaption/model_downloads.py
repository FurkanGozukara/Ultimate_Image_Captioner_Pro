from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .common import BASE_DIR, log_event
from .model_catalog import ModelSpec, get_model_spec, model_is_ready, model_readiness_error


@dataclass(frozen=True)
class ModelAvailability:
    spec: ModelSpec
    downloaded: bool


_DOWNLOAD_LOCKS_GUARD = threading.Lock()
_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}


def _download_lock(model_key: str) -> threading.Lock:
    with _DOWNLOAD_LOCKS_GUARD:
        return _DOWNLOAD_LOCKS.setdefault(model_key, threading.Lock())


def model_downloader_path() -> Path:
    override = os.environ.get("ULTIMATE_CAPTION_MODEL_DOWNLOADER")
    path = Path(override).expanduser() if override else BASE_DIR.parent / "HF_model_downloader.py"
    path = path.resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(
            "The model downloader was not found one directory above the app. "
            f"Expected: {path}"
        )
    return path


def download_required(model_key: str) -> bool:
    return not model_is_ready(model_key)


def ensure_model_available(model_key: str) -> ModelAvailability:
    spec = get_model_spec(model_key)
    if model_is_ready(model_key):
        return ModelAvailability(spec=spec, downloaded=False)

    with _download_lock(model_key):
        if model_is_ready(model_key):
            return ModelAvailability(spec=spec, downloaded=False)

        downloader = model_downloader_path()
        command = [
            sys.executable,
            str(downloader),
            "--model",
            spec.key,
            "--target-root",
            str(BASE_DIR),
        ]
        log_event(f"Downloading {spec.label} with {downloader}.", "Model Downloader")
        completed = subprocess.run(command, cwd=str(downloader.parent), check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"The model downloader exited with code {completed.returncode} while downloading {spec.label}. "
                "Review the downloader output, then press the caption button again to resume."
            )

        readiness_error = model_readiness_error(model_key)
        if readiness_error is not None:
            raise RuntimeError(f"{spec.label} download finished but validation failed: {readiness_error}")
        log_event(f"{spec.label} download and validation complete.", "Model Downloader")
        return ModelAvailability(spec=spec, downloaded=True)
