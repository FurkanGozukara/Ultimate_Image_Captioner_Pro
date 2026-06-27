from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import PRESETS_DIR, ensure_runtime_dirs


LAST_USED_FILE = ".last_used_preset.txt"


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "default"


class UniversalPresetStore:
    def __init__(self, base_dir: Path = PRESETS_DIR) -> None:
        ensure_runtime_dirs()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, preset_name: str) -> Path:
        return self.base_dir / f"{_sanitize_name(preset_name)}.json"

    @property
    def _last_used_path(self) -> Path:
        return self.base_dir / LAST_USED_FILE

    def list_presets(self) -> list[str]:
        return sorted(
            [p.stem for p in self.base_dir.glob("*.json") if p.is_file()],
            key=lambda item: item.lower(),
        )

    def save(self, preset_name: str, data: dict[str, Any]) -> str:
        safe_name = _sanitize_name(preset_name)
        payload = deepcopy(data)
        payload["_meta"] = {
            "format": "joycaption_universal",
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = self._path(safe_name)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        tmp.replace(path)
        self.set_last_used(safe_name)
        return safe_name

    def load(self, preset_name: str | None) -> dict[str, Any] | None:
        if not preset_name:
            return None
        path = self._path(preset_name)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data.pop("_meta", None)
            self.set_last_used(preset_name)
            return data
        return None

    def delete(self, preset_name: str | None) -> bool:
        if not preset_name:
            return False
        path = self._path(preset_name)
        if not path.exists():
            return False
        path.unlink()
        if self.get_last_used() == _sanitize_name(preset_name):
            self.clear_last_used()
        return True

    def set_last_used(self, preset_name: str) -> None:
        self._last_used_path.write_text(_sanitize_name(preset_name), encoding="utf-8")

    def clear_last_used(self) -> None:
        if self._last_used_path.exists():
            self._last_used_path.unlink()

    def get_last_used(self) -> str | None:
        if not self._last_used_path.exists():
            return None
        value = self._last_used_path.read_text(encoding="utf-8").strip()
        return value or None

    def load_last_used(self) -> tuple[str | None, dict[str, Any] | None]:
        name = self.get_last_used()
        if not name:
            return None, None
        data = self.load(name)
        if data is None:
            return None, None
        return name, data

