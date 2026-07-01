from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joycaption.engines.qwen import QwenEngine


RAW_WITH_EXTRA_IMAGE_KEY = json.dumps(
    {
        "image": "https://i.imgur.com/7ZQZQZQ.jpg",
        "aspect_ratio": "1:1",
        "high_level_description": "A simple caption.",
        "compositional_deconstruction": {
            "background": "Plain background.",
            "elements": [],
        },
    }
)


def _finalize(settings: dict[str, object]) -> tuple[str, dict[str, object] | None, list[str]]:
    engine = QwenEngine(Path("unused-model-path"))
    image = Image.new("RGB", (8, 8), "white")
    return engine._finalize_output(image, RAW_WITH_EXTRA_IMAGE_KEY, settings)


def main() -> None:
    official_settings = {
        "id": "i4_json_auto_best",
        "output_format": "json",
        "extension": ".json",
        "beautify_saved_json": True,
        "compact_json": False,
        "json_retries": 0,
    }
    final, parsed, warnings = _finalize(official_settings)
    assert warnings == []
    assert parsed is not None
    assert "image" not in parsed
    assert "i.imgur.com" not in final
    assert list(parsed.keys()) == [
        "aspect_ratio",
        "high_level_description",
        "compositional_deconstruction",
    ]

    non_official_settings = {
        "id": "txt_flux2_general",
        "output_format": "json",
        "extension": ".json",
        "beautify_saved_json": True,
        "compact_json": False,
        "json_retries": 0,
    }
    final, parsed, warnings = _finalize(non_official_settings)
    assert warnings == []
    assert parsed is not None
    assert parsed["image"] == "https://i.imgur.com/7ZQZQZQ.jpg"
    assert "i.imgur.com" in final
    print("Qwen JSON finalization verification passed.")


if __name__ == "__main__":
    main()
