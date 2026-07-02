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

RAW_TEXT_PRESET_JSON_WITH_EXTRA_IMAGE_KEY = json.dumps(
    {
        "image": "https://i.imgur.com/7ZQZQZQ.jpg",
        "description": "A woman stands in a studio portrait.",
        "objects": [
            {
                "label": "woman",
                "bbox": [100, 100, 900, 900],
            }
        ],
    }
)


def _finalize(raw_caption: str, settings: dict[str, object]) -> tuple[str, dict[str, object] | None, list[str]]:
    engine = QwenEngine(Path("unused-model-path"))
    image = Image.new("RGB", (8, 8), "white")
    return engine._finalize_output(image, raw_caption, settings)


class FakeStructuredQwenEngine(QwenEngine):
    def __init__(self) -> None:
        super().__init__(Path("unused-model-path"))

    def generate_caption(self, image: Image.Image, settings: dict[str, object]) -> str:
        return json.dumps(
            {
                "image": "https://i.imgur.com/7ZQZQZQ.jpg",
                "caption": "A woman stands in a studio portrait.",
                "elements": [
                    {
                        "label": "woman",
                        "bbox": [100, 100, 900, 900],
                        "description": "A woman standing against a plain studio background.",
                    }
                ],
            }
        )


def main() -> None:
    official_settings = {
        "id": "i4_json_auto_best",
        "output_format": "json",
        "extension": ".json",
        "beautify_saved_json": True,
        "compact_json": False,
        "json_retries": 0,
    }
    final, parsed, warnings = _finalize(RAW_WITH_EXTRA_IMAGE_KEY, official_settings)
    assert warnings == []
    assert parsed is not None
    assert "image" not in parsed
    assert "i.imgur.com" not in final
    assert list(parsed.keys()) == [
        "aspect_ratio",
        "high_level_description",
        "compositional_deconstruction",
    ]

    text_preset_json_settings = {
        "id": "txt_flux2_subject_person_lora",
        "output_format": "json",
        "extension": ".json",
        "beautify_saved_json": True,
        "compact_json": False,
        "json_retries": 0,
        "replace_pairs": [["woman", "maxianer"]],
        "replace_case_sensitive": False,
        "replace_single_word": False,
    }
    final, parsed, warnings = _finalize(RAW_TEXT_PRESET_JSON_WITH_EXTRA_IMAGE_KEY, text_preset_json_settings)
    assert warnings == []
    assert parsed is not None
    assert "image" not in parsed
    assert "i.imgur.com" not in final
    assert "maxianer" in parsed["description"]
    assert parsed["objects"][0]["label"] == "maxianer"

    raw_text = "A woman stands in a studio portrait."
    engine = QwenEngine(Path("unused-model-path"))
    image = Image.new("RGB", (8, 8), "white")
    final, parsed, warnings = engine._finalize_output(image, raw_text, text_preset_json_settings)
    assert warnings == []
    assert parsed == {"caption": "A maxianer stands in a studio portrait."}
    assert json.loads(final) == parsed

    raw_text_with_url = "A woman stands in a studio portrait. https://i.imgur.com/7ZQZQZQ.jpg"
    final, parsed, warnings = engine._finalize_output(image, raw_text_with_url, text_preset_json_settings)
    assert warnings == []
    assert parsed == {"caption": "A maxianer stands in a studio portrait."}
    assert "i.imgur.com" not in final

    structured_settings = dict(text_preset_json_settings)
    structured_settings["json_retries"] = 1
    final, parsed, warnings = FakeStructuredQwenEngine()._finalize_output(image, raw_text, structured_settings)
    assert warnings == []
    assert parsed is not None
    assert "image" not in parsed
    assert "i.imgur.com" not in final
    assert parsed["caption"] == "A maxianer stands in a studio portrait."
    assert parsed["elements"][0]["label"] == "maxianer"
    assert "maxianer" in parsed["elements"][0]["description"]
    assert json.loads(final) == parsed
    print("Qwen JSON finalization verification passed.")


if __name__ == "__main__":
    main()
