from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "bbox_verification"
sys.path.insert(0, str(ROOT))

from joycaption.json_tools import json_to_element_rows, normalize_json_output, save_boxed_image


GEOMETRY_CASES: list[dict[str, Any]] = [
    {
        "name": "square_two_objects",
        "size": (1000, 1000),
        "elements": [
            {"type": "obj", "bbox": [80, 80, 360, 360], "desc": "A square object in the upper-left quadrant."},
            {"type": "obj", "bbox": [580, 620, 920, 940], "desc": "A square object in the lower-right quadrant."},
        ],
    },
    {
        "name": "wide_vehicle_person_text",
        "size": (1600, 900),
        "elements": [
            {"type": "obj", "bbox": [220, 95, 935, 360], "desc": "A standing man on the left side."},
            {"type": "obj", "bbox": [520, 360, 880, 940], "desc": "A long red car across the right side."},
            {"type": "text", "bbox": [75, 520, 160, 900], "text": "GARAGE", "desc": "Bold text sign near the top right."},
        ],
    },
    {
        "name": "portrait_person_lamp",
        "size": (900, 1600),
        "elements": [
            {"type": "obj", "bbox": [55, 330, 990, 690], "desc": "A full-body woman standing in the center."},
            {"type": "obj", "bbox": [285, 50, 820, 190], "desc": "A tall floor lamp on the far left."},
        ],
    },
    {
        "name": "four_three_product",
        "size": (1200, 900),
        "elements": [
            {"type": "obj", "bbox": [210, 360, 850, 625], "desc": "A tall glass bottle centered on a table."},
            {"type": "text", "bbox": [420, 405, 500, 580], "text": "PURE", "desc": "Bottle label text."},
        ],
    },
    {
        "name": "three_four_poster",
        "size": (900, 1200),
        "elements": [
            {"type": "text", "bbox": [80, 110, 205, 890], "text": "SUMMER SALE", "desc": "Large headline across the top."},
            {"type": "obj", "bbox": [345, 215, 760, 790], "desc": "A centered illustrated sneaker product."},
        ],
    },
    {
        "name": "ultrawide_multi_subject",
        "size": (1800, 600),
        "elements": [
            {"type": "obj", "bbox": [120, 55, 920, 225], "desc": "A standing person at far left."},
            {"type": "obj", "bbox": [430, 360, 805, 875], "desc": "A long bus in the mid-right area."},
        ],
    },
    {
        "name": "edge_clamped_boxes",
        "size": (1000, 1400),
        "elements": [
            {"type": "obj", "bbox": [0, 0, 260, 190], "desc": "A cropped plant at the top-left edge."},
            {"type": "obj", "bbox": [735, 805, 1000, 1000], "desc": "A cropped chair at the bottom-right edge."},
        ],
    },
]


REPAIR_CASES = [
    {
        "name": "qwen_xyxy_multi_object",
        "text": {
            "aspect_ratio": "1:1",
            "high_level_description": "A man stands in front of a gold Mercedes-AMG S63.",
            "compositional_deconstruction": {
                "background": "Luxury room interior.",
                "elements": [
                    {
                        "type": "obj",
                        "bbox": [330, 56, 640, 994],
                        "desc": "A man with dark curly hair and glasses, wearing a dark suit, white shirt, and black dress shoes, standing with hands adjusting his jacket.",
                    },
                    {
                        "type": "obj",
                        "bbox": [168, 412, 818, 856],
                        "desc": "A gold Mercedes-AMG S63 with a black grille and illuminated headlights.",
                    },
                    {
                        "type": "obj",
                        "bbox": [108, 352, 218, 670],
                        "desc": "A tall floor lamp with a white lampshade and ornate golden base.",
                    },
                    {
                        "type": "obj",
                        "bbox": [0, 278, 134, 722],
                        "desc": "A tall potted plant with green leaves in a black pot.",
                    },
                ],
            },
        },
        "expected_boxes": [[56, 330, 994, 640], [412, 168, 856, 818], [352, 108, 670, 218], [278, 0, 722, 134]],
    },
    {
        "name": "official_yxyx_preserved",
        "text": {
            "aspect_ratio": "16:9",
            "high_level_description": "A woman stands beside a long bus.",
            "compositional_deconstruction": {
                "background": "Street scene.",
                "elements": [
                    {"type": "obj", "bbox": [90, 180, 980, 380], "desc": "A standing woman in a blue coat."},
                    {"type": "obj", "bbox": [480, 420, 850, 960], "desc": "A long city bus parked on the road."},
                ],
            },
        },
        "expected_boxes": [[90, 180, 980, 380], [480, 420, 850, 960]],
    },
    {
        "name": "single_strong_xyxy_person",
        "text": {
            "aspect_ratio": "1:1",
            "high_level_description": "A standing man.",
            "compositional_deconstruction": {
                "background": "Plain studio.",
                "elements": [
                    {"type": "obj", "bbox": [350, 55, 640, 980], "desc": "A standing man wearing a black suit."},
                ],
            },
        },
        "expected_boxes": [[55, 350, 980, 640]],
    },
]


def _caption(elements: list[dict[str, Any]], aspect_ratio: str = "1:1") -> dict[str, Any]:
    return {
        "aspect_ratio": aspect_ratio,
        "high_level_description": "A deterministic bbox verification image.",
        "compositional_deconstruction": {
            "background": "Plain white background.",
            "elements": elements,
        },
    }


def _inside_sample(width: int, height: int, bbox: list[int]) -> tuple[int, int]:
    y_min, x_min, y_max, x_max = bbox
    x = int(round(((x_min + x_max) / 2000) * width))
    y = int(round(((y_min + y_max) / 2000) * height))
    return min(width - 1, max(0, x)), min(height - 1, max(0, y))


def _write_base_image(path: Path, size: tuple[int, int], elements: list[dict[str, Any]]) -> None:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for element in elements:
        y_min, x_min, y_max, x_max = element["bbox"]
        left = int(round((x_min / 1000) * width))
        top = int(round((y_min / 1000) * height))
        right = int(round((x_max / 1000) * width))
        bottom = int(round((y_max / 1000) * height))
        draw.rectangle((left, top, right, bottom), fill=(245, 245, 245), outline=(220, 220, 220), width=2)
    image.save(path)


def run_geometry_cases() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in GEOMETRY_CASES:
        image_path = OUT_DIR / f"{case['name']}.png"
        boxed_path = OUT_DIR / f"{case['name']}_boxed.png"
        _write_base_image(image_path, case["size"], case["elements"])
        caption = _caption(case["elements"])
        rows = json_to_element_rows(caption, bbox_order="yxyx")
        saved = save_boxed_image(image_path, rows, boxed_path, bbox_order="yxyx")
        if saved is None:
            raise AssertionError(f"{case['name']} did not render a boxed image")
        with Image.open(saved) as boxed:
            pixels = boxed.convert("RGB").load()
            width, height = boxed.size
            for element in case["elements"]:
                sample_x, sample_y = _inside_sample(width, height, element["bbox"])
                if pixels[sample_x, sample_y] == (245, 245, 245):
                    raise AssertionError(f"{case['name']} box fill missing at {sample_x},{sample_y}")
        results.append(
            {
                "name": case["name"],
                "size": case["size"],
                "box_count": len(case["elements"]),
                "image": str(image_path),
                "boxed": str(boxed_path),
                "status": "passed",
            }
        )
    return results


def run_repair_cases() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in REPAIR_CASES:
        text = json.dumps(case["text"], ensure_ascii=False)
        _final, parsed, warnings = normalize_json_output(text, preset_id="i4_official_v1_app_compare", compact=True)
        if parsed is None:
            raise AssertionError(f"{case['name']} did not parse: {warnings}")
        boxes = [element.get("bbox") for element in parsed["compositional_deconstruction"]["elements"]]
        if boxes != case["expected_boxes"]:
            raise AssertionError(f"{case['name']} boxes {boxes} != expected {case['expected_boxes']}")
        results.append({"name": case["name"], "boxes": boxes, "warnings": warnings, "status": "passed"})
    return results


def main() -> None:
    report = {
        "geometry_cases": run_geometry_cases(),
        "repair_cases": run_repair_cases(),
    }
    report["summary"] = {
        "geometry_cases": len(report["geometry_cases"]),
        "geometry_boxes": sum(item["box_count"] for item in report["geometry_cases"]),
        "repair_cases": len(report["repair_cases"]),
        "status": "passed",
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
