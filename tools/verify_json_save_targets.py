from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joycaption.json_tools import _file_url, apply_rows_to_json
from joycaption.tabs import json_builder, qwen


def _write_image(path: Path, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 90), color).save(path)


def _json_for(rows: list[list[object]], description: str = "A test image.") -> str:
    elements: list[dict[str, object]] = []
    for row in rows:
        values = list(row) + [""] * max(0, 8 - len(row))
        item: dict[str, object] = {
            "type": values[0],
            "bbox": [int(values[1]), int(values[2]), int(values[3]), int(values[4])],
            "desc": values[5],
        }
        if values[0] == "text":
            item["text"] = values[7]
        elements.append(item)
    return json.dumps(
        {
            "aspect_ratio": "4:3",
            "high_level_description": description,
            "compositional_deconstruction": {"background": "plain", "elements": elements},
        },
        ensure_ascii=False,
        indent=2,
    )


def _patch_output_roots(root: Path):
    original_json_builder_outputs = json_builder.OUTPUTS_DIR
    original_qwen_outputs = qwen.OUTPUTS_DIR
    json_builder.OUTPUTS_DIR = root
    qwen.OUTPUTS_DIR = root
    return original_json_builder_outputs, original_qwen_outputs


def _restore_output_roots(values) -> None:
    json_builder.OUTPUTS_DIR, qwen.OUTPUTS_DIR = values


def verify_qwen_duplicate_name_autosave(root: Path) -> None:
    previous_rows = [["obj", 10, 20, 100, 200, "old desc", "subject", ""]]
    table_rows = [["obj", 11, 22, 111, 222, "table desc", "subject", ""]]
    preview_rows = [["obj", 300, 400, 700, 800, "ignored preview desc", "subject", ""]]

    selected = qwen._box_choices(previous_rows)
    merged = qwen._merge_visible_rows(previous_rows, selected, table_rows)
    merged = qwen._merge_preview_bbox_rows(merged, qwen._preview_rows_snapshot_value(preview_rows, [0]))
    assert merged[0][1:5] == [300, 400, 700, 800]
    assert merged[0][5] == "table desc"

    saved_json, _parsed, warnings = apply_rows_to_json(_json_for(previous_rows), merged, bbox_order="yxyx")
    assert not warnings

    run_a = root / "qwen_a"
    run_b = root / "qwen_b"
    run_a.mkdir()
    run_b.mkdir()
    image_a = run_a / "same.png"
    image_b = run_b / "same.png"
    _write_image(image_a, "white")
    _write_image(image_b, "gray")
    json_a = run_a / "same.json"
    json_b = run_b / "same.json"
    json_a.write_text(_json_for(previous_rows, "old a"), encoding="utf-8")
    json_b.write_text(_json_for(previous_rows, "old b"), encoding="utf-8")
    metadata_b = run_b / "metadata.json"
    metadata_b.write_text(json.dumps({"caption_final": "old b"}, indent=2), encoding="utf-8")

    target = {
        "caption_path": str(json_b),
        "output_run_dir": str(run_b),
        "metadata_path": str(metadata_b),
        "output_image_path": str(image_b),
        "boxed_image_path": str(run_b / "same_boxed.png"),
    }
    status = qwen._autosave_json_caption(saved_json, target, image=image_a, rows=merged, bbox_order="yxyx")
    assert "autosaved" in status
    assert json.loads(json_a.read_text(encoding="utf-8"))["high_level_description"] == "old a"
    saved_b = json.loads(json_b.read_text(encoding="utf-8"))
    element_b = saved_b["compositional_deconstruction"]["elements"][0]
    assert element_b["bbox"] == [300, 400, 700, 800]
    assert element_b["desc"] == "table desc"
    assert not (run_a / "same_boxed.png").exists()
    assert (run_b / "same_boxed.png").exists()
    metadata = json.loads(metadata_b.read_text(encoding="utf-8"))
    assert metadata["caption_path"] == str(json_b)
    assert metadata["output_run_dir"] == str(run_b)


def verify_builder_save_targets(root: Path) -> None:
    rows = [["obj", 80, 90, 500, 600, "box desc", "box 1", ""]]
    first_json = _json_for(rows, "first builder save")
    second_json = _json_for(rows, "second builder save")

    status, saved_path_text = json_builder._save_json_builder_edits(None, "", first_json, rows, "yxyx")
    saved_path = Path(saved_path_text)
    assert "saved" in status
    assert saved_path == root / "0001" / "image.json"
    assert json.loads(saved_path.read_text(encoding="utf-8"))["high_level_description"] == "first builder save"
    assert (saved_path.parent / "metadata.json").exists()

    _status, second_path_text = json_builder._save_json_builder_edits(None, saved_path_text, second_json, rows, "yxyx")
    assert Path(second_path_text) == saved_path
    numeric_dirs = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    assert numeric_dirs == ["0001"]
    assert json.loads(saved_path.read_text(encoding="utf-8"))["high_level_description"] == "second builder save"

    upload = root.parent / "upload_same.png"
    _write_image(upload, "blue")
    assert json_builder._output_dropdown_update(upload)["value"] == json_builder.OUTPUT_NOT_SELECTED_VALUE
    status, image_saved_json_text = json_builder._save_json_builder_edits(upload, "", first_json, rows, "yxyx")
    image_saved_json = Path(image_saved_json_text)
    assert "saved" in status
    assert image_saved_json == root / "0002" / "upload_same.json"
    assert (root / "0002" / "upload_same.png").exists()
    assert (root / "0002" / "upload_same_boxed.png").exists()

    run_a = root / "loaded_a"
    run_b = root / "loaded_b"
    run_a.mkdir()
    run_b.mkdir()
    builder_image_name = "builder_same.png"
    builder_json_name = "builder_same.json"
    builder_boxed_name = "builder_same_boxed.png"
    _write_image(run_a / builder_image_name, "white")
    _write_image(run_b / builder_image_name, "gray")
    (run_a / builder_json_name).write_text(_json_for(rows, "old a"), encoding="utf-8")
    (run_b / builder_json_name).write_text(_json_for(rows, "old b"), encoding="utf-8")
    _status, loaded_path_text = json_builder._save_json_builder_edits(
        run_a / builder_image_name,
        run_b / builder_json_name,
        second_json,
        rows,
        "yxyx",
    )
    assert Path(loaded_path_text) == run_b / builder_json_name
    assert json.loads((run_a / builder_json_name).read_text(encoding="utf-8"))["high_level_description"] == "old a"
    assert json.loads((run_b / builder_json_name).read_text(encoding="utf-8"))["high_level_description"] == "second builder save"
    gradio_echo = Path(tempfile.gettempdir()) / "gradio" / "unit_echo" / builder_image_name
    gradio_echo.parent.mkdir(parents=True, exist_ok=True)
    _write_image(gradio_echo, "gray")
    gradio_echo.with_suffix(".json").write_text((run_b / builder_json_name).read_text(encoding="utf-8"), encoding="utf-8")
    Image.new("RGB", (5, 5), "red").save(gradio_echo.with_name(builder_boxed_name))
    assert json_builder._is_loaded_output_image_echo(gradio_echo, run_b / builder_json_name)
    assert not json_builder._is_loaded_output_image_echo(run_b / builder_image_name, run_b / builder_json_name)
    assert (
        json_builder._gradio_temp_output_echo_path(gradio_echo, run_a / builder_json_name, run_a / builder_image_name)
        == run_b / builder_image_name
    )
    assert json_builder._gradio_temp_output_echo_path(gradio_echo, "", run_b / builder_image_name) == run_b / builder_image_name
    assert json_builder._gradio_temp_output_echo_path(gradio_echo, "", json_builder.OUTPUT_NOT_SELECTED_VALUE) == run_b / builder_image_name
    assert json_builder._sidecar_json_for_image(gradio_echo) is None
    assert json_builder._json_save_path_for_image(gradio_echo) is None
    assert json_builder._output_dropdown_update(gradio_echo)["value"] == json_builder.OUTPUT_NOT_SELECTED_VALUE
    gradio_echo.with_suffix(".json").write_text(_json_for(rows, "temp must not load"), encoding="utf-8")
    assert json_builder._gradio_temp_output_echo_path(gradio_echo, "", json_builder.OUTPUT_NOT_SELECTED_VALUE) == run_b / builder_image_name
    run_c = root / "loaded_c"
    run_c.mkdir()
    _write_image(run_c / builder_image_name, "gray")
    (run_c / builder_json_name).write_text(_json_for(rows, "old c"), encoding="utf-8")
    assert json_builder._gradio_temp_output_echo_path(gradio_echo, "", json_builder.OUTPUT_NOT_SELECTED_VALUE) is None
    assert "Folder: loaded_b" in json_builder._loaded_output_label(run_b / builder_image_name, run_b / builder_json_name)
    Image.new("RGB", (5, 5), "red").save(run_b / builder_boxed_name)
    preview_html, preview_note = json_builder._render_boxed_preview_from_json(run_b / builder_json_name, rows, "yxyx")
    assert builder_boxed_name in preview_html
    assert f"loaded_b/{builder_json_name}" in preview_html
    assert "matching JSON" in preview_note
    assert Image.open(run_b / builder_boxed_name).size == (120, 90)

    _write_image(root / "filter" / "visible.png", "white")
    _write_image(root / "filter" / "visible_boxed.png", "white")
    choices = json_builder._outputs_image_choices()
    labels = [label for label, _value in choices]
    assert labels[0] == json_builder.OUTPUT_NOT_SELECTED_LABEL
    assert "filter/visible.png" in labels
    assert all("_boxed" not in label for label in labels)
    assert json_builder._resolve_output_image_selection("filter/visible.png") == root / "filter" / "visible.png"
    assert json_builder._resolve_output_image_selection(json_builder.OUTPUT_NOT_SELECTED_VALUE) is None

    boxed_url = _file_url(root / "0002" / "upload_same_boxed.png")
    time.sleep(0.02)
    _write_image(root / "0002" / "upload_same_boxed.png", "red")
    assert _file_url(root / "0002" / "upload_same_boxed.png") != boxed_url


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "outputs"
        root.mkdir()
        originals = _patch_output_roots(root)
        try:
            verify_qwen_duplicate_name_autosave(root)
            verify_builder_save_targets(root)
        finally:
            _restore_output_roots(originals)
    print("JSON save target verification passed.")


if __name__ == "__main__":
    main()
