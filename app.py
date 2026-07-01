from __future__ import annotations

import argparse
import os
import string
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import gradio as gr
from starlette.exceptions import StarletteDeprecationWarning

from joycaption import APP_NAME
from joycaption.common import BASE_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR, ensure_runtime_dirs, html_message
from joycaption.lazy_engines import LazyBetaEngine, LazyLegacyEngine, LazyQwenEngine
from joycaption.presets import UniversalPresetStore
from joycaption.styles import CUSTOM_CSS
from joycaption.tabs import alpha_one, alpha_two, beta_one, json_builder, output_browser, pre_alpha, qwen
from joycaption.tabs.shared import TabUI, values_for_settings
from joycaption.vram import gpu_summary_html


warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)

GLOBAL_ORDER = ["theme_mode"]
GLOBAL_DEFAULTS = {"theme_mode": "dark"}
FAVICON_PATH = BASE_DIR / "assets" / "favicon.svg"


def _normalize_allowed_path(path: str | Path) -> str | None:
    try:
        candidate = Path(path).expanduser()
        if not candidate.exists():
            return None
        return str(candidate.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None


def _decode_mount_path(path_text: str) -> str:
    return (
        path_text.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _windows_drive_roots() -> list[str]:
    roots: list[str] = []
    try:
        import ctypes

        mask = int(ctypes.windll.kernel32.GetLogicalDrives())
        for index, letter in enumerate(string.ascii_uppercase):
            if mask & (1 << index):
                roots.append(f"{letter}:\\")
    except Exception:
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if os.path.exists(root):
                roots.append(root)
    return roots


def _posix_mount_roots() -> list[str]:
    roots = {"/"}
    mountinfo = Path("/proc/self/mountinfo")
    mounts = Path("/proc/mounts")
    try:
        if mountinfo.exists():
            for line in mountinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                fields = line.split()
                if len(fields) >= 5:
                    roots.add(_decode_mount_path(fields[4]))
        elif mounts.exists():
            for line in mounts.read_text(encoding="utf-8", errors="ignore").splitlines():
                fields = line.split()
                if len(fields) >= 2:
                    roots.add(_decode_mount_path(fields[1]))
    except OSError:
        pass

    volumes_dir = Path("/Volumes")
    if volumes_dir.is_dir():
        try:
            roots.update(str(path) for path in volumes_dir.iterdir() if path.is_dir())
        except OSError:
            pass
    return sorted(roots)


def discover_gradio_allowed_paths() -> list[str]:
    candidates: list[str | Path] = [BASE_DIR, OUTPUTS_DIR]
    if TEST_IMAGES_DIR.exists():
        candidates.append(TEST_IMAGES_DIR)
    if os.name == "nt":
        candidates.extend(_windows_drive_roots())
    else:
        candidates.extend(_posix_mount_roots())

    allowed_paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_allowed_path(candidate)
        if normalized is None:
            continue
        key = os.path.normcase(os.path.normpath(normalized))
        if key not in seen:
            seen.add(key)
            allowed_paths.append(normalized)
    return allowed_paths


def build_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )


def _split_values(sections: list[TabUI], flat_values: list[Any]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    idx = 0
    for section in sections:
        count = len(section.order)
        values = flat_values[idx : idx + count]
        payload[section.key] = {key: value for key, value in zip(section.order, values)}
        idx += count
    return payload


def _flat_values_from_payload(sections: list[TabUI], payload: dict[str, Any] | None) -> list[Any]:
    values: list[Any] = []
    payload = payload or {}
    for section in sections:
        section_data = payload.get(section.key, {})
        if not isinstance(section_data, dict):
            section_data = {}
        values.extend(values_for_settings(section.order, section.defaults, section_data))
    return values


def _default_payload(sections: list[TabUI]) -> dict[str, dict[str, Any]]:
    return {section.key: deepcopy(section.defaults) for section in sections}


def _numbered_preset_choices(preset_names: list[str]) -> list[tuple[str, str]]:
    return [(f"{index}. {preset_name}", preset_name) for index, preset_name in enumerate(preset_names, start=1)]


def build_app() -> gr.Blocks:
    ensure_runtime_dirs()
    preset_store = UniversalPresetStore()

    pre_engine = LazyLegacyEngine("pre_alpha", BASE_DIR)
    alpha1_engine = LazyLegacyEngine("alpha_one", BASE_DIR)
    alpha2_engine = LazyLegacyEngine("alpha_two", BASE_DIR)
    beta_engine = LazyBetaEngine(BASE_DIR / "model_files_beta_one")
    qwen_engine = LazyQwenEngine(BASE_DIR / "model_files_qwen3_vl3_8b_instruct")

    with gr.Blocks(title=APP_NAME) as demo:
        with gr.Column(elem_id="jc-shell"):
            with gr.Row(elem_classes=["jc-topbar"]):
                with gr.Column(scale=1, elem_classes=["jc-brand"]):
                    gr.HTML(
                        """
                        <h1>Ultimate Image Captioner Pro V1.0 : <a href="https://www.patreon.com/SECourses/posts/162527725">https://www.patreon.com/SECourses/posts/162527725</a></h1>
                        <p>Unified Qwen3 VL, Ideogram JSON, Pre-Alpha, Alpha, and Beta captioning workspace.</p>
                        """
                    )
                    with gr.Column(elem_classes=["jc-header-status"]):
                        preset_status = gr.HTML("")
                        gr.HTML(gpu_summary_html(), elem_classes=["jc-gpu-summary"])
                with gr.Column(scale=1, elem_classes=["jc-preset-panel"]):
                    gr.Markdown("**Universal Preset**")
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            choices=_numbered_preset_choices(preset_store.list_presets()),
                            label="Preset",
                            allow_custom_value=False,
                            scale=2,
                        )
                        preset_name = gr.Textbox(label="Save As", placeholder="my_caption_settings", scale=2)
                        theme_mode = gr.Radio(choices=["dark", "light"], value=GLOBAL_DEFAULTS["theme_mode"], label="Theme", scale=1)
                    with gr.Row():
                        save_preset_btn = gr.Button("Save", elem_classes=["btn-save-preset"])
                        load_preset_btn = gr.Button("Load", elem_classes=["btn-load-preset"])
                        reset_preset_btn = gr.Button("Reset", elem_classes=["btn-reset-preset"])
                        delete_preset_btn = gr.Button("Delete", elem_classes=["btn-delete-preset"])

            tabs: list[TabUI] = [
                TabUI("global", GLOBAL_ORDER, GLOBAL_DEFAULTS, [theme_mode]),
            ]

            with gr.Tabs(elem_id="jc-main-tabs", elem_classes=["jc-main-tabs"]):
                with gr.Tab("Qwen3 VL 8B Instruct", render_children=True):
                    tabs.append(qwen.build_tab(qwen_engine))
                with gr.Tab("Joy Caption Beta 1", render_children=True):
                    tabs.append(beta_one.build_tab(beta_engine))
                with gr.Tab("Joy Caption Alpha 2", render_children=True):
                    tabs.append(alpha_two.build_tab(alpha2_engine))
                with gr.Tab("Joy Caption Alpha 1", render_children=True):
                    tabs.append(alpha_one.build_tab(alpha1_engine))
                with gr.Tab("Joy Caption Pre Alpha", render_children=True):
                    tabs.append(pre_alpha.build_tab(pre_engine))
                with gr.Tab("JSON Prompt Builder", render_children=True):
                    tabs.append(json_builder.build_tab())
                with gr.Tab("Saved Outputs", render_children=True):
                    tabs.append(output_browser.build_tab())

            flat_inputs: list[gr.components.Component] = []
            for tab in tabs:
                flat_inputs.extend(tab.inputs)

            def save_preset(name: str, selected_name: str | None, *values):
                target = (name or "").strip() or (selected_name or "").strip()
                if not target:
                    return gr.update(), html_message("error", "Enter a preset name or select an existing preset."), ""
                payload = _split_values(tabs, list(values))
                saved_name = preset_store.save(target, payload)
                choices = _numbered_preset_choices(preset_store.list_presets())
                return gr.update(choices=choices, value=saved_name), html_message("success", f"Saved universal preset '{saved_name}'."), ""

            def load_preset(selected_name: str | None):
                if not selected_name:
                    return [* _flat_values_from_payload(tabs, _default_payload(tabs)), html_message("info", "No preset selected."), gr.update(choices=_numbered_preset_choices(preset_store.list_presets()), value=None)]
                payload = preset_store.load(selected_name)
                if not payload:
                    return [* _flat_values_from_payload(tabs, _default_payload(tabs)), html_message("error", f"Preset '{selected_name}' was not found."), gr.update(choices=_numbered_preset_choices(preset_store.list_presets()), value=None)]
                return [
                    *_flat_values_from_payload(tabs, payload),
                    html_message("success", f"Loaded universal preset '{selected_name}'."),
                    gr.update(choices=_numbered_preset_choices(preset_store.list_presets()), value=selected_name),
                ]

            def reset_defaults():
                preset_store.clear_last_used()
                return [
                    *_flat_values_from_payload(tabs, _default_payload(tabs)),
                    html_message("success", "Reset all tabs to defaults."),
                    gr.update(choices=_numbered_preset_choices(preset_store.list_presets()), value=None),
                ]

            def delete_preset(selected_name: str | None):
                if not selected_name:
                    return gr.update(), html_message("error", "No preset selected.")
                if preset_store.delete(selected_name):
                    return gr.update(choices=_numbered_preset_choices(preset_store.list_presets()), value=None), html_message("success", f"Deleted preset '{selected_name}'.")
                return gr.update(), html_message("error", f"Could not delete preset '{selected_name}'.")

            def load_startup_preset():
                selected_name, payload = preset_store.load_last_used()
                if selected_name and payload:
                    return [
                        *_flat_values_from_payload(tabs, payload),
                        gr.update(choices=_numbered_preset_choices(preset_store.list_presets()), value=selected_name),
                        html_message("success", f"Loaded last used preset '{selected_name}'."),
                    ]
                return [
                    *_flat_values_from_payload(tabs, _default_payload(tabs)),
                    gr.update(choices=_numbered_preset_choices(preset_store.list_presets()), value=None),
                    html_message("info", "No preset selected."),
                ]

            save_preset_btn.click(
                save_preset,
                inputs=[preset_name, preset_dropdown] + flat_inputs,
                outputs=[preset_dropdown, preset_status, preset_name],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            load_preset_btn.click(
                load_preset,
                inputs=[preset_dropdown],
                outputs=flat_inputs + [preset_status, preset_dropdown],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            preset_dropdown.change(
                load_preset,
                inputs=[preset_dropdown],
                outputs=flat_inputs + [preset_status, preset_dropdown],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            reset_preset_btn.click(
                reset_defaults,
                outputs=flat_inputs + [preset_status, preset_dropdown],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            delete_preset_btn.click(
                delete_preset,
                inputs=[preset_dropdown],
                outputs=[preset_dropdown, preset_status],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )
            theme_mode.change(
                fn=lambda value: value,
                inputs=[theme_mode],
                outputs=[theme_mode],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
                js="""
                (value) => {
                  const url = new URL(window.location.href);
                  url.searchParams.set("__theme", value || "dark");
                  window.location.href = url.toString();
                  return value;
                }
                """,
            )
            demo.load(
                load_startup_preset,
                outputs=flat_inputs + [preset_dropdown, preset_status],
                queue=False,
                show_progress="hidden",
                show_progress_on=[],
            )

    demo.queue(default_concurrency_limit=1)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ultimate Image Captioner Pro")
    parser.add_argument("--gradio", action="store_true", help="Compatibility flag; this entrypoint always launches Gradio")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--server-name", "--server", dest="server_name", default=None, help="Server host")
    parser.add_argument("--server-port", "--port", dest="server_port", type=int, default=None, help="Server port")
    parser.set_defaults(inbrowser=True)
    parser.add_argument("--inbrowser", dest="inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--no-inbrowser", dest="inbrowser", action="store_false", help="Do not open browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_paths = discover_gradio_allowed_paths()
    demo = build_app()
    launch_kwargs: dict[str, Any] = {
        "share": args.share,
        "inbrowser": args.inbrowser,
        "allowed_paths": allowed_paths,
        "favicon_path": FAVICON_PATH if FAVICON_PATH.exists() else None,
        "css": CUSTOM_CSS,
        "quiet": True,
    }
    if args.server_name:
        launch_kwargs["server_name"] = args.server_name
    if args.server_port is not None:
        launch_kwargs["server_port"] = args.server_port
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
