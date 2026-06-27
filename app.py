from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import gradio as gr
from starlette.exceptions import StarletteDeprecationWarning

from joycaption import APP_NAME
from joycaption.common import BASE_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR, ensure_runtime_dirs, html_message
from joycaption.lazy_engines import LazyBetaEngine, LazyLegacyEngine
from joycaption.presets import UniversalPresetStore
from joycaption.styles import CUSTOM_CSS
from joycaption.tabs import alpha_one, alpha_two, beta_one, pre_alpha
from joycaption.tabs.shared import TabUI, values_for_settings


warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)

GLOBAL_ORDER = ["theme_mode"]
GLOBAL_DEFAULTS = {"theme_mode": "dark"}


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


def build_app() -> gr.Blocks:
    ensure_runtime_dirs()
    preset_store = UniversalPresetStore()

    pre_engine = LazyLegacyEngine("pre_alpha", BASE_DIR)
    alpha1_engine = LazyLegacyEngine("alpha_one", BASE_DIR)
    alpha2_engine = LazyLegacyEngine("alpha_two", BASE_DIR)
    beta_engine = LazyBetaEngine(BASE_DIR / "model_files_beta_one")

    with gr.Blocks(title=APP_NAME) as demo:
        with gr.Column(elem_id="jc-shell"):
            with gr.Row(elem_classes=["jc-topbar"]):
                with gr.Column(elem_classes=["jc-brand"]):
                    gr.HTML(
                        """
                        <h1>Ultimate Image Captioner Pro</h1>
                        <p>Unified Pre-Alpha, Alpha, and Beta captioning workspace.</p>
                        """
                    )
                with gr.Column(elem_classes=["jc-preset-panel"]):
                    gr.Markdown("**Universal Preset**")
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            choices=preset_store.list_presets(),
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
                    preset_status = gr.HTML("")

            tabs: list[TabUI] = [
                TabUI("global", GLOBAL_ORDER, GLOBAL_DEFAULTS, [theme_mode]),
            ]

            with gr.Tabs(elem_id="jc-main-tabs", elem_classes=["jc-main-tabs"]):
                with gr.Tab("Joy Caption Pre Alpha", render_children=True):
                    tabs.append(pre_alpha.build_tab(pre_engine))
                with gr.Tab("Joy Caption Alpha 1", render_children=True):
                    tabs.append(alpha_one.build_tab(alpha1_engine))
                with gr.Tab("Joy Caption Alpha 2", render_children=True):
                    tabs.append(alpha_two.build_tab(alpha2_engine))
                with gr.Tab("Joy Caption Beta 1", render_children=True):
                    tabs.append(beta_one.build_tab(beta_engine))

            flat_inputs: list[gr.components.Component] = []
            for tab in tabs:
                flat_inputs.extend(tab.inputs)

            def save_preset(name: str, selected_name: str | None, *values):
                target = (name or "").strip() or (selected_name or "").strip()
                if not target:
                    return gr.update(), html_message("error", "Enter a preset name or select an existing preset."), ""
                payload = _split_values(tabs, list(values))
                saved_name = preset_store.save(target, payload)
                choices = preset_store.list_presets()
                return gr.update(choices=choices, value=saved_name), html_message("success", f"Saved universal preset '{saved_name}'."), ""

            def load_preset(selected_name: str | None):
                if not selected_name:
                    return [* _flat_values_from_payload(tabs, _default_payload(tabs)), html_message("info", "No preset selected."), gr.update(choices=preset_store.list_presets(), value=None)]
                payload = preset_store.load(selected_name)
                if not payload:
                    return [* _flat_values_from_payload(tabs, _default_payload(tabs)), html_message("error", f"Preset '{selected_name}' was not found."), gr.update(choices=preset_store.list_presets(), value=None)]
                return [
                    *_flat_values_from_payload(tabs, payload),
                    html_message("success", f"Loaded universal preset '{selected_name}'."),
                    gr.update(choices=preset_store.list_presets(), value=selected_name),
                ]

            def reset_defaults():
                preset_store.clear_last_used()
                return [
                    *_flat_values_from_payload(tabs, _default_payload(tabs)),
                    html_message("success", "Reset all tabs to defaults."),
                    gr.update(choices=preset_store.list_presets(), value=None),
                ]

            def delete_preset(selected_name: str | None):
                if not selected_name:
                    return gr.update(), html_message("error", "No preset selected.")
                if preset_store.delete(selected_name):
                    return gr.update(choices=preset_store.list_presets(), value=None), html_message("success", f"Deleted preset '{selected_name}'.")
                return gr.update(), html_message("error", f"Could not delete preset '{selected_name}'.")

            def load_startup_preset():
                return gr.update(choices=preset_store.list_presets(), value=None), html_message("info", "No preset selected.")

            save_preset_btn.click(
                save_preset,
                inputs=[preset_name, preset_dropdown] + flat_inputs,
                outputs=[preset_dropdown, preset_status, preset_name],
                queue=False,
            )
            load_preset_btn.click(
                load_preset,
                inputs=[preset_dropdown],
                outputs=flat_inputs + [preset_status, preset_dropdown],
                queue=False,
            )
            preset_dropdown.change(
                load_preset,
                inputs=[preset_dropdown],
                outputs=flat_inputs + [preset_status, preset_dropdown],
                queue=False,
            )
            reset_preset_btn.click(
                reset_defaults,
                outputs=flat_inputs + [preset_status, preset_dropdown],
                queue=False,
            )
            delete_preset_btn.click(
                delete_preset,
                inputs=[preset_dropdown],
                outputs=[preset_dropdown, preset_status],
                queue=False,
            )
            theme_mode.change(
                fn=lambda value: value,
                inputs=[theme_mode],
                outputs=[theme_mode],
                queue=False,
                js="""
                (value) => {
                  const url = new URL(window.location.href);
                  url.searchParams.set("__theme", value || "dark");
                  window.location.href = url.toString();
                  return value;
                }
                """,
            )
            demo.load(load_startup_preset, outputs=[preset_dropdown, preset_status], queue=False)

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
    allowed_paths = [str(BASE_DIR), str(OUTPUTS_DIR)]
    if TEST_IMAGES_DIR.exists():
        allowed_paths.append(str(TEST_IMAGES_DIR))
    demo = build_app()
    launch_kwargs: dict[str, Any] = {
        "share": args.share,
        "inbrowser": args.inbrowser,
        "allowed_paths": allowed_paths,
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
