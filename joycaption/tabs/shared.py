from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gradio as gr

from ..common import format_exception, open_folder, ordered_values, values_from_components


@dataclass
class TabUI:
    key: str
    order: list[str]
    defaults: dict[str, Any]
    inputs: list[gr.components.Component]


def settings_from_values(order: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    return values_from_components(order, values)


def values_for_settings(order: Sequence[str], defaults: dict[str, Any], data: dict[str, Any] | None) -> list[Any]:
    return ordered_values(order, data or {}, defaults)


def run_open_folder(output_folder: str, input_folder: str = "") -> str:
    return open_folder(output_folder or input_folder or None)


def run_open_outputs() -> str:
    return open_folder(None)


def error_pair(exc: BaseException) -> tuple[str, str]:
    return "Failed", f"Error: {format_exception(exc)}"


def error_triple(exc: BaseException) -> tuple[str, str, str]:
    return "Error", "Failed", f"Error: {format_exception(exc)}"
