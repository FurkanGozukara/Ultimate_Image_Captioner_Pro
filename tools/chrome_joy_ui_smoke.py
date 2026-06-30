from __future__ import annotations

import json
import re
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "chrome_manual_full"
APP_OUTPUTS_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "joy_playwright_results.json"

DEFAULT_URL = "http://127.0.0.1:7875/"
CHROME_EXE = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")

IMAGES = [
    ("poster", Path("C:/Users/Furkan/Downloads/24c11c5c-97cf-44d5-831f-182b5b3e0c5d.jpeg")),
    ("owl_text", Path("C:/Users/Furkan/Downloads/ChatGPT Image Jun 29, 2026, 04_19_47 PM (4).png")),
    ("person_car", Path("C:/Users/Furkan/Pictures/Untitled - Copy (2).jfif")),
]

VARIANTS = [
    ("pre_alpha", "Joy Caption Pre Alpha", "button.btn-pre-caption"),
    ("alpha_one", "Joy Caption Alpha 1", "button.btn-alpha1-caption"),
    ("alpha_two", "Joy Caption Alpha 2", "button.btn-alpha2-caption"),
    ("beta_one", "Joy Caption Beta 1", "button.btn-beta-caption"),
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_results(results: list[dict[str, Any]], url: str = DEFAULT_URL) -> None:
    RESULTS_PATH.write_text(
        json.dumps({"updated_at": now(), "url": url, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def status_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, label in (
        ("caption_path", "Caption saved to"),
        ("image_path", "Image output"),
        ("metadata_path", "Metadata saved to"),
    ):
        match = re.search(re.escape(label) + r":\s*([^\n]+)", text)
        if match:
            fields[key] = match.group(1).strip()
    return fields


def output_dirs() -> set[str]:
    return {path.name for path in APP_OUTPUTS_DIR.iterdir() if path.is_dir() and path.name.isdigit()}


def metadata_matches_variant(metadata: dict[str, Any], variant: str) -> bool:
    if variant == "beta_one":
        return metadata.get("engine") == "beta_one"
    return metadata.get("engine") == "legacy_siglip" and metadata.get("variant") == variant


def wait_for_new_output(before_dirs: set[str], variant: str, timeout_seconds: int = 1200) -> dict[str, Any]:
    started = time.time()
    while time.time() - started < timeout_seconds:
        candidates = [
            path
            for path in APP_OUTPUTS_DIR.iterdir()
            if path.is_dir() and path.name.isdigit() and path.name not in before_dirs
        ]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for run_dir in candidates:
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not metadata_matches_variant(metadata, variant):
                continue
            caption_path = None
            for ext in ("*.txt", "*.json", "*.caption"):
                matches = [path for path in run_dir.glob(ext) if path.name != "metadata.json"]
                if matches:
                    caption_path = matches[0]
                    break
            return {
                "run_dir": str(run_dir),
                "caption_path": str(caption_path) if caption_path else "",
                "image_path": str(metadata.get("output_image_path") or ""),
                "metadata_path": str(metadata_path),
                "caption": str(metadata.get("caption_final") or ""),
                "metadata": metadata,
            }
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for new {variant} output.")


def textareas(page: Page) -> list[dict[str, Any]]:
    return page.evaluate(
        """() => Array.from(document.querySelectorAll('textarea')).map((el, idx) => ({
            idx,
            value: el.value,
            disabled: el.disabled,
            aria: el.getAttribute('aria-label') || '',
            placeholder: el.getAttribute('placeholder') || ''
        }))"""
    )


def extract_caption(values: list[dict[str, Any]]) -> str:
    for item in values:
        value = str(item.get("value") or "").strip()
        if not item.get("disabled") or not value:
            continue
        if "Caption saved to:" in value or "Metadata saved to:" in value or "Token speed:" in value:
            continue
        if "Batch processing" in value:
            continue
        return value
    for item in values:
        value = str(item.get("value") or "").strip()
        if value and not value.startswith("Write a detailed description") and "Caption saved to:" not in value:
            return value
    return ""


def click_tab(page: Page, name: str) -> None:
    tab = page.get_by_role("tab", name=name)
    if tab.count() != 1:
        raise RuntimeError(f"Expected one tab named {name!r}, found {tab.count()}.")
    tab.click()
    page.wait_for_timeout(800)


def upload_image(page: Page, image_path: Path) -> None:
    drop = page.get_by_role("button", name="Drop an image file here to upload")
    if drop.count() != 1:
        raise RuntimeError(f"Expected one visible image drop button, found {drop.count()}.")
    with page.expect_file_chooser(timeout=10_000) as chooser_info:
        drop.click()
    chooser_info.value.set_files(str(image_path))
    page.wait_for_timeout(1200)


def run_case(
    browser: Any,
    url: str,
    variant: str,
    tab_name: str,
    button_selector: str,
    image_label: str,
    image_path: Path,
) -> dict[str, Any]:
    page = browser.new_page(viewport={"width": 1920, "height": 1200})
    page.goto(url, wait_until="load", timeout=60_000)
    page.wait_for_timeout(1500)
    click_tab(page, tab_name)
    upload_image(page, image_path)
    before_dirs = output_dirs()
    button = page.locator(button_selector)
    if button.count() != 1:
        raise RuntimeError(f"Expected one caption button for {variant}, found {button.count()}.")
    started = time.time()
    button.click()
    output = wait_for_new_output(before_dirs, variant)
    page.wait_for_timeout(1000)
    values = textareas(page)
    page.screenshot(path=str(OUT_DIR / f"joy_{variant}_{image_label}.png"), full_page=True)
    page.close()
    ui_caption = extract_caption(values)
    caption = output["caption"] or ui_caption
    ok = bool(caption)
    return {
        "engine": "joy",
        "variant": variant,
        "image_label": image_label,
        "image": str(image_path),
        "status": "pass" if ok else "fail",
        "detail": "non-empty caption" if ok else "empty caption after saved output",
        "caption_chars": len(caption),
        "caption_path": output["caption_path"],
        "image_path": output["image_path"],
        "metadata_path": output["metadata_path"],
        "run_dir": output["run_dir"],
        "elapsed_seconds": round(time.time() - started, 2),
        "timestamp": now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    write_results(results, args.url)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=str(CHROME_EXE),
            headless=False,
            args=["--start-maximized", "--disable-features=CalculateNativeWinOcclusion"],
        )
        for variant, tab_name, button_selector in VARIANTS:
            for image_label, image_path in IMAGES:
                try:
                    result = run_case(browser, args.url, variant, tab_name, button_selector, image_label, image_path)
                except Exception as exc:
                    result = {
                        "engine": "joy",
                        "variant": variant,
                        "image_label": image_label,
                        "image": str(image_path),
                        "status": "fail",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "timestamp": now(),
                    }
                results.append(result)
                write_results(results, args.url)
        browser.close()
    return 1 if any(item.get("status") == "fail" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
