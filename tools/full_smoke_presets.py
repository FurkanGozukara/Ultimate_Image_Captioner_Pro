from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joycaption.common import BASE_DIR
from joycaption.json_tools import json_to_element_rows, normalize_json_output
from joycaption.lazy_engines import LazyBetaEngine, LazyLegacyEngine, LazyQwenEngine
from joycaption.qwen_presets import load_qwen_presets, preset_payload
from joycaption.tabs import alpha_one, alpha_two, beta_one, pre_alpha, qwen


IMAGE_LABELS = ("poster", "owl_text", "person_car")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def strip_html(value: Any) -> str:
    text = re.sub(r"<br\s*/?>", "\n", str(value or ""), flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def parse_status_paths(status_html: str) -> dict[str, str]:
    text = strip_html(status_html)
    out: dict[str, str] = {}
    patterns = {
        "caption_path": r"Caption saved to:\s*(.+)",
        "image_path": r"Image output:\s*(.+)",
        "boxed_image_path": r"Boxed image:\s*(.+)",
        "metadata_path": r"Metadata saved to:\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value and not value.lower().startswith("no ") and value != "Image copy disabled.":
                out[key] = value
    return out


def gpu_memory() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "used_mb": int(parts[2]) if parts[2].isdigit() else parts[2],
                    "total_mb": int(parts[3]) if parts[3].isdigit() else parts[3],
                }
            )
    return rows


class SmokeReport:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = output_dir / "progress.jsonl"
        self.summary_path = output_dir / "summary.json"
        self.html_path = output_dir / "report.html"
        self.events: list[dict[str, Any]] = []

    def event(self, **payload: Any) -> None:
        payload.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        self.events.append(payload)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def write_summary(self, extra: dict[str, Any]) -> None:
        summary = {
            "output_dir": str(self.output_dir),
            "progress_path": str(self.progress_path),
            "html_path": str(self.html_path),
            "events": self.events,
            **extra,
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.write_html(summary)

    def write_html(self, summary: dict[str, Any]) -> None:
        rows = []
        for event in self.events:
            if event.get("type") not in {"case", "skip", "checkpoint"}:
                continue
            status = event.get("status", event.get("type", ""))
            status_class = "pass" if status in {"pass", "ok"} else "fail" if status == "fail" else "skip"
            detail = event.get("detail") or event.get("error") or event.get("note") or ""
            links = []
            for key in ("caption_path", "metadata_path", "boxed_image_path"):
                value = event.get(key)
                if value:
                    links.append(f"{html.escape(key)}: {html.escape(str(value))}")
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(event.get('type', '')))}</td>"
                f"<td>{html.escape(str(event.get('engine', event.get('phase', ''))))}</td>"
                f"<td>{html.escape(str(event.get('preset_id', event.get('variant', ''))))}</td>"
                f"<td>{html.escape(str(event.get('image_label', '')))}</td>"
                f"<td class='{status_class}'>{html.escape(str(status))}</td>"
                f"<td>{html.escape(str(event.get('elapsed_seconds', '')))}</td>"
                f"<td>{html.escape(str(event.get('rows', '')))}</td>"
                f"<td>{html.escape(str(detail))}<br>{'<br>'.join(links)}</td>"
                "</tr>"
            )
        body = "\n".join(rows)
        html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Caption Preset Smoke Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #111; color: #eee; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #333; padding: 8px; vertical-align: top; }}
    th {{ background: #222; text-align: left; position: sticky; top: 0; }}
    .pass {{ color: #37d67a; font-weight: 700; }}
    .fail {{ color: #ff5c5c; font-weight: 700; }}
    .skip {{ color: #f0b429; font-weight: 700; }}
    code {{ color: #9ad; }}
  </style>
</head>
<body>
  <h1>Caption Preset Smoke Report</h1>
  <p><code>{html.escape(str(self.output_dir))}</code></p>
  <p>Total events: {len(self.events)}</p>
  <table>
    <thead>
      <tr><th>Type</th><th>Engine</th><th>Preset / Variant</th><th>Image</th><th>Status</th><th>Seconds</th><th>Rows</th><th>Detail</th></tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
        self.html_path.write_text(html_doc, encoding="utf-8")


def collect_generator(generator: Iterable[Any]) -> Any:
    last = None
    for item in generator:
        last = item
    return last


def validate_qwen_caption(preset_id: str, output_format: str, caption: str, rows: list[list[Any]]) -> tuple[bool, str, int]:
    if not str(caption or "").strip():
        return False, "empty caption", 0
    if output_format != "json":
        return True, "non-empty text", 0
    normalized, parsed, warnings = normalize_json_output(caption, preset_id=preset_id, compact=False)
    del normalized
    if parsed is None:
        return False, "; ".join(warnings) or "json parse failed", 0
    parsed_rows = json_to_element_rows(parsed, bbox_order="yxyx")
    row_count = len(parsed_rows or rows or [])
    if preset_id.startswith(("i4_official_v1", "i4_json_")):
        if warnings:
            return False, "; ".join(warnings), row_count
        if row_count <= 0:
            return False, "official v1 JSON produced no previewable bbox rows", row_count
    return True, "valid JSON", row_count


def qwen_settings_for_preset(preset_id: str, device_id: str, max_tokens_cap: int | None) -> dict[str, Any]:
    payload = preset_payload(preset_id)
    settings = dict(qwen.DEFAULTS)
    settings.update(
        {
            "preset_id": preset_id,
            "output_format": payload["output_format"],
            "extension": payload["extension"],
            "system_prompt": payload["system_prompt"],
            "prompt": payload["prompt"],
            "temperature": payload["temperature"],
            "max_new_tokens": payload["max_new_tokens"],
            "image_long_edge": payload["image_long_edge"],
            "app_side_only": payload["app_side_only"],
        }
    )
    if max_tokens_cap is not None:
        settings["max_new_tokens"] = min(int(settings["max_new_tokens"]), int(max_tokens_cap))
    settings.update(
        {
            "device_id": device_id,
            "model_quantization": "bf16",
            "unload_model": False,
            "use_subprocess": False,
            "compact_json": False,
            "json_retries": 1,
            "auto_save_boxed_image": True,
            "save_image": True,
            "console_progress": False,
        }
    )
    return settings


def run_qwen_matrix(report: SmokeReport, image_paths: list[Path], device_id: str, max_tokens_cap: int | None) -> list[dict[str, Any]]:
    engine = LazyQwenEngine(BASE_DIR / "model_files_qwen3_vl3_8b_instruct")
    results: list[dict[str, Any]] = []
    presets = load_qwen_presets()
    report.event(type="checkpoint", phase="qwen_start", status="ok", count=len(presets), gpu_memory=gpu_memory())
    for preset in presets.values():
        if preset.app_side_only:
            event = {
                "type": "skip",
                "engine": "qwen",
                "preset_id": preset.id,
                "preset_label": preset.label,
                "status": "skip",
                "note": "app-side utility preset",
            }
            report.event(**event)
            results.append(event)
            continue
        settings = qwen_settings_for_preset(preset.id, device_id, max_tokens_cap)
        for image_label, image_path in zip(IMAGE_LABELS, image_paths):
            started = time.time()
            event: dict[str, Any] = {
                "type": "case",
                "engine": "qwen",
                "preset_id": preset.id,
                "preset_label": preset.label,
                "image_label": image_label,
                "image_path": str(image_path),
                "output_format": preset.output_format,
            }
            try:
                final = collect_generator(engine.caption_single(str(image_path), dict(settings)))
                if not final:
                    raise RuntimeError("Qwen generator did not yield a final result.")
                status_html, caption, _overlay, rows, error_html = final
                if error_html:
                    raise RuntimeError(strip_html(error_html))
                ok, detail, row_count = validate_qwen_caption(preset.id, preset.output_format, str(caption), rows or [])
                event.update(parse_status_paths(str(status_html)))
                event.update(
                    {
                        "status": "pass" if ok else "fail",
                        "detail": detail,
                        "rows": row_count,
                        "caption_chars": len(str(caption or "")),
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                )
            except Exception as exc:
                event.update(
                    {
                        "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                )
            report.event(**event)
            results.append(event)
    engine.clear_models()
    report.event(type="checkpoint", phase="qwen_cleared", status="ok", gpu_memory=gpu_memory())
    return results


def run_legacy_case(report: SmokeReport, engine: LazyLegacyEngine, variant: str, defaults: dict[str, Any], image_label: str, image_path: Path, device_id: str, max_tokens: int | None) -> dict[str, Any]:
    settings = dict(defaults)
    settings.update({"device_id": device_id, "gpu_ids": device_id, "use_subprocess": False, "save_image": True})
    if max_tokens is not None:
        settings["max_new_tokens"] = min(int(settings.get("max_new_tokens", max_tokens)), int(max_tokens))
    started = time.time()
    event: dict[str, Any] = {
        "type": "case",
        "engine": "joy",
        "variant": variant,
        "image_label": image_label,
        "image_path": str(image_path),
    }
    try:
        result = engine.caption_single(str(image_path), settings)
        ok = bool(str(result.caption or "").strip())
        event.update(
            {
                "status": "pass" if ok else "fail",
                "detail": "non-empty caption" if ok else "empty caption",
                "caption_chars": len(str(result.caption or "")),
                "caption_path": str(result.caption_path) if result.caption_path else "",
                "metadata_path": str(result.metadata_path) if result.metadata_path else "",
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
    except Exception as exc:
        event.update(
            {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
    report.event(**event)
    return event


def run_beta_case(report: SmokeReport, engine: LazyBetaEngine, image_label: str, image_path: Path, device_id: str, max_tokens: int | None) -> dict[str, Any]:
    defaults = dict(beta_one.DEFAULTS)
    max_new_tokens = int(defaults["max_tokens"] if max_tokens is None else min(defaults["max_tokens"], max_tokens))
    started = time.time()
    event: dict[str, Any] = {
        "type": "case",
        "engine": "joy",
        "variant": "beta_one",
        "image_label": image_label,
        "image_path": str(image_path),
    }
    try:
        final = collect_generator(
            engine.caption_single(
                str(image_path),
                defaults["single_prompt"],
                defaults["temperature"],
                defaults["top_p"],
                max_new_tokens,
                defaults["model_quantization"],
                device_id,
                False,
                True,
                False,
                defaults["allow_tf32"],
                defaults["clear_cuda_cache"],
                defaults["low_cpu_mem_usage"],
                defaults["attention_backend"],
                defaults["use_liger_kernel"],
            )
        )
        if not final:
            raise RuntimeError("Beta generator did not yield a final result.")
        status_html, caption, error_html = final
        if error_html:
            raise RuntimeError(strip_html(error_html))
        ok = bool(str(caption or "").strip())
        event.update(parse_status_paths(str(status_html)))
        event.update(
            {
                "status": "pass" if ok else "fail",
                "detail": "non-empty caption" if ok else "empty caption",
                "caption_chars": len(str(caption or "")),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
    except Exception as exc:
        event.update(
            {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
    report.event(**event)
    return event


def run_joy_matrix(report: SmokeReport, image_paths: list[Path], device_id: str, max_tokens: int | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    engines = [
        ("pre_alpha", LazyLegacyEngine("pre_alpha", BASE_DIR), pre_alpha.DEFAULTS),
        ("alpha_one", LazyLegacyEngine("alpha_one", BASE_DIR), alpha_one.DEFAULTS),
        ("alpha_two", LazyLegacyEngine("alpha_two", BASE_DIR), alpha_two.DEFAULTS),
    ]
    report.event(type="checkpoint", phase="joy_start", status="ok", gpu_memory=gpu_memory())
    for variant, engine, defaults in engines:
        for image_label, image_path in zip(IMAGE_LABELS, image_paths):
            results.append(run_legacy_case(report, engine, variant, defaults, image_label, image_path, device_id, max_tokens))
        engine.clear_models()
        report.event(type="checkpoint", phase=f"{variant}_cleared", status="ok", gpu_memory=gpu_memory())
    beta_engine = LazyBetaEngine(BASE_DIR / "model_files_beta_one")
    for image_label, image_path in zip(IMAGE_LABELS, image_paths):
        results.append(run_beta_case(report, beta_engine, image_label, image_path, device_id, max_tokens))
    beta_engine.clear_models()
    report.event(type="checkpoint", phase="beta_one_cleared", status="ok", gpu_memory=gpu_memory())
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--qwen-max-tokens", type=int, default=None)
    parser.add_argument("--joy-max-tokens", type=int, default=None)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-joy", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "smoke_runs" / f"full_{now_stamp()}")
    parser.add_argument("images", nargs=3, type=Path)
    args = parser.parse_args()

    image_paths = [path.expanduser().resolve() for path in args.images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing image(s): " + ", ".join(missing))

    report = SmokeReport(args.output_dir)
    report.event(
        type="checkpoint",
        phase="start",
        status="ok",
        device_id=args.device_id,
        images=[str(path) for path in image_paths],
        gpu_memory=gpu_memory(),
    )
    qwen_results: list[dict[str, Any]] = []
    joy_results: list[dict[str, Any]] = []
    if not args.skip_qwen:
        qwen_results = run_qwen_matrix(report, image_paths, args.device_id, args.qwen_max_tokens)
    if not args.skip_joy:
        joy_results = run_joy_matrix(report, image_paths, args.device_id, args.joy_max_tokens)
    all_results = qwen_results + joy_results
    failed = [item for item in all_results if item.get("status") == "fail"]
    skipped = [item for item in all_results if item.get("status") == "skip"]
    passed = [item for item in all_results if item.get("status") == "pass"]
    report.event(
        type="checkpoint",
        phase="finish",
        status="ok" if not failed else "fail",
        passed=len(passed),
        failed=len(failed),
        skipped=len(skipped),
        gpu_memory=gpu_memory(),
    )
    report.write_summary(
        {
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
            "device_id": args.device_id,
        }
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
