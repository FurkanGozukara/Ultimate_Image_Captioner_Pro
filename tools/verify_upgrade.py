from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


warnings.filterwarnings("ignore", category=ResourceWarning)


APP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _load_downloader():
    path = ROOT_DIR / "HF_model_downloader.py"
    spec = importlib.util.spec_from_file_location("ultimate_caption_test_downloader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import downloader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UpgradeVerification(unittest.TestCase):
    online = False

    def test_downloader_catalog_matches_app(self) -> None:
        from joycaption.model_catalog import MODEL_SPECS

        downloader = _load_downloader()
        self.assertEqual(set(MODEL_SPECS), set(downloader.MODEL_DOWNLOAD_SPECS))
        self.assertEqual(
            set(downloader.DEFAULT_MODEL_KEYS),
            {"joycaption_beta_one", "qwen3_vl_8b_instruct"},
        )
        for key, app_spec in MODEL_SPECS.items():
            download_spec = downloader.MODEL_DOWNLOAD_SPECS[key]
            self.assertEqual(app_spec.repo_id, download_spec["repo_id"], key)
            if download_spec.get("target_subdir"):
                self.assertEqual(app_spec.subdir, download_spec["target_subdir"], key)
            else:
                self.assertEqual(tuple(download_spec["include_prefixes"]), (f"{app_spec.subdir}/",), key)

    def test_existing_default_models_are_complete(self) -> None:
        from joycaption.model_catalog import model_readiness_error

        for key in ("joycaption_beta_one", "qwen3_vl_8b_instruct"):
            self.assertIsNone(model_readiness_error(key), key)

    def test_real_compile_benchmark_report_is_consistent(self) -> None:
        data = json.loads((APP_DIR / "reports" / "torch_compile_benchmark.json").read_text(encoding="utf-8"))
        report_html = (APP_DIR / "reports" / "torch_compile_benchmark.html").read_text(encoding="utf-8")
        self.assertGreater(data["summary"]["speedup"], 1.0)
        self.assertGreaterEqual(data["summary"]["quality"]["word_sequence_similarity_percent"], 95.0)
        self.assertGreaterEqual(data["summary"]["quality"]["common_prefix_percent"], 85.0)
        worker_comparison = data["compile_worker_comparison"]
        self.assertEqual(worker_comparison["baseline_workers"], 1)
        self.assertEqual(worker_comparison["parallel_workers"], 8)
        self.assertTrue(worker_comparison["outputs_match"])
        self.assertGreater(worker_comparison["cold_compile_speed_gain_percent"], 0.0)
        self.assertEqual(len(data["runs"]["eager"]), 3)
        self.assertEqual(len(data["runs"]["compiled"]), 3)
        self.assertIn("color-scheme: dark", report_html)
        self.assertIn("Parallel compilation", report_html)
        self.assertIn("Output quality", report_html)

    def test_downloader_is_resolved_from_app_parent(self) -> None:
        from joycaption.model_downloads import model_downloader_path

        self.assertEqual(model_downloader_path(), (APP_DIR.parent / "HF_model_downloader.py").resolve())

    def test_optional_download_uses_current_venv_and_parent_downloader(self) -> None:
        from joycaption.model_downloads import ensure_model_available

        with (
            patch("joycaption.model_downloads.model_is_ready", return_value=False),
            patch("joycaption.model_downloads.model_readiness_error", return_value=None),
            patch("joycaption.model_downloads.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run,
        ):
            availability = ensure_model_available("qwen3_vl_2b_instruct")

        self.assertTrue(availability.downloaded)
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]).resolve(), (ROOT_DIR / "HF_model_downloader.py").resolve())
        self.assertEqual(command[2:4], ["--model", "qwen3_vl_2b_instruct"])
        self.assertEqual(command[4:6], ["--target-root", str(APP_DIR)])
        self.assertEqual(Path(run.call_args.kwargs["cwd"]).resolve(), ROOT_DIR.resolve())

    def test_lazy_qwen_reports_first_use_download_progress(self) -> None:
        from PIL import Image
        from joycaption.lazy_engines import LazyQwenEngine
        from joycaption.model_catalog import get_model_spec
        from joycaption.model_downloads import ModelAvailability

        spec = get_model_spec("qwen3_vl_2b_instruct")

        class FakeEngine:
            def caption_single(self, _image_path, _settings):
                yield "done", "caption", "", [], "", {}

            def clear_models(self) -> None:
                pass

        engine = LazyQwenEngine(spec.path)
        engine._engine = FakeEngine()
        engine._engine_model_key = spec.key
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            with (
                patch("joycaption.lazy_engines.download_required", return_value=True),
                patch(
                    "joycaption.lazy_engines.ensure_model_available",
                    return_value=ModelAvailability(spec=spec, downloaded=True),
                ),
            ):
                results = list(
                    engine.caption_single(
                        image_path,
                        {"model_key": spec.key, "use_subprocess": False},
                    )
                )

        self.assertEqual(len(results), 3)
        self.assertIn("Downloading it now", results[0][0])
        self.assertIn("download verified", results[1][0])
        self.assertEqual(results[2][1], "caption")

    def test_enum_compatibility_emits_no_deprecation(self) -> None:
        code = (
            "import joycaption; import transformers; "
            "print(transformers.Qwen3VLForConditionalGeneration); "
            "print(transformers.Qwen3VLMoeForConditionalGeneration); "
            "print(transformers.Qwen3_5ForConditionalGeneration)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, combined)
        self.assertNotIn("register_constant() on Enum subclasses", combined)
        self.assertNotIn("KernelPreference'> is an Enum subclass", combined)
        self.assertNotIn("ScaleCalculationMode'> is an Enum subclass", combined)

    def test_compile_toolchain_probe(self) -> None:
        from joycaption.torch_compile import inspect_compile_environment

        report = inspect_compile_environment(force=True)
        self.assertTrue(report.available, report.summary())
        self.assertTrue(report.compiler_probe_ok, report)
        self.assertTrue(report.compiler, report)
        self.assertNotEqual(report.triton_version, "unknown", report)

    def test_compile_worker_settings_are_portable_and_bounded(self) -> None:
        from joycaption.torch_compile_workers import (
            COMPILE_THREADS_ENV,
            DEFAULT_COMPILE_THREADS,
            WORKER_START_ENV,
            normalize_compile_threads,
            prepare_compile_worker_env,
        )

        self.assertEqual(normalize_compile_threads(None, env={}), DEFAULT_COMPILE_THREADS)
        self.assertEqual(normalize_compile_threads(0), 1)
        self.assertEqual(normalize_compile_threads(99), 32)
        windows_env: dict[str, str] = {WORKER_START_ENV: "subprocess"}
        windows = prepare_compile_worker_env(windows_env, 12, platform="win32")
        self.assertEqual((windows.threads, windows.worker_start), (12, "spawn"))
        self.assertEqual(windows_env[COMPILE_THREADS_ENV], "12")
        linux_env: dict[str, str] = {WORKER_START_ENV: "fork"}
        linux = prepare_compile_worker_env(linux_env, 6, platform="linux")
        self.assertEqual((linux.threads, linux.worker_start), (6, "fork"))

    def test_generation_compile_waits_for_parallel_workers(self) -> None:
        from joycaption.torch_compile import generation_compile_kwargs

        settings = {
            "torch_compile": True,
            "compile_backend": "inductor",
            "compile_mode": "default",
            "compile_dynamic": "false",
            "compile_fullgraph": False,
            "compile_threads": 8,
        }
        with patch("joycaption.torch_compile.prepare_torch_compile") as prepare:
            result = generation_compile_kwargs(settings, SimpleNamespace(hf_quantizer=None))

        prepare.assert_called_once_with(settings, wait_for_workers=True)
        self.assertEqual(result["cache_implementation"], "static")

    def test_large_qwen_models_receive_safe_vram_defaults(self) -> None:
        from joycaption.vram import VRAM_PRESET_CHOICES, qwen_vram_settings

        self.assertIn("80 GB", VRAM_PRESET_CHOICES)
        for model_key in (
            "qwen3_vl_30b_a3b_instruct",
            "qwen3_6_27b",
            "huihui_qwen3_6_27b_abliterated",
        ):
            self.assertEqual(qwen_vram_settings("32 GB", model_key)["model_quantization"], "nf4")
            self.assertEqual(qwen_vram_settings("64 GB", model_key)["model_quantization"], "int8")
            self.assertEqual(qwen_vram_settings("80 GB", model_key)["model_quantization"], "bf16")

    def test_thinking_budget_preserves_visible_answer_tokens(self) -> None:
        import torch
        from joycaption.engines.qwen import ThinkingBudgetCriteria

        criteria = ThinkingBudgetCriteria(think_end_token_id=99, answer_tokens=2, batch_size=2)
        scores = torch.empty(2, 1)
        self.assertEqual(criteria(torch.tensor([[1, 2], [3, 4]]), scores).tolist(), [False, False])
        self.assertEqual(criteria(torch.tensor([[1, 2, 99], [3, 4, 99]]), scores).tolist(), [False, False])
        self.assertEqual(criteria(torch.tensor([[1, 2, 99, 5], [3, 4, 99, 6]]), scores).tolist(), [False, False])
        self.assertEqual(criteria(torch.tensor([[1, 2, 99, 5, 7], [3, 4, 99, 6, 8]]), scores).tolist(), [True, True])

    def test_app_exposes_models_and_compile_controls(self) -> None:
        import app
        from joycaption.model_catalog import qwen_model_choices

        demo = app.build_app()
        config = demo.get_config_file()
        components = config.get("components", [])
        labels = [component.get("props", {}).get("label") for component in components]
        self.assertEqual(labels.count("Enable torch.compile"), 5)
        self.assertEqual(labels.count("Compile Workers"), 5)
        compile_worker_components = [
            component for component in components if component.get("props", {}).get("label") == "Compile Workers"
        ]
        self.assertTrue(all(component["props"].get("value") == 8 for component in compile_worker_components))
        model_component = next(
            component for component in components if component.get("props", {}).get("label") == "Caption Model"
        )
        actual_choices = [tuple(choice) for choice in model_component["props"]["choices"]]
        self.assertEqual(actual_choices, qwen_model_choices())

    def test_online_qwen_architectures_are_supported(self) -> None:
        if not self.online:
            self.skipTest("pass --online to validate Hugging Face configs")

        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModelForImageTextToText
        from joycaption.model_catalog import QWEN_MODEL_SPECS

        for model_spec in QWEN_MODEL_SPECS:
            config_path = hf_hub_download(model_spec.repo_id, "config.json")
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
            self.assertIn(model_spec.architecture, payload.get("architectures") or [], model_spec.repo_id)
            config = AutoConfig.from_pretrained(model_spec.repo_id, trust_remote_code=True)
            resolved_class = AutoModelForImageTextToText._model_mapping[type(config)]
            self.assertEqual(resolved_class.__name__, model_spec.architecture, model_spec.repo_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args, unittest_args = parser.parse_known_args()
    UpgradeVerification.online = bool(args.online)
    program = unittest.main(argv=[sys.argv[0], *unittest_args], exit=False, verbosity=2)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
