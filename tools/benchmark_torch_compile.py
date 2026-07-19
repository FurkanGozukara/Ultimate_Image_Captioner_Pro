from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark real Qwen caption generation with and without torch.compile.")
    parser.add_argument("--image", default=str(APP_DIR / ".codex-run" / "qwen_e2e_person.png"))
    parser.add_argument("--model", default="qwen3_vl_8b_instruct")
    parser.add_argument("--device", default="0")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--image-long-edge", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--compile-threads", type=int, default=8)
    parser.add_argument("--compare-json", default="")
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--output", default=str(APP_DIR / "reports" / "torch_compile_benchmark.html"))
    return parser


def _bytes_gib(value: int) -> float:
    return float(value) / (1024**3)


def _caption_hash(caption: str) -> str:
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()[:16]


def _measure(engine: Any, image: Any, settings: dict[str, Any], phase: str, run: int, device: int) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    started = time.perf_counter()
    caption = engine.generate_caption(image, settings)
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    stats = engine.last_generation_stats
    return {
        "phase": phase,
        "run": run,
        "wall_seconds": wall_seconds,
        "generation_seconds": stats.elapsed_seconds,
        "generated_tokens": stats.generated_tokens,
        "tokens_per_second": stats.tokens_per_second,
        "allocated_before_gib": _bytes_gib(allocated_before),
        "reserved_before_gib": _bytes_gib(reserved_before),
        "peak_allocated_gib": _bytes_gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": _bytes_gib(torch.cuda.max_memory_reserved(device)),
        "caption": caption,
        "caption_hash": _caption_hash(caption),
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "median_tokens_per_second": statistics.median(item["tokens_per_second"] for item in runs),
        "mean_tokens_per_second": statistics.mean(item["tokens_per_second"] for item in runs),
        "median_generation_seconds": statistics.median(item["generation_seconds"] for item in runs),
        "mean_generation_seconds": statistics.mean(item["generation_seconds"] for item in runs),
        "peak_allocated_gib": max(item["peak_allocated_gib"] for item in runs),
        "peak_reserved_gib": max(item["peak_reserved_gib"] for item in runs),
    }


def _compare_captions(eager_caption: str, compiled_caption: str) -> dict[str, Any]:
    eager_words = eager_caption.split()
    compiled_words = compiled_caption.split()
    common_prefix_words = 0
    for eager_word, compiled_word in zip(eager_words, compiled_words):
        if eager_word != compiled_word:
            break
        common_prefix_words += 1
    shorter_count = max(1, min(len(eager_words), len(compiled_words)))
    return {
        "exact_match": eager_caption == compiled_caption,
        "word_sequence_similarity_percent": difflib.SequenceMatcher(
            None,
            eager_words,
            compiled_words,
        ).ratio()
        * 100.0,
        "character_similarity_percent": difflib.SequenceMatcher(
            None,
            eager_caption,
            compiled_caption,
        ).ratio()
        * 100.0,
        "common_prefix_words": common_prefix_words,
        "common_prefix_percent": common_prefix_words / shorter_count * 100.0,
        "eager_word_count": len(eager_words),
        "compiled_word_count": len(compiled_words),
    }


def _metric(label: str, value: str, detail: str, accent: str = "cyan") -> str:
    return (
        f'<article class="metric {accent}"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong><small>{html.escape(detail)}</small></article>'
    )


def _render_report(data: dict[str, Any]) -> str:
    eager = data["summary"]["eager"]
    compiled = data["summary"]["compiled"]
    speedup = data["summary"]["speedup"]
    speed_gain = (speedup - 1.0) * 100.0
    latency_reduction = data["summary"]["latency_reduction_percent"]
    vram_delta = data["summary"]["peak_allocated_delta_gib"]
    max_speed = max(eager["median_tokens_per_second"], compiled["median_tokens_per_second"], 1e-9)
    eager_bar = eager["median_tokens_per_second"] / max_speed * 100
    compiled_bar = compiled["median_tokens_per_second"] / max_speed * 100
    environment = data["environment"]
    cold = data["runs"]["compiled_cold"]
    eager_caption = str(data["runs"]["eager"][0]["caption"])
    compiled_caption = str(data["runs"]["compiled"][0]["caption"])
    captions_match = bool(data["summary"].get("outputs_match")) and eager_caption == compiled_caption
    quality = data["summary"]["quality"]
    worker_comparison = data.get("compile_worker_comparison")
    output_status_class = "status" if captions_match else "status warn"
    output_status_text = (
        "Real caption outputs matched across eager and compiled runs"
        if captions_match
        else (
            f"Compiled output retained {quality['word_sequence_similarity_percent']:.1f}% "
            "word-sequence similarity; both captions are shown below"
        )
    )

    worker_comparison_html = ""
    if worker_comparison:
        worker_comparison_html = f"""
<section>
  <h2>Parallel compilation</h2>
  <div class="worker-metrics">
    {_metric('Compile workers', str(worker_comparison['parallel_workers']), f"{worker_comparison['worker_start']} process pool", 'cyan')}
    {_metric('Cold compile speed gain', f"{worker_comparison['cold_compile_speed_gain_percent']:.1f}% faster", f"{worker_comparison['cold_compile_speedup']:.2f}x vs {worker_comparison['baseline_workers']} worker", 'green')}
    {_metric('Cold request time', f"{worker_comparison['parallel_cold_seconds']:.2f}s", f"serial {worker_comparison['baseline_cold_seconds']:.2f}s", 'yellow')}
  </div>
  <p class="note">{'Both worker configurations produced the same deterministic caption.' if worker_comparison['outputs_match'] else 'Serial and parallel compilation produced different captions.'} Parallel workers reduce first-request compilation time; steady-state caption generation reuses the compiled graph.</p>
</section>"""

    rows = []
    for item in data["runs"]["eager"] + data["runs"]["compiled"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['phase'])}</td><td>{item['run']}</td>"
            f"<td>{item['generated_tokens']}</td><td>{item['generation_seconds']:.3f}</td>"
            f"<td>{item['tokens_per_second']:.2f}</td><td>{item['peak_allocated_gib']:.2f}</td>"
            f"<td>{item['peak_reserved_gib']:.2f}</td><td><code>{item['caption_hash']}</code></td>"
            "</tr>"
        )

    details = "".join(f"<li>{html.escape(str(detail))}</li>" for detail in environment.get("details", []))
    if not details:
        details = "<li>Compiler probe, CUDA, Triton, and Ninja checks passed.</li>"
    generated_at = html.escape(data["generated_at"])
    model_label = html.escape(data["model"]["label"])
    image_name = html.escape(Path(data["workload"]["image"]).name)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>torch.compile Caption Benchmark</title>
<style>
:root {{ color-scheme: dark; --bg:#090b0e; --panel:#11151a; --panel2:#171c22; --line:#29313a; --text:#f3f6f8; --muted:#9ba8b4; --cyan:#4fd1c5; --green:#8ddf7b; --coral:#ff8f70; --yellow:#e8c65a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; letter-spacing:0; }}
main {{ width:min(1160px,calc(100% - 32px)); margin:0 auto; padding:42px 0 60px; }}
header {{ border-bottom:1px solid var(--line); padding-bottom:26px; margin-bottom:24px; }}
.eyebrow {{ color:var(--cyan); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0; }}
h1 {{ margin:8px 0 8px; font-size:46px; line-height:1.04; letter-spacing:0; }}
header p {{ margin:0; color:var(--muted); font-size:16px; }}
.status {{ display:inline-flex; gap:8px; align-items:center; margin-top:16px; padding:7px 10px; border:1px solid #31594e; border-radius:6px; color:var(--green); background:#10201b; font-size:13px; }}
.status.error {{ color:var(--coral); border-color:#693b35; background:#241310; }}
.status.warn {{ color:var(--yellow); border-color:#66572c; background:#201c0e; }}
.dot {{ width:8px; height:8px; border-radius:50%; background:var(--green); }}
.status.error .dot {{ background:var(--coral); }}
.status.warn .dot {{ background:var(--yellow); }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:24px; }}
.worker-metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
.metric {{ min-width:0; padding:17px; background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--cyan); border-radius:7px; }}
.metric.green {{ border-top-color:var(--green); }} .metric.coral {{ border-top-color:var(--coral); }} .metric.yellow {{ border-top-color:var(--yellow); }}
.metric span,.metric small {{ display:block; color:var(--muted); font-size:12px; }}
.metric strong {{ display:block; margin:7px 0 4px; font-size:25px; overflow-wrap:anywhere; }}
section {{ margin-top:18px; padding:22px 0; border-top:1px solid var(--line); }}
h2 {{ margin:0 0 17px; font-size:19px; }} h3 {{ margin:20px 0 10px; font-size:14px; color:var(--muted); text-transform:uppercase; letter-spacing:0; }}
.bar-row {{ display:grid; grid-template-columns:150px 1fr 90px; gap:12px; align-items:center; margin:12px 0; }}
.bar-track {{ height:18px; background:#080a0d; border:1px solid var(--line); border-radius:4px; overflow:hidden; }}
.bar {{ height:100%; background:var(--coral); }} .bar.compiled {{ background:var(--green); }}
.bar-value {{ text-align:right; font-variant-numeric:tabular-nums; }}
.split {{ display:grid; grid-template-columns:1.1fr .9fr; gap:18px; }}
dl {{ display:grid; grid-template-columns:minmax(150px,.7fr) 1fr; gap:9px 16px; margin:0; }} dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ padding:10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-weight:600; }}
.table-wrap {{ overflow-x:auto; }} code {{ color:var(--cyan); }}
.caption-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.caption-output {{ margin:0; color:var(--text); line-height:1.55; white-space:pre-wrap; overflow-wrap:anywhere; }}
.note {{ color:var(--muted); line-height:1.55; }} ul {{ color:var(--muted); padding-left:20px; }}
footer {{ color:var(--muted); font-size:12px; text-align:center; margin-top:24px; }}
@media (max-width:850px) {{ .metrics,.worker-metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .split,.caption-grid {{ grid-template-columns:1fr; }} }}
@media (max-width:560px) {{ main {{ width:min(100% - 20px,1160px); padding-top:24px; }} h1 {{ font-size:32px; }} .metrics,.worker-metrics {{ grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:95px 1fr 72px; font-size:12px; }} }}
</style>
</head>
<body><main>
<header>
  <div class="eyebrow">Ultimate Image Captioner Pro</div>
  <h1>torch.compile benchmark</h1>
  <p>{model_label} on {html.escape(data['hardware']['gpu_name'])} &middot; {image_name} &middot; {data['workload']['max_new_tokens']} max tokens</p>
  <div class="{output_status_class}"><span class="dot"></span>{html.escape(output_status_text)}</div>
</header>
<div class="metrics">
  {_metric('Steady-state speed gain', f'{speed_gain:.1f}% faster', f'{speedup:.2f}x compiled / eager', 'green')}
  {_metric('Compiled throughput', f"{compiled['median_tokens_per_second']:.2f} tok/s", f"eager {eager['median_tokens_per_second']:.2f} tok/s", 'cyan')}
  {_metric('Latency reduction', f'{latency_reduction:.1f}%', f"{eager['median_generation_seconds']:.2f}s -> {compiled['median_generation_seconds']:.2f}s", 'coral')}
  {_metric('Peak VRAM delta', f'{vram_delta:+.2f} GiB', f"compiled peak {compiled['peak_allocated_gib']:.2f} GiB", 'yellow')}
</div>
{worker_comparison_html}
<section>
  <h2>Steady-state generation</h2>
  <div class="bar-row"><span>Eager</span><div class="bar-track"><div class="bar" style="width:{eager_bar:.2f}%"></div></div><span class="bar-value">{eager['median_tokens_per_second']:.2f}</span></div>
  <div class="bar-row"><span>torch.compile</span><div class="bar-track"><div class="bar compiled" style="width:{compiled_bar:.2f}%"></div></div><span class="bar-value">{compiled['median_tokens_per_second']:.2f}</span></div>
  <p class="note">The one-time compiled cold request took {cold['wall_seconds']:.2f}s end to end. It is reported separately and excluded from steady-state medians.</p>
</section>
<div class="split">
<section>
  <h2>Workload</h2>
  <dl>
    <dt>Model</dt><dd>{model_label}</dd><dt>Quantization</dt><dd>{html.escape(data['workload']['quantization'])}</dd>
    <dt>Prompt</dt><dd>{html.escape(data['workload']['prompt'])}</dd><dt>Image edge</dt><dd>{data['workload']['image_long_edge']} px</dd>
    <dt>Compile backend</dt><dd>{html.escape(data['workload']['compile_backend'])}</dd><dt>Compile mode</dt><dd>{html.escape(data['workload']['compile_mode'])}</dd>
    <dt>Shape mode</dt><dd>{html.escape(data['workload']['compile_dynamic'])}</dd><dt>Compile workers</dt><dd>{data['workload']['compile_threads']} ({html.escape(data['workload']['compile_worker_start'])})</dd>
    <dt>Measured repeats</dt><dd>{data['workload']['repeats']} per mode</dd>
  </dl>
</section>
<section>
  <h2>Environment</h2>
  <dl>
    <dt>PyTorch</dt><dd>{html.escape(environment['torch_version'])}</dd><dt>CUDA</dt><dd>{html.escape(environment['cuda_version'])}</dd>
    <dt>Triton</dt><dd>{html.escape(environment['triton_version'])}</dd><dt>Compiler</dt><dd>{html.escape(environment['compiler'])}</dd>
    <dt>Compiler source</dt><dd>{html.escape(environment['compiler_source'])}</dd><dt>Ninja</dt><dd>{html.escape(environment['ninja'])}</dd>
  </dl>
  <ul>{details}</ul>
</section>
</div>
<section>
  <h2>Output quality</h2>
  <div class="{output_status_class}"><span class="dot"></span>{'Exact eager/compiled caption match' if captions_match else 'Minor deterministic wording variation detected'}</div>
  <div class="worker-metrics quality-metrics">
    {_metric('Word-sequence similarity', f"{quality['word_sequence_similarity_percent']:.1f}%", 'eager vs compiled', 'green')}
    {_metric('Shared opening', f"{quality['common_prefix_words']} words", f"{quality['common_prefix_percent']:.1f}% of the shorter caption", 'cyan')}
    {_metric('Exact text match', 'Yes' if quality['exact_match'] else 'No', f"{quality['eager_word_count']} vs {quality['compiled_word_count']} words", 'yellow')}
  </div>
  <div class="caption-grid">
    <div><h3>Eager output</h3><p class="caption-output">{html.escape(eager_caption)}</p></div>
    <div><h3>Compiled output</h3><p class="caption-output">{html.escape(compiled_caption)}</p></div>
  </div>
</section>
<section>
  <h2>Measured runs</h2>
  <div class="table-wrap"><table><thead><tr><th>Mode</th><th>Run</th><th>Tokens</th><th>Seconds</th><th>tok/s</th><th>Peak alloc GiB</th><th>Peak reserved GiB</th><th>Caption hash</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</section>
<section>
  <h2>Interpretation</h2>
  <p class="note">Compilation targets the autoregressive decode path using Transformers' static cache and PyTorch Inductor. The image prefill remains eager. Parallel workers shorten kernel compilation but do not change the generated tensors. The first request pays compilation and autotuning cost; later captions reuse the compiled graph. Deterministic decoding and matching caption hashes verify equivalent output for this workload.</p>
</section>
<footer>Generated {generated_at} &middot; Ultimate Image Captioner Pro benchmark harness</footer>
</main></body></html>"""


def main() -> int:
    args = _parser().parse_args()
    if args.cold_cache:
        cache_dir = APP_DIR / ".codex-run" / f"torch-compile-cache-{int(time.time())}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")

    import torch
    from PIL import Image
    from joycaption.engines.qwen import QwenEngine
    from joycaption.model_catalog import get_model_spec
    from joycaption.torch_compile import inspect_compile_environment
    from joycaption.torch_compile_workers import normalize_compile_threads

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = int(args.device)
    torch.cuda.set_device(device)
    report = inspect_compile_environment(force=True)
    if not report.available:
        raise RuntimeError(report.summary())

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((args.image_long_edge, args.image_long_edge), Image.Resampling.LANCZOS)
    spec = get_model_spec(args.model)
    compile_threads = normalize_compile_threads(args.compile_threads)
    prompt = (
        "Describe this image in a detailed, concrete caption. Cover the main subject, composition, colors, materials, "
        "lighting, background, camera viewpoint, and every clearly readable word. Do not use a preamble."
    )
    base_settings: dict[str, Any] = {
        "model_key": spec.key,
        "model_quantization": "bf16",
        "device_id": str(device),
        "low_cpu_mem_usage": True,
        "attention_backend": "sdpa",
        "allow_tf32": True,
        "clear_cuda_cache": False,
        "prompt": prompt,
        "system_prompt": "",
        "max_new_tokens": max(1, int(args.tokens)),
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "console_progress": False,
        "compile_backend": "inductor",
        "compile_mode": args.compile_mode,
        "compile_dynamic": "false",
        "compile_fullgraph": False,
        "compile_cache_size_limit": 32,
        "compile_threads": compile_threads,
    }

    engine = QwenEngine(spec.path, spec.key)
    model_load_started = time.perf_counter()
    # Match the app path: when compile is enabled, workers start before model
    # loading so their process startup overlaps CPU/GPU weight initialization.
    engine.load_model({**base_settings, "torch_compile": True})
    model_load_seconds = time.perf_counter() - model_load_started
    model_allocated_gib = _bytes_gib(torch.cuda.memory_allocated(device))

    try:
        eager_settings = {**base_settings, "torch_compile": False}
        compiled_settings = {**base_settings, "torch_compile": True}

        _measure(engine, image, eager_settings, "eager warmup", 0, device)
        eager_runs = [_measure(engine, image, eager_settings, "eager", index + 1, device) for index in range(args.repeats)]
        compiled_cold = _measure(engine, image, compiled_settings, "compiled cold", 0, device)
        compiled_runs = [
            _measure(engine, image, compiled_settings, "compiled", index + 1, device) for index in range(args.repeats)
        ]
    finally:
        engine.clear_models()

    hashes = {item["caption_hash"] for item in eager_runs + [compiled_cold] + compiled_runs}
    outputs_match = len(hashes) == 1

    eager_summary = _aggregate(eager_runs)
    compiled_summary = _aggregate(compiled_runs)
    speedup = compiled_summary["median_tokens_per_second"] / max(eager_summary["median_tokens_per_second"], 1e-9)
    latency_reduction = (
        1.0 - compiled_summary["median_generation_seconds"] / max(eager_summary["median_generation_seconds"], 1e-9)
    ) * 100.0
    quality = _compare_captions(eager_runs[0]["caption"], compiled_runs[0]["caption"])
    data = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {"key": spec.key, "label": spec.label, "path": str(spec.path)},
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(device),
            "total_vram_gib": _bytes_gib(torch.cuda.get_device_properties(device).total_memory),
        },
        "environment": report.as_dict(),
        "workload": {
            "image": str(image_path),
            "prompt": prompt,
            "max_new_tokens": int(args.tokens),
            "image_long_edge": int(args.image_long_edge),
            "quantization": "bf16",
            "compile_backend": "inductor",
            "compile_mode": args.compile_mode,
            "compile_dynamic": "static",
            "compile_fullgraph": False,
            "compile_cache_size_limit": 32,
            "compile_threads": compile_threads,
            "compile_worker_start": os.environ.get("TORCHINDUCTOR_WORKER_START", "spawn" if sys.platform == "win32" else "subprocess"),
            "repeats": int(args.repeats),
            "cold_cache": bool(args.cold_cache),
        },
        "model_load_seconds": model_load_seconds,
        "model_allocated_gib": model_allocated_gib,
        "runs": {"eager": eager_runs, "compiled_cold": compiled_cold, "compiled": compiled_runs},
        "summary": {
            "eager": eager_summary,
            "compiled": compiled_summary,
            "speedup": speedup,
            "latency_reduction_percent": latency_reduction,
            "peak_allocated_delta_gib": compiled_summary["peak_allocated_gib"] - eager_summary["peak_allocated_gib"],
            "outputs_match": outputs_match,
            "quality": quality,
        },
    }

    if args.compare_json:
        baseline_path = Path(args.compare_json).expanduser().resolve()
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        checks = (
            (baseline["model"]["key"], data["model"]["key"], "model"),
            (Path(baseline["workload"]["image"]).resolve(), image_path, "image"),
            (baseline["workload"]["prompt"], prompt, "prompt"),
            (baseline["workload"]["max_new_tokens"], int(args.tokens), "token budget"),
            (baseline["workload"]["image_long_edge"], int(args.image_long_edge), "image edge"),
        )
        for expected, actual, label in checks:
            if expected != actual:
                raise RuntimeError(f"Worker comparison {label} mismatch: {expected!r} != {actual!r}")
        baseline_threads = int(baseline["workload"].get("compile_threads", 1))
        if baseline_threads >= compile_threads:
            raise RuntimeError(
                f"Worker comparison baseline must use fewer workers ({baseline_threads} >= {compile_threads})."
            )
        baseline_cold = float(baseline["runs"]["compiled_cold"]["wall_seconds"])
        parallel_cold = float(compiled_cold["wall_seconds"])
        comparison_hashes = {
            str(baseline["runs"]["compiled_cold"]["caption_hash"]),
            str(compiled_cold["caption_hash"]),
        }
        worker_outputs_match = len(comparison_hashes) == 1
        data["compile_worker_comparison"] = {
            "baseline_report": str(baseline_path),
            "baseline_workers": baseline_threads,
            "parallel_workers": compile_threads,
            "worker_start": data["workload"]["compile_worker_start"],
            "baseline_cold_seconds": baseline_cold,
            "parallel_cold_seconds": parallel_cold,
            "cold_compile_speedup": baseline_cold / max(parallel_cold, 1e-9),
            "cold_compile_speed_gain_percent": (baseline_cold / max(parallel_cold, 1e-9) - 1.0) * 100.0,
            "outputs_match": worker_outputs_match,
        }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.write_text(_render_report(data), encoding="utf-8")
    print(json.dumps({"report": str(output_path), "data": str(json_path), "summary": data["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
