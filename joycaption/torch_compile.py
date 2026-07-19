from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .torch_compile_workers import (
    DEFAULT_COMPILE_THREADS,
    CompileWorkerSettings,
    CompileWorkerWarmupHandle,
    CompileWorkerWarmupResult,
    configure_compile_workers,
    finish_compile_worker_warmup,
    normalize_compile_threads,
    start_compile_worker_warmup,
)


COMPILE_BACKEND_CHOICES = ["inductor", "cudagraphs", "aot_eager", "eager"]
COMPILE_MODE_CHOICES = ["max-autotune-no-cudagraphs", "reduce-overhead", "default", "max-autotune"]
COMPILE_DYNAMIC_CHOICES = [("Auto", "auto"), ("Static", "false"), ("Dynamic", "true")]

DEFAULT_COMPILE_SETTINGS: dict[str, Any] = {
    "torch_compile": False,
    "compile_backend": "inductor",
    "compile_mode": "max-autotune-no-cudagraphs",
    "compile_dynamic": "false",
    "compile_fullgraph": False,
    "compile_cache_size_limit": 32,
    "compile_threads": DEFAULT_COMPILE_THREADS,
}


@dataclass(frozen=True)
class CompileEnvironmentReport:
    available: bool
    platform: str
    torch_version: str
    cuda_version: str
    triton_version: str
    compiler: str
    compiler_source: str
    compiler_probe_ok: bool
    ninja: str
    details: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if not self.available:
            return "torch.compile unavailable: " + "; ".join(self.details)
        return (
            f"torch.compile ready ({self.platform}; Torch {self.torch_version}; CUDA {self.cuda_version}; "
            f"Triton {self.triton_version}; compiler {self.compiler})"
        )


_REPORT_LOCK = threading.Lock()
_REPORT_CACHE: CompileEnvironmentReport | None = None
_VS_ENV_CACHE: dict[str, str] | None = None
_VS_ENV_CACHE_FAILED = False
_WORKER_WARMUP_LOCK = threading.RLock()
_WORKER_WARMUP_KEY: tuple[int, str] | None = None
_WORKER_WARMUP_HANDLE: CompileWorkerWarmupHandle | None = None
_WORKER_WARMUP_RESULT: CompileWorkerWarmupResult | None = None
_LOGGER = logging.getLogger(__name__)

_ENV_VS_INSTALL_CANDIDATES = (
    ("ULTIMATE_CAPTION_VS_INSTALLDIR", 0),
    ("VSINSTALLDIR", 0),
    ("VS170COMNTOOLS", 2),
    ("VS160COMNTOOLS", 2),
    ("VS150COMNTOOLS", 2),
    ("VCINSTALLDIR", 1),
    ("VCToolsInstallDir", 3),
)
_ENV_VS_DEV_CMD_CANDIDATES = ("ULTIMATE_CAPTION_VS_DEV_CMD", "VS_DEV_CMD")


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _normalize_path(value: str | None) -> str:
    if not value:
        return ""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(value).strip().strip('"').strip("'"))))


def _has_openmp_header(env: dict[str, str]) -> bool:
    for item in env.get("INCLUDE", "").split(os.pathsep):
        candidate = item.strip()
        if candidate and Path(candidate, "omp.h").is_file():
            return True
    return False


def _vs_installation_from_env(env: dict[str, str]) -> tuple[str, str] | None:
    for variable, levels_up in _ENV_VS_INSTALL_CANDIDATES:
        candidate = _normalize_path(env.get(variable))
        if not candidate:
            continue
        for _ in range(levels_up):
            candidate = os.path.dirname(candidate)
        if os.path.isdir(candidate):
            return candidate, variable
    return None


def _resolve_vswhere() -> str | None:
    path_candidate = shutil.which("vswhere.exe") or shutil.which("vswhere")
    if path_candidate and os.path.isfile(path_candidate):
        return path_candidate
    for root in (
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramFiles"),
        r"C:\Program Files (x86)",
        r"C:\Program Files",
    ):
        if not root:
            continue
        candidate = os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _query_vswhere(vswhere: str, env: dict[str, str]) -> str | None:
    try:
        completed = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return next((line.strip() for line in completed.stdout.splitlines() if line.strip()), None)


def _default_vs_installations() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for root in (
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramFiles"),
        r"C:\Program Files (x86)",
        r"C:\Program Files",
    ):
        base = Path(root or "") / "Microsoft Visual Studio"
        if not base.is_dir():
            continue
        try:
            versions = sorted((path for path in base.iterdir() if path.is_dir()), reverse=True)
        except OSError:
            continue
        for version in versions:
            try:
                editions = sorted((path for path in version.iterdir() if path.is_dir()), reverse=True)
            except OSError:
                continue
            for edition in editions:
                normalized = os.path.normcase(str(edition))
                if normalized not in seen:
                    seen.add(normalized)
                    candidates.append(str(edition))
    return candidates


def _locate_vs_dev_batch(installation: str, env: dict[str, str]) -> str | None:
    for variable in _ENV_VS_DEV_CMD_CANDIDATES:
        candidate = _normalize_path(env.get(variable))
        if candidate and os.path.isfile(candidate):
            return candidate
    for relative in (
        ("Common7", "Tools", "VsDevCmd.bat"),
        ("VC", "Auxiliary", "Build", "vcvars64.bat"),
        ("VC", "Auxiliary", "Build", "vcvarsall.bat"),
    ):
        candidate = os.path.join(installation, *relative)
        if os.path.isfile(candidate):
            return candidate
    return None


def _capture_batch_environment(batch_file: str, base_env: dict[str, str]) -> dict[str, str]:
    batch_name = os.path.basename(batch_file).lower()
    args = " -arch=amd64 -host_arch=amd64" if batch_name == "vsdevcmd.bat" else " amd64" if batch_name == "vcvarsall.bat" else ""
    script_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="utf-8", newline="") as script:
            script_path = script.name
            script.write("@echo off\r\n")
            script.write(f'call "{batch_file}"{args}\r\n')
            script.write("if %errorlevel% neq 0 exit /b %errorlevel%\r\n")
            script.write("set\r\n")
        completed = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=base_env,
            timeout=120,
            check=False,
        )
    finally:
        if script_path:
            try:
                os.remove(script_path)
            except OSError:
                pass
    if completed.returncode != 0:
        raise RuntimeError(f"{batch_name} exited with code {completed.returncode}")
    captured: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            # cmd.exe can expose drive-current-directory pseudo variables such
            # as ``=C:=...``. They are not legal Python environment keys.
            if key and "=" not in key and "\x00" not in key:
                captured[key] = value
    return captured


def _bootstrap_visual_studio_env(base_env: dict[str, str]) -> tuple[dict[str, str] | None, str]:
    global _VS_ENV_CACHE, _VS_ENV_CACHE_FAILED
    if _VS_ENV_CACHE is not None:
        return dict(_VS_ENV_CACHE), "cached Visual Studio developer environment"
    if _VS_ENV_CACHE_FAILED:
        return None, "previous Visual Studio discovery failed"

    installation: str | None = None
    source = ""
    env_result = _vs_installation_from_env(base_env)
    if env_result:
        installation, source = env_result
    if not installation:
        vswhere = _resolve_vswhere()
        if vswhere:
            installation = _query_vswhere(vswhere, base_env)
            source = "vswhere"

    installations = [installation] if installation else []
    installations.extend(path for path in _default_vs_installations() if path not in installations)
    for candidate in installations:
        if not candidate:
            continue
        batch_file = _locate_vs_dev_batch(candidate, base_env)
        if not batch_file:
            continue
        try:
            captured = _capture_batch_environment(batch_file, base_env)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            continue
        delta = {key: value for key, value in captured.items() if base_env.get(key) != value}
        _VS_ENV_CACHE = delta
        return dict(delta), f"{source or 'filesystem'}: {batch_file}"

    _VS_ENV_CACHE_FAILED = True
    return None, "environment variables, vswhere, and filesystem search"


def _select_posix_compiler(env: dict[str, str]) -> tuple[str | None, str]:
    configured = env.get("CXX")
    if configured:
        resolved = shutil.which(configured, path=env.get("PATH"))
        if resolved:
            return resolved, "CXX"
    for candidate in ("c++", "g++", "clang++"):
        resolved = shutil.which(candidate, path=env.get("PATH"))
        if resolved:
            env["CXX"] = resolved
            if "CC" not in env:
                cc_name = "clang" if "clang" in Path(resolved).name else "gcc"
                cc_path = shutil.which(cc_name, path=env.get("PATH"))
                if cc_path:
                    env["CC"] = cc_path
            return resolved, f"PATH ({candidate})"
    return None, "CXX and PATH search"


def _probe_compiler(compiler: str, env: dict[str, str]) -> tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory(prefix="ultimate-caption-compile-probe-") as temp_dir:
            source = Path(temp_dir) / "probe.cpp"
            source.write_text("int ultimate_caption_compile_probe() { return 0; }\n", encoding="ascii")
            if os.name == "nt":
                command = [compiler, "/nologo", "/c", str(source), f"/Fo{Path(temp_dir) / 'probe.obj'}"]
            else:
                command = [compiler, "-c", str(source), "-o", str(Path(temp_dir) / "probe.o")]
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env,
                timeout=60,
                check=False,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = completed.stdout.strip()
    return completed.returncode == 0, output[-500:]


def inspect_compile_environment(force: bool = False) -> CompileEnvironmentReport:
    global _REPORT_CACHE
    with _REPORT_LOCK:
        if _REPORT_CACHE is not None and not force:
            return _REPORT_CACHE

        details: list[str] = []
        try:
            import torch
        except ImportError as exc:
            report = CompileEnvironmentReport(False, sys.platform, "missing", "missing", "missing", "", "", False, "", (str(exc),))
            _REPORT_CACHE = report
            return report

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "unavailable")
        if not callable(getattr(torch, "compile", None)):
            details.append("this PyTorch build does not expose torch.compile")
        if not torch.cuda.is_available():
            details.append("CUDA is unavailable")
        if importlib.util.find_spec("triton") is None:
            details.append("Triton is not installed")
        triton_version = _package_version("triton", "triton-windows")

        env = os.environ.copy()
        compiler = ""
        compiler_source = ""
        if os.name == "nt":
            compiler = shutil.which("cl.exe", path=env.get("PATH", "")) or ""
            if compiler and _has_openmp_header(env):
                compiler_source = "existing PATH and INCLUDE"
            else:
                delta, compiler_source = _bootstrap_visual_studio_env(env)
                if delta:
                    env.update(delta)
                    os.environ.update(delta)
                compiler = shutil.which("cl.exe", path=env.get("PATH", "")) or ""
            if not compiler:
                details.append("MSVC cl.exe was not found")
            elif not _has_openmp_header(env):
                details.append("MSVC was found but omp.h is not available through INCLUDE")
        else:
            selected, compiler_source = _select_posix_compiler(env)
            compiler = selected or ""
            if compiler:
                os.environ.update({key: value for key, value in env.items() if key in {"CC", "CXX"}})
            else:
                details.append("no C++ compiler was found through CXX or PATH")

        compiler_probe_ok = False
        if compiler:
            compiler_probe_ok, probe_detail = _probe_compiler(compiler, env)
            if not compiler_probe_ok:
                details.append(f"C++ compiler probe failed: {probe_detail or 'unknown error'}")

        ninja = shutil.which("ninja", path=env.get("PATH", "")) or ""
        if not ninja and importlib.util.find_spec("ninja") is None:
            details.append("Ninja is not installed")

        report = CompileEnvironmentReport(
            available=not details,
            platform=sys.platform,
            torch_version=torch_version,
            cuda_version=cuda_version,
            triton_version=triton_version,
            compiler=compiler,
            compiler_source=compiler_source,
            compiler_probe_ok=compiler_probe_ok,
            ninja=ninja or "Python ninja package",
            details=tuple(details),
        )
        _REPORT_CACHE = report
        return report


def require_compile_environment() -> CompileEnvironmentReport:
    report = inspect_compile_environment()
    if not report.available:
        raise RuntimeError(report.summary())
    return report


def compile_enabled(settings: dict[str, Any] | None) -> bool:
    return bool((settings or {}).get("torch_compile", False))


def compile_dynamic_value(settings: dict[str, Any]) -> bool | None:
    value = str(settings.get("compile_dynamic", "false") or "false").strip().lower()
    if value in {"true", "1", "yes", "dynamic"}:
        return True
    if value in {"false", "0", "no", "static"}:
        return False
    return None


def compile_settings_key(settings: dict[str, Any] | None) -> tuple[Any, ...]:
    values = {**DEFAULT_COMPILE_SETTINGS, **(settings or {})}
    return (
        bool(values["torch_compile"]),
        str(values["compile_backend"]),
        str(values["compile_mode"]),
        str(values["compile_dynamic"]),
        bool(values["compile_fullgraph"]),
        int(values["compile_cache_size_limit"]),
        normalize_compile_threads(values["compile_threads"]),
    )


def _prepare_compile_worker_pool(
    settings: dict[str, Any],
    *,
    wait_until_ready: bool,
) -> tuple[CompileWorkerSettings, CompileWorkerWarmupResult | None]:
    global _WORKER_WARMUP_HANDLE, _WORKER_WARMUP_KEY, _WORKER_WARMUP_RESULT

    worker_settings = configure_compile_workers(settings.get("compile_threads"))
    key = (worker_settings.threads, worker_settings.worker_start)
    with _WORKER_WARMUP_LOCK:
        if _WORKER_WARMUP_KEY != key:
            _WORKER_WARMUP_KEY = key
            _WORKER_WARMUP_HANDLE = None
            _WORKER_WARMUP_RESULT = None
        if _WORKER_WARMUP_HANDLE is None and _WORKER_WARMUP_RESULT is None:
            _WORKER_WARMUP_HANDLE = start_compile_worker_warmup(worker_settings.threads)
            _LOGGER.info(
                "Starting TorchInductor worker pool: requested=%s start=%s",
                worker_settings.threads,
                worker_settings.worker_start,
            )
        if wait_until_ready and _WORKER_WARMUP_RESULT is None and _WORKER_WARMUP_HANDLE is not None:
            _WORKER_WARMUP_RESULT = finish_compile_worker_warmup(_WORKER_WARMUP_HANDLE)
            _WORKER_WARMUP_HANDLE = None
            log = _LOGGER.info if _WORKER_WARMUP_RESULT.ready else _LOGGER.warning
            log(
                "TorchInductor worker pool ready=%s active=%s/%s start=%s warmup=%.2fs detail=%s",
                _WORKER_WARMUP_RESULT.ready,
                _WORKER_WARMUP_RESULT.active_workers,
                _WORKER_WARMUP_RESULT.requested_threads,
                _WORKER_WARMUP_RESULT.worker_start,
                _WORKER_WARMUP_RESULT.elapsed_seconds,
                _WORKER_WARMUP_RESULT.detail,
            )
        return worker_settings, _WORKER_WARMUP_RESULT


def prepare_torch_compile(
    settings: dict[str, Any],
    *,
    wait_for_workers: bool = False,
) -> CompileEnvironmentReport | None:
    if not compile_enabled(settings):
        return None
    report = require_compile_environment()
    import torch

    cache_limit = max(1, int(settings.get("compile_cache_size_limit", 32) or 32))
    torch._dynamo.config.cache_size_limit = cache_limit
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(cache_limit * 8, cache_limit)
    try:
        torch._inductor.config.fx_graph_cache = True
    except (AttributeError, RuntimeError):
        pass
    # Caption generation is token-sensitive: a tiny BF16 fusion difference can
    # change an argmax and branch the rest of a caption. Preserve eager-style
    # rounding across fused low-precision kernels before compiling the graph.
    try:
        torch._inductor.config.emulate_precision_casts = True
        torch._inductor.config.eager_numerics.division_rounding = True
        torch._inductor.config.eager_numerics.use_pytorch_libdevice = True
    except (AttributeError, RuntimeError):
        pass
    if str(settings.get("compile_backend") or "inductor") == "inductor":
        _prepare_compile_worker_pool(settings, wait_until_ready=wait_for_workers)
    return report


def generation_compile_kwargs(settings: dict[str, Any], model: Any) -> dict[str, Any]:
    if not compile_enabled(settings):
        return {}
    prepare_torch_compile(settings, wait_for_workers=True)
    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is not None and not bool(getattr(quantizer, "is_compileable", False)):
        raise RuntimeError(
            "The selected quantization backend does not support torch.compile. "
            "Use BF16/FP16 or disable torch.compile."
        )

    from transformers import CompileConfig

    compile_config = CompileConfig(
        backend=str(settings.get("compile_backend") or DEFAULT_COMPILE_SETTINGS["compile_backend"]),
        mode=str(settings.get("compile_mode") or DEFAULT_COMPILE_SETTINGS["compile_mode"]),
        dynamic=compile_dynamic_value(settings),
        fullgraph=bool(settings.get("compile_fullgraph", False)),
    )
    return {"cache_implementation": "static", "compile_config": compile_config}


def compile_status_text(settings: dict[str, Any] | None) -> str:
    if not compile_enabled(settings):
        return "torch.compile disabled"
    values = {**DEFAULT_COMPILE_SETTINGS, **(settings or {})}
    worker_detail = (
        f", workers={normalize_compile_threads(values['compile_threads'])}"
        if str(values["compile_backend"]) == "inductor"
        else ""
    )
    return (
        "torch.compile enabled "
        f"({values['compile_backend']}, {values['compile_mode']}, dynamic={values['compile_dynamic']}, "
        f"fullgraph={bool(values['compile_fullgraph'])}, cache={int(values['compile_cache_size_limit'])}"
        f"{worker_detail})"
    )
