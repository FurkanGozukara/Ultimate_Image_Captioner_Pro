from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Sequence

from .common import BASE_DIR, log_event


_ACTIVE_WORKERS: dict[int, tuple[str, subprocess.Popen[str]]] = {}
_CANCELLED_PIDS: set[int] = set()
_WORKER_LOCK = threading.Lock()


def _stream_worker_output(pipe: Any, tail: list[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line.rstrip())
            if len(tail) > 80:
                del tail[:-80]
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _register_worker(command: str, proc: subprocess.Popen[str]) -> None:
    with _WORKER_LOCK:
        _ACTIVE_WORKERS[proc.pid] = (command, proc)


def _finish_worker(proc: subprocess.Popen[str]) -> bool:
    with _WORKER_LOCK:
        _ACTIVE_WORKERS.pop(proc.pid, None)
        was_cancelled = proc.pid in _CANCELLED_PIDS
        _CANCELLED_PIDS.discard(proc.pid)
        return was_cancelled


def cancel_active_workers(commands: Sequence[str] | None = None) -> tuple[int, str]:
    command_set = set(commands or [])
    with _WORKER_LOCK:
        targets = [
            (pid, command, proc)
            for pid, (command, proc) in _ACTIVE_WORKERS.items()
            if not command_set or command in command_set
        ]
        for pid, _command, _proc in targets:
            _CANCELLED_PIDS.add(pid)

    if not targets:
        return 0, "No matching subprocess worker is active."

    for pid, command, proc in targets:
        log_event(f"Terminating subprocess worker: {command} pid={pid}", "Subprocess")
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            else:
                proc.terminate()
        except Exception as exc:
            log_event(f"Could not terminate pid={pid}: {exc}", "Subprocess")

    target_text = ", ".join(f"{command} pid={pid}" for pid, command, _proc in targets)
    return len(targets), f"Cancellation requested for subprocess worker(s): {target_text}"


def run_worker(command: str, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
    payload_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    result_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    payload_path = Path(payload_file.name)
    result_path = Path(result_file.name)
    try:
        json.dump(payload, payload_file, ensure_ascii=False)
        payload_file.close()
        result_file.close()

        log_event(f"Starting subprocess worker: {command}", "Subprocess")
        proc = subprocess.Popen(
            [sys.executable, "-m", "joycaption.worker", command, str(payload_path), str(result_path)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        _register_worker(command, proc)
        tail: list[str] = []
        reader = threading.Thread(target=_stream_worker_output, args=(proc.stdout, tail), daemon=True)
        reader.start()
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with _WORKER_LOCK:
                _CANCELLED_PIDS.add(proc.pid)
            proc.kill()
            reader.join(timeout=5)
            _finish_worker(proc)
            raise RuntimeError(f"Worker timed out while running {command}.")
        reader.join(timeout=5)
        was_cancelled = _finish_worker(proc)
        log_event(f"Subprocess worker finished: {command} (exit code {returncode})", "Subprocess")

        result: dict[str, Any] | None = None
        if result_path.exists() and result_path.stat().st_size > 0:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result = None

        if returncode != 0:
            if was_cancelled:
                raise RuntimeError("Worker was cancelled.")
            tail_text = "\n".join(tail[-30:])
            message = tail_text or f"Worker exited with code {returncode}"
            if result and result.get("error"):
                message = str(result.get("error"))
            raise RuntimeError(message)

        if not result:
            tail_text = "\n".join(tail[-30:])
            raise RuntimeError(f"Worker did not write a result file. {tail_text}")
        if not result.get("ok", False):
            raise RuntimeError(str(result.get("error") or "Worker failed."))
        return result
    finally:
        for path in (payload_path, result_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
