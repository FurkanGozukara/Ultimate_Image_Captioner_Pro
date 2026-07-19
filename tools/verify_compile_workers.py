from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Start and verify the real TorchInductor compile worker pool.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    from joycaption.torch_compile import inspect_compile_environment
    from joycaption.torch_compile_workers import finish_compile_worker_warmup, start_compile_worker_warmup

    environment = inspect_compile_environment(force=True)
    if not environment.available:
        raise RuntimeError(environment.summary())
    handle = start_compile_worker_warmup(args.threads)
    result = finish_compile_worker_warmup(handle, timeout_seconds=args.timeout)
    print(json.dumps({"environment": environment.as_dict(), "workers": asdict(result)}, indent=2))
    if not result.ready or result.active_workers < result.requested_threads:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
