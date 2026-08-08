#!/usr/bin/env python3
"""Launch Chromium after removing agent-browser's automatic sandbox bypass."""

import os
from pathlib import Path
import sys
from typing import NoReturn


_REAL_CHROMIUM_ENV = "HERMES_CHROMIUM_EXECUTABLE"
_BYPASS_FLAGS = frozenset({
    "--no-sandbox",
    "--no-zygote-sandbox",
    "--disable-setuid-sandbox",
    "--disable-namespace-sandbox",
    "--disable-gpu-sandbox",
    "--disable-seccomp-filter-sandbox",
    "--single-process",
    "--in-process-gpu",
})
_BYPASS_FLAG_ENV_VARS = frozenset({
    "CHROMIUM_FLAGS",
    "CHROME_FLAGS",
    "CHROME_USER_FLAGS",
    "CHROMIUM_USER_FLAGS",
})


def _fail(message: str) -> NoReturn:
    print(f"Hermes sandboxed Chromium launch failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        _fail("refusing to launch as root")

    configured = os.environ.get(_REAL_CHROMIUM_ENV, "").strip()
    if not configured:
        _fail(f"{_REAL_CHROMIUM_ENV} is missing")

    try:
        executable = Path(configured).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail(f"real Chromium executable cannot be resolved: {exc}")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        _fail(f"real Chromium executable is not executable: {executable}")

    filtered_args = [
        arg for arg in sys.argv[1:]
        if arg.split("=", 1)[0] not in _BYPASS_FLAGS
    ]
    child_env = os.environ.copy()
    child_env.pop(_REAL_CHROMIUM_ENV, None)
    for name in _BYPASS_FLAG_ENV_VARS:
        child_env.pop(name, None)
    try:
        os.execve(str(executable), [str(executable), *filtered_args], child_env)
    except OSError as exc:
        _fail(f"could not execute {executable}: {exc}")


if __name__ == "__main__":
    main()
