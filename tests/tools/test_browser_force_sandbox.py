"""Regression coverage for the explicit Chromium sandbox opt-in (#81540)."""

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from tools import _sandboxed_chromium as sandboxed_chromium
from tools import browser_tool as bt


_LAUNCHER_FLAG_ENV_VARS = frozenset({
    "CHROMIUM_FLAGS",
    "CHROME_FLAGS",
    "CHROME_USER_FLAGS",
    "CHROMIUM_USER_FLAGS",
})


def _make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_force_sandbox_suppresses_hermes_docker_bypass(monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_FORCE_SANDBOX", "1")
    monkeypatch.setattr(bt, "_running_in_docker", lambda: True)
    monkeypatch.setattr(bt.os, "geteuid", lambda: 10000)

    assert bt._needs_chromium_sandbox_bypass() is False


def test_force_sandbox_config_true_is_authoritative():
    with patch(
        "hermes_cli.config.read_raw_config",
        return_value={"browser": {"force_sandbox": True}},
    ):
        assert (
            bt._force_chromium_sandbox({"AGENT_BROWSER_FORCE_SANDBOX": "0"})
            is True
        )


def test_force_sandbox_config_false_overrides_env_true():
    with patch(
        "hermes_cli.config.read_raw_config",
        return_value={"browser": {"force_sandbox": False}},
    ):
        assert (
            bt._force_chromium_sandbox({"AGENT_BROWSER_FORCE_SANDBOX": "1"})
            is False
        )


@pytest.mark.parametrize(
    ("user_value", "managed_value", "expected"),
    [
        pytest.param(False, True, True, id="managed-true"),
        pytest.param(True, False, False, id="managed-false"),
    ],
)
def test_force_sandbox_managed_config_overrides_user_config(
    monkeypatch, tmp_path, user_value, managed_value, expected
):
    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        f"browser:\n  force_sandbox: {str(user_value).lower()}\n",
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(
        f"browser:\n  force_sandbox: {str(managed_value).lower()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    from hermes_cli import config, managed_scope

    config._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    assert bt._force_chromium_sandbox({}) is expected


def test_force_sandbox_env_fallback_when_config_key_is_absent():
    with patch(
        "hermes_cli.config.read_raw_config",
        return_value={"browser": {"headed": False}},
    ):
        assert (
            bt._force_chromium_sandbox({"AGENT_BROWSER_FORCE_SANDBOX": "1"})
            is True
        )


def test_force_sandbox_has_canonical_false_default():
    assert bt.DEFAULT_CONFIG["browser"]["force_sandbox"] is False


def test_force_sandbox_from_browser_env_suppresses_hermes_docker_bypass(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_FORCE_SANDBOX", raising=False)
    monkeypatch.setattr(bt, "_running_in_docker", lambda: True)
    monkeypatch.setattr(bt.os, "geteuid", lambda: 10000)

    assert (
        bt._needs_chromium_sandbox_bypass({"AGENT_BROWSER_FORCE_SANDBOX": "1"}) is False
    )


def test_force_sandbox_from_browser_env_changes_timeout_guidance(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_FORCE_SANDBOX", raising=False)

    error = bt._format_browser_timeout_error(
        "open",
        120,
        "",
        "Chromium sandbox failed",
        {"AGENT_BROWSER_FORCE_SANDBOX": "1"},
    )

    assert "browser.force_sandbox=true" in error
    assert "AGENT_BROWSER_ARGS='--no-sandbox" not in error


def test_default_docker_bypass_is_preserved(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_FORCE_SANDBOX", raising=False)
    monkeypatch.setattr(bt, "_running_in_docker", lambda: True)
    monkeypatch.setattr(bt.os, "geteuid", lambda: 10000)

    assert bt._needs_chromium_sandbox_bypass() is True


@pytest.mark.parametrize(
    "unsafe_arg",
    [
        "--no-sandbox",
        "--no-zygote-sandbox",
        "--disable-setuid-sandbox",
        "--disable-namespace-sandbox",
        "--disable-gpu-sandbox",
        "--disable-seccomp-filter-sandbox",
        "--single-process",
        "--in-process-gpu",
    ],
)
def test_forced_sandbox_error_removes_bypass_suggestion(unsafe_arg):
    error = bt._format_forced_sandbox_launch_error(
        f"Chrome sandbox failed\nHint: try passing --args '{unsafe_arg}'"
    )

    assert "Chrome sandbox failed" in error
    assert "AppArmor/seccomp" in error
    assert "did not retry" in error
    assert unsafe_arg not in error


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("Hint: try --no-sandbox=true", True),
        ("Hint: try '--single-process'", True),
        ("Navigation failed: net::ERR_NAME_NOT_RESOLVED", False),
        ("Unknown option --no-sandboxed", False),
    ],
)
def test_unsafe_sandbox_guidance_requires_an_exact_flag_token(error, expected):
    assert bt._contains_unsafe_sandbox_guidance(error) is expected


def test_preflight_and_wrapper_reject_the_same_sandbox_bypasses():
    assert bt._UNSAFE_CHROMIUM_SANDBOX_ARGS == sandboxed_chromium._BYPASS_FLAGS
    assert sandboxed_chromium._BYPASS_FLAG_ENV_VARS == _LAUNCHER_FLAG_ENV_VARS


def test_prepare_force_sandbox_wraps_real_chromium(monkeypatch, tmp_path):
    real_chromium = _make_executable(tmp_path / "chromium")
    browser_env = {
        "AGENT_BROWSER_FORCE_SANDBOX": "true",
        "AGENT_BROWSER_EXECUTABLE_PATH": str(real_chromium),
        "PATH": os.environ.get("PATH", ""),
    }
    monkeypatch.setattr(bt.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(bt.sys, "platform", "linux")

    error = bt._prepare_forced_chromium_sandbox(browser_env)

    assert error is None
    assert browser_env["HERMES_CHROMIUM_EXECUTABLE"] == str(real_chromium.resolve())
    wrapper = Path(browser_env["AGENT_BROWSER_EXECUTABLE_PATH"])
    assert wrapper.name == "_sandboxed_chromium.py"
    assert wrapper.is_file()


@pytest.mark.parametrize(
    "unsafe_arg",
    [
        "--no-sandbox",
        "--no-zygote-sandbox",
        "--disable-setuid-sandbox",
        "--disable-namespace-sandbox",
        "--disable-gpu-sandbox=true",
        "--disable-seccomp-filter-sandbox",
        "--single-process",
        "--in-process-gpu=true",
    ],
)
def test_prepare_force_sandbox_rejects_conflicting_user_args(
    monkeypatch, tmp_path, unsafe_arg
):
    real_chromium = _make_executable(tmp_path / "chromium")
    browser_env = {
        "AGENT_BROWSER_FORCE_SANDBOX": "1",
        "AGENT_BROWSER_EXECUTABLE_PATH": str(real_chromium),
        "AGENT_BROWSER_ARGS": f"--disable-dev-shm-usage,{unsafe_arg}",
    }
    monkeypatch.setattr(bt.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(bt.sys, "platform", "linux")

    error = bt._prepare_forced_chromium_sandbox(browser_env)

    assert error is not None
    assert unsafe_arg.split("=", 1)[0] in error
    assert "AGENT_BROWSER_ARGS" in error
    assert browser_env["AGENT_BROWSER_EXECUTABLE_PATH"] == str(real_chromium)


@pytest.mark.parametrize(
    "unsafe_arg",
    ["--single-process=true", "--in-process-gpu"],
)
def test_prepare_force_sandbox_names_legacy_conflicting_arg_source(
    monkeypatch, tmp_path, unsafe_arg
):
    real_chromium = _make_executable(tmp_path / "chromium")
    browser_env = {
        "AGENT_BROWSER_FORCE_SANDBOX": "1",
        "AGENT_BROWSER_EXECUTABLE_PATH": str(real_chromium),
        "AGENT_BROWSER_CHROME_FLAGS": unsafe_arg,
    }
    monkeypatch.setattr(bt.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(bt.sys, "platform", "linux")

    error = bt._prepare_forced_chromium_sandbox(browser_env)

    assert error is not None
    assert unsafe_arg.split("=", 1)[0] in error
    assert "AGENT_BROWSER_CHROME_FLAGS" in error
    assert "in AGENT_BROWSER_ARGS" not in error


def test_prepare_force_sandbox_fails_closed_as_root(monkeypatch, tmp_path):
    real_chromium = _make_executable(tmp_path / "chromium")
    browser_env = {
        "AGENT_BROWSER_FORCE_SANDBOX": "1",
        "AGENT_BROWSER_EXECUTABLE_PATH": str(real_chromium),
    }
    monkeypatch.setattr(bt.os, "geteuid", lambda: 0)
    monkeypatch.setattr(bt.sys, "platform", "linux")

    error = bt._prepare_forced_chromium_sandbox(browser_env)

    assert error is not None
    assert "non-root" in error
    assert browser_env["AGENT_BROWSER_EXECUTABLE_PATH"] == str(real_chromium)


def test_prepare_force_sandbox_fails_closed_without_chromium(monkeypatch):
    browser_env = {"AGENT_BROWSER_FORCE_SANDBOX": "1", "PATH": ""}
    monkeypatch.setattr(bt.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(bt.sys, "platform", "linux")
    monkeypatch.setattr(bt, "_resolve_chromium_executable", lambda _env: None)

    error = bt._prepare_forced_chromium_sandbox(browser_env)

    assert error is not None
    assert "executable" in error.lower()
    assert "AGENT_BROWSER_EXECUTABLE_PATH" in error


def test_resolver_finds_official_docker_playwright_headless_shell(
    monkeypatch, tmp_path
):
    browser_root = tmp_path / "playwright"
    chromium = _make_executable(
        browser_root
        / "chromium_headless_shell-1234"
        / "chrome-linux"
        / "headless_shell"
    )
    monkeypatch.setattr(bt.shutil, "which", lambda *_args, **_kwargs: None)

    resolved = bt._resolve_chromium_executable({
        "HOME": str(tmp_path / "home"),
        "PATH": "",
        "PLAYWRIGHT_BROWSERS_PATH": str(browser_root),
    })

    assert resolved == chromium.resolve()


@pytest.mark.skipif(os.name != "posix", reason="wrapper test uses a POSIX shell stub")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="sandbox wrapper intentionally refuses root",
)
def test_wrapper_removes_agent_browser_sandbox_bypass(tmp_path):
    capture_path = tmp_path / "argv.txt"
    env_capture_path = tmp_path / "launcher-env.txt"
    real_chromium = _make_executable(
        tmp_path / "chromium",
        "#!/bin/sh\n"
        "env | grep -E '^(CHROMIUM_FLAGS|CHROME_FLAGS|CHROME_USER_FLAGS|"
        "CHROMIUM_USER_FLAGS)=' > \"$HERMES_CHROMIUM_ENV_CAPTURE\"\n"
        "printf '%s\\n' \"$@\" > \"$HERMES_CHROMIUM_ARGV_CAPTURE\"\n",
    )
    env = os.environ.copy()
    env["HERMES_CHROMIUM_EXECUTABLE"] = str(real_chromium)
    env["HERMES_CHROMIUM_ARGV_CAPTURE"] = str(capture_path)
    env["HERMES_CHROMIUM_ENV_CAPTURE"] = str(env_capture_path)
    for name in _LAUNCHER_FLAG_ENV_VARS:
        env[name] = "--no-sandbox --single-process"
    wrapper = Path(bt.__file__).with_name("_sandboxed_chromium.py")

    proc = subprocess.run(
        [
            sys.executable,
            str(wrapper),
            "--headless=new",
            "--no-sandbox=true",
            "--no-zygote-sandbox",
            "--disable-setuid-sandbox=1",
            "--disable-namespace-sandbox",
            "--disable-gpu-sandbox=enabled",
            "--disable-seccomp-filter-sandbox",
            "--single-process",
            "--single-process=1",
            "--in-process-gpu",
            "--in-process-gpu=true",
            "--disable-dev-shm-usage",
            "--disable-gpu=false",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    captured_args = capture_path.read_text(encoding="utf-8").splitlines()
    assert captured_args == [
        "--headless=new",
        "--disable-dev-shm-usage",
        "--disable-gpu=false",
    ]
    assert env_capture_path.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("raw_config", "legacy_force"),
    [
        pytest.param({"browser": {"force_sandbox": True}}, None, id="config"),
        pytest.param({}, "1", id="legacy-env"),
    ],
)
def test_local_command_routes_agent_browser_through_wrapper(
    monkeypatch, tmp_path, raw_config, legacy_force
):
    real_chromium = _make_executable(tmp_path / "chromium")
    captured_env = {}
    fake_proc = MagicMock(returncode=0)
    fake_proc.wait.return_value = 0

    def _capture_popen(_cmd, **kwargs):
        captured_env.update(kwargs["env"])
        return fake_proc

    browser_env = {
        "AGENT_BROWSER_EXECUTABLE_PATH": str(real_chromium),
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
    }
    if legacy_force is not None:
        browser_env["AGENT_BROWSER_FORCE_SANDBOX"] = legacy_force

    fake_session = {"session_name": "force-sandbox", "cdp_url": None}
    with (
        patch("hermes_cli.config.read_raw_config", return_value=raw_config),
        patch.object(bt, "_find_agent_browser", return_value="/usr/bin/agent-browser"),
        patch.object(bt, "_chromium_installed", return_value=True),
        patch.object(bt, "_get_session_info", return_value=fake_session),
        patch.object(bt, "_get_browser_engine", return_value="auto"),
        patch.object(
            bt,
            "_build_browser_env",
            return_value=browser_env,
        ),
        patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmp_path)),
        patch.object(bt, "_write_owner_pid"),
        patch.object(bt.os, "geteuid", return_value=10000),
        patch.object(bt.subprocess, "Popen", side_effect=_capture_popen),
        patch.object(bt.os, "open", return_value=99),
        patch.object(bt.os, "close"),
        patch("tools.interrupt.is_interrupted", return_value=False),
        patch("builtins.open", mock_open(read_data='{"success": true}')),
        patch.dict(
            os.environ,
            {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
            clear=True,
        ),
    ):
        result = bt._run_browser_command("task-81540", "open", ["about:blank"])

    assert result["success"] is True
    assert captured_env["HERMES_CHROMIUM_EXECUTABLE"] == str(real_chromium.resolve())
    assert Path(captured_env["AGENT_BROWSER_EXECUTABLE_PATH"]).name == (
        "_sandboxed_chromium.py"
    )
    assert "AGENT_BROWSER_ARGS" not in captured_env


def test_lightpanda_chrome_fallback_routes_every_command_through_wrapper(
    monkeypatch, tmp_path
):
    real_chromium = _make_executable(tmp_path / "chromium")
    captured_envs = []
    fake_proc = MagicMock(returncode=0)
    fake_proc.wait.return_value = 0

    def _capture_popen(_cmd, **kwargs):
        captured_envs.append(dict(kwargs["env"]))
        return fake_proc

    with (
        patch.object(
            bt,
            "_run_browser_command",
            return_value={
                "success": True,
                "data": {"result": '"https://example.com/"'},
            },
        ),
        patch.object(bt, "_find_agent_browser", return_value="/usr/bin/agent-browser"),
        patch.object(bt, "_chromium_installed", return_value=True),
        patch.object(
            bt,
            "_build_browser_env",
            return_value={
                "AGENT_BROWSER_FORCE_SANDBOX": "1",
                "AGENT_BROWSER_EXECUTABLE_PATH": str(real_chromium),
                "HOME": str(tmp_path),
                "PATH": "/usr/bin:/bin",
            },
        ),
        patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmp_path)),
        patch.object(bt.os, "geteuid", return_value=10000),
        patch.object(bt.subprocess, "Popen", side_effect=_capture_popen),
        patch.object(bt.os, "open", return_value=99),
        patch.object(bt.os, "close"),
        patch("builtins.open", mock_open(read_data='{"success": true}')),
    ):
        result = bt._run_chrome_fallback_command(
            "task-81540", "snapshot", [], timeout=30
        )

    assert result["success"] is True
    assert len(captured_envs) == 3  # open, requested command, close
    for browser_env in captured_envs:
        assert browser_env["HERMES_CHROMIUM_EXECUTABLE"] == str(
            real_chromium.resolve()
        )
        wrapper = Path(browser_env["AGENT_BROWSER_EXECUTABLE_PATH"])
        assert wrapper.name == "_sandboxed_chromium.py"
        assert wrapper != real_chromium.resolve()


def test_lightpanda_chrome_fallback_fails_closed_before_launch(monkeypatch, tmp_path):
    real_chromium = _make_executable(tmp_path / "chromium")

    with (
        patch.object(
            bt,
            "_run_browser_command",
            return_value={
                "success": True,
                "data": {"result": '"https://example.com/"'},
            },
        ),
        patch.object(bt, "_find_agent_browser", return_value="/usr/bin/agent-browser"),
        patch.object(bt, "_chromium_installed", return_value=True),
        patch.object(
            bt,
            "_build_browser_env",
            return_value={
                "AGENT_BROWSER_FORCE_SANDBOX": "1",
                "AGENT_BROWSER_EXECUTABLE_PATH": str(real_chromium),
                "HOME": str(tmp_path),
                "PATH": "/usr/bin:/bin",
            },
        ),
        patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmp_path)),
        patch.object(bt.os, "geteuid", return_value=0),
        patch.object(bt.subprocess, "Popen") as popen,
    ):
        result = bt._run_chrome_fallback_command(
            "task-81540", "snapshot", [], timeout=30
        )

    assert result["success"] is False
    assert "browser.force_sandbox=true" in result["error"]
    assert "non-root" in result["error"]
    popen.assert_not_called()
