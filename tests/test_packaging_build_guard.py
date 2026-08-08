"""Behavioral regression coverage for the wheel/sdist distribution guard."""

import os
import stat
import subprocess
import sys
from pathlib import Path
import zipfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_artifact(kind: str, tmp_path, *, nix_build: bool) -> subprocess.CompletedProcess[str]:
    """Invoke a real PEP 517 build hook as a subprocess.

    The wheel and sdist guards live in SEPARATE cmdclass entries in setup.py
    (the bdist_wheel one behind a try/except ImportError), so each hook needs
    its own regression coverage — a passing sdist test proves nothing about
    the wheel path. The same harness exercises build_editable without the Nix
    marker so development installs stay supported.
    """
    env = os.environ.copy()
    # nix develop exports this too, so it must not grant permission to build
    # a distributable artifact.
    env["NIX_BUILD_TOP"] = "/build/devshell"
    if nix_build:
        env["HERMES_NIX_BUILD"] = "1"
    else:
        env.pop("HERMES_NIX_BUILD", None)
    # Redirect setuptools' scratch dirs (build/, *.egg-info) into tmp_path so
    # the allowed-marker build doesn't litter the real worktree.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    extra_cfg = tmp_path / "dist-extra.cfg"
    extra_cfg.write_text(
        f"[build]\nbuild_base = {scratch / 'build'}\n\n[egg_info]\negg_base = {scratch}\n",
        encoding="utf-8",
    )
    env["DIST_EXTRA_CONFIG"] = str(extra_cfg)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_{kind}; build_{kind}(r'{out}')".format(
                kind=kind, out=tmp_path
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_editable_build_skips_copied_wrapper_mode_guard(tmp_path):
    """PEP 660 has no copied build_lib wrapper to chmod or require."""
    build_meta = pytest.importorskip("setuptools.build_meta")
    if not hasattr(build_meta, "build_editable"):
        pytest.skip("installed setuptools does not expose PEP 660 build_editable")

    result = _build_artifact("editable", tmp_path, nix_build=False)

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob("hermes_agent-*.editable-*.whl"))
    output = result.stdout + result.stderr
    assert "built Chromium sandbox wrapper is missing" not in output
    assert "Customization incompatible with editable install" not in output


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
def test_artifact_build_rejects_nix_development_shell_environment(kind, tmp_path):
    result = _build_artifact(kind, tmp_path, nix_build=False)

    assert result.returncode != 0
    assert "Building wheels or sdists for hermes-agent is not supported" in result.stderr


@pytest.mark.parametrize(
    ("kind", "artifact_glob"),
    [("sdist", "hermes_agent-*.tar.gz"), ("wheel", "hermes_agent-*.whl")],
)
def test_artifact_build_allows_explicit_nix_package_build_marker(kind, artifact_glob, tmp_path):
    result = _build_artifact(kind, tmp_path, nix_build=True)

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob(artifact_glob))


@pytest.mark.skipif(os.name != "posix", reason="supported Nix wheel path is POSIX-only")
def test_nix_wheel_installs_executable_browser_wrapper(tmp_path):
    """The supported wheel path preserves and directly runs the native shim."""
    pytest.importorskip("setuptools.build_meta")
    pytest.importorskip("wheel")
    pytest.importorskip("pip")

    result = _build_artifact("wheel", tmp_path, nix_build=True)

    assert result.returncode == 0, result.stderr
    wheel_path = next(tmp_path.glob("hermes_agent-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel_file:
        wrapper_info = wheel_file.getinfo("tools/_sandboxed_chromium.py")
        packaged_mode = (wrapper_info.external_attr >> 16) & 0o777
    assert packaged_mode == 0o755

    install_root = tmp_path / "installed"
    install_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--ignore-requires-python",
            "--target",
            str(install_root),
            str(wheel_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert install_result.returncode == 0, install_result.stderr

    wrapper = install_root / "tools" / "_sandboxed_chromium.py"
    assert wrapper.stat().st_mode & stat.S_IXUSR
    assert os.access(wrapper, os.X_OK)

    capture_path = tmp_path / "packaged-argv.txt"
    env_capture_path = tmp_path / "packaged-launcher-env.txt"
    real_chromium = tmp_path / "chromium-stub"
    real_chromium.write_text(
        "#!/bin/sh\n"
        "env | grep -E '^(CHROMIUM_FLAGS|CHROME_FLAGS|CHROME_USER_FLAGS|"
        "CHROMIUM_USER_FLAGS)=' > \"$HERMES_CHROMIUM_ENV_CAPTURE\"\n"
        "printf '%s\\n' \"$@\" > \"$HERMES_CHROMIUM_ARGV_CAPTURE\"\n",
        encoding="utf-8",
    )
    real_chromium.chmod(0o755)
    env = os.environ.copy()
    env["HERMES_CHROMIUM_EXECUTABLE"] = str(real_chromium)
    env["HERMES_CHROMIUM_ARGV_CAPTURE"] = str(capture_path)
    env["HERMES_CHROMIUM_ENV_CAPTURE"] = str(env_capture_path)
    for name in (
        "CHROMIUM_FLAGS",
        "CHROME_FLAGS",
        "CHROME_USER_FLAGS",
        "CHROMIUM_USER_FLAGS",
    ):
        env[name] = "--no-sandbox --single-process"

    invocation = subprocess.run(
        [
            str(wrapper),
            "--headless=new",
            "--no-sandbox",
            "--single-process=true",
            "--in-process-gpu",
            "--disable-dev-shm-usage",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert invocation.returncode == 0, invocation.stderr
    assert capture_path.read_text(encoding="utf-8").splitlines() == [
        "--headless=new",
        "--disable-dev-shm-usage",
    ]
    assert env_capture_path.read_text(encoding="utf-8") == ""
