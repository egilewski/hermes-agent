"""Regression coverage for GHSA-5qr3-c538-wm9j (#29156) — Remote Code
Execution via the ``HERMES_ENABLE_PROJECT_PLUGINS`` bypass in the web
server's dashboard plugin loader.

Two primitives combined into the original advisory chain:

1. ``hermes_cli.web_server._discover_dashboard_plugins`` opted into
   the untrusted ``./.hermes/plugins/`` source via
   ``os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS")`` — truthy for
   any non-empty string, so ``=0`` / ``=false`` / ``=no`` (all of
   which the agent loader treats as off, and which operators set to
   *disable* project plugins) silently *enabled* the source.
2. ``hermes_cli.web_server._mount_plugin_api_routes`` then imported
   each plugin's manifest ``api`` field as a Python module via
   ``importlib.util.spec_from_file_location``.  The field was used
   raw, with no path-traversal check, so a single manifest line
   ``{"api": "/tmp/payload.py"}`` was enough to redirect the
   importer at any Python file on disk (``Path('safe') / '/abs'``
   resolves to ``/abs`` in Python).

These tests pin each layer of the new defence:

* Truthy env semantics now match the agent loader.
* ``_safe_plugin_api_relpath`` rejects absolute paths, ``..``
  traversal, and non-string / empty values.
* ``_mount_plugin_api_routes`` re-validates at import time and
  refuses project-source plugins outright.
* End-to-end the original PoC manifest no longer triggers
  ``importlib`` for ``/tmp/payload.py``.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import uvicorn

from hermes_cli import web_server


@pytest.fixture(autouse=True)
def _reset_plugin_cache(monkeypatch):
    """The plugin scanner caches its result per-process.  Bust the
    cache before *and* after each test so leakage between tests can't
    mask a regression — and so the production cache the import-time
    ``_mount_plugin_api_routes()`` populated doesn't bleed in."""
    web_server._dashboard_plugins_cache = None
    web_server._snapshot_insecure_user_plugin_api_opt_in(False)
    web_server._plugin_api_user_routes_mounted = False
    web_server._plugin_api_user_module_names.clear()
    web_server._plugin_api_user_prefixes.clear()
    yield
    web_server._dashboard_plugins_cache = None
    web_server._snapshot_insecure_user_plugin_api_opt_in(False)
    web_server._plugin_api_user_routes_mounted = False
    web_server._plugin_api_user_module_names.clear()
    web_server._plugin_api_user_prefixes.clear()


def _write_plugin_manifest(root: Path, name: str, manifest: dict) -> Path:
    """Drop a manifest under ``root/<name>/dashboard/manifest.json`` and
    return the dashboard dir path."""
    dashboard_dir = root / name / "dashboard"
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "manifest.json").write_text(json.dumps(manifest))
    return dashboard_dir


def _set_insecure_user_plugin_api_opt_in(monkeypatch, enabled) -> None:
    config = {
        "dashboard": {
            "allow_insecure_user_plugin_api": enabled,
        },
    }
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: config,
    )


def _configure_and_snapshot_public_insecure_mode(
    monkeypatch, enabled, public_insecure: bool = True
) -> None:
    _set_insecure_user_plugin_api_opt_in(monkeypatch, enabled)
    web_server._snapshot_insecure_user_plugin_api_opt_in(
        public_insecure=public_insecure
    )


def _plugin_module_names(plugin_name: str) -> list[str]:
    name = f"hermes_dashboard_plugin_{plugin_name}"
    return [name] if name in sys.modules else []


def _pop_plugin_modules(plugin_name: str) -> None:
    for name in _plugin_module_names(plugin_name):
        sys.modules.pop(name, None)


def _stub_uvicorn_run(monkeypatch) -> dict:
    """Stub uvicorn.Server so start_server performs setup without binding."""
    captured: dict = {}

    class _FakeConfig:
        loaded = True
        host = "127.0.0.1"
        port = 8000

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def load(self):
            pass

        class lifespan_class:
            should_exit = False
            state: dict = {}

            def __init__(self, *args, **kwargs):
                pass

            async def startup(self):
                pass

            async def shutdown(self):
                pass

    class _FakeServer:
        should_exit = False
        started = True
        servers: list = []
        lifespan = None

        @staticmethod
        def capture_signals():
            return contextlib.nullcontext()

        async def startup(self, sockets=None):
            pass

        async def main_loop(self):
            pass

        async def shutdown(self, sockets=None):
            pass

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", lambda config: _FakeServer())
    return captured


# ---------------------------------------------------------------------------
# Layer 1 — HERMES_ENABLE_PROJECT_PLUGINS env gate uses truthy semantics.
# ---------------------------------------------------------------------------


class TestProjectPluginsEnvGate:
    """Project plugins must only be discovered when the env var is set
    to a documented truthy value.  Pre-#29156 any non-empty string —
    including ``0`` / ``false`` / ``no`` — silently enabled the source."""

    @pytest.fixture
    def project_plugin(self, tmp_path, monkeypatch):
        """Plant a project-source plugin under CWD's ``.hermes/plugins``
        and isolate the user-plugins dir to an empty tmp tree."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        cwd = tmp_path / "evil-repo"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        _write_plugin_manifest(
            cwd / ".hermes" / "plugins",
            "evil",
            {
                "name": "evil",
                "label": "Evil",
                "entry": "dist/index.js",
            },
        )
        return cwd

    @pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off", "False"])
    def test_falsy_values_keep_project_plugins_disabled(
        self, project_plugin, monkeypatch, value
    ):
        if value == "":
            monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
        else:
            monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", value)

        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        names = {p["name"] for p in plugins}
        assert "evil" not in names, (
            f"HERMES_ENABLE_PROJECT_PLUGINS={value!r} must NOT enable the "
            "project source — that's the GHSA-5qr3-c538-wm9j env bypass."
        )

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "YES"])
    def test_truthy_values_enable_project_plugins(
        self, project_plugin, monkeypatch, value
    ):
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", value)
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        evil = next((p for p in plugins if p["name"] == "evil"), None)
        assert evil is not None
        assert evil["source"] == "project"


# ---------------------------------------------------------------------------
# Layer 2 — _safe_plugin_api_relpath rejects path-traversal payloads.
# ---------------------------------------------------------------------------


class TestApiPathSanitizer:
    """Unit-level coverage for the new ``_safe_plugin_api_relpath``
    helper.  Anything that escapes the plugin's dashboard directory
    must come back as ``None``."""

    def _dashboard_dir(self, tmp_path):
        d = tmp_path / "plug" / "dashboard"
        d.mkdir(parents=True)
        return d

    def test_simple_relative_path_accepted(self, tmp_path):
        d = self._dashboard_dir(tmp_path)
        (d / "api.py").write_text("router = None\n")
        assert web_server._safe_plugin_api_relpath("api.py", dashboard_dir=d) == "api.py"

    def test_nested_relative_path_accepted(self, tmp_path):
        d = self._dashboard_dir(tmp_path)
        (d / "backend").mkdir()
        (d / "backend" / "routes.py").write_text("router = None\n")
        out = web_server._safe_plugin_api_relpath(
            "backend/routes.py", dashboard_dir=d
        )
        assert out == "backend/routes.py"

    @pytest.mark.parametrize("payload", [
        "/etc/passwd",
        "/tmp/payload.py",
        "/usr/bin/python",
        # NT-style absolute on POSIX is a relative path — covered by traversal below.
    ])
    def test_absolute_path_rejected(self, tmp_path, payload):
        d = self._dashboard_dir(tmp_path)
        assert web_server._safe_plugin_api_relpath(payload, dashboard_dir=d) is None

    @pytest.mark.parametrize("payload", [
        "../../../etc/passwd",
        "../neighbour/api.py",
        "../../../../tmp/evil.py",
        "subdir/../../../../etc/passwd",
    ])
    def test_traversal_rejected(self, tmp_path, payload):
        d = self._dashboard_dir(tmp_path)
        assert web_server._safe_plugin_api_relpath(payload, dashboard_dir=d) is None

    @pytest.mark.parametrize("payload", [None, "", "   ", 42, [], {}])
    def test_non_string_or_empty_rejected(self, tmp_path, payload):
        d = self._dashboard_dir(tmp_path)
        assert web_server._safe_plugin_api_relpath(payload, dashboard_dir=d) is None


# ---------------------------------------------------------------------------
# Layer 3 — _discover_dashboard_plugins scrubs ``_api_file`` early.
# ---------------------------------------------------------------------------


class TestDiscoveryScrubsApiField:
    """The cached plugin entry must NEVER carry an unsanitised api path.
    A regression here would re-arm the RCE for any caller that uses
    ``plugin['_api_file']`` directly."""

    @pytest.fixture
    def user_plugin_factory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

        def _make(name: str, manifest: dict) -> None:
            _write_plugin_manifest(tmp_path / "plugins", name, manifest)

        return _make

    def test_absolute_api_path_in_manifest_is_scrubbed(self, user_plugin_factory):
        user_plugin_factory("evil", {
            "name": "evil",
            "label": "Evil",
            "api": "/tmp/payload.py",
            "entry": "dist/index.js",
        })
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        evil = next(p for p in plugins if p["name"] == "evil")
        assert evil["_api_file"] is None
        assert evil["has_api"] is False

    def test_traversal_api_path_in_manifest_is_scrubbed(self, user_plugin_factory):
        user_plugin_factory("traverse", {
            "name": "traverse",
            "label": "Traverse",
            "api": "../../../../tmp/evil.py",
            "entry": "dist/index.js",
        })
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "traverse")
        assert entry["_api_file"] is None
        assert entry["has_api"] is False

    def test_safe_api_path_survives(self, user_plugin_factory, tmp_path):
        user_plugin_factory("safe", {
            "name": "safe",
            "label": "Safe",
            "api": "api.py",
            "entry": "dist/index.js",
        })
        # Make the api file actually exist so a downstream mount could
        # in principle proceed — we're only testing the discovery scrub.
        (tmp_path / "plugins" / "safe" / "dashboard" / "api.py").write_text(
            "router = None\n"
        )
        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "safe")
        assert entry["_api_file"] == "api.py"
        assert entry["has_api"] is True


class TestPublicInsecureDashboardMode:
    """Public-insecure mode controls when user plugin APIs are suppressed."""

    def test_configured_public_insecure_mode_starts_disabled(self):
        web_server._snapshot_insecure_user_plugin_api_opt_in(False)

        assert web_server._public_insecure_dashboard_mode() is False

    def test_configured_public_insecure_mode_can_be_enabled(self, monkeypatch):
        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)

        assert web_server._public_insecure_dashboard_mode() is True

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "localhost.",
            "LOCALHOST",
            "LOCALHOST.",
            "ip6-localhost",
            "127.0.0.1",
            "[127.0.0.1.]",
            "127.0.0.2",
            "::1",
            "[::1]",
            "[ ::1 ]",
            "[::1].",
            "::ffff:127.0.0.1",
            "[::ffff:127.0.0.1]",
            "[::ffff:127.0.0.1].",
            "[::ffff:127.0.0.1.]",
        ],
    )
    def test_loopback_hosts_keep_plugin_policy_and_auth_gate_local(
        self, monkeypatch, host
    ):
        assert web_server.should_require_auth(host, allow_public=False) is False
        assert web_server._set_public_insecure_dashboard_mode(
            host, allow_public=True
        ) is False
        assert web_server._public_insecure_dashboard_mode() is False

    def test_non_loopback_insecure_sets_public_plugin_policy(self, monkeypatch):
        assert web_server.should_require_auth("0.0.0.0", allow_public=False) is True
        assert web_server._set_public_insecure_dashboard_mode(
            "0.0.0.0", allow_public=True
        ) is True
        assert web_server._public_insecure_dashboard_mode() is True

    def test_ipv4_mapped_non_loopback_stays_public(self, monkeypatch):
        assert web_server.should_require_auth(
            "::ffff:192.168.1.10", allow_public=False
        ) is True
        assert web_server._set_public_insecure_dashboard_mode(
            "::ffff:192.168.1.10", allow_public=True
        ) is True
        assert web_server._public_insecure_dashboard_mode() is True

    @pytest.mark.parametrize("host", ["[::1", "::1]"])
    def test_unmatched_ipv6_brackets_are_not_normalized_as_loopback(
        self, monkeypatch, host
    ):
        assert web_server.should_require_auth(host, allow_public=False) is True

    def test_ws_loopback_bind_uses_same_ip_loopback_classifier(self, monkeypatch):
        class _Client:
            def __init__(self, host: str):
                self.host = host

        class _WebSocket:
            def __init__(self, host: str):
                self.client = _Client(host)

        monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.2", raising=False)

        assert web_server._ws_client_is_allowed(_WebSocket("127.0.0.3")) is True
        assert web_server._ws_client_reason(_WebSocket("127.0.0.3")) is None
        assert web_server._ws_client_is_allowed(_WebSocket("192.168.1.10")) is False
        assert web_server._ws_auth_mode() == "loopback"


# ---------------------------------------------------------------------------
# Layer 4 — _mount_plugin_api_routes refuses project-source + traversal.
# ---------------------------------------------------------------------------


class TestMountApiRoutesRefusesUntrusted:
    """The mount routine is the actual ``importlib`` call site — these
    tests poke synthetic plugin entries directly into the cache and
    assert the importer is *not* invoked."""

    def _payload_plugin(self, tmp_path, *, source: str, api_file: str = "api.py"):
        dash = tmp_path / "plug" / "dashboard"
        dash.mkdir(parents=True)
        # Write a benign router file; the test asserts it's NOT imported
        # regardless of whether it exists, since the source/path checks
        # short-circuit before the importer runs.
        (dash / "api.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        return {
            "name": "synthetic",
            "label": "Synthetic",
            "tab": {"path": "/synthetic", "position": "end"},
            "slots": [],
            "entry": "dist/index.js",
            "css": None,
            "has_api": True,
            "source": source,
            "_dir": str(dash),
            "_api_file": api_file,
        }

    def test_project_source_api_is_not_imported(self, tmp_path):
        plugin = self._payload_plugin(tmp_path, source="project")
        web_server._dashboard_plugins_cache = [plugin]
        with patch("importlib.util.spec_from_file_location") as spec:
            web_server._mount_plugin_api_routes()
        assert spec.call_count == 0, (
            "project-source plugin's api file was imported — "
            "GHSA-5qr3-c538-wm9j defence-in-depth regression"
        )

    def test_bundled_source_api_imports_normally(self, tmp_path):
        plugin = self._payload_plugin(tmp_path, source="bundled")
        web_server._dashboard_plugins_cache = [plugin]
        with patch("importlib.util.spec_from_file_location") as spec:
            spec.return_value = None  # loader is None -> early continue, safe
            web_server._mount_plugin_api_routes()
        assert spec.call_count == 1
        # First positional arg after module_name is the resolved api path.
        called_path = Path(spec.call_args.args[1])
        assert called_path.name == "api.py"
        assert called_path.is_absolute()

    def test_traversal_api_caught_at_mount_time(self, tmp_path):
        """Defence-in-depth: if discovery is bypassed (e.g. cache
        tampering), mount-time validation still refuses to import a
        file outside the dashboard dir."""
        plugin = self._payload_plugin(tmp_path, source="user",
                                       api_file="../../../tmp/evil.py")
        web_server._dashboard_plugins_cache = [plugin]
        with patch("importlib.util.spec_from_file_location") as spec:
            web_server._mount_plugin_api_routes()
        assert spec.call_count == 0

    def test_user_source_api_not_imported_on_public_insecure_dashboard(
        self, tmp_path, monkeypatch
    ):
        """Public ``--insecure`` dashboards expose the legacy session-token
        surface to the bound network. In that mode, user-installed dashboard
        plugin API modules must not auto-import server-side Python code.
        """
        plugin = self._payload_plugin(tmp_path, source="user")
        web_server._dashboard_plugins_cache = [plugin]

        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)

        with patch("importlib.util.spec_from_file_location") as spec:
            web_server._mount_plugin_api_routes()

        assert spec.call_count == 0

    def test_user_source_api_imports_on_public_insecure_explicit_opt_in(
        self, tmp_path, monkeypatch
    ):
        """Trusted user plugin APIs can be re-enabled with explicit opt-in."""
        plugin = self._payload_plugin(tmp_path, source="user")
        web_server._dashboard_plugins_cache = [plugin]

        _configure_and_snapshot_public_insecure_mode(monkeypatch, True)

        with patch("importlib.util.spec_from_file_location") as spec:
            spec.return_value = None
            web_server._mount_plugin_api_routes()

        assert spec.call_count == 1

    def test_disabled_config_opt_in_does_not_enable_user_api(
        self, tmp_path, monkeypatch
    ):
        plugin = self._payload_plugin(tmp_path, source="user")
        web_server._dashboard_plugins_cache = [plugin]

        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)

        with patch("importlib.util.spec_from_file_location") as spec:
            web_server._mount_plugin_api_routes()

        assert spec.call_count == 0

    def test_enabled_config_opt_in_enables_user_api(
        self, tmp_path, monkeypatch
    ):
        plugin = self._payload_plugin(tmp_path, source="user")
        web_server._dashboard_plugins_cache = [plugin]

        _configure_and_snapshot_public_insecure_mode(monkeypatch, True)

        with patch("importlib.util.spec_from_file_location") as spec:
            spec.return_value = None
            web_server._mount_plugin_api_routes()

        assert spec.call_count == 1

    @pytest.mark.parametrize("opt_in_value", ["1", "true", "yes", 1])
    def test_non_boolean_config_opt_in_does_not_enable_user_api(
        self, tmp_path, monkeypatch, opt_in_value
    ):
        plugin = self._payload_plugin(tmp_path, source="user")
        web_server._dashboard_plugins_cache = [plugin]

        _configure_and_snapshot_public_insecure_mode(monkeypatch, opt_in_value)

        with patch("importlib.util.spec_from_file_location") as spec:
            web_server._mount_plugin_api_routes()

        assert spec.call_count == 0

    def test_bundled_source_api_imports_on_public_insecure_without_opt_in(
        self, tmp_path, monkeypatch
    ):
        """Bundled plugin APIs are unaffected by the user-plugin gate."""
        plugin = self._payload_plugin(tmp_path, source="bundled")
        web_server._dashboard_plugins_cache = [plugin]

        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)

        with patch("importlib.util.spec_from_file_location") as spec:
            spec.return_value = None
            web_server._mount_plugin_api_routes()

        assert spec.call_count == 1

    def test_start_server_mounts_user_routes_in_safe_mode(
        self, tmp_path, monkeypatch
    ):
        """Safe dashboard launches preserve trusted user plugin backend APIs."""
        original_routes = list(web_server.app.router.routes)
        plugin = self._payload_plugin(tmp_path, source="user")
        plugin["name"] = "safeuser"
        api_path = Path(plugin["_dir"]) / "api.py"
        api_path.write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/ping')\n"
            "def ping():\n"
            "    return {'ok': True}\n"
        )
        monkeypatch.setattr(
            web_server,
            "_get_dashboard_plugins",
            lambda force_rescan=False: [plugin],
        )
        web_server._snapshot_insecure_user_plugin_api_opt_in(False)
        _set_insecure_user_plugin_api_opt_in(monkeypatch, False)

        try:
            captured = _stub_uvicorn_run(monkeypatch)
            web_server.start_server(
                host="127.0.0.1",
                port=9119,
                open_browser=False,
                allow_public=False,
            )

            assert captured["host"] == "127.0.0.1"
            paths = [
                getattr(route, "path", None)
                for route in web_server.app.router.routes
            ]
            assert "/api/plugins/safeuser/ping" in paths
            assert paths.index("/api/plugins/safeuser/ping") < paths.index(
                "/{full_path:path}"
            )
            assert _plugin_module_names("safeuser")
        finally:
            web_server.app.router.routes[:] = original_routes
            _pop_plugin_modules("safeuser")

    def test_start_server_does_not_mount_user_routes_public_insecure(
        self, tmp_path, monkeypatch
    ):
        original_routes = list(web_server.app.router.routes)
        plugin = self._payload_plugin(tmp_path, source="user")
        plugin["name"] = "publicuser"
        api_path = Path(plugin["_dir"]) / "api.py"
        api_path.write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/ping')\n"
            "def ping():\n"
            "    return {'ok': True}\n"
        )
        monkeypatch.setattr(
            web_server,
            "_get_dashboard_plugins",
            lambda force_rescan=False: [plugin],
        )
        web_server._snapshot_insecure_user_plugin_api_opt_in(False)
        _set_insecure_user_plugin_api_opt_in(monkeypatch, False)

        try:
            captured = _stub_uvicorn_run(monkeypatch)
            web_server.start_server(
                host="0.0.0.0",
                port=9119,
                open_browser=False,
                allow_public=True,
            )

            assert captured["host"] == "0.0.0.0"
            assert not _plugin_module_names("publicuser")
            assert not any(
                getattr(route, "path", None) == "/api/plugins/publicuser/ping"
                for route in web_server.app.router.routes
            )
        finally:
            web_server.app.router.routes[:] = original_routes
            web_server._dashboard_plugins_cache = None
            _pop_plugin_modules("publicuser")

    def test_public_insecure_launch_removes_existing_user_routes(
        self, tmp_path, monkeypatch
    ):
        """A reused process must not retain safe-mode user API routes."""
        original_routes = list(web_server.app.router.routes)
        plugin = self._payload_plugin(tmp_path, source="user")
        plugin["name"] = "modeflip"
        api_path = Path(plugin["_dir"]) / "api.py"
        api_path.write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/ping')\n"
            "def ping():\n"
            "    return {'ok': True}\n"
        )
        monkeypatch.setattr(
            web_server,
            "_get_dashboard_plugins",
            lambda force_rescan=False: [plugin],
        )
        web_server._snapshot_insecure_user_plugin_api_opt_in(False)
        _set_insecure_user_plugin_api_opt_in(monkeypatch, False)

        try:
            captured = _stub_uvicorn_run(monkeypatch)
            web_server.start_server(
                host="127.0.0.1",
                port=9119,
                open_browser=False,
                allow_public=False,
            )
            assert captured["host"] == "127.0.0.1"
            assert _plugin_module_names("modeflip")
            assert any(
                getattr(route, "path", None) == "/api/plugins/modeflip/ping"
                for route in web_server.app.router.routes
            )

            captured = _stub_uvicorn_run(monkeypatch)
            web_server.start_server(
                host="0.0.0.0",
                port=9119,
                open_browser=False,
                allow_public=True,
            )

            assert captured["host"] == "0.0.0.0"
            assert web_server._plugin_api_user_routes_mounted is False
            assert not _plugin_module_names("modeflip")
            assert not any(
                getattr(route, "path", None) == "/api/plugins/modeflip/ping"
                for route in web_server.app.router.routes
            )
        finally:
            web_server.app.router.routes[:] = original_routes
            web_server._dashboard_plugins_cache = None
            _pop_plugin_modules("modeflip")

    def test_mount_cleans_module_when_api_has_no_router(self, tmp_path):
        original_routes = list(web_server.app.router.routes)
        plugin = self._payload_plugin(tmp_path, source="user")
        plugin["name"] = "norouter"
        (Path(plugin["_dir"]) / "api.py").write_text("VALUE = 1\n")

        try:
            web_server._dashboard_plugins_cache = [plugin]
            web_server._mount_plugin_api_routes()

            assert not _plugin_module_names("norouter")
            assert not any(
                str(getattr(route, "path", "")).startswith(
                    "/api/plugins/norouter/"
                )
                for route in web_server.app.router.routes
            )
        finally:
            web_server.app.router.routes[:] = original_routes
            web_server._dashboard_plugins_cache = None
            _pop_plugin_modules("norouter")


class TestDiscoveryDisablesUserApiOnPublicInsecure:
    """Discovery hides user plugin API metadata in public-insecure mode."""

    def test_real_user_api_not_imported_when_dashboard_public_insecure(
        self, tmp_path, monkeypatch
    ):
        original_routes = list(web_server.app.router.routes)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "bundled"))

        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)
        dashboard = _write_plugin_manifest(
            tmp_path / "home" / "plugins",
            "real-public-risk",
            {
                "name": "real-public-risk",
                "label": "Real Public Risk",
                "api": "api.py",
                "entry": "dist/index.js",
            },
        )
        marker = tmp_path / "imported.txt"
        dashboard.joinpath("api.py").write_text(
            "from fastapi import APIRouter\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported')\n"
            "router = APIRouter()\n"
            "@router.get('/ping')\n"
            "def ping():\n"
            "    return {'ok': True}\n"
        )

        try:
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            entry = next(p for p in plugins if p["name"] == "real-public-risk")
            assert entry["_api_file"] is None
            assert entry["has_api"] is False

            web_server._mount_plugin_api_routes()

            assert not marker.exists()
            assert not any(
                getattr(route, "path", None)
                == "/api/plugins/real-public-risk/ping"
                for route in web_server.app.router.routes
            )
        finally:
            web_server.app.router.routes[:] = original_routes
            web_server.app.openapi_schema = None
            web_server._dashboard_plugins_cache = None
            _pop_plugin_modules("real-public-risk")

    def test_real_user_api_imports_when_public_insecure_opt_in_enabled(
        self, tmp_path, monkeypatch
    ):
        original_routes = list(web_server.app.router.routes)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "bundled"))

        _configure_and_snapshot_public_insecure_mode(monkeypatch, True)
        dashboard = _write_plugin_manifest(
            tmp_path / "home" / "plugins",
            "real-opt-in",
            {
                "name": "real-opt-in",
                "label": "Real Opt In",
                "api": "api.py",
                "entry": "dist/index.js",
            },
        )
        marker = tmp_path / "imported.txt"
        dashboard.joinpath("api.py").write_text(
            "from fastapi import APIRouter\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported')\n"
            "router = APIRouter()\n"
            "@router.get('/ping')\n"
            "def ping():\n"
            "    return {'ok': True}\n"
        )

        try:
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            entry = next(p for p in plugins if p["name"] == "real-opt-in")
            assert entry["_api_file"] == "api.py"
            assert entry["has_api"] is True

            web_server._mount_plugin_api_routes()

            assert marker.read_text() == "imported"
            assert any(
                getattr(route, "path", None) == "/api/plugins/real-opt-in/ping"
                for route in web_server.app.router.routes
            )
        finally:
            web_server.app.router.routes[:] = original_routes
            web_server.app.openapi_schema = None
            web_server._dashboard_plugins_cache = None
            _pop_plugin_modules("real-opt-in")

    def test_user_api_scrubbed_when_dashboard_public_insecure(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)
        dashboard = _write_plugin_manifest(
            tmp_path / "plugins",
            "public-risk",
            {
                "name": "public-risk",
                "label": "Public Risk",
                "api": "api.py",
                "entry": "dist/index.js",
            },
        )
        (dashboard / "api.py").write_text("router = None\n")

        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "public-risk")

        assert entry["_api_file"] is None
        assert entry["has_api"] is False

    def test_runtime_config_flip_does_not_reenable_user_api(
        self, tmp_path, monkeypatch
    ):
        original_routes = list(web_server.app.router.routes)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "bundled").mkdir()
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "bundled"))

        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)
        dashboard = _write_plugin_manifest(
            tmp_path / "plugins",
            "late-opt-in",
            {
                "name": "late-opt-in",
                "label": "Late Opt In",
                "api": "api.py",
                "entry": "dist/index.js",
            },
        )
        (dashboard / "api.py").write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/ping')\n"
            "def ping():\n"
            "    return {'ok': True}\n"
        )

        try:
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            entry = next(p for p in plugins if p["name"] == "late-opt-in")
            assert entry["_api_file"] is None
            assert entry["has_api"] is False

            _set_insecure_user_plugin_api_opt_in(monkeypatch, True)
            with patch("importlib.util.spec_from_file_location") as spec:
                rescanned = web_server._get_dashboard_plugins(force_rescan=True)

            entry = next(p for p in rescanned if p["name"] == "late-opt-in")
            assert entry["_api_file"] is None
            assert entry["has_api"] is False
            for call in spec.call_args_list:
                assert not call.args[0].startswith(
                    "hermes_dashboard_plugin_late-opt-in"
                )
                assert Path(call.args[1]) != dashboard / "api.py"
            assert not any(
                getattr(route, "path", None) == "/api/plugins/late-opt-in/ping"
                for route in web_server.app.router.routes
            )
        finally:
            web_server.app.router.routes[:] = original_routes
            web_server.app.openapi_schema = None
            web_server._dashboard_plugins_cache = None
            _pop_plugin_modules("late-opt-in")

    def test_user_api_preserved_when_public_insecure_opt_in_enabled(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        _configure_and_snapshot_public_insecure_mode(monkeypatch, True)
        dashboard = _write_plugin_manifest(
            tmp_path / "plugins",
            "opt-in",
            {
                "name": "opt-in",
                "label": "Opt In",
                "api": "api.py",
                "entry": "dist/index.js",
            },
        )
        (dashboard / "api.py").write_text("router = None\n")

        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "opt-in")

        assert entry["_api_file"] == "api.py"
        assert entry["has_api"] is True

    def test_bundled_api_preserved_when_dashboard_public_insecure(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "bundled"))

        _configure_and_snapshot_public_insecure_mode(monkeypatch, False)
        dashboard = _write_plugin_manifest(
            tmp_path / "bundled",
            "bundled-risk",
            {
                "name": "bundled-risk",
                "label": "Bundled Risk",
                "api": "api.py",
                "entry": "dist/index.js",
            },
        )
        (dashboard / "api.py").write_text("router = None\n")

        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        entry = next(p for p in plugins if p["name"] == "bundled-risk")

        assert entry["source"] == "bundled"
        assert entry["_api_file"] == "api.py"
        assert entry["has_api"] is True

# ---------------------------------------------------------------------------
# Layer 5 — End-to-end: the original PoC manifest no longer triggers RCE.
# ---------------------------------------------------------------------------


class TestEndToEndPocBlocked:
    """Reproduces the original advisory PoC shape: untrusted CWD with a
    manifest pointing ``api`` at an attacker-chosen Python file, with
    ``HERMES_ENABLE_PROJECT_PLUGINS=0`` (so the operator believed the
    project source was disabled).  Post-fix, the importer must never
    be invoked for the payload path, regardless of how the bypass is
    framed (``=0`` truthy-string bypass, absolute path bypass,
    project-source bypass)."""

    def test_full_chain_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        cwd = tmp_path / "evil-repo"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        # The original bypass: operator sets the var to a "disabled"
        # string the web server pre-fix treated as enabled.
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
        # Payload: absolute path inside a manifest dropped in CWD.
        payload_py = tmp_path / "payload.py"
        payload_py.write_text("OWNED = True\n")
        _write_plugin_manifest(
            cwd / ".hermes" / "plugins",
            "evil",
            {
                "name": "evil",
                "label": "Evil",
                "api": str(payload_py),
                "entry": "dist/index.js",
            },
        )

        with patch("importlib.util.spec_from_file_location") as spec:
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            web_server._mount_plugin_api_routes()

        # The project source must stay disabled because ``0`` is no
        # longer truthy.  Even if the operator *had* opted in, the
        # absolute-path api would be scrubbed at discovery, and even
        # if discovery missed it the project-source guard in mount
        # would refuse the import.
        assert "evil" not in {p["name"] for p in plugins}
        # Bundled plugins shipped with the repo may legitimately have
        # ``api`` files and so ``spec_from_file_location`` can fire for
        # those — the regression is specifically that the *payload*
        # path / *evil* module are never targeted.
        for call in spec.call_args_list:
            module_name = call.args[0]
            target = Path(call.args[1])
            assert module_name != "hermes_dashboard_plugin_evil"
            assert target != payload_py
            assert "evil-repo" not in target.parts
        assert not _plugin_module_names("evil")
