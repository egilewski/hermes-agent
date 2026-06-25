"""Tests for subprocess env sanitization in LocalEnvironment.

Verifies that Hermes-managed provider, tool, and gateway env vars are
stripped from subprocess environments so external CLIs are not silently
misrouted or handed Hermes secrets.

See: https://github.com/NousResearch/hermes-agent/issues/1002
See: https://github.com/NousResearch/hermes-agent/issues/1264
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _HERMES_PROVIDER_ENV_FORCE_PREFIX,
    _GATEWAY_RUNTIME_CREDENTIAL_ENV_VARS,
    _is_credential_env_name,
)


def _make_fake_popen(captured: dict):
    """Return a fake Popen constructor that records the env kwarg."""
    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.stdout = MagicMock(__iter__=lambda s: iter([]), __next__=lambda s: (_ for _ in ()).throw(StopIteration))
        proc.stdin = MagicMock()
        return proc
    return fake_popen


def _run_with_env(extra_os_env=None, self_env=None):
    """Execute a command via LocalEnvironment with mocked Popen
    and return the env dict passed to the subprocess."""
    captured = {}
    fake_interrupt = threading.Event()
    test_environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "USER": "testuser",
    }
    if extra_os_env:
        test_environ.update(extra_os_env)

    env = LocalEnvironment(cwd="/tmp", timeout=10, env=self_env)

    with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
         patch("subprocess.Popen", side_effect=_make_fake_popen(captured)), \
         patch("tools.terminal_tool._interrupt_event", fake_interrupt), \
         patch.dict(os.environ, test_environ, clear=True):
        env.execute("echo hello")

    return captured.get("env", {})


class TestProviderEnvBlocklist:
    """Provider env vars loaded from ~/.hermes/.env must not leak."""

    def test_blocked_vars_are_stripped(self):
        """OPENAI_BASE_URL and other provider vars must not appear in subprocess env."""
        leaked_vars = {
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OPENAI_API_KEY": "sk-fake-key",
            "OPENROUTER_API_KEY": "or-fake-key",
            "ANTHROPIC_API_KEY": "ant-fake-key",
            "LLM_MODEL": "anthropic/claude-opus-4-6",
        }
        result_env = _run_with_env(extra_os_env=leaked_vars)

        for var in leaked_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_registry_derived_vars_are_stripped(self):
        """Vars from the provider registry (ANTHROPIC_TOKEN, ZAI_API_KEY, etc.)
        must also be blocked — not just the hand-written extras."""
        registry_vars = {
            "ANTHROPIC_TOKEN": "ant-tok",
            "CLAUDE_CODE_OAUTH_TOKEN": "cc-tok",
            "ZAI_API_KEY": "zai-key",
            "Z_AI_API_KEY": "z-ai-key",
            "GLM_API_KEY": "glm-key",
            "KIMI_API_KEY": "kimi-key",
            "MINIMAX_API_KEY": "mm-key",
            "MINIMAX_CN_API_KEY": "mmcn-key",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "NVIDIA_API_KEY": "nvidia-key",
        }
        result_env = _run_with_env(extra_os_env=registry_vars)

        for var in registry_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_bedrock_bearer_token_is_stripped(self):
        """The Bedrock-specific bearer token is a Hermes inference secret
        (analogous to OPENAI_API_KEY) and must not leak into subprocesses.

        Regression for #32314: AWS_BEARER_TOKEN_BEDROCK leaked into terminal /
        execute_code children because the ``bedrock`` ProviderConfig declares
        ``api_key_env_vars=()`` (auth_type="aws_sdk") and the blocklist builder
        only consulted that field. The reporter caught it when ``opencode
        models`` run inside a Hermes terminal enumerated the entire Bedrock
        catalog off the leaked bearer token.
        """
        result_env = _run_with_env(extra_os_env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-bearer-secret",
        })

        assert "AWS_BEARER_TOKEN_BEDROCK" not in result_env, (
            "AWS_BEARER_TOKEN_BEDROCK leaked into subprocess env (see #32314)"
        )

    def test_general_aws_credential_chain_is_preserved(self):
        """The GENERAL AWS credential chain must STILL pass through to
        subprocesses — this is the no-regression guard for #32314.

        Per SECURITY.md §3.2 the local terminal is the user's trusted operator
        shell. A user running ``aws``/``terraform``/``cdk``/``boto3`` in the
        agent terminal must keep the same AWS access their own shell has.
        Stripping these would (a) break every user who does AWS work in the
        agent terminal — not just Bedrock users, since the registry is iterated
        unconditionally — and (b) be unrecoverable, because env_passthrough.py
        refuses to re-allow anything in _HERMES_PROVIDER_ENV_BLOCKLIST
        (GHSA-rhgp-j443-p4rf). Only the Bedrock inference bearer token is
        Hermes-managed; the rest belongs to the user.
        """
        general_chain = {
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "AWS_SESSION_TOKEN": "session-token",
            "AWS_PROFILE": "production",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_REGION": "us-east-1",
            "AWS_SHARED_CREDENTIALS_FILE": "/home/user/.aws/credentials",
            "AWS_CONFIG_FILE": "/home/user/.aws/config",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/secrets/token",
            "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/example",
        }
        result_env = _run_with_env(extra_os_env=general_chain)

        for var, value in general_chain.items():
            assert result_env.get(var) == value, (
                f"{var} was stripped from subprocess env — this is a "
                f"capability regression (see #32314 discussion)"
            )

    def test_non_registry_provider_vars_are_stripped(self):
        """Extra provider vars not in PROVIDER_REGISTRY must also be blocked."""
        extra_provider_vars = {
            "GOOGLE_API_KEY": "google-key",
            "MISTRAL_API_KEY": "mistral-key",
            "GROQ_API_KEY": "groq-key",
            "TOGETHER_API_KEY": "together-key",
            "PERPLEXITY_API_KEY": "perplexity-key",
            "COHERE_API_KEY": "cohere-key",
            "FIREWORKS_API_KEY": "fireworks-key",
            "XAI_API_KEY": "xai-key",
            "HELICONE_API_KEY": "helicone-key",
        }
        result_env = _run_with_env(extra_os_env=extra_provider_vars)

        for var in extra_provider_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_tool_and_gateway_vars_are_stripped(self):
        """Tool and gateway secrets/config must not leak into subprocess env."""
        leaked_vars = {
            "TELEGRAM_BOT_TOKEN": "bot-token",
            "TELEGRAM_HOME_CHANNEL": "12345",
            "DISCORD_HOME_CHANNEL": "67890",
            "SLACK_APP_TOKEN": "xapp-secret",
            "WHATSAPP_ALLOWED_USERS": "+15555550123",
            "SIGNAL_ACCOUNT": "+15555550124",
            "HASS_TOKEN": "ha-secret",
            "EMAIL_PASSWORD": "email-secret",
            "MATRIX_PASSWORD": "matrix-secret",
            "TWILIO_ACCOUNT_SID": "twilio-sid",
            "TWILIO_AUTH_TOKEN": "twilio-secret",
            "FIRECRAWL_API_KEY": "fc-secret",
            "HERMES_DASHBOARD_SESSION_TOKEN": "dashboard-session-secret",
            "BROWSERBASE_PROJECT_ID": "bb-project",
            "ELEVENLABS_API_KEY": "el-secret",
            "DINGTALK_CLIENT_SECRET": "dingtalk-secret",
            "FEISHU_APP_SECRET": "feishu-secret",
            "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery-key",
            "GATEWAY_RELAY_SECRET": "relay-secret",
            "GATEWAY_RELAY_ENROLL_TOKEN": "relay-enroll-token",
            "MSGRAPH_CLIENT_SECRET": "msgraph-client-secret",
            "MSGRAPH_WEBHOOK_CLIENT_STATE": "msgraph-client-state",
            "PHOTON_SIDECAR_TOKEN": "photon-sidecar-token",
            "TEAMS_GRAPH_ACCESS_TOKEN": "teams-graph-token",
            "TEAMS_INCOMING_WEBHOOK_URL": "https://example.webhook.office.com/secret",
            "TELEGRAM_WEBHOOK_SECRET": "telegram-webhook-secret",
            "WECOM_SECRET": "wecom-secret",
            "WECOM_CALLBACK_CORP_SECRET": "wecom-corp-secret",
            "WEIXIN_TOKEN": "weixin-token",
            "WHATSAPP_CLOUD_ACCESS_TOKEN": "whatsapp-cloud-token",
            "WHATSAPP_CLOUD_APP_SECRET": "whatsapp-cloud-secret",
            "WHATSAPP_CLOUD_VERIFY_TOKEN": "whatsapp-cloud-verify",
            "YUANBAO_APP_SECRET": "yuanbao-secret",
            "GITHUB_TOKEN": "ghp_secret",
            "GH_TOKEN": "gh_alias_secret",
            "GATEWAY_ALLOW_ALL_USERS": "true",
            "GATEWAY_ALLOWED_USERS": "alice,bob",
            "MODAL_TOKEN_ID": "modal-id",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "DAYTONA_API_KEY": "daytona-key",
        }
        result_env = _run_with_env(extra_os_env=leaked_vars)

        for var in leaked_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_safe_vars_are_preserved(self):
        """Standard env vars (PATH, HOME, USER) must still be passed through."""
        result_env = _run_with_env()

        assert "HOME" in result_env
        assert result_env["HOME"] == "/home/user"
        assert "USER" in result_env
        assert "PATH" in result_env

    def test_self_env_blocked_vars_also_stripped(self):
        """Blocked vars in self.env are stripped; non-blocked vars pass through."""
        result_env = _run_with_env(self_env={
            "OPENAI_BASE_URL": "http://custom:9999/v1",
            "MY_CUSTOM_VAR": "keep-this",
        })

        assert "OPENAI_BASE_URL" not in result_env
        assert "MY_CUSTOM_VAR" in result_env
        assert result_env["MY_CUSTOM_VAR"] == "keep-this"


class TestForceEnvOptIn:
    """Callers can opt in to passing a blocked var via _HERMES_FORCE_ prefix."""

    def test_force_prefix_passes_blocked_var(self):
        """_HERMES_FORCE_OPENAI_API_KEY in self.env should inject OPENAI_API_KEY."""
        result_env = _run_with_env(self_env={
            f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_API_KEY": "sk-explicit",
        })

        assert "OPENAI_API_KEY" in result_env
        assert result_env["OPENAI_API_KEY"] == "sk-explicit"
        # The force-prefixed key itself must not appear
        assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_API_KEY" not in result_env

    def test_force_prefix_overrides_os_environ_block(self):
        """Force-prefix in self.env wins even when os.environ has the blocked var."""
        result_env = _run_with_env(
            extra_os_env={"OPENAI_BASE_URL": "http://leaked/v1"},
            self_env={f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_BASE_URL": "http://intended/v1"},
        )

        assert result_env["OPENAI_BASE_URL"] == "http://intended/v1"


class TestBlocklistCoverage:
    """Sanity checks that the blocklist covers all known providers."""

    def test_issue_1002_offenders(self):
        """Blocklist includes the main offenders from issue #1002."""
        must_block = {
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "LLM_MODEL",
        }
        assert must_block.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_registry_vars_are_in_blocklist(self):
        """Every api_key_env_var and base_url_env_var from PROVIDER_REGISTRY
        must appear in the blocklist — ensures no drift."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        for pconfig in PROVIDER_REGISTRY.values():
            for var in pconfig.api_key_env_vars:
                assert var in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Registry var {var} (provider={pconfig.id}) missing from blocklist"
                )
            if pconfig.base_url_env_var:
                assert pconfig.base_url_env_var in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Registry base_url_env_var {pconfig.base_url_env_var} "
                    f"(provider={pconfig.id}) missing from blocklist"
                )

    def test_bedrock_bearer_token_is_in_blocklist(self):
        """auth_type='aws_sdk' providers contribute their Hermes-managed
        inference token (the Bedrock bearer) to the blocklist, keyed off
        auth_type so any future SDK-cred provider is covered automatically."""
        assert "AWS_BEARER_TOKEN_BEDROCK" in _HERMES_PROVIDER_ENV_BLOCKLIST

    def test_general_aws_chain_not_in_blocklist(self):
        """The general AWS credential chain must NOT be in the blocklist —
        no-regression guard for #32314. These belong to the user's trusted
        operator shell (SECURITY.md §3.2), not to Hermes, and blocklisting
        them would be unrecoverable via env_passthrough (GHSA-rhgp-j443-p4rf).
        """
        general_chain = {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_REGION",
            "AWS_REGION",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
        }
        leaked_block = general_chain & _HERMES_PROVIDER_ENV_BLOCKLIST
        assert not leaked_block, (
            f"General AWS chain vars must stay inheritable, but these are "
            f"blocklisted: {sorted(leaked_block)} (capability regression, #32314)"
        )

    def test_extra_auth_vars_covered(self):
        """Non-registry auth vars (ANTHROPIC_TOKEN, CLAUDE_CODE_OAUTH_TOKEN)
        must also be in the blocklist."""
        extras = {"ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_non_registry_provider_vars_are_in_blocklist(self):
        extras = {
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "PERPLEXITY_API_KEY",
            "COHERE_API_KEY",
            "FIREWORKS_API_KEY",
            "XAI_API_KEY",
            "HELICONE_API_KEY",
        }
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_optional_tool_and_messaging_vars_are_in_blocklist(self):
        """Tool/messaging vars from OPTIONAL_ENV_VARS should stay covered."""
        from hermes_cli.config import OPTIONAL_ENV_VARS

        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                assert name in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Optional env var {name} (category={category}) missing from blocklist"
                )
            elif category == "setting" and metadata.get("password"):
                assert name in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Secret setting env var {name} missing from blocklist"
                )

    def test_secret_extra_env_keys_are_in_blocklist(self):
        """Secret-shaped config extras should stay covered even when they are
        not listed in user-facing OPTIONAL_ENV_VARS."""
        from hermes_cli.config import _EXTRA_ENV_KEYS

        credential_extras = {
            name for name in _EXTRA_ENV_KEYS if _is_credential_env_name(name)
        }
        assert credential_extras, "expected at least one credential-shaped extra env key"
        missing = {
            name for name in credential_extras
            if name not in _HERMES_PROVIDER_ENV_BLOCKLIST
        }
        assert not missing

    def test_credential_env_name_helper_covers_app_keys(self):
        assert _is_credential_env_name("EXAMPLE_APP_KEY")
        assert not _is_credential_env_name("EXAMPLE_CLIENT_ID")

    def test_gateway_runtime_vars_are_in_blocklist(self):
        extras = {
            "TELEGRAM_HOME_CHANNEL",
            "TELEGRAM_HOME_CHANNEL_NAME",
            "TELEGRAM_WEBHOOK_SECRET",
            "DISCORD_HOME_CHANNEL",
            "DISCORD_HOME_CHANNEL_NAME",
            "DISCORD_REQUIRE_MENTION",
            "DISCORD_FREE_RESPONSE_CHANNELS",
            "DISCORD_AUTO_THREAD",
            "SLACK_HOME_CHANNEL",
            "SLACK_HOME_CHANNEL_NAME",
            "SLACK_ALLOWED_USERS",
            "WHATSAPP_ENABLED",
            "WHATSAPP_MODE",
            "WHATSAPP_ALLOWED_USERS",
            "SIGNAL_HTTP_URL",
            "SIGNAL_ACCOUNT",
            "SIGNAL_ALLOWED_USERS",
            "SIGNAL_GROUP_ALLOWED_USERS",
            "SIGNAL_HOME_CHANNEL",
            "SIGNAL_HOME_CHANNEL_NAME",
            "SIGNAL_IGNORE_STORIES",
            "HASS_TOKEN",
            "HASS_URL",
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
            "EMAIL_HOME_ADDRESS",
            "EMAIL_HOME_ADDRESS_NAME",
            "HERMES_DASHBOARD_SESSION_TOKEN",
            "MATRIX_PASSWORD",
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_PHONE_NUMBER",
            "TWILIO_PHONE_NUMBER_SID",
            "GATEWAY_ALLOWED_USERS",
            "GATEWAY_RELAY_DELIVERY_KEY",
            "GATEWAY_RELAY_ENROLL_TOKEN",
            "GATEWAY_RELAY_SECRET",
            "MSGRAPH_CLIENT_SECRET",
            "MSGRAPH_WEBHOOK_CLIENT_STATE",
            "PHOTON_SIDECAR_TOKEN",
            "TEAMS_GRAPH_ACCESS_TOKEN",
            "TEAMS_INCOMING_WEBHOOK_URL",
            "DINGTALK_CLIENT_ID",
            "DINGTALK_CLIENT_SECRET",
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_ENCRYPT_KEY",
            "FEISHU_VERIFICATION_TOKEN",
            "WECOM_BOT_ID",
            "WECOM_SECRET",
            "WECOM_CALLBACK_CORP_ID",
            "WECOM_CALLBACK_CORP_SECRET",
            "WECOM_CALLBACK_AGENT_ID",
            "WECOM_CALLBACK_TOKEN",
            "WECOM_CALLBACK_ENCODING_AES_KEY",
            "WEIXIN_TOKEN",
            "WEIXIN_ACCOUNT_ID",
            "YUANBAO_APP_ID",
            "YUANBAO_APP_KEY",
            "YUANBAO_APP_SECRET",
            "YUANBAO_BOT_ID",
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            "WHATSAPP_CLOUD_APP_SECRET",
            "WHATSAPP_CLOUD_VERIFY_TOKEN",
            "GH_TOKEN",
            "GITHUB_APP_ID",
            "GITHUB_APP_PRIVATE_KEY_PATH",
            "GITHUB_APP_INSTALLATION_ID",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "DAYTONA_API_KEY",
        }
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_gateway_ghsa_m4m8_xjp4_5rmm_credentials_are_stripped_from_all_local_env_paths(self):
        """GHSA-m4m8-xjp4-5rmm: gateway credentials must not leak to subprocesses."""
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        secrets = {
            "DINGTALK_CLIENT_SECRET": "dingtalk-secret",
            "FEISHU_APP_SECRET": "feishu-secret",
            "FEISHU_ENCRYPT_KEY": "feishu-encrypt-key",
            "FEISHU_VERIFICATION_TOKEN": "feishu-verification-token",
            "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery-key",
            "GATEWAY_RELAY_ENROLL_TOKEN": "relay-enroll-token",
            "GATEWAY_RELAY_SECRET": "relay-secret",
            "MATRIX_PASSWORD": "matrix-password",
            "MSGRAPH_CLIENT_SECRET": "msgraph-client-secret",
            "MSGRAPH_WEBHOOK_CLIENT_STATE": "msgraph-client-state",
            "PHOTON_SIDECAR_TOKEN": "photon-sidecar-token",
            "TEAMS_GRAPH_ACCESS_TOKEN": "teams-graph-token",
            "TEAMS_INCOMING_WEBHOOK_URL": "https://example.webhook.office.com/secret",
            "TELEGRAM_WEBHOOK_SECRET": "telegram-webhook-secret",
            "TWILIO_AUTH_TOKEN": "twilio-auth-token",
            "WECOM_CALLBACK_CORP_SECRET": "wecom-callback-secret",
            "WECOM_CALLBACK_TOKEN": "wecom-callback-token",
            "WECOM_CALLBACK_ENCODING_AES_KEY": "wecom-callback-aes-key",
            "WECOM_SECRET": "wecom-secret",
            "WEIXIN_TOKEN": "weixin-token",
            "WHATSAPP_CLOUD_ACCESS_TOKEN": "whatsapp-cloud-token",
            "WHATSAPP_CLOUD_APP_SECRET": "whatsapp-cloud-secret",
            "WHATSAPP_CLOUD_VERIFY_TOKEN": "whatsapp-cloud-verify",
            "YUANBAO_APP_SECRET": "yuanbao-secret",
        }

        with patch.dict(os.environ, secrets | {"PATH": "/usr/bin:/bin"}, clear=True):
            run_env = _make_run_env({})
        bg_env = _sanitize_subprocess_env(secrets | {"PATH": "/usr/bin:/bin"})

        for env_name in secrets:
            assert env_name not in run_env, f"{env_name} leaked through _make_run_env"
            assert env_name not in bg_env, (
                f"{env_name} leaked through _sanitize_subprocess_env"
            )

    def test_gateway_runtime_credential_set_is_stripped_from_all_local_env_paths(self):
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        secrets = {
            name: f"secret-{index}"
            for index, name in enumerate(_GATEWAY_RUNTIME_CREDENTIAL_ENV_VARS)
        }
        assert secrets, "_GATEWAY_RUNTIME_CREDENTIAL_ENV_VARS must not be empty"

        with patch.dict(os.environ, secrets | {"PATH": "/usr/bin:/bin"}, clear=True):
            run_env = _make_run_env({})
        bg_env = _sanitize_subprocess_env(secrets | {"PATH": "/usr/bin:/bin"})

        for env_name in secrets:
            assert env_name not in run_env, f"{env_name} leaked through _make_run_env"
            assert env_name not in bg_env, (
                f"{env_name} leaked through _sanitize_subprocess_env"
            )

    def test_teams_runtime_credentials_are_stripped_from_all_local_env_paths(self):
        """Teams delivery credentials are read from process env but must not leak."""
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        secrets = {
            "TEAMS_CLIENT_SECRET": "teams-client-secret",
            "TEAMS_GRAPH_ACCESS_TOKEN": "teams-graph-token",
            "TEAMS_INCOMING_WEBHOOK_URL": "https://example.webhook.office.com/secret",
        }

        with patch.dict(os.environ, secrets | {"PATH": "/usr/bin:/bin"}, clear=True):
            run_env = _make_run_env({})
        bg_env = _sanitize_subprocess_env(secrets | {"PATH": "/usr/bin:/bin"})

        for env_name in secrets:
            assert env_name not in run_env, f"{env_name} leaked through _make_run_env"
            assert env_name not in bg_env, (
                f"{env_name} leaked through _sanitize_subprocess_env"
            )

    def test_runtime_non_secret_gateway_vars_are_preserved(self):
        """Runtime-only gateway knobs should not be swept up by secret suffixes."""
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        operational = {
            "GATEWAY_RELAY_URL": "wss://connector.example/relay",
            "GATEWAY_RELAY_ID": "gw-example",
            "GATEWAY_RELAY_PLATFORM": "relay",
            "GATEWAY_RELAY_ROUTE_KEYS": "tenant-a,tenant-b",
            "MSGRAPH_AUTHORITY_URL": "https://login.microsoftonline.com",
            "MSGRAPH_CLIENT_ID": "client-123",
            "MSGRAPH_SCOPE": "https://graph.microsoft.com/.default",
            "MSGRAPH_TENANT_ID": "tenant-123",
            "MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES": "communications/onlineMeetings",
            "MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS": "52.96.0.0/14",
            "MSGRAPH_WEBHOOK_PORT": "8646",
            "MSGRAPH_WEBHOOK_STORE_PATH": "/tmp/hermes-msgraph.sqlite3",
            "TEAMS_CHANNEL_ID": "channel-123",
            "TEAMS_CHAT_ID": "chat-123",
            "TEAMS_DELIVERY_MODE": "graph",
            "TEAMS_SERVICE_URL": "https://smba.trafficmanager.net/teams/",
            "TEAMS_TEAM_ID": "team-123",
            "TELEGRAM_WEBHOOK_URL": "https://example.com/telegram",
            "WHATSAPP_CLOUD_PHONE_NUMBER_ID": "7794189252778687",
            "WHATSAPP_CLOUD_WEBHOOK_PORT": "8090",
        }

        with patch.dict(os.environ, operational | {"PATH": "/usr/bin:/bin"}, clear=True):
            run_env = _make_run_env({})
        bg_env = _sanitize_subprocess_env(operational | {"PATH": "/usr/bin:/bin"})

        for env_name, value in operational.items():
            assert run_env.get(env_name) == value
            assert bg_env.get(env_name) == value


class TestSanePathIncludesHomebrew:
    """Verify _SANE_PATH includes macOS Homebrew directories."""

    @pytest.fixture(autouse=True)
    def _disable_hermes_bin_injection(self):
        """These tests assert the sane-path merge in isolation. Disable the
        hermes-install-dir prepend (a separate concern, covered by
        TestHermesBinDirOnPath) so a real ``hermes`` on the test runner's PATH
        doesn't shift the asserted PATH layout."""
        from tools.environments import local as local_mod
        saved = local_mod._HERMES_BIN_DIR
        local_mod._HERMES_BIN_DIR = None  # resolved -> no dir to inject
        yield
        local_mod._HERMES_BIN_DIR = saved

    def test_sane_path_includes_homebrew_bin(self):
        from tools.environments.local import _SANE_PATH
        assert "/opt/homebrew/bin" in _SANE_PATH

    def test_sane_path_includes_homebrew_sbin(self):
        from tools.environments.local import _SANE_PATH
        assert "/opt/homebrew/sbin" in _SANE_PATH

    def test_make_run_env_appends_homebrew_on_minimal_path(self):
        """When PATH is minimal, _make_run_env appends missing sane entries."""
        from tools.environments.local import _SANE_PATH, _make_run_env
        minimal_env = {"PATH": "/some/custom/bin"}
        with patch.dict(os.environ, minimal_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert path_entries[0] == "/some/custom/bin"
        for entry in _SANE_PATH.split(":"):
            assert entry in path_entries

    def test_make_run_env_fills_missing_homebrew_when_usr_bin_present(self):
        """macOS launchd PATH can include /usr/bin while missing Homebrew."""
        from tools.environments.local import _make_run_env
        launchd_env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        with patch.dict(os.environ, launchd_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert "/opt/homebrew/bin" in path_entries
        assert "/opt/homebrew/sbin" in path_entries

    def test_make_run_env_does_not_duplicate_existing_sane_entries(self):
        from tools.environments.local import _make_run_env
        existing_env = {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
        with patch.dict(os.environ, existing_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert path_entries.count("/opt/homebrew/bin") == 1
        assert path_entries.count("/usr/local/bin") == 1
        assert path_entries.count("/usr/bin") == 1

    def test_make_run_env_real_launchd_path_gains_homebrew(self):
        """The literal macOS launchd PATH is the production trigger for #35613."""
        from tools.environments.local import _make_run_env
        launchd_env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
        with patch.dict(os.environ, launchd_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert "/opt/homebrew/bin" in path_entries
        assert "/opt/homebrew/sbin" in path_entries
        # Original entries keep their leading precedence.
        assert path_entries[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]

    def test_make_run_env_collapses_duplicate_caller_entries(self):
        """Duplicates already present in the caller PATH are de-duplicated."""
        from tools.environments.local import _make_run_env
        dup_env = {"PATH": "/usr/bin:/usr/bin:/custom/bin:/custom/bin:/bin"}
        with patch.dict(os.environ, dup_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert path_entries.count("/usr/bin") == 1
        assert path_entries.count("/custom/bin") == 1
        # First-occurrence order is preserved for the caller entries.
        assert path_entries[:3] == ["/usr/bin", "/custom/bin", "/bin"]

    def test_make_run_env_strips_empty_path_entries(self):
        """Leading/trailing/double colons (== CWD on POSIX) are dropped."""
        from tools.environments.local import _make_run_env
        empty_env = {"PATH": "/usr/bin::/bin:"}
        with patch.dict(os.environ, empty_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert "" not in path_entries
        assert "/usr/bin" in path_entries
        assert "/opt/homebrew/bin" in path_entries

    def test_make_run_env_leaves_windows_path_unchanged(self, monkeypatch):
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        windows_env = {"PATH": r"C:\Windows\System32;C:\Program Files\Git\bin"}
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        with patch.dict(os.environ, windows_env, clear=True):
            result = _make_run_env({})
        assert result["PATH"] == windows_env["PATH"]

    def test_make_run_env_preserves_windows_mixed_case_path_key(self, monkeypatch):
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        windows_env = {"Path": r"C:\Windows\System32;C:\Program Files\Git\bin"}
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        with patch.object(local_mod.os, "environ", windows_env):
            result = _make_run_env({})
        assert result["Path"] == windows_env["Path"]
        assert "PATH" not in result


class TestHermesBinDirOnPath:
    """The hermes install dir is reachable in the terminal subshell PATH.

    Plugins shelling out to bare ``hermes`` via the terminal tool must work
    even when the gateway was launched without the hermes install dir on
    PATH (systemd, service managers, cron). See the discussion that motivated
    _resolve_hermes_bin_dir / _prepend_hermes_bin_dir.
    """

    def _reset_cache(self):
        from tools.environments import local as local_mod
        local_mod._HERMES_BIN_DIR = local_mod._SENTINEL

    def test_resolves_via_which(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        monkeypatch.setattr(local_mod.shutil, "which",
                            lambda name: "/opt/hermes/bin/hermes" if name == "hermes" else None)
        monkeypatch.setattr(local_mod.os.path, "isdir", lambda p: p == "/opt/hermes/bin")
        assert local_mod._resolve_hermes_bin_dir() == "/opt/hermes/bin"

    def test_resolves_via_sys_executable_dir(self, monkeypatch, tmp_path):
        from tools.environments import local as local_mod
        self._reset_cache()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "hermes").write_text("#!/bin/sh\n")
        monkeypatch.setattr(local_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(local_mod.sys, "argv", ["python"])
        monkeypatch.setattr(local_mod.sys, "executable", str(venv_bin / "python"))
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        assert local_mod._resolve_hermes_bin_dir() == str(venv_bin)

    def test_returns_none_when_unresolvable(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        monkeypatch.setattr(local_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(local_mod.sys, "argv", ["python"])
        monkeypatch.setattr(local_mod.sys, "executable", "/nonexistent/python")
        assert local_mod._resolve_hermes_bin_dir() is None

    def test_prepend_adds_missing_dir_at_front(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = "/opt/hermes/bin"
        out = local_mod._prepend_hermes_bin_dir("/usr/bin:/bin")
        assert out.split(os.pathsep)[0] == "/opt/hermes/bin"
        assert "/usr/bin" in out.split(os.pathsep)

    def test_prepend_is_idempotent(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = "/opt/hermes/bin"
        once = local_mod._prepend_hermes_bin_dir("/usr/bin:/bin")
        twice = local_mod._prepend_hermes_bin_dir(once)
        assert twice == once
        assert once.split(os.pathsep).count("/opt/hermes/bin") == 1

    def test_prepend_noop_when_unresolved(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = None
        assert local_mod._prepend_hermes_bin_dir("/usr/bin:/bin") == "/usr/bin:/bin"

    def test_make_run_env_injects_hermes_bin_dir(self, monkeypatch):
        """A gateway env missing the hermes dir gets it back in the subshell PATH."""
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = "/opt/hermes/bin"
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True):
            result = _make_run_env({})
        entries = result["PATH"].split(os.pathsep)
        assert entries[0] == "/opt/hermes/bin"
        assert "/usr/bin" in entries
