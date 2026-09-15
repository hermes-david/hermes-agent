"""Tests for Ollama Cloud provider integration."""

import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import _PROVIDER_MODELS, _PROVIDER_LABELS, _PROVIDER_ALIASES, normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider
from agent.model_metadata import _URL_TO_PROVIDER, _PROVIDER_PREFIXES
from agent.models_dev import PROVIDER_TO_MODELS_DEV, list_agentic_models


# ── Provider Registry ──

class TestOllamaCloudProviderRegistry:
    def test_ollama_cloud_in_registry(self):
        assert "ollama-cloud" in PROVIDER_REGISTRY

    def test_ollama_cloud_config(self):
        pconfig = PROVIDER_REGISTRY["ollama-cloud"]
        assert pconfig.id == "ollama-cloud"
        assert pconfig.name == "Ollama Cloud"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://ollama.com/v1"


# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestOllamaCloudAliases:

    def test_alias_ollama_underscore(self):
        """ollama_cloud (underscore) is the unambiguous cloud alias."""
        assert resolve_provider("ollama_cloud") == "ollama-cloud"


    def test_models_py_aliases(self):
        assert _PROVIDER_ALIASES.get("ollama_cloud") == "ollama-cloud"
        # bare "ollama" stays local
        assert _PROVIDER_ALIASES.get("ollama") == "custom"


# ── Auto-detection ──

class TestOllamaCloudAutoDetection:
    def test_auto_detects_ollama_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-key")
        assert resolve_provider("auto") == "ollama-cloud"


# ── Credential Resolution ──

class TestOllamaCloudCredentials:
    def test_resolve_with_ollama_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
        creds = resolve_api_key_provider_credentials("ollama-cloud")
        assert creds["provider"] == "ollama-cloud"
        assert creds["api_key"] == "ollama-secret"
        assert creds["base_url"] == "https://ollama.com/v1"


    def test_runtime_ollama_cloud(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="ollama-cloud")
        assert result["provider"] == "ollama-cloud"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "ollama-key"
        assert result["base_url"] == "https://ollama.com/v1"


# ── Model Catalog (dynamic — no static list) ──

class TestOllamaCloudModelCatalog:


    def test_provider_model_ids_returns_dynamic_models(self, tmp_path, monkeypatch):
        """provider_model_ids('ollama-cloud') should call fetch_ollama_cloud_models()."""
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "qwen3.5:397b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5:397b"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = provider_model_ids("ollama-cloud", force_refresh=True)

        assert len(result) > 0
        assert "qwen3.5:397b" in result


# ── Model Picker (list_authenticated_providers) ──

class TestOllamaCloudModelPicker:
    def test_ollama_cloud_shows_model_count(self, tmp_path, monkeypatch):
        """Ollama Cloud should show non-zero model count in provider picker."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "qwen3.5:397b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5:397b"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            providers = list_authenticated_providers(current_provider="ollama-cloud")

        ollama = next((p for p in providers if p["slug"] == "ollama-cloud"), None)
        assert ollama is not None, "ollama-cloud should appear when OLLAMA_API_KEY is set"
        assert ollama["total_models"] > 0, "ollama-cloud should show non-zero model count"

    def test_ollama_cloud_not_shown_without_creds(self, monkeypatch):
        """Ollama Cloud should not appear without credentials."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

        providers = list_authenticated_providers(current_provider="openrouter")
        ollama = next((p for p in providers if p["slug"] == "ollama-cloud"), None)
        assert ollama is None, "ollama-cloud should not appear without OLLAMA_API_KEY"


# ── Merged Model Discovery ──

class TestOllamaCloudMergedDiscovery:
    def test_merges_live_and_models_dev(self, tmp_path, monkeypatch):
        """Live API models appear first, models.dev additions fill gaps."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "glm-5": {"tool_call": True},
                    "kimi-k2.5": {"tool_call": True},
                    "nemotron-3-super": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5:397b", "glm-5"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev), \
             patch("hermes_cli.models.ollama_cloud_model_is_servable", return_value=True):
            result = fetch_ollama_cloud_models(force_refresh=True)

        # Live models first, then models.dev additions (deduped)
        assert result[0] == "qwen3.5:397b"    # from live API
        assert result[1] == "glm-5"          # from live API (also in models.dev)
        assert "kimi-k2.5" in result         # from models.dev only, servable
        assert "nemotron-3-super" in result  # from models.dev only, servable
        assert result.count("glm-5") == 1    # no duplicates


class TestOllamaCloudRetiredModelFiltering:
    """models.dev is a *gap-fill* source, not an additive one.

    Ollama Cloud retires a model by removing it from ``/v1/models`` and returning HTTP 410
    ("was retired at <date>") from every inference call. models.dev keeps listing it, so the
    unconditional union served retired IDs in the picker: selecting one produced a hard 410
    error rather than a usable model.
    """

    def _patch_servability(self, monkeypatch, servable_ids):
        """Wire the /api/show servability probe to a fixed set of model IDs.

        Patches the real seam (the probe helper), not the module under test.
        """
        monkeypatch.setattr(
            "hermes_cli.models.ollama_cloud_model_is_servable",
            lambda model, *a, **k: model in servable_ids,
        )

    def test_retired_model_is_excluded_when_live_probe_succeeds(self, tmp_path, monkeypatch):
        """A models.dev-only model the API refuses (410) must not be offered."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        # kimi-k2.5 is retired upstream; deepseek-v4-flash is still served.
        self._patch_servability(monkeypatch, {"deepseek-v4-flash"})

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "kimi-k2.5": {"tool_call": True},
                    "deepseek-v4-flash": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5:397b"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert "kimi-k2.5" not in result, (
            "retired model must not reach the picker — selecting it 410s"
        )
        assert "deepseek-v4-flash" in result, (
            "a servable models.dev-only model must survive the filter"
        )
        assert "qwen3.5:397b" in result, "live models must be preserved"

    def test_live_models_are_not_probe_filtered(self, tmp_path, monkeypatch):
        """Live /v1/models is already authoritative — never probe-filter it.

        The probe returns False for everything here. Live models must still survive: a
        transient probe fault must not silently shrink the catalog.
        """
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        self._patch_servability(monkeypatch, set())

        mock_mdev = {"ollama-cloud": {"models": {"kimi-k2.5": {"tool_call": True}}}}
        with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.5:397b", "glm-5"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result == ["qwen3.5:397b", "glm-5"]

    def test_live_failure_keeps_models_dev_fallback_unfiltered(self, tmp_path, monkeypatch):
        """With no live catalog there is nothing to validate against.

        Keep today's behaviour: serve the models.dev set so an authenticated user still gets
        a usable picker during a transient outage.
        """
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        self._patch_servability(monkeypatch, set())

        mock_mdev = {"ollama-cloud": {"models": {"kimi-k2.5": {"tool_call": True}}}}
        with patch("hermes_cli.models.fetch_api_models", return_value=[]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result == ["kimi-k2.5"]

class TestOllamaCloudServabilityProbe:
    """The /api/show status mapping that decides servability.

    These run at the real boundary (httpx.Client), mirroring the established probe-test
    pattern, so the status-code mapping itself is covered rather than mocked away.
    """

    def _patch_show(self, monkeypatch, *, status=200, raise_exc=None):
        import httpx

        class _Resp:
            status_code = status

            def json(self):
                return {}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, *a, **k):
                if raise_exc:
                    raise raise_exc
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Client)

    def test_status_200_is_servable(self, monkeypatch):
        from hermes_cli.models import ollama_cloud_model_is_servable

        self._patch_show(monkeypatch, status=200)
        assert ollama_cloud_model_is_servable(
            "deepseek-v4-flash", "https://ollama.com/v1", "k") is True

    def test_status_410_retired_is_not_servable(self, monkeypatch):
        """410 = '<model> was retired at <date>' — the reported bug."""
        from hermes_cli.models import ollama_cloud_model_is_servable

        self._patch_show(monkeypatch, status=410)
        assert ollama_cloud_model_is_servable(
            "kimi-k2.5", "https://ollama.com/v1", "k") is False

    def test_status_404_unknown_is_not_servable(self, monkeypatch):
        """404 = 'model <id> not found' — also unservable."""
        from hermes_cli.models import ollama_cloud_model_is_servable

        self._patch_show(monkeypatch, status=404)
        assert ollama_cloud_model_is_servable(
            "no-such-model", "https://ollama.com/v1", "k") is False

    def test_transport_failure_fails_open(self, monkeypatch):
        """A transient probe fault must never drop a candidate."""
        from hermes_cli.models import ollama_cloud_model_is_servable

        self._patch_show(monkeypatch, raise_exc=RuntimeError("boom"))
        assert ollama_cloud_model_is_servable(
            "kimi-k2.6", "https://ollama.com/v1", "k") is True

    def test_strips_v1_suffix_for_native_endpoint(self, monkeypatch):
        """The probe must hit <root>/api/show, not <root>/v1/api/show."""
        import httpx
        from hermes_cli.models import ollama_cloud_model_is_servable

        seen = {}

        class _Resp:
            status_code = 200

            def json(self):
                return {}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, *a, **k):
                seen["url"] = url
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Client)
        ollama_cloud_model_is_servable("kimi-k2.6", "https://ollama.com/v1", "k")
        assert seen["url"] == "https://ollama.com/api/show"


class TestOllamaCloudServableFilter:
    """filter_servable_ollama_cloud_models — fan-out + ordering."""

    def test_preserves_order_and_duplicates(self, monkeypatch):
        from hermes_cli.models import filter_servable_ollama_cloud_models

        monkeypatch.setattr(
            "hermes_cli.models.ollama_cloud_model_is_servable",
            lambda m, *a, **k: m != "retired",
        )
        assert filter_servable_ollama_cloud_models(
            ["a", "retired", "b", "a"]) == ["a", "b", "a"]

    def test_empty_input(self):
        from hermes_cli.models import filter_servable_ollama_cloud_models

        assert filter_servable_ollama_cloud_models([]) == []


class TestOllamaCloudModelsDevFallback:
    def test_models_dev_only_when_live_unavailable(self, tmp_path, monkeypatch):
        """models.dev supplies the catalog when the live probe returns nothing.

        The probe is still ATTEMPTED without an env key — the catalog is anonymous — so it
        is mocked to an empty result here to represent an unreachable endpoint.
        """
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "glm-5": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=[]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result == ["glm-5"]






# ── Model Normalization ──

class TestOllamaCloudModelNormalization:


    def test_passthrough_no_tag(self):
        assert normalize_model_for_provider("glm-5", "ollama-cloud") == "glm-5"


# ── URL-to-Provider Mapping ──


# ── models.dev Integration ──

class TestOllamaCloudModelsDev:
    def test_ollama_cloud_mapped(self):
        assert PROVIDER_TO_MODELS_DEV.get("ollama-cloud") == "ollama-cloud"

    def test_list_agentic_models_with_mock_data(self):
        """list_agentic_models filters correctly from mock models.dev data."""
        mock_data = {
            "ollama-cloud": {
                "models": {
                    "qwen3.5:397b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                    "nemotron-3-nano:30b": {"tool_call": True},
                    "some-embedding:latest": {"tool_call": False},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            result = list_agentic_models("ollama-cloud")
        assert "qwen3.5:397b" in result
        assert "glm-5" in result
        assert "nemotron-3-nano:30b" in result
        assert "some-embedding:latest" not in result  # no tool_call


# ── Agent Init (no SyntaxError) ──

class TestOllamaCloudAgentInit:
    def test_agent_imports_without_error(self):
        """Verify run_agent.py has no SyntaxError."""
        import importlib
        import run_agent
        importlib.reload(run_agent)

    def test_ollama_cloud_agent_uses_chat_completions(self, monkeypatch):
        """Ollama Cloud falls through to chat_completions — no special elif needed."""
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        with patch("agent.process_bootstrap.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from run_agent import AIAgent
            agent = AIAgent(
                model="qwen3.5:397b",
                provider="ollama-cloud",
                api_key="test-key",
                base_url="https://ollama.com/v1",
            )
            assert agent.api_mode == "chat_completions"
            assert agent.provider == "ollama-cloud"


# ── providers.py New System ──

class TestOllamaCloudProvidersNew:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "ollama-cloud" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["ollama-cloud"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "OLLAMA_BASE_URL"

    def test_alias_resolves(self):
        from hermes_cli.providers import normalize_provider as np
        assert np("ollama") == "custom"  # bare "ollama" = local
        assert np("ollama-cloud") == "ollama-cloud"


    def test_get_provider(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("ollama-cloud")
        assert pdef is not None
        assert pdef.id == "ollama-cloud"
        assert pdef.transport == "openai_chat"


# ── Cloud Suffix Stripping ──

class TestOllamaCloudSuffixStripping:
    """models.dev appends :cloud / -cloud suffixes that the live API omits.

    fetch_ollama_cloud_models() must normalise these before the dedup merge so
    users never see broken IDs like 'kimi-k2.6:cloud' in the model picker.
    """


    def test_no_duplicate_when_live_clean_and_mdev_suffixed(self, tmp_path, monkeypatch):
        """Live API returns clean ID; mdev has :cloud variant — result has exactly one entry."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        mock_mdev = {
            "ollama-cloud": {
                "models": {
                    "kimi-k2.6:cloud": {"tool_call": True},
                    "glm-5.1:cloud": {"tool_call": True},
                }
            }
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["kimi-k2.6", "glm-5.1"]), \
             patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert result.count("kimi-k2.6") == 1
        assert result.count("glm-5.1") == 1
        assert "kimi-k2.6:cloud" not in result
        assert "glm-5.1:cloud" not in result

    def test_unsuffixed_model_id_unchanged(self, tmp_path, monkeypatch):
        """Model IDs without :cloud / -cloud suffix are passed through unchanged."""
        from hermes_cli.models import fetch_ollama_cloud_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

        mock_mdev = {
            "ollama-cloud": {
                "models": {"nemotron-3-nano:30b": {"tool_call": True}}
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_mdev):
            result = fetch_ollama_cloud_models(force_refresh=True)

        assert "nemotron-3-nano:30b" in result

    def test_strip_suffix_helper(self):
        """Unit test for the _strip_ollama_cloud_suffix helper."""
        from hermes_cli.models_local import _strip_ollama_cloud_suffix

        assert _strip_ollama_cloud_suffix("kimi-k2.6:cloud") == "kimi-k2.6"
        assert _strip_ollama_cloud_suffix("glm-5.1:cloud") == "glm-5.1"
        assert _strip_ollama_cloud_suffix("qwen3-coder:480b-cloud") == "qwen3-coder:480b"
        assert _strip_ollama_cloud_suffix("nemotron-3-nano:30b") == "nemotron-3-nano:30b"
        assert _strip_ollama_cloud_suffix("") == ""
