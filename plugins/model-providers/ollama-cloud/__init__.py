"""Ollama Cloud provider profile.

Ollama Cloud's OpenAI-compatible ``/v1/chat/completions`` endpoint
supports top-level ``reasoning_effort`` with values ``none``, ``low``,
``medium``, ``high``, and ``max`` (the last being undocumented but
empirically confirmed for DeepSeek V4 — ``max`` produces ~2.5× more
thinking tokens than ``high``).

This profile maps Hermes's ``xhigh`` → ``max`` to unlock DeepSeek V4's
"Max thinking" tier through Ollama Cloud.  ``low`` / ``medium`` / ``high``
pass through unchanged.

When reasoning is explicitly disabled (``enabled: false`` or
``effort: "none"``), ``reasoning_effort`` is omitted entirely so the
model runs in non-thinking mode.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    """Ollama Cloud — maps xhigh→max via top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit top-level ``reasoning_effort`` for Ollama Cloud thinking models.

        Gated on ``supports_reasoning``, which the transport resolves from the
        model's native ``/api/show`` ``capabilities`` (``thinking``). Models
        without the thinking capability (e.g. ``gemma3``, ``qwen3-coder``) get
        no ``reasoning_effort`` at all — emitting it there is a no-op the API
        ignores, and gating avoids sending a meaningless field.
        """
        top_level: dict[str, Any] = {}

        if not supports_reasoning:
            return {}, {}

        if reasoning_config and isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled", True)
            if enabled is False:
                # Ollama Cloud defaults to thinking ON, and ignores the
                # extra_body.thinking:{type:disabled} shape (verified live).
                # The ONLY way to actually suppress thinking on its
                # /v1/chat/completions endpoint is top-level
                # reasoning_effort:"none" — omitting the field leaves
                # thinking on.
                return {}, {"reasoning_effort": "none"}

            effort = (reasoning_config.get("effort") or "").strip().lower()
            if not effort:
                # No explicit effort requested — let the model decide
                # (Ollama Cloud's server default is thinking ON).
                return {}, {}
            if effort == "none":
                return {}, {"reasoning_effort": "none"}  # explicit off switch
            # Accepted set {none, low, medium, high, max} is declared in
            # agent.reasoning_effort ("minimal" is rejected with HTTP 400 →
            # clamps to low; xhigh rounds up to max). Bespoke levels outside
            # the ladder are omitted so the model applies its own default
            # rather than triggering a hard 400.
            from agent.reasoning_effort import (
                OLLAMA_CLOUD_EFFORTS,
                OLLAMA_CLOUD_OVERRIDES,
                clamp_effort,
            )

            clamped = clamp_effort(
                effort, OLLAMA_CLOUD_EFFORTS, OLLAMA_CLOUD_OVERRIDES
            )
            if clamped in OLLAMA_CLOUD_EFFORTS:
                top_level["reasoning_effort"] = clamped

        return {}, top_level

    def supported_reasoning_efforts(self, model: str | None = None) -> list[str] | None:
        """Ollama Cloud's accepted reasoning-effort contract.

        The /v1/chat/completions endpoint accepts {low, medium, high, max,
        none} and rejects ``minimal``/``xhigh``/``ultra`` with HTTP 400.
        Empirically (live probing 2026-08-20), DeepSeek V4's low/medium/high
        produce statistically indistinguishable reasoning volume — only ``max``
        (and ``none``) are behaviorally distinct, matching the model page's
        documented Non-Think / High / Max modes. So:

        - DeepSeek V4 family → the honest 3-mode set {high, max, none}
          (what the UI should advertise).
        - Other models → the full wire-accepted set, since their per-tier
          behavior is not known to collapse.

        Note: this reports what the backend ACCEPTS; whether a specific model
        supports thinking at all is resolved separately (the /api/show
        capability probe), and callers gate on that first.
        """
        bare = str(model or "").strip()
        # Strip Ollama's `:cloud` / `:<date>-cloud` suffix shapes.
        bare = bare.split("/")[-1]
        for suffix in (":cloud", "-cloud"):
            if bare.endswith(suffix):
                bare = bare[: -len(suffix)]
                break
        bare = bare.lower()
        if "deepseek" in bare and "v4" in bare:
            return ["high", "max", "none"]
        return ["low", "medium", "high", "max", "none"]


ollama_cloud = OllamaCloudProfile(
    name="ollama-cloud",
    aliases=("ollama_cloud",),
    default_aux_model="nemotron-3-nano:30b",
    env_vars=("OLLAMA_API_KEY",),
    base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)
