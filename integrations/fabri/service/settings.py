"""Gateway settings: resolve the fabri LLM overrides from a git-ignored .env.

Pure stdlib + typing only — this module deliberately does NOT import fabri, so
the test suite runs green even when fabri is not installed (fabri is BUSL and
must be installed separately by whoever deploys this; see
integrations/fabri/README.md). Everything provider/secret-related is driven by
the process env (populated from integrations/fabri/.env via python-dotenv at the
FastAPI app's entrypoint), never hardcoded.

The user configures EITHER a Gemini key OR an OpenAI key and it "just works":
`load_settings` picks the provider from FABRI_LLM_PROVIDER if set, else prefers
OpenAI, else Gemini, else fails loudly. `build_overrides` turns the resolved
Settings into the deep-merge overrides dict that FabriService.submit() applies
onto the template fabri_agent.yaml per run.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# --- Paths (this file lives at integrations/fabri/service/settings.py) --------
SERVICE_DIR = Path(__file__).resolve().parent
FABRI_INTEGRATION_DIR = SERVICE_DIR.parent
TEMPLATE_CONFIG = FABRI_INTEGRATION_DIR / ".agent/fabri_agent.yaml"
ORCHESTRATOR_PROMPT = FABRI_INTEGRATION_DIR / ".agent/prompts/orchestrator.md"
TOOLS_DIR = FABRI_INTEGRATION_DIR / "tools/video_editing"

# --- Provider knowledge (mirrors fabri, but never imported from fabri) --------
# Matches fabri's Provider StrEnum (src/fabri/core/llm.py:22-26).
KNOWN_PROVIDERS = {"gemini", "anthropic", "openai", "openrouter", "bedrock"}

# The NAME of the env var each provider's backend reads at construction time
# (fabri llm.py:177, os.environ.get(api_key_env)). bedrock is intentionally
# absent: it authenticates via the AWS credential chain, not a single key var.
PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# We only auto-fill a default model for the two providers the user actually
# targets from .env (OpenAI now, Gemini later). Any other provider must set
# FABRI_LLM_MODEL explicitly (see load_settings).
DEFAULT_MODEL = {"gemini": "gemini-2.5-pro", "openai": "gpt-4o"}
DEFAULT_NARRATOR_MODEL = {
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4o-mini",
}

# WHY this must be inline JSON and never a bare "1":
# vid_ground_text.py:93-123 json-parses the VALUE of WEBSAM_GROUND_TEXT_STUB and
# requires the payload to carry one of box/boxes/point/points/chosen. A bare "1"
# parses as the int 1, has none of those keys, and the tool raises TypeError ->
# tool_error. A minimal well-formed inline stub keeps the offline path valid.
DEFAULT_GROUND_TEXT_STUB = '{"point": {"x": 128, "y": 128}}'


class SettingsError(RuntimeError):
    """Raised when the gateway cannot resolve a valid LLM configuration."""


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    api_key_env: str | None
    narrator_model: str
    cors_origins: list[str]
    port: int
    runs_dir: Path
    ground_text_stub_forced: bool


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    """Resolve gateway Settings from the process env (loaded from .env).

    Provider decision order is exact and load-bearing:
      1. explicit FABRI_LLM_PROVIDER (validated + its key var required)
      2. else OpenAI if OPENAI_API_KEY present
      3. else Gemini if GEMINI_API_KEY present
      4. else fail loudly naming the missing vars
    """
    # (1) explicit provider override.
    provider = env.get("FABRI_LLM_PROVIDER", "").strip().lower() or None
    if provider is not None:
        if provider not in KNOWN_PROVIDERS:
            raise SettingsError(
                f"FABRI_LLM_PROVIDER={provider!r} is not a known provider; "
                f"allowed: {sorted(KNOWN_PROVIDERS)}"
            )
        key_env = PROVIDER_KEY_ENV.get(provider)
        # bedrock has no key_env (AWS credential chain) -> nothing to require.
        if key_env is not None and not env.get(key_env):
            raise SettingsError(
                f"FABRI_LLM_PROVIDER={provider!r} requires {key_env} to be set "
                f"and non-empty in integrations/fabri/.env"
            )
    # (2) prefer OpenAI, (3) then Gemini.
    elif env.get("OPENAI_API_KEY"):
        provider = "openai"
    elif env.get("GEMINI_API_KEY"):
        provider = "gemini"
    # (4) nothing configured -> fail loudly.
    else:
        raise SettingsError(
            "No LLM credentials configured: set OPENAI_API_KEY or GEMINI_API_KEY "
            "(or FABRI_LLM_PROVIDER plus its provider key) in "
            "integrations/fabri/.env"
        )

    # Model: explicit override, else our per-provider default. Required because
    # the template YAML pins model: gemini-2.5-pro which would otherwise be sent
    # verbatim to an overridden provider (e.g. OpenAI would reject it).
    model = env.get("FABRI_LLM_MODEL") or DEFAULT_MODEL.get(provider)
    if model is None:
        raise SettingsError(
            f"FABRI_LLM_MODEL is required for provider {provider!r}"
        )

    api_key_env = PROVIDER_KEY_ENV.get(provider)  # None for bedrock.

    # Narrator model: explicit override, else our per-provider narrator default,
    # else fall back to the main model. REQUIRED because fabri's role
    # normalization (config.py _normalize_llm_roles) makes llm.narrator inherit
    # the overridden provider while the template pins narrator.model
    # gemini-2.5-flash-lite; without this override an OpenAI-only run would send
    # a Gemini model id to OpenAI and silently lose ALL narration (narrator
    # errors are swallowed at debug level, agent.py:389-404). Note we override
    # narrator.MODEL rather than nulling the block: 'narrator: null' cannot be
    # merged over the template's dict — fabri config.py _deep_merge:276-280
    # raises ConfigError on a scalar-over-mapping merge.
    narrator_model = (
        env.get("FABRI_LLM_NARRATOR_MODEL")
        or DEFAULT_NARRATOR_MODEL.get(provider)
        or model
    )

    # CORS/Origin matching in the gateway is exact-string; the default covers
    # both the localhost and 127.0.0.1 spellings of Vite's dev origin.
    cors_origins = [
        o.strip()
        for o in env.get(
            "FABRI_GATEWAY_CORS_ORIGIN",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if o.strip()
    ]

    port = int(env.get("FABRI_GATEWAY_PORT", "8787"))

    # Runs dir: relative FABRI_RUNS_DIR resolves against the integration dir,
    # NEVER the process cwd — this prevents uploaded user videos landing outside
    # the git-ignored integrations/fabri/runs/ when the gateway is launched from
    # the repo root.
    raw_runs = env.get("FABRI_RUNS_DIR")
    if raw_runs is None:
        runs_dir = FABRI_INTEGRATION_DIR / "runs"
    else:
        candidate = Path(raw_runs)
        if candidate.is_absolute():
            runs_dir = candidate
        else:
            runs_dir = FABRI_INTEGRATION_DIR / candidate
    runs_dir = runs_dir.resolve()

    # Force the offline text-grounding stub when there's no Gemini key (the tool
    # calls Gemini vision directly) and the operator hasn't already set a stub.
    ground_text_stub_forced = (not env.get("GEMINI_API_KEY")) and (
        not env.get("WEBSAM_GROUND_TEXT_STUB")
    )

    return Settings(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        narrator_model=narrator_model,
        cors_origins=cors_origins,
        port=port,
        runs_dir=runs_dir,
        ground_text_stub_forced=ground_text_stub_forced,
    )


def build_overrides(settings: Settings, sandbox_root: Path) -> dict:
    """Build the deep-merge overrides dict for FabriService.submit().

    Applied per run onto the template fabri_agent.yaml (fabri
    service/binding.py merge_overrides/bind_run_config).
    """
    return {
        # agent.system_prompt is REQUIRED because fabri has NO --system-prompt-file
        # CLI flag (verified against cli.py) — it's a literal string per
        # run_config.py:79, and FabriService launches plain
        # `python -m fabri.cli run --config` (launcher.py:31-53). So we read the
        # orchestrator prompt file here and inline its contents.
        "agent": {
            "system_prompt": ORCHESTRATOR_PROMPT.read_text(encoding="utf-8"),
        },
        "llm": {
            "provider": settings.provider,
            "model": settings.model,
            **(
                {"api_key_env": settings.api_key_env}
                if settings.api_key_env
                else {}
            ),
            "narrator": {"model": settings.narrator_model},
        },
        "tools": {
            # manifest_dir must be ABSOLUTE because fabri resolves it
            # cwd-relatively (runtime.py:47-48); keep the 'builtin' sentinel
            # first so builtin tools stay registered.
            "manifest_dir": ["builtin", str(TOOLS_DIR)],
            "sandbox_root": str(sandbox_root),
        },
        # sqlite_path absolute so fabri's sqlite memory persists under
        # FABRI_RUNS_DIR instead of whatever cwd the gateway runs from.
        "memory": {"sqlite_path": str(settings.runs_dir / "memory.db")},
    }
