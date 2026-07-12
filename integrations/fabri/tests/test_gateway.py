"""Gateway service tests (integrations/fabri/service/app.py + settings.py).

Runs with fabri NOT installed — mirrors the rest of this suite's zero-fabri
stance (conftest.py imports no fabri package either). Only FastAPI is needed,
so the whole module `importorskip`s on it; settings resolution and the HTTP
surface that never touches FabriService (health, the artifact path-jail, the
POST /runs origin guard) are exercised directly against the process env.

`create_app()` must build FabriService lazily (startup/lifespan), so a plain
`TestClient(app)` — used WITHOUT the context-manager form on purpose — never
triggers the fabri-dependent startup path.
"""

from __future__ import annotations

import pathlib
import sys

FABRI_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FABRI_ROOT))

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from service.app import create_app
from service.settings import (
    TOOLS_DIR,
    SettingsError,
    build_overrides,
    load_settings,
)


@pytest.fixture()
def gateway_env(monkeypatch, tmp_path):
    """Neutralize every provider/model env var and point the gateway at a
    throwaway runs dir + a non-existent dotenv, so a developer's real
    integrations/fabri/.env can never leak into the test's decisions.
    """
    for var in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "FABRI_LLM_PROVIDER",
        "FABRI_LLM_MODEL",
        "FABRI_LLM_NARRATOR_MODEL",
        "WEBSAM_GROUND_TEXT_STUB",
        "FABRI_GATEWAY_CORS_ORIGIN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FABRI_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FABRI_GATEWAY_DOTENV", str(tmp_path / "no.env"))
    return monkeypatch


# --- /health: provider auto-selection from the process env ------------------


def test_health_openai_only(gateway_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app = create_app()
    resp = TestClient(app).get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o"
    # no GEMINI key -> vid_ground_text is forced onto the offline stub.
    assert body["groundTextStub"] is True


def test_health_gemini_only(gateway_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    app = create_app()
    body = TestClient(app).get("/health").json()

    assert body["provider"] == "gemini"
    # GEMINI key present -> real text-grounding, stub off.
    assert body["groundTextStub"] is False


# --- load_settings: provider resolution + failure modes ---------------------


def test_startup_fails_without_any_key(gateway_env):
    with pytest.raises(SettingsError) as exc:
        load_settings({})
    message = str(exc.value)
    assert "OPENAI_API_KEY" in message
    assert "GEMINI_API_KEY" in message


def test_provider_preference_openai_wins_when_both(gateway_env):
    settings = load_settings({"OPENAI_API_KEY": "a", "GEMINI_API_KEY": "b"})
    assert settings.provider == "openai"


def test_explicit_provider_override(gateway_env):
    settings = load_settings(
        {
            "FABRI_LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "g",
            "OPENAI_API_KEY": "o",
        }
    )
    assert settings.provider == "gemini"
    assert settings.narrator_model == "gemini-2.5-flash-lite"


def test_explicit_provider_missing_key_fails(gateway_env):
    with pytest.raises(SettingsError) as exc:
        load_settings({"FABRI_LLM_PROVIDER": "anthropic"})
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_provider_without_default_model_fails(gateway_env):
    # anthropic has no baked-in default model in the gateway, so an explicit
    # FABRI_LLM_MODEL is required even once the key is present.
    with pytest.raises(SettingsError) as exc:
        load_settings(
            {"FABRI_LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "x"}
        )
    assert "FABRI_LLM_MODEL" in str(exc.value)


# --- build_overrides: the deep-merge payload handed to service.submit -------


def test_build_overrides_shape(gateway_env, tmp_path):
    s = load_settings({"OPENAI_API_KEY": "x", "FABRI_RUNS_DIR": str(tmp_path)})
    o = build_overrides(s, tmp_path / "sb")

    assert o["llm"] == {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "narrator": {"model": "gpt-4o-mini"},
    }
    assert o["tools"]["manifest_dir"] == ["builtin", str(TOOLS_DIR)]
    assert o["tools"]["sandbox_root"] == str(tmp_path / "sb")
    assert o["agent"]["system_prompt"].startswith("# ABSOLUTE SCOPE")


def test_relative_runs_dir_resolves_against_integration_dir(gateway_env):
    s = load_settings({"OPENAI_API_KEY": "x", "FABRI_RUNS_DIR": "runs"})
    assert s.runs_dir == (FABRI_ROOT / "runs").resolve()


# --- /runs/{session}/artifact: path-jail (never touches FabriService) -------


def test_artifact_path_jail(gateway_env, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app = create_app()

    run_dir = tmp_path / "run-fake"
    run_dir.mkdir()
    (run_dir / "out.mp4").write_bytes(b"\x00\x00")
    app.state.run_dirs["fake-session"] = run_dir

    client = TestClient(app)

    assert (
        client.get("/runs/fake-session/artifact?path=out.mp4").status_code == 200
    )
    # traversal out of the run dir -> jailed.
    assert (
        client.get(
            "/runs/fake-session/artifact?path=../../../etc/passwd"
        ).status_code
        == 403
    )
    # absolute path outside the run dir -> jailed.
    assert (
        client.get("/runs/fake-session/artifact?path=/etc/passwd").status_code
        == 403
    )
    # in-jail but non-existent -> 404.
    assert (
        client.get("/runs/fake-session/artifact?path=missing.mp4").status_code
        == 404
    )
    # unknown session -> 404.
    assert (
        client.get("/runs/unknown-session/artifact?path=out.mp4").status_code
        == 404
    )


# --- POST /runs: CORS/Origin guard runs before any fabri import -------------


def test_post_runs_rejects_disallowed_origin(gateway_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app = create_app()
    client = TestClient(app)

    files = {"video": ("clip.mp4", b"\x00\x00", "video/mp4")}
    data = {"task": "cut out the car"}

    # Disallowed origin: rejected up front, before FabriService is ever used.
    rejected = client.post(
        "/runs",
        headers={"Origin": "http://evil.example"},
        files=files,
        data=data,
    )
    assert rejected.status_code == 403

    # Allowed origin (Vite default): passes the guard, then 503s because fabri
    # is not installed in this env — proving the guard let it through.
    allowed = client.post(
        "/runs",
        headers={"Origin": "http://localhost:5173"},
        files=files,
        data=data,
    )
    assert allowed.status_code in {503}
