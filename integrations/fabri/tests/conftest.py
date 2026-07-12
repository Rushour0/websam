"""Shared fixtures for the fabri video_editing tool-script test suite.

Every test in this directory invokes the tool scripts exactly the way
fabri's runner does: `subprocess.run([*manifest["command"]], input=json,
cwd=<tools/video_editing dir>, env={FABRI_SANDBOX_ROOT: <tmp sandbox>})`,
reading one JSON object back off stdout. No fabri package is imported here.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent
FABRI_ROOT = TESTS_DIR.parent
TOOLS_DIR = FABRI_ROOT / "tools" / "video_editing"
REPO_ROOT = FABRI_ROOT.parents[1]
GOLDEN_VIDEO_DIR = REPO_ROOT / "tools" / "goldens" / "fixtures" / "video"
GOLDEN_CLIP = GOLDEN_VIDEO_DIR / "clip-256.mp4"

IOU_GATE = 0.90

sys.path.insert(0, str(TOOLS_DIR))
import _websam_ort as ort_core  # noqa: E402


def pytest_configure(config):
    # Registers the `e2e_agent` marker (test_agent_e2e.py) without needing to
    # touch pyproject.toml (not owned by this task). test_agent_e2e.py's own
    # skipif already keeps it out of the default run whenever fabri isn't
    # installed / GEMINI_API_KEY isn't set, so no addopts deselect is needed
    # for a plain `pytest -q` to stay green.
    config.addinivalue_line(
        "markers", "e2e_agent: full fabri-agent run, requires fabri + GEMINI_API_KEY"
    )


def load_manifest(tool_name: str) -> dict:
    """Tiny local re-implementation of ToolManifest.from_file's JSON load —
    does not import fabri (per docs/fabri-contracts.md §5.1)."""
    path = TOOLS_DIR / f"{tool_name}.json"
    return json.loads(path.read_text())


def run_tool(tool_name: str, args: dict, sandbox_root: pathlib.Path, extra_env: dict | None = None,
             timeout: float | None = None) -> tuple[int, dict, str]:
    """Invoke a fabri tool script the way the runner would.

    Returns (returncode, parsed_stdout_json, raw_stderr).
    """
    manifest = load_manifest(tool_name)
    command = manifest["command"]
    # command[0] is "python3" -- use the current interpreter (via uv run) instead
    # so this works inside the project's uv-managed venv without relying on a
    # system `python3` having the right deps installed.
    cmd = [sys.executable, *command[1:]]

    env = dict(os.environ)
    env["FABRI_SANDBOX_ROOT"] = str(sandbox_root)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        cmd,
        input=json.dumps(args),
        cwd=str(TOOLS_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout or manifest.get("timeout_s", 30),
    )
    try:
        out = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"{tool_name} did not print one JSON object on stdout.\n"
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr}"
        ) from e
    return proc.returncode, out, proc.stderr


@pytest.fixture()
def sandbox(tmp_path) -> pathlib.Path:
    """A fresh sandbox root with the golden clip copied in (paths are
    sandbox-jailed, so fixtures living under the repo's tools/goldens/ must be
    copied in first, per the task brief)."""
    root = tmp_path / "sandbox"
    root.mkdir()
    shutil.copy(GOLDEN_CLIP, root / "clip-256.mp4")
    return root


@pytest.fixture(scope="session")
def golden_meta() -> dict:
    return json.loads((GOLDEN_VIDEO_DIR / "golden-video-meta.json").read_text())


@pytest.fixture(scope="session")
def golden_masks(golden_meta) -> list:
    import numpy as np  # noqa: F401

    out = []
    for name in golden_meta["masks"]:
        rle = json.loads((GOLDEN_VIDEO_DIR / name).read_text())
        out.append(ort_core.rle_decode(rle))
    return out
