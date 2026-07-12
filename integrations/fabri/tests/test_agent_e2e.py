"""OPTIONAL full fabri-agent end-to-end run (docs/fabri-contracts.md §5.3).

Skipped unless BOTH `fabri` is importable (pip install -e ~/gba/fabri) AND
GEMINI_API_KEY is set. Excluded from the default `pytest -q` gate by the
project's pytest markers config; run explicitly with `pytest -m e2e_agent`.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil

import pytest

pytestmark = pytest.mark.e2e_agent

FABRI_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = FABRI_ROOT.parents[1]
GOLDEN_CLIP = REPO_ROOT / "tools" / "goldens" / "fixtures" / "video" / "clip-256.mp4"

_SKIP_REASON = "requires `pip install -e ~/gba/fabri` + GEMINI_API_KEY (see README.md)"
_SKIP = importlib.util.find_spec("fabri") is None or not os.environ.get("GEMINI_API_KEY")


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_track_and_export_matte(tmp_path):
    from fabri import build_llm, build_tool_defs, build_tools, run_agent
    from fabri.config import load_config

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    shutil.copy(GOLDEN_CLIP, project_dir / "clip-256.mp4")

    os.environ["FABRI_SANDBOX_ROOT"] = str(project_dir)

    config = load_config(str(FABRI_ROOT / ".agent" / "fabri_agent.yaml"))
    llm = build_llm(config)
    tool_defs = build_tool_defs(config)
    tools = build_tools(config, tool_defs)

    result = run_agent(
        "track the object in clip-256.mp4 and export a matte",
        llm,
        tools,
        store=None,
        max_steps=config["agent"]["max_steps"],
    )

    assert result["outcome"] == "success", result
    output_path = result["response"]["outputPath"]
    assert (project_dir / output_path).exists()
