"""vid_ground_text — text -> prompt grounding for a single video frame.

Reads one JSON object from stdin: {frame, phrase}. Writes one JSON object to
stdout: {boxes, points, chosen} (SOURCE-pixel coords of `frame`). Exit 0 on
success, exit 1 with {"error": "..."} on failure — mirrors the fetch_url.py /
write_file.py contract in fabri-source (scripts print their raw result
payload; the runner wraps it in {ok, result, ...}, so this script must NOT
self-wrap).

Two paths, chosen at call time (no manifest flag — matches
docs/fabri-contracts.md §3.2):

1. STUB (WEBSAM_GROUND_TEXT_STUB env var set) — fully deterministic, zero
   network, zero API key. The env var value is either:
     - inline JSON, e.g. '{"box": [10, 20, 110, 220]}', or
     - a path (sandbox-relative or absolute) to a JSON fixture file with the
       same shape.
   Accepted shapes (all normalized into the full {boxes, points, chosen}
   output below): {"box": [x1,y1,x2,y2]}, {"boxes": [[...], ...]},
   {"point": {"x":.., "y":.., "label"?:..}}, {"points": [...]}, or an
   already-full {"boxes":..., "points":..., "chosen":...}.

2. REAL — calls Gemini (google-genai) via structured JSON output to locate
   the phrase in the image, then rescales the model's normalized [0,1]
   fractional box back to the frame's real pixel dimensions (PIL-read
   width/height). Requires GEMINI_API_KEY (same env var the fabri
   orchestrator itself uses — one key covers both roles). The google-genai
   import is LAZY (deferred into the real-call function) so this file still
   parses and the stub path still works with google-genai NOT installed;
   only a real (non-stub) call without the package fails, with a clear
   tool_error message.

NOTE for the wave-2 pyproject owner: this tool needs `google-genai` added to
integrations/fabri/pyproject.toml for the REAL path to work. Not added here
— I don't own pyproject.toml per my task brief. The stub path has zero
dependency on it.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys


def _sandbox_root() -> pathlib.Path:
    root = os.environ.get("FABRI_SANDBOX_ROOT")
    return pathlib.Path(root).resolve() if root else pathlib.Path.cwd().resolve()


def _resolve_in_sandbox(raw: str) -> pathlib.Path:
    """Resolve `raw` against $FABRI_SANDBOX_ROOT and refuse escapes.
    Mirrors the read_file.py/write_file.py `target.is_relative_to(root)`
    pattern referenced in docs/fabri-contracts.md §1/§3."""
    root = _sandbox_root()
    candidate = pathlib.Path(raw)
    target = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"refused: path {raw!r} escapes sandbox root {root}")
    return target


def _normalize_box(box) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2, y2]


def _normalize_point(point) -> dict:
    if isinstance(point, dict):
        return {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "label": point.get("label", "positive"),
        }
    x, y = point
    return {"x": float(x), "y": float(y), "label": "positive"}


def _build_result(boxes=None, points=None, chosen=None) -> dict:
    boxes = [_normalize_box(b) for b in (boxes or [])]
    points = [_normalize_point(p) for p in (points or [])]
    if chosen is None:
        if boxes:
            chosen = {"box": boxes[0]}
        elif points:
            chosen = {"point": points[0]}
        else:
            raise ValueError("no box or point produced")
    return {"boxes": boxes, "points": points, "chosen": chosen}


def _load_stub_payload(stub_value: str) -> dict:
    """`stub_value` is either inline JSON or a path to a JSON fixture file
    (sandbox-relative or absolute)."""
    try:
        return json.loads(stub_value)
    except (json.JSONDecodeError, TypeError):
        pass
    # Not inline JSON -> treat as a path.
    candidate = pathlib.Path(stub_value)
    if not candidate.is_absolute():
        candidate = _resolve_in_sandbox(stub_value)
    with open(candidate, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_stub(stub_value: str) -> dict:
    payload = _load_stub_payload(stub_value)
    # Already a full/near-full result?
    if "boxes" in payload or "points" in payload or "chosen" in payload:
        return _build_result(
            boxes=payload.get("boxes"),
            points=payload.get("points"),
            chosen=payload.get("chosen"),
        )
    if "box" in payload:
        return _build_result(boxes=[payload["box"]])
    if "point" in payload:
        return _build_result(points=[payload["point"]])
    raise ValueError(
        f"WEBSAM_GROUND_TEXT_STUB payload has none of box/boxes/point/points/chosen: {payload!r}"
    )


def _run_real(frame_path: pathlib.Path, phrase: str) -> dict:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY not set and WEBSAM_GROUND_TEXT_STUB not set — "
            "vid_ground_text needs one of the two to run"
        )
    if not frame_path.exists():
        raise FileNotFoundError(f"frame not found: {frame_path}")

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "google-genai is not installed. Add it to integrations/fabri/pyproject.toml "
            "(this tool's real, non-stub path needs it) or set WEBSAM_GROUND_TEXT_STUB "
            "for an offline/deterministic run."
        ) from e

    from PIL import Image

    with Image.open(frame_path) as im:
        width, height = im.size

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    schema = {
        "type": "OBJECT",
        "properties": {
            "box": {
                "type": "ARRAY",
                "description": (
                    "Tight bounding box around the described object, as fractional "
                    "[x1, y1, x2, y2] in [0, 1], top-left origin, x2>x1 and y2>y1."
                ),
                "items": {"type": "NUMBER"},
                "minItems": 4,
                "maxItems": 4,
            },
            "point": {
                "type": "OBJECT",
                "description": "Fractional [0,1] center point of the object, as a fallback.",
                "properties": {"x": {"type": "NUMBER"}, "y": {"type": "NUMBER"}},
            },
        },
        "required": ["box"],
    }

    prompt = (
        f"Locate the object described as: {phrase!r} in this image. "
        "Return a tight bounding box as fractional coordinates in [0, 1] "
        "(top-left origin)."
    )

    with open(frame_path, "rb") as f:
        image_bytes = f.read()
    mime = "image/png" if frame_path.suffix.lower() == ".png" else "image/jpeg"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    parsed = json.loads(response.text)
    frac_box = parsed["box"]
    px_box = [
        frac_box[0] * width,
        frac_box[1] * height,
        frac_box[2] * width,
        frac_box[3] * height,
    ]

    points = None
    if parsed.get("point"):
        points = [
            {
                "x": parsed["point"]["x"] * width,
                "y": parsed["point"]["y"] * height,
                "label": "positive",
            }
        ]

    return _build_result(boxes=[px_box], points=points, chosen={"box": px_box})


def main() -> int:
    try:
        args = json.loads(sys.stdin.read())
        frame = args["frame"]
        phrase = args["phrase"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(json.dumps({"error": f"bad input: {e}"}))
        return 1

    try:
        frame_path = _resolve_in_sandbox(frame)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        return 1

    stub_value = os.environ.get("WEBSAM_GROUND_TEXT_STUB")
    try:
        if stub_value:
            result = _run_stub(stub_value)
        else:
            result = _run_real(frame_path, phrase)
    except Exception as e:  # noqa: BLE001 - tool_error contract: always report, never crash silently
        print(json.dumps({"error": f"tool_error: {e}"}))
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
