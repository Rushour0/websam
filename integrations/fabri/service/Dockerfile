# syntax=docker/dockerfile:1
#
# websam-fabri-gateway — generic self-host image for the fabri video-editing
# gateway. No cloud provider, container registry, or hosting account is
# referenced anywhere: build it here, run it wherever you like.
#
# BUILD (run from the REPO ROOT). fabri is BUSL-licensed and is NOT vendored
# into this repo; you supply it at build time from your own checkout via a
# BuildKit named context called exactly `fabri-src` (the sibling compose file
# uses the same name):
#
#   docker build -f integrations/fabri/service/Dockerfile \
#     --build-context fabri-src=/path/to/your/fabri \
#     -t websam-fabri-gateway .
#
# REQUIREMENTS:
#   (b) BuildKit is required (named build contexts + `# syntax` above).
#       It is the default since Docker 23; on older Docker export
#       DOCKER_BUILDKIT=1 before building.
#   (c) tools/goldens/models-cache/edgetam/ holds the EdgeTAM ONNX weights.
#       Those files are git-ignored and are NOT in the repo — populate them
#       locally before building or the tracking core has no model to load.
#   (d) Secrets and local state are kept OUT of the build context by the
#       sibling integrations/fabri/service/Dockerfile.dockerignore: the built
#       image NEVER contains any .env, runs/, or .venv (verifier-blocker
#       resolution — API keys and run outputs must never bake into a layer).
#
# API keys are NOT baked in. There are deliberately no *_API_KEY ENV lines
# below — keys arrive at run time, e.g.:
#   docker run --env-file integrations/fabri/.env ... websam-fabri-gateway
# or via `env_file:` in the sibling compose file.

FROM python:3.12-slim

# fabri (BUSL) supplied by the deployer's own checkout via the `fabri-src`
# named build context — never vendored into this repo.
COPY --from=fabri-src . /opt/fabri
RUN pip install --no-cache-dir /opt/fabri

# Runtime deps for the gateway + the video_editing tools. Keep this list in
# sync with integrations/fabri/pyproject.toml [project].dependencies: that
# pyproject sets [tool.uv] package = false and has NO [build-system] table,
# so pip cannot install integrations/fabri itself as a project — its deps are
# installed explicitly here instead.
RUN pip install --no-cache-dir \
    onnxruntime numpy pillow imageio imageio-ffmpeg google-genai \
    fastapi 'uvicorn[standard]' python-multipart python-dotenv

# CRITICAL layout constraint: _websam_ort.py locates the EdgeTAM weights via
#   Path(__file__).resolve().parents[4] / tools/goldens/models-cache/edgetam
# i.e. it walks four parents up from integrations/fabri/tools/video_editing/
# to the repo root, then down into tools/goldens/... . The COPYs below MUST
# preserve that repo depth so parents[4] still resolves — do not flatten.
COPY integrations/fabri /repo/integrations/fabri
COPY tools/goldens/models-cache/edgetam /repo/tools/goldens/models-cache/edgetam

WORKDIR /repo/integrations/fabri

# Per-run sandbox/output root inside the container. Mount a volume here to
# persist run artifacts; the compose file wires this up.
ENV FABRI_RUNS_DIR=/data/runs

EXPOSE 8787

# Bind 0.0.0.0 — this is correct ONLY inside a container, where the container
# boundary is the exposure control. Local bare-metal runs bind 127.0.0.1.
CMD ["sh", "-c", "uvicorn --factory service.app:create_app --host 0.0.0.0 --port ${FABRI_GATEWAY_PORT:-8787}"]
