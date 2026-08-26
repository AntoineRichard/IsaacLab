# Rendered once by tools/odin/image.py + tools/odin/osmo_build/render.py, then
# hand-adapted for OSMO's kaniko git context (see tools/odin/osmo_build/README.md).
# This file is committed and reused for every future OSMO build, so it must not
# hardcode anything specific to the commit that first rendered it.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_CACHE_DIR=/opt/uv-cache
ENV PATH=/root/.local/bin:$PATH

# libgl1/libegl1 provide the GL loader Newton's headless video recorder reaches
# through pyglet. The CUDA runtime image ships no libGL at all, so without them
# play fails with `Library "GL" not found`, surfaced as "requires pyglet to be
# installed" even though pyglet is present. The NVIDIA driver's own GL is
# injected at run time and needs the graphics capability, set in dispatch.yaml.j2.
# libopengl0 carries libOpenGL.so.0, which OvPhysX links directly; libxt6 silences
# a loader warning from the Isaac Sim stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        libgl1 libegl1 libopengl0 libxt6 \
    && rm -rf /var/lib/apt/lists/*

# uv manages its own Python (pyproject sets python-preference = "only-managed"),
# so the base image needs no interpreter.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# The git context is the repository at the pinned commit, including .git, so
# capture_versions() still records real provenance. Replaces the bundle clone.
# (Dockerfile.odin.dockerignore, not the root .dockerignore, governs what this
# COPY sees for this file -- the root file excludes .git/ for the production
# container images, which must not gain a real .git.)
COPY . /workspace/isaaclab

WORKDIR /workspace/isaaclab

# ODIN_COMMIT_SHA is supplied per build by render_build_workflow(); this file is
# committed once and reused for every future build, so the pinned commit cannot
# be hardcoded here. Fails loudly if the git context resolved to the wrong commit.
ARG ODIN_COMMIT_SHA
RUN test "$(git rev-parse HEAD)" = "$ODIN_COMMIT_SHA"

# uv sync resolves against this commit's committed uv.lock, so a commit that
# changes a dependency is tested with the changed dependency.
#
# UV_HTTP_TIMEOUT is raised well above the 30s default because large wheels
# behind a VPN routinely stall longer than that without the transfer being dead.
ENV UV_HTTP_TIMEOUT=180

# Cache mounts removed: the OSMO builder does not support them (see
# tools/odin/osmo_build/README.md). A dropped transfer restarts this step
# from zero, so treat a network failure here as expected-flaky, not a bug.
RUN uv sync --frozen --extra isaacsim --extra ovphysx --extra ovrtx --extra rsl-rl --extra skrl --extra rl-games --extra sb3 --extra rerun --extra video --extra tetrahedralization

# KNOWN GAP: the nvdataset CLI install is dropped for OSMO builds. It used to
# live here as a `uv tool install` from artifactory.pdx.nvidia.com (the NGC
# data platform index; the package is not on public PyPI). The OSMO kaniko
# build pod cannot resolve artifactory.pdx.nvidia.com (confirmed: every other
# host this Dockerfile touches -- github.com, pypi.org, nvcr.io,
# api.ngc.nvidia.com -- resolves fine; this is one host, not a network
# boundary). Images built from this Dockerfile therefore do NOT have
# `nvdataset` and CANNOT run the DSS upload path in dispatch.yaml.j2 (the
# `nvdataset upload` call). Restore this step once artifactory.pdx DNS is
# reachable from OSMO pools -- see tools/odin/osmo_build/README.md.
