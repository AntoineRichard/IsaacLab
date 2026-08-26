# Rendered once from tools/odin/templates/Dockerfile.j2 via tools/odin/image.py,
# then hand-adapted for OSMO's kaniko git context (see
# tools/odin/osmo_build/README.md). tools/odin/osmo_build/render.py renders the
# OSMO workflow that builds this file; it does not render Dockerfiles.
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
# from zero, but the measured build (see the README's "Measured sizing")
# needed a single attempt with zero retries -- kaniko's full-filesystem
# snapshot and the registry push, not this step, are where a slow OSMO build
# is actually spent.
#
# Without a cache mount, anything uv writes under UV_CACHE_DIR lands in this
# layer permanently -- templates/Dockerfile.j2 avoids that with
# `RUN --mount=type=cache`, which also keeps the cache out of the image; that
# mount is unavailable here, so `rm -rf` the cache in the same RUN instead.
RUN uv sync --frozen --extra isaacsim --extra ovphysx --extra ovrtx --extra rsl-rl --extra skrl --extra rl-games --extra sb3 --extra rerun --extra video --extra tetrahedralization \
    && rm -rf "$UV_CACHE_DIR"

# The nvdataset CLI, used by the DSS upload path in dispatch.yaml.j2. It is
# NOT installed from an index here: OSMO's build pods cannot resolve
# artifactory.pdx.nvidia.com and have no route to artifactory.nvidia.com, and
# the package is not on public PyPI. They can reach nvcr.io, so it arrives
# pre-installed in a carrier image instead.
#
# TEMPORARY. Delete this and restore the plain `uv tool install` once OSMO can
# reach artifactory. See tools/odin/osmo_build/README.md for the carrier's
# rebuild command and the full rationale.
#
# `COPY --from` needs no extra credential: kaniko pulls it with the same
# /kaniko/.docker/config.json injected for the push, which carries pull scope.
#
# The copied tree is self-contained: the launcher's shebang points at
# /root/.local/share/uv/tools/nvdataset/bin/python, which travels with it, so
# this image needs no Python of its own. The carrier shares this base for ABI
# compatibility -- that interpreter is a real binary linked against the
# carrier's glibc. `nvdataset --version` below turns any mismatch into a build
# failure rather than an image whose CLI cannot start.
COPY --from=nvcr.io/nvidian/antoiner-isaac-lab:nvdataset-0.96.0 /root/.local /root/.local

RUN ln -sf /root/.local/bin/nvdataset /usr/local/bin/nvdataset \
    && nvdataset --version
