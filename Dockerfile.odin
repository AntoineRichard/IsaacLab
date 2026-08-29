# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Builds the benchmark/training image for OSMO dispatches, from a kaniko git
# context pinned to one commit. The Odin dispatcher that renders the build
# workflow around this file lives outside this repository, on the
# `antoiner/feat/odin-nvdataset-carrier` branch of isaac-sim/IsaacLab
# (`tools/odin/osmo_build/`); this file is the only piece that has to travel
# with the commit under test, because the builder fetches it from the context.
#
# It is committed once and reused for every future build, so it must not
# hardcode anything specific to the commit that first rendered it.
#
# The CUDA minor version must track the torch index [tool.uv.sources] selects
# for linux/x86_64, which is cu128.
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

# A second environment, holding the standalone importers and deliberately NOT
# `isaacsim`. Assets that are generated rather than committed -- the MicroDuck
# USDs, whose converter is scripts/tools/convert_microduck.py -- have to be built
# in the job's prologue, and that converter only takes its kit-less path when
# `isaaclab.utils.version.standalone_importers_available()` is true.
#
# The two extras cannot share one environment, which is not a resolver conflict
# and so is invisible to uv.lock: `isaacsim-asset-isolated` contributes
# `isaacsim.asset` as a PEP 420 namespace portion, while the Isaac Sim runtime
# ships `isaacsim` as a regular package, and a regular package discards every
# namespace portion for that name. Installed together, the wheel is present but
# unreachable, the converter falls back to demanding a full Kit runtime, and Kit
# then fails to start in this image on a USD ABI mismatch -- exiting 0 while
# writing no asset. Measured on workflow microduck-convert-probe-1.
#
# Training keeps using the main environment; only the prologue uses this one.
ENV ODIN_CONVERT_VENV=/opt/convenv
RUN UV_PROJECT_ENVIRONMENT="$ODIN_CONVERT_VENV" uv sync --frozen --extra importers \
    && rm -rf "$UV_CACHE_DIR" \
    && UV_PROJECT_ENVIRONMENT="$ODIN_CONVERT_VENV" uv run --frozen --extra importers \
         python -c "from isaaclab.utils.version import standalone_importers_available as s; assert s(), 'standalone importers unreachable in the convert env'; print('convert env OK')"

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
