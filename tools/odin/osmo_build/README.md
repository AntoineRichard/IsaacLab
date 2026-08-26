# Building Odin benchmark images inside OSMO

Odin's local build path (`odin build-image`) shells out to `docker build` and
ships the commit as a `git bundle`. Neither works on OSMO: tasks get no Docker
daemon and no local repository. This directory holds the OSMO-native build.

## Chosen builder

**kaniko.** Two single-task OSMO workflows (`builder-probe-kaniko-1`,
`builder-probe-buildkit-1`) each built a minimal Dockerfile containing
`RUN --mount=type=cache,target=/opt/probe-cache,sharing=locked` and pushed the
result to `nvcr.io/nvidian/antoiner-isaac-lab:probe-{builder}`.

`builder-probe-kaniko-1` reached **COMPLETED**. Kaniko unpacked the rootfs to
honor the mount, ran the command, and pushed
`nvcr.io/nvidian/antoiner-isaac-lab@sha256:6c4b75809a5554ac2652d2be56299071b76b47c8664659e53f29b375ef715f51`.
The deciding log line:

```
Unpacking rootfs as cmd RUN --mount=type=cache,target=/opt/probe-cache,sharing=locked     echo cache-mount-ok > /probe.txt requires it.
```

`builder-probe-buildkit-1` reached **FAILED** before it ever reached the
Dockerfile: rootless `buildkitd` could not start inside the OSMO task sandbox.
The deciding log line:

```
[rootlesskit:child ] error: failed to share mount point: /: permission denied
```

Per the decision rule (BuildKit FAILED, kaniko COMPLETED -> kaniko), Odin's
OSMO build path uses kaniko. Because kaniko's cache-mount support here is
handled by unpacking the full rootfs rather than a real cache-preserving
mount, the OSMO Dockerfile drops both `RUN --mount=type=cache` flags —
kaniko runs the mounted command but does not persist the cache mount's
contents between builds the way BuildKit would.

## Which Dockerfile is which

- **`Dockerfile.odin`** (repo root, committed) — the OSMO path. Rendered once
  from `templates/Dockerfile.j2` and then hand-adapted for a kaniko git
  context: `COPY . /workspace/isaaclab` instead of the bundle clone, an
  `ARG ODIN_COMMIT_SHA` instead of a hardcoded commit, and no cache mounts, for
  the reason above. `tools/odin/osmo_build/render.py` renders the OSMO
  workflow that builds this file; it does not touch `Dockerfile.j2`.
- **`templates/Dockerfile.j2`** — the local path (`odin build-image`), unchanged.
  It keeps both `RUN --mount=type=cache` flags on purpose: local builds run
  under real Docker/BuildKit, where the mount persists `/opt/uv-cache` across
  builds and makes a dropped transfer resumable instead of a full restart.
  Do not strip the mounts here to "match" `Dockerfile.odin` — that would
  regress local build resumability for no OSMO-side benefit.

`Dockerfile.odin.dockerignore` (repo root, committed) exists so that
`Dockerfile.odin`'s `COPY .` keeps `.git` — kaniko prefers a
`<dockerfile-path>.dockerignore` over the repo's root `.dockerignore`, which
deliberately excludes `.git/` for the production container images. Isaac
Lab's `capture_versions()` needs a real `.git` for provenance; the production
images must not gain one.

## Usage

There is no CLI subcommand for this path yet (the local path has
`odin build-image`; this one does not). Render and submit by hand:

```python
import pathlib

from tools.odin.osmo_build.render import read_push_auth, render_build_workflow

auth_b64 = read_push_auth()  # reads ~/.docker/config.json by default
commit_sha = "<full 40-char commit SHA, pushed to the antoine fork>"

rendered = render_build_workflow(
    builder="kaniko",
    commit_sha=commit_sha,
    git_remote="https://github.com/antoine/IsaacLab.git",
    destination=f"nvcr.io/nvidian/antoiner-isaac-lab:{commit_sha[:7]}",
    auth_b64=auth_b64,
    cpu=16,
    memory="64Gi",
    storage="256Gi",
)

out_path = pathlib.Path("odin_runs") / f"osmo-build-{commit_sha[:7]}.yaml"
out_path.write_text(rendered)
```

```sh
osmo workflow submit odin_runs/osmo-build-<short-sha>.yaml \
    --pool isaac-lab-test-l40-07 --priority NORMAL
```

The rendered file embeds a live `nvcr.io` push credential in plain text
(`auth_b64`, under `files:`). It must stay under `odin_runs/`, which is
gitignored, and must not be committed or pasted elsewhere.

## Why the choice matters

`templates/Dockerfile.j2` uses `RUN --mount=type=cache` so that a dropped
transfer during a multi-gigabyte `uv sync` resumes instead of restarting, and
so the wheel cache never lands in an image layer. `Dockerfile.odin` cannot use
the mount, so it loses only the resumability half of that; the "keep the
cache out of the image" half is recovered with a plain `rm -rf` in the same
`RUN` (see the comment above `uv sync` in `Dockerfile.odin`).

In practice the lost resumability has not mattered: per "Measured sizing"
below, `uv sync` needed a single attempt with zero retries and was only ~14%
of a 26m07s build. The real cost is kaniko's full-filesystem snapshot
(11m27s) plus the registry push (8m50s) -- together ~78% of the build -- so
that is where future optimisation should be aimed, not at `uv sync`.

## Known gap: `nvdataset` is absent from OSMO-built images

`Dockerfile.odin` used to install the `nvdataset` CLI (the DSS upload path
used by `dispatch.yaml.j2`) from `artifactory.pdx.nvidia.com`, an internal
NVIDIA index that does not carry the package on public PyPI. The OSMO kaniko
build pod cannot resolve that host — every other host the Dockerfile touches
(`github.com`, `pypi.org`, `nvcr.io`, `api.ngc.nvidia.com`) resolves fine, so
this is a single unreachable host, not a network boundary. The step is
dropped for OSMO builds; images built from this Dockerfile do **not** have
`nvdataset` and cannot run the DSS upload path. Restore the step once
`artifactory.pdx.nvidia.com` is reachable from OSMO pools; see the comment at
the same spot in `Dockerfile.odin`.

This does **not** mean a dispatch trains successfully and then silently loses
its results. `dispatch.yaml.j2` (the `nvdataset upload` call, ~lines 178-191)
retries the upload three times and, on failure, sets `rc=91` so the row goes
red rather than reporting success. A missing `nvdataset` binary exits 127 on
the first attempt. The actual symptom is a red row, three wasted retry
sleeps, and an `rc=91` that reads like a DSS outage -- easy to misdiagnose as
a dataset-service problem when the real cause is a missing binary in the
image.

## Measured sizing

Recorded from the first successful real build (Task A3).

| Setting | Value | How it was determined |
|---|---|---|
| `cpu` | 16 | first try; never raised, no CPU-bound symptom observed |
| `memory` | 64Gi | first try; never raised, no OOM signal |
| `storage` | 256Gi | first try; never raised, no `no space left on device` signal |
| wall-clock | 26m07s | from `osmo workflow query` (kaniko log's own elapsed markers) |

Phase breakdown of the 26m07s, so a future reader optimizes the right thing:

| Phase | Duration | Share |
|---|---|---|
| base image + apt + uv installer | ~2m | ~8% |
| `uv sync` (`Prepared 297 packages in 3m 35s`, `Installed ... in 2.13s`) | ~3m35s | ~14% |
| kaniko full-filesystem snapshot after `uv sync` | 11m27s | ~44% |
| push to `nvcr.io` | 8m50s | ~34% |

Dependency resolution (`uv sync`) is only ~14% of the wall clock. Snapshot +
push together are ~78% — both are kaniko/registry overhead rather than work
this Dockerfile controls, so that is where future speedups should be sought,
not in `uv sync` itself. `uv sync` needed a single attempt with zero retries;
the concern that the non-resumable dependency layer would be flaky did not
materialize on either build attempt in this task.

An earlier attempt at this same commit-and-sizing combination (before the
`nvdataset` step was dropped, see above) failed after `uv sync` on a DNS
failure resolving `artifactory.pdx.nvidia.com` — that failure was unrelated
to sizing.

Build workflow ID: `odin-build-c20a57a-1`. Verify workflow ID:
`odin-build-verify-1`. Both COMPLETED. Pushed image:
`nvcr.io/nvidian/antoiner-isaac-lab:c20a57a` →
`sha256:628c942b225ecfa62f170bf87e8eebc84dfd44f12d06627db9513f81273d5c67`.
