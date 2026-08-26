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

## `nvdataset` and the carrier image

`dispatch.yaml.j2` uploads results with the `nvdataset` CLI, so OSMO-built
images need it. It cannot be installed from an index during an OSMO build:
the package is not on public PyPI, `artifactory.pdx.nvidia.com` has no DNS
from OSMO pods, and `artifactory.nvidia.com` resolves but has no route
(TCP never connects). `gitlab-master.nvidia.com`, `github.com`, `pypi.org`,
`nvcr.io` and `api.ngc.nvidia.com` are all reachable, so this is two
unreachable hosts, not a network boundary.

`nvcr.io` being reachable is what the carrier exploits. `nvdataset` is
installed once into a small image built from a machine that *does* have
artifactory access, that image is pushed to `nvcr.io`, and `Dockerfile.odin`
pulls the installed tree in with `COPY --from`. No credential beyond the one
already injected for the push: kaniko resolves `COPY --from` using the same
`/kaniko/.docker/config.json`, which carries pull scope.

### This is temporary

It exists only because OSMO cannot reach artifactory. When that is fixed,
delete `Dockerfile.nvdataset`, drop the two constants from `render.py`, drop
the two carrier tests, and restore the plain `uv tool install` in
`Dockerfile.odin`. Nothing else depends on it.

### Rebuilding the carrier

Needs Docker **and** NVIDIA-internal network access, so it cannot run in CI or
on OSMO — it is a deliberate manual step. Run it from a repo checkout on a
machine with both:

```bash
VERSION=0.96.0   # must match NVDATASET_VERSION in tools/odin/osmo_build/render.py
docker build \
  --build-arg "NVDATASET_VERSION=${VERSION}" \
  -f tools/odin/osmo_build/Dockerfile.nvdataset \
  -t "nvcr.io/nvidian/antoiner-isaac-lab:nvdataset-${VERSION}" \
  tools/odin/osmo_build
docker push "nvcr.io/nvidian/antoiner-isaac-lab:nvdataset-${VERSION}"
```

To move to a new `nvdataset`: bump `NVDATASET_VERSION` in `render.py`, update
the `COPY --from` tag in `Dockerfile.odin`, rebuild and push the carrier, then
run the tests. `test_dockerfile_odin_copies_the_pinned_nvdataset_carrier`
fails if the constant and the Dockerfile disagree, which is cheaper than
discovering a missing tag 20 minutes into an OSMO build.

### Why the base images must match

The carrier hands over `/root/.local`. `nvdataset` is pure Python installed
through `console_scripts`, so its launcher carries a shebang baked to the
interpreter path `uv tool install` used, and that path only resolves in the
consuming image if both were built from the same base. Both Dockerfiles
therefore use `DEFAULT_CUDA_IMAGE`, and
`test_nvdataset_carrier_shares_dockerfile_odin_base_image` holds them
together. The `nvdataset --version` call in `Dockerfile.odin` is the runtime
backstop: a mismatch fails the build there rather than shipping an image
whose CLI cannot start.

### If the carrier tag is missing

`COPY --from` on an unpushed tag fails the OSMO build with a manifest error
after the expensive layers have already run. If that happens, the carrier was
never pushed for the pinned version — rebuild it with the command above.

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
