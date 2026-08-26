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

## Why the choice matters

`templates/Dockerfile.j2` uses `RUN --mount=type=cache` so that a dropped
transfer during a multi-gigabyte `uv sync` resumes instead of restarting.
`Dockerfile.odin` cannot use it: a builder without cache-mount support forces
that flag out, making long OSMO builds materially more fragile than local ones.
