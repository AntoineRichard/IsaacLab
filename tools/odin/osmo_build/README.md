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
mount, `templates/Dockerfile.j2` must drop both `RUN --mount=type=cache`
flags (Task A2, Step 2) — kaniko runs the mounted command but does not
persist the cache mount's contents between builds the way BuildKit would.

## Why the choice matters

`templates/Dockerfile.j2` uses `RUN --mount=type=cache` so that a dropped
transfer during a multi-gigabyte `uv sync` resumes instead of restarting. A
builder without cache-mount support forces that flag out of the Dockerfile and
makes long builds materially more fragile.
