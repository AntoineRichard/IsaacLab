# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render an OSMO workflow that builds a commit-pinned Isaac Lab image.

The local path in :mod:`tools.odin.image` ships the commit as a ``git bundle``
into ``docker build``. Neither is available on OSMO: a task has no Docker
daemon and no local repository, so the commit is fetched from a pushed remote
by the builder's own git context.

Only ``"kaniko"`` has been validated end to end on OSMO. ``"buildkit"`` failed
the OSMO builder probe (rootless ``buildkitd`` could not get a mount inside
the task sandbox; see ``tools/odin/osmo_build/README.md``) and is kept here as
a reference point, not a supported option -- its git-credential delivery
(``/home/user/.git-credentials``) has never actually run on OSMO.
"""

from __future__ import annotations

import json
import pathlib

__all__ = [
    "BUILDER_IMAGES",
    "NVDATASET_CARRIER_IMAGE",
    "NVDATASET_VERSION",
    "read_push_auth",
    "render_build_workflow",
]

BUILDER_IMAGES: dict[str, str] = {
    "kaniko": "gcr.io/kaniko-project/executor:v1.23.2",
    "buildkit": "moby/buildkit:v0.18.2-rootless",
}

# The nvdataset CLI reaches the OSMO build through a carrier image rather than
# an index, because OSMO cannot route to artifactory. These two constants are
# the single source of truth: ``Dockerfile.nvdataset`` is built with the
# version, ``Dockerfile.odin`` copies from the image, and a test asserts the
# Dockerfile still names this exact tag. Bumping the version here without
# rebuilding and pushing the carrier fails that test rather than failing a
# 26-minute OSMO build.
NVDATASET_VERSION = "0.96.0"
NVDATASET_CARRIER_IMAGE = f"nvcr.io/nvidian/antoiner-isaac-lab:nvdataset-{NVDATASET_VERSION}"


def read_push_auth(path: pathlib.Path | None = None) -> str:
    """Return the base64 nvcr.io credential that carries push scope.

    Args:
        path: Docker config to read. Defaults to ``~/.docker/config.json``.

    Returns:
        The base64 ``user:password`` blob for ``nvcr.io``.

    Raises:
        FileNotFoundError: If no nvcr.io credential is present in *path*.
    """
    path = path or pathlib.Path.home() / ".docker" / "config.json"
    auth = json.loads(path.read_text()).get("auths", {}).get("nvcr.io", {}).get("auth")
    if not auth:
        raise FileNotFoundError(f"no nvcr.io auth in {path}")
    return auth


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def render_build_workflow(
    *,
    builder: str,
    commit_sha: str,
    git_remote: str,
    destination: str,
    auth_b64: str,
    git_token: str | None = None,
    cpu: int = 16,
    memory: str = "64Gi",
    storage: str = "256Gi",
) -> str:
    """Render a single-task OSMO workflow that builds and pushes the image.

    Args:
        builder: ``"kaniko"`` or ``"buildkit"``; see :data:`BUILDER_IMAGES`.
            Only ``"kaniko"`` passed the OSMO builder probe; ``"buildkit"``
            failed it (rootless mount permission error) and is retained as a
            reference point with its git-credential delivery unverified.
        commit_sha: Full commit SHA to pin. Must exist on *git_remote*.
        git_remote: HTTPS clone URL the builder fetches the context from.
        destination: Full image reference to push.
        auth_b64: Base64 ``user:password`` for nvcr.io, with push scope.
        git_token: Token for a private remote, delivered via the credential
            file rather than the context URL so it stays out of build logs.
        cpu: CPU cores requested.
        memory: Memory request, e.g. ``"64Gi"``.
        storage: Ephemeral storage request. The builder snapshots the whole
            filesystem, so this must exceed the built image's size.

    Returns:
        Rendered workflow YAML.

    Raises:
        KeyError: If *builder* is not a known builder.
    """
    image = BUILDER_IMAGES[builder]
    docker_auths: dict[str, dict[str, str]] = {"nvcr.io": {"auth": auth_b64}}

    files: list[tuple[str, str]] = []
    context_url = f"{git_remote}#{commit_sha}"

    if builder == "kaniko":
        files.append(("/kaniko/.docker/config.json", json.dumps({"auths": docker_auths}, indent=2)))
        command = '["/kaniko/executor"]'
        args = [
            f"--context=git://{context_url.removeprefix('https://')}",
            "--dockerfile=Dockerfile.odin",
            f"--destination={destination}",
            f"--build-arg=ODIN_COMMIT_SHA={commit_sha}",
            "--verbosity=info",
        ]
        env: dict[str, str] = {}
        if git_token:
            # Kaniko reads GIT_TOKEN for its git context; keeping it here rather
            # than in the URL keeps it out of the echoed context argument.
            env["GIT_TOKEN"] = git_token
    else:
        files.append(("/home/user/.docker/config.json", json.dumps({"auths": docker_auths}, indent=2)))
        command = '["buildctl-daemonless.sh"]'
        args = [
            "build",
            "--frontend=dockerfile.v0",
            f"--opt=context={context_url}",
            "--opt=filename=Dockerfile.odin",
            f"--opt=build-arg:ODIN_COMMIT_SHA={commit_sha}",
            f"--output=type=image,name={destination},push=true",
        ]
        # Rootless buildkitd otherwise requires privileges OSMO will not grant.
        env = {"BUILDKITD_FLAGS": "--oci-worker-no-process-sandbox"}
        if git_token:
            files.append(("/home/user/.git-credentials", f"https://x-access-token:{git_token}@github.com\n"))

    rendered_args = "\n".join(f'      - "{a}"' for a in args)
    rendered_env = ""
    if env:
        lines = "\n".join(f'        {k}: "{v}"' for k, v in env.items())
        rendered_env = f"      environment:\n{lines}\n"
    rendered_files = "\n".join(
        f"      - path: {path}\n        contents: |\n{_indent(contents, 10)}" for path, contents in files
    )

    return f"""workflow:
  name: odin-build-{commit_sha[:7]}
  timeout:
    exec_timeout: 14400s
    queue_timeout: 172800s
  resources:
    default:
      cpu: {cpu}
      gpu: 0
      memory: {memory}
      storage: {storage}
      platform: ovx-l40
  groups:
  - name: g-odin-build
    tasks:
    - name: odin-build-{commit_sha[:7]}
      image: {image}
      command: {command}
      args:
{rendered_args}
{rendered_env}      files:
{rendered_files}
"""
