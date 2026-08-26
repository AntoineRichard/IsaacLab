# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe which OSMO-runnable builder supports ``RUN --mount=type=cache``.

Odin's benchmark Dockerfile depends on a BuildKit cache mount to make a
multi-gigabyte ``uv sync`` resumable. Kaniko's support for that flag is
uncertain, so the builder is chosen by running both rather than by assumption.
"""

from __future__ import annotations

import json
import pathlib

__all__ = ["BUILDERS", "PROBE_DOCKERFILE", "read_push_auth", "render_probe"]

BUILDERS: dict[str, str] = {
    "kaniko": "gcr.io/kaniko-project/executor:v1.23.2",
    "buildkit": "moby/buildkit:v0.18.2-rootless",
}

PROBE_DOCKERFILE = """FROM busybox:latest
RUN --mount=type=cache,target=/opt/probe-cache,sharing=locked \\
    echo cache-mount-ok > /probe.txt
"""


def read_push_auth(path: pathlib.Path | None = None) -> str:
    """Return the base64 nvcr.io credential that carries push scope.

    Args:
        path: Docker config to read. Defaults to ``~/.docker/config.json``.

    Returns:
        The base64 ``user:password`` blob for ``nvcr.io``.

    Raises:
        SystemExit: If no nvcr.io credential is present.
    """
    path = path or pathlib.Path.home() / ".docker" / "config.json"
    auth = json.loads(path.read_text()).get("auths", {}).get("nvcr.io", {}).get("auth")
    if not auth:
        raise SystemExit(f"no nvcr.io auth in {path}")
    return auth


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def render_probe(builder: str, auth_b64: str, destination: str) -> str:
    """Render a single-task OSMO workflow that builds the probe Dockerfile.

    Args:
        builder: Key into :data:`BUILDERS`.
        auth_b64: Base64 nvcr.io credential with push scope.
        destination: Full image reference to push.

    Returns:
        Rendered workflow YAML.

    Raises:
        KeyError: If *builder* is unknown.
    """
    image = BUILDERS[builder]
    docker_config = json.dumps({"auths": {"nvcr.io": {"auth": auth_b64}}}, indent=2)

    if builder == "kaniko":
        config_path = "/kaniko/.docker/config.json"
        command = '["/kaniko/executor"]'
        args = [
            "--dockerfile=/workspace/Dockerfile",
            "--context=dir:///workspace",
            f"--destination={destination}",
            "--verbosity=info",
        ]
        env = ""
    else:
        # Rootless BuildKit runs as uid 1000 with HOME=/home/user.
        config_path = "/home/user/.docker/config.json"
        command = '["buildctl-daemonless.sh"]'
        args = [
            "build",
            "--frontend=dockerfile.v0",
            "--local=context=/workspace",
            "--local=dockerfile=/workspace",
            f"--output=type=image,name={destination},push=true",
        ]
        # Without this, rootless buildkitd needs privileges OSMO will not grant.
        env = '      environment:\n        BUILDKITD_FLAGS: "--oci-worker-no-process-sandbox"\n'

    rendered_args = "\n".join(f'      - "{a}"' for a in args)

    return f"""workflow:
  name: builder-probe-{builder}
  timeout:
    exec_timeout: 1800s
    queue_timeout: 7200s
  resources:
    default:
      cpu: 4
      gpu: 0
      memory: 16Gi
      storage: 32Gi
      platform: ovx-l40
  groups:
  - name: g-builder-probe
    tasks:
    - name: builder-probe-{builder}
      image: {image}
      command: {command}
      args:
{rendered_args}
{env}      files:
      - path: {config_path}
        contents: |
{_indent(docker_config, 10)}
      - path: /workspace/Dockerfile
        contents: |
{_indent(PROBE_DOCKERFILE, 10)}
"""
