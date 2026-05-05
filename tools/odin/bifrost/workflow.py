# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workflow rendering for Bifrost.

This module:
- builds DNS-1123-safe OSMO task names from Odin run_ids
  (:func:`osmo_safe_task_name`),
- (later) renders the Jinja workflow template from a row list.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.odin.bifrost.config import BifrostConfig

__all__ = [
    "osmo_safe_task_name",
    "RenderRow",
    "render_workflow_yaml",
]


_DNS_1123_LABEL_MAX = 63
_HASH_SUFFIX_LEN = 7  # "-" + 6 hex chars
_NON_ALNUM_DASH = re.compile(r"[^a-z0-9-]")
_RUN_OF_DASHES = re.compile(r"-+")


def osmo_safe_task_name(run_id: str) -> str:
    """Convert an Odin ``run_id`` into a DNS-1123-compliant OSMO task name.

    Constraints (per Kubernetes' DNS-1123 label rules):

    - At most 63 characters.
    - Only lowercase alphanumerics and ``-``.
    - Must not start or end with ``-``.

    On truncation, a 6-hex-char hash of the full ``run_id`` is appended so
    distinct long inputs produce distinct outputs.

    Args:
        run_id: Odin run_id (e.g. ``rsl-rl_physx_Isaac-Ant_seed42``).

    Returns:
        A DNS-1123-safe label.
    """
    lowered = run_id.lower()
    dashed = re.sub(r"[_.\s]+", "-", lowered)
    only_safe = _NON_ALNUM_DASH.sub("-", dashed)
    collapsed = _RUN_OF_DASHES.sub("-", only_safe).strip("-")
    if not collapsed:
        # Degenerate input: emit a stable hash-only label.
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:6]
        return f"odin-{digest}"
    if len(collapsed) <= _DNS_1123_LABEL_MAX:
        return collapsed
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:6]
    keep = _DNS_1123_LABEL_MAX - _HASH_SUFFIX_LEN
    return f"{collapsed[:keep].rstrip('-')}-{digest}"


_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class RenderRow:
    """One row of the workflow render context.

    Maps 1:1 to a single task in the OSMO workflow YAML.
    """

    run_id: str
    osmo_task_name: str
    framework: str  # rsl-rl | skrl
    framework_runner: str  # hugin | munin
    task_id: str  # gym task id, e.g. Isaac-Ant-Direct-v0
    backend: str  # physx | newton
    seed: int
    num_envs: int
    max_iterations: int


def render_workflow_yaml(
    *,
    dispatch_id: str,
    rows: list[RenderRow],
    cfg: BifrostConfig,
    tarball_path: str | None,
) -> str:
    """Render the OSMO workflow YAML for one dispatch.

    Args:
        dispatch_id: Odin dispatch id (``YYYYMMDD-HHMMSS``).
        rows: One per ``(task, seed)`` to dispatch.
        cfg: Validated config from :func:`load_bifrost_config`.
        tarball_path: Controller-local path to the source tarball; required
            when ``cfg.code_delivery.mode == "files_upload"``, ignored
            otherwise.

    Returns:
        The rendered workflow YAML as a string. Caller writes it to disk
        and passes the path to ``osmo workflow submit``.
    """
    if cfg.code_delivery.mode == "files_upload" and not tarball_path:
        raise ValueError("tarball_path is required when code_delivery.mode == files_upload")
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    template = env.get_template("dispatch.yaml.j2")
    return template.render(
        dispatch_id=dispatch_id,
        rows=rows,
        cfg=cfg,
        tarball_path=tarball_path,
    )
