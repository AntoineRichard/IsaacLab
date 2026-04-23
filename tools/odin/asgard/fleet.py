# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fleet configuration — list of Valkyrie hosts + per-host SSH / path config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Fleet", "ValkyrieConfig", "load_fleet"]


@dataclass
class ValkyrieConfig:
    """Per-host configuration for a Valkyrie."""

    host: str
    ssh_user: str
    ssh_key: Path | None = None
    isaaclab_path: str = "~/IsaacLab"
    container_name: str = "isaac-lab-base"
    labels: list[str] = field(default_factory=list)


@dataclass
class Fleet:
    """The whole fleet."""

    fleet_name: str
    hosts: list[ValkyrieConfig]


def _resolve_ssh_key(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()


def load_fleet(path: Path) -> Fleet:
    """Parse ``fleet.yaml`` into a :class:`Fleet` with defaults resolved.

    Per-host fields override fleet-level ``default_*`` fields. ``ssh_key`` is
    ``pathlib.Path``-resolved with ``~`` expansion.

    Args:
        path: Path to ``fleet.yaml``.

    Returns:
        Populated :class:`Fleet`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On missing required fields or empty host list.
    """
    if not path.exists():
        raise FileNotFoundError(f"fleet.yaml not found: {path}")
    with path.open("r") as fh:
        payload: dict[str, Any] = yaml.safe_load(fh) or {}

    fleet_name = str(payload.get("fleet_name") or "")
    default_ssh_user = payload.get("default_ssh_user")
    default_ssh_key = payload.get("default_ssh_key")
    hosts_raw = payload.get("hosts") or []

    if not hosts_raw:
        raise ValueError(f"fleet.yaml must list at least one host: {path}")

    hosts: list[ValkyrieConfig] = []
    for idx, raw in enumerate(hosts_raw):
        if not isinstance(raw, dict) or "host" not in raw:
            raise ValueError(f"fleet.yaml host entry #{idx} missing required 'host' field: {raw!r}")
        ssh_user = raw.get("ssh_user") or default_ssh_user
        if ssh_user is None:
            raise ValueError(f"fleet.yaml host {raw['host']!r} has no ssh_user and no default_ssh_user is set")
        ssh_key_value = raw.get("ssh_key") if "ssh_key" in raw else default_ssh_key
        hosts.append(
            ValkyrieConfig(
                host=str(raw["host"]),
                ssh_user=str(ssh_user),
                ssh_key=_resolve_ssh_key(ssh_key_value),
                isaaclab_path=str(raw.get("isaaclab_path", "~/IsaacLab")),
                container_name=str(raw.get("container_name", "isaac-lab-base")),
                labels=list(raw.get("labels") or []),
            )
        )

    return Fleet(fleet_name=fleet_name, hosts=hosts)
