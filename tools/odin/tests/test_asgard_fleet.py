# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.fleet`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig, load_fleet


def _write_fleet(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "fleet.yaml"
    path.write_text(text)
    return path


def test_load_fleet_applies_defaults(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: test-fleet
default_ssh_user: odin
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: valkyrie-01
  - host: valkyrie-02
    ssh_user: other_user
""",
    )
    fleet = load_fleet(path)
    assert isinstance(fleet, Fleet)
    assert fleet.fleet_name == "test-fleet"
    assert len(fleet.hosts) == 2
    assert fleet.hosts[0].host == "valkyrie-01"
    assert fleet.hosts[0].ssh_user == "odin"
    assert str(fleet.hosts[0].ssh_key) == str(Path("~/.ssh/id_ed25519").expanduser())
    # Per-host override wins over default.
    assert fleet.hosts[1].ssh_user == "other_user"


def test_load_fleet_container_name_default(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts:
  - host: h1
""",
    )
    fleet = load_fleet(path)
    # Spec default for profile=base.
    assert fleet.hosts[0].container_name == "isaac-lab-base"


def test_load_fleet_per_host_overrides(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: odin
hosts:
  - host: h1
    isaaclab_path: /mnt/scratch/IsaacLab
    container_name: isaac-lab-ros
    labels: [h100-80gb]
""",
    )
    fleet = load_fleet(path)
    h = fleet.hosts[0]
    assert h.isaaclab_path == "/mnt/scratch/IsaacLab"
    assert h.container_name == "isaac-lab-ros"
    assert h.labels == ["h100-80gb"]


def test_load_fleet_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_fleet(tmp_path / "nope.yaml")


def test_load_fleet_empty_hosts_raises(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts: []
""",
    )
    with pytest.raises(ValueError, match="at least one host"):
        load_fleet(path)


def test_load_fleet_missing_host_field_raises(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts:
  - ssh_user: xx
""",
    )
    with pytest.raises(ValueError, match="host"):
        load_fleet(path)


def test_valkyrie_config_ssh_key_none_when_no_default(tmp_path: Path):
    path = _write_fleet(
        tmp_path,
        """
fleet_name: t
default_ssh_user: u
hosts:
  - host: h1
""",
    )
    fleet = load_fleet(path)
    assert fleet.hosts[0].ssh_key is None
