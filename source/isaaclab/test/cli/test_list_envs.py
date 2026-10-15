# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed environment-listing command."""

from __future__ import annotations

import importlib.metadata
from unittest import mock

import gymnasium as gym

from isaaclab.cli.commands.list_envs import command_list_envs


def test_command_lists_non_isaac_prefixed_task_from_current_project(tmp_path, monkeypatch, capsys):
    """Downstream tasks must not need an ``Isaac`` prefix to appear in project-scoped output.

    Registry packages installed into the project (its ``isaaclab-registry`` group) count as part of it.
    """
    tasks = {"Example-Balance-Robot": "example.tasks", "Registry-Walk-Robot": "registry_walk.tasks"}
    for task_id, module in tasks.items():
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{module}:EnvCfg"},
        )
    project_dir = tmp_path / "project"
    working_dir = project_dir / "scripts" / "nested"
    working_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "example"\n[project.entry-points."isaaclab.tasks"]\nexample = "example.tasks"\n'
        '[dependency-groups]\nisaaclab-registry = ["registry-walk"]\n'
    )
    registry_walk = mock.Mock()
    registry_walk.entry_points = [
        importlib.metadata.EntryPoint(name="registry_walk", value="registry_walk.tasks", group="isaaclab.tasks")
    ]
    monkeypatch.chdir(working_dir)
    monkeypatch.setattr("isaaclab.cli.importlib.metadata.entry_points", lambda **kwargs: [])
    distribution = importlib.metadata.distribution
    monkeypatch.setattr(
        "isaaclab.cli.commands.list_envs.importlib.metadata.distribution",
        lambda name: registry_walk if name == "registry-walk" else distribution(name),
    )

    try:
        command_list_envs([])
    finally:
        for task_id in tasks:
            del gym.registry[task_id]

    output = capsys.readouterr().out
    assert all(task_id in output for task_id in tasks)
    assert "Isaac-Cartpole" not in output
