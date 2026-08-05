# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for first-party actuator command and telemetry consumers."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from isaaclab.envs.mdp import observations, rewards, terminations
from isaaclab.managers import SceneEntityCfg
from isaaclab.benchmark.asset_suites import get_asset_benchmark_suite, resolve_method_benchmarks
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv


class _DeprecatedActuatorAccess:
    """Raise when a consumer reaches a deprecated actuator API."""

    @property
    def applied_torque(self):
        raise AssertionError("consumer accessed deprecated applied_torque")

    @property
    def computed_torque(self):
        raise AssertionError("consumer accessed deprecated computed_torque")

    @property
    def set_joint_effort_target_index(self):
        raise AssertionError("consumer called deprecated set_joint_effort_target_index")


def _proxy(values: list[list[float]]) -> SimpleNamespace:
    """Build a minimal ProxyArray-shaped value for a consumer boundary test."""
    return SimpleNamespace(torch=torch.tensor(values))


def test_effort_consumers_use_canonical_actuator_telemetry() -> None:
    """Catch a regression that redirects telemetry consumers through torque aliases."""
    telemetry = _DeprecatedActuatorAccess()
    telemetry.applied_effort = _proxy([[2.0, 3.0]])
    telemetry.computed_effort = _proxy([[2.0, 4.0]])
    asset = SimpleNamespace(data=telemetry, actuators=telemetry)
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

    torch.testing.assert_close(rewards.joint_torques_l2(env, asset_cfg), torch.tensor([13.0]))
    torch.testing.assert_close(observations.joint_effort(env, asset_cfg), torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(terminations.joint_effort_out_of_limit(env, asset_cfg), torch.tensor([True]))


def test_cartpole_action_uses_canonical_effort_command() -> None:
    """Catch a regression that routes Cartpole actions through a deprecated setter."""
    command = SimpleNamespace(calls=[])

    def set_effort_index(*, value, joint_ids) -> None:
        command.calls.append((value, joint_ids))

    command.set_effort_index = set_effort_index
    cartpole = _DeprecatedActuatorAccess()
    cartpole.actuators = SimpleNamespace(command=command)
    environment = object.__new__(CartpoleEnv)
    environment._is_closed = True
    environment.cartpole = cartpole
    environment.actions = torch.tensor([[1.5]])
    environment._cart_dof_idx = [1]

    environment._apply_action()

    assert command.calls == [(environment.actions, [1])]


def test_articulation_benchmark_commands_target_the_canonical_command_api() -> None:
    """Catch benchmark workloads that invoke deprecated articulation command setters."""
    definitions = resolve_method_benchmarks(
        get_asset_benchmark_suite("articulation"),
        SimpleNamespace(capabilities=frozenset({"warp_mask"}), generator_overrides={}),
    )
    command_definitions = {
        definition.method_name: definition
        for definition in definitions
        if definition.method_name.startswith("actuators.command.set_")
    }

    assert set(command_definitions) == {
        "actuators.command.set_position_index",
        "actuators.command.set_velocity_index",
        "actuators.command.set_effort_index",
        "actuators.command.set_position_mask",
        "actuators.command.set_velocity_mask",
        "actuators.command.set_effort_mask",
    }
    config = SimpleNamespace(num_instances=1, num_joints=1, num_bodies=1, device="cpu")
    assert all(
        "value" in next(iter(definition.input_generators.values()))(config)
        for definition in command_definitions.values()
    )


_DEPRECATED_ATTRIBUTES = {
    "applied_torque",
    "computed_torque",
    "set_joint_position_target",
    "set_joint_position_target_index",
    "set_joint_position_target_mask",
    "set_joint_velocity_target",
    "set_joint_velocity_target_index",
    "set_joint_velocity_target_mask",
    "set_joint_effort_target",
    "set_joint_effort_target_index",
    "set_joint_effort_target_mask",
}
_COMPATIBILITY_SOURCES = {
    "source/isaaclab/isaaclab/actuators/actuator_collection.py",
    "source/isaaclab/isaaclab/assets/articulation/base_articulation.py",
    "source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py",
    "source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py",
    "source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py",
    "source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py",
    "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py",
    "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py",
}


def test_non_experimental_runtime_sources_do_not_use_deprecated_actuator_apis() -> None:
    """Catch first-party runtime calls that would emit actuator deprecation warnings."""
    root = Path(__file__).resolve().parents[4]
    sources = [*root.glob("source/**/*.py"), *root.glob("tools/**/*.py"), *root.glob("scripts/**/*.py")]
    offenders: list[str] = []
    for source in sources:
        relative_source = source.relative_to(root).as_posix()
        if (
            "/test/" in relative_source
            or relative_source in _COMPATIBILITY_SOURCES
            or "/isaaclab_experimental/" in relative_source
            or "/isaaclab_tasks_experimental/" in relative_source
        ):
            continue
        tree = ast.parse(source.read_text(), filename=relative_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _DEPRECATED_ATTRIBUTES:
                offenders.append(f"{relative_source}:{node.lineno}:{node.attr}")

    assert not offenders, "Deprecated actuator APIs remain in runtime sources:\n" + "\n".join(sorted(offenders))
