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

from isaaclab.benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner, MethodBenchmarkRunnerConfig
from isaaclab.benchmark.asset_suites import get_asset_benchmark_suite, resolve_method_benchmarks
from isaaclab.envs.mdp import observations, rewards, terminations
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.anymal_c_direct.anymal_c_env import AnymalCEnv
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


def test_anymal_action_preserves_default_command_selectors() -> None:
    """Catch a regression that adds selectors to a full Anymal command."""
    command = SimpleNamespace(calls=[])

    def set_position_index(**kwargs) -> None:
        command.calls.append(kwargs)

    command.set_position_index = set_position_index
    robot = _DeprecatedActuatorAccess()
    robot.actuators = SimpleNamespace(command=command)
    environment = object.__new__(AnymalCEnv)
    environment._is_closed = True
    environment._robot = robot
    environment._processed_actions = torch.tensor([[1.0, 2.0]])

    environment._apply_action()

    assert command.calls == [{"value": environment._processed_actions}]


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


def test_method_benchmark_runner_invokes_dotted_index_and_mask_commands() -> None:
    """Catch dotted command paths that resolve to skipped benchmarks instead of real methods."""
    calls: dict[str, list[dict[str, object]]] = {"index": [], "mask": []}

    def set_position_index(**kwargs) -> None:
        calls["index"].append(kwargs)

    def set_position_mask(**kwargs) -> None:
        calls["mask"].append(kwargs)

    target = SimpleNamespace(
        actuators=SimpleNamespace(
            command=SimpleNamespace(set_position_index=set_position_index, set_position_mask=set_position_mask)
        )
    )
    index_inputs = {"value": object(), "env_ids": object(), "joint_ids": object()}
    mask_inputs = {"value": object(), "env_mask": object(), "joint_mask": object()}
    definitions = [
        MethodBenchmarkDefinition(
            name="position_index",
            method_name="actuators.command.set_position_index",
            input_generators={"literal": lambda _config: index_inputs},
        ),
        MethodBenchmarkDefinition(
            name="position_mask",
            method_name="actuators.command.set_position_mask",
            input_generators={"literal": lambda _config: mask_inputs},
        ),
    ]
    runner = object.__new__(MethodBenchmarkRunner)
    runner._config = MethodBenchmarkRunnerConfig(num_iterations=1, warmup_steps=0, device="cpu")
    runner._modes_to_run = None
    runner.update_manual_recorders = lambda: None
    runner.add_measurement = lambda *_args, **_kwargs: None

    runner.run_benchmarks(definitions, target)

    assert calls == {"index": [index_inputs, index_inputs], "mask": [mask_inputs, mask_inputs]}


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
    "source/isaaclab/isaaclab/test/mock_interfaces/assets/mock_articulation.py",
}
_COMPATIBILITY_TEST_SCOPES = {
    # These scopes exercise the collection storage object's own output fields. Their names coincide
    # with deprecated articulation-level aliases, but they are not deprecated collection consumers.
    "source/isaaclab/test/actuators/test_actuator_collection.py": {
        "_assert_collection_outputs_match_exactly",
        "test_implicit_batch_bypasses_torch_actuator_compute",
        "test_collection_exports_proxy_arrays",
    },
    # These tests intentionally preserve released compatibility behavior.
    "source/isaaclab/test/assets/test_articulation_ordering_iface.py": {
        "test_newton_data_aliases_are_bound_to_nested_actuator_view",
    },
    "source/isaaclab/test/test_mock_interfaces/test_mock_assets.py": {
        "test_set_joint_position_target",
    },
}


def test_non_experimental_runtime_sources_do_not_use_deprecated_actuator_apis() -> None:
    """Catch first-party runtime calls that would emit actuator deprecation warnings."""
    root = Path(__file__).resolve().parents[4]
    sources = [*root.glob("source/**/*.py"), *root.glob("tools/**/*.py"), *root.glob("scripts/**/*.py")]
    offenders: list[str] = []
    for source in sources:
        relative_source = source.relative_to(root).as_posix()
        if (
            relative_source in _COMPATIBILITY_SOURCES
            or "/isaaclab_experimental/" in relative_source
            or "/isaaclab_tasks_experimental/" in relative_source
        ):
            continue
        tree = ast.parse(source.read_text(), filename=relative_source)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _DEPRECATED_ATTRIBUTES:
                scope = node
                while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope = parents[scope]
                scope_name = scope.name if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
                if scope_name in _COMPATIBILITY_TEST_SCOPES.get(relative_source, set()):
                    continue
                offenders.append(f"{relative_source}:{node.lineno}:{node.attr}")

    assert not offenders, "Deprecated actuator APIs remain in runtime sources:\n" + "\n".join(sorted(offenders))
