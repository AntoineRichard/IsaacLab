# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for capability-based actuator-gain randomization."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import events


class _FakeActuator:
    """Logical actuator group with typed gain storage and recorded writes."""

    def __init__(
        self,
        parameter_names: frozenset[str],
        joint_indices: torch.Tensor | slice,
        num_envs: int,
        num_joints: int,
    ):
        self.parameter_names = parameter_names
        self.joint_indices = joint_indices
        self._parameters = {
            name: torch.full((num_envs, num_joints), float(index + 1)) for index, name in enumerate(parameter_names)
        }
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor | None, torch.Tensor]] = []

    @property
    def stiffness(self) -> torch.Tensor:
        if "stiffness" not in self.parameter_names:
            raise AssertionError("neural actuator stiffness must not be read")
        return self._parameters["stiffness"]

    @property
    def damping(self) -> torch.Tensor:
        if "damping" not in self.parameter_names:
            raise AssertionError("neural actuator damping must not be read")
        return self._parameters["damping"]

    def set_parameter_index(
        self, name: str, value: torch.Tensor, *, env_ids: torch.Tensor | None, joint_ids: torch.Tensor
    ) -> None:
        self.calls.append((name, value.clone(), env_ids, joint_ids))


class _FakeImplicitActuator(_FakeActuator):
    """Marker type for the immediate implicit-solver write path."""


class _FakeAsset:
    """Minimal articulation boundary used by the randomizer."""

    def __init__(self, actuators: OrderedDict[str, _FakeActuator], num_envs: int, num_joints: int):
        self.actuators = actuators
        self.device = "cpu"
        self.num_joints = num_joints
        self.data = SimpleNamespace(
            joint_stiffness=SimpleNamespace(torch=torch.full((num_envs, num_joints), 10.0)),
            joint_damping=SimpleNamespace(torch=torch.full((num_envs, num_joints), 20.0)),
        )
        self.solver_calls: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def write_joint_stiffness_to_sim_index(self, *, stiffness, joint_ids, env_ids) -> None:
        self.solver_calls.append(("stiffness", stiffness.clone(), joint_ids, env_ids))

    def write_joint_damping_to_sim_index(self, *, damping, joint_ids, env_ids) -> None:
        self.solver_calls.append(("damping", damping.clone(), joint_ids, env_ids))


def _make_term(asset: _FakeAsset, joint_ids: list[int] | slice = slice(None)) -> events.randomize_actuator_gains:
    """Build a randomizer without invoking its environment-facing constructor."""
    term = object.__new__(events.randomize_actuator_gains)
    term.asset = asset
    term.asset_cfg = SimpleNamespace(joint_ids=joint_ids)
    term.default_joint_stiffness = asset.data.joint_stiffness.torch.clone()
    term.default_joint_damping = asset.data.joint_damping.torch.clone()
    return term


def test_randomize_actuator_gains_uses_group_setters_and_skips_neural_sidecars(monkeypatch) -> None:
    """Only logical groups declaring a gain are randomized, with implicit solver writes retained."""
    monkeypatch.setattr(events, "ImplicitActuator", _FakeImplicitActuator)
    num_envs, num_joints = 3, 4
    implicit = _FakeImplicitActuator(frozenset({"stiffness", "damping"}), torch.tensor([0, 2]), num_envs, 2)
    explicit = _FakeActuator(frozenset({"stiffness"}), torch.tensor([1, 3]), num_envs, 2)
    neural = _FakeActuator(frozenset({"effort_limit", "velocity_limit"}), torch.tensor([2]), num_envs, 1)
    asset = _FakeAsset(OrderedDict(implicit=implicit, explicit=explicit, neural=neural), num_envs, num_joints)
    term = _make_term(asset)
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=num_envs))
    env_ids = torch.tensor([2, 0])

    term(
        env,
        env_ids,
        SimpleNamespace(),
        stiffness_distribution_params=(7.0, 7.0),
        damping_distribution_params=(9.0, 9.0),
        operation="abs",
    )

    assert [call[0] for call in implicit.calls] == ["stiffness", "damping"]
    assert [call[0] for call in explicit.calls] == ["stiffness"]
    assert neural.calls == []
    assert [(name, joint_ids.tolist()) for name, _, joint_ids, _ in asset.solver_calls] == [
        ("stiffness", [0, 2]),
        ("damping", [0, 2]),
    ]
    assert all(call[2].tolist() == [2, 0] for call in implicit.calls + explicit.calls)
    assert all(
        torch.all(call[1] == (7.0 if call[0] == "stiffness" else 9.0)) for call in implicit.calls + explicit.calls
    )


def test_randomize_actuator_gains_caches_only_declared_explicit_gains_without_newton_snapshots(monkeypatch) -> None:
    """Default caching uses declared actuator gains without Newton snapshot sidecars."""
    monkeypatch.setattr(events, "ImplicitActuator", _FakeImplicitActuator)
    monkeypatch.setattr(events.ManagerTermBase, "__init__", lambda self, cfg, env: None)
    explicit = _FakeActuator(frozenset({"stiffness"}), torch.tensor([1]), 2, 1)
    explicit.stiffness.fill_(30.0)
    neural = _FakeActuator(frozenset({"effort_limit", "velocity_limit"}), torch.tensor([0]), 2, 1)
    asset = _FakeAsset(OrderedDict(explicit=explicit, neural=neural), 2, 2)
    assert not hasattr(asset, "newton_default_stiffness")
    assert not hasattr(asset, "newton_default_damping")
    environment = SimpleNamespace(scene={"robot": asset})
    cfg = SimpleNamespace(params={"asset_cfg": SimpleNamespace(name="robot"), "operation": "abs"})

    term = events.randomize_actuator_gains(cfg, environment)

    torch.testing.assert_close(term.default_joint_stiffness[:, 1], torch.full((2,), 30.0))
    torch.testing.assert_close(term.default_joint_damping, torch.full((2, 2), 20.0))
    assert neural.calls == []


@pytest.mark.parametrize("operation", ["abs", "add", "scale"])
@pytest.mark.parametrize("distribution", ["uniform", "log_uniform", "gaussian"])
def test_randomize_actuator_gains_routes_each_sampling_mode_through_group_setters(
    monkeypatch, operation: str, distribution: str
) -> None:
    """Every existing operation and distribution retains the typed group write path."""
    monkeypatch.setattr(events, "ImplicitActuator", _FakeImplicitActuator)
    actuator = _FakeActuator(frozenset({"stiffness"}), slice(None), 2, 2)
    asset = _FakeAsset(OrderedDict(pd=actuator), 2, 2)
    term = _make_term(asset)
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=2))

    term(
        env,
        None,
        SimpleNamespace(),
        stiffness_distribution_params=(1.0, 1.0),
        operation=operation,
        distribution=distribution,
    )

    assert len(actuator.calls) == 1
    assert actuator.calls[0][0] == "stiffness"
    assert actuator.calls[0][2].tolist() == [0, 1]


def test_randomize_actuator_gains_preserves_nonmonotonic_world_and_joint_selectors(monkeypatch) -> None:
    """Group setters receive compact values with articulation-coordinate writer selectors."""
    monkeypatch.setattr(events, "ImplicitActuator", _FakeImplicitActuator)
    actuator = _FakeActuator(frozenset({"stiffness", "damping"}), torch.tensor([3, 0, 2]), 4, 3)
    asset = _FakeAsset(OrderedDict(pd=actuator), 4, 4)
    term = _make_term(asset, joint_ids=[2, 3])
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=4))
    env_ids = torch.tensor([3, 1])

    term(
        env,
        env_ids,
        SimpleNamespace(),
        stiffness_distribution_params=(2.0, 2.0),
        operation="abs",
    )

    name, values, recorded_env_ids, writer_joint_ids = actuator.calls[0]
    assert name == "stiffness"
    assert recorded_env_ids.tolist() == [3, 1]
    assert writer_joint_ids.tolist() == [3, 2]
    torch.testing.assert_close(values, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
