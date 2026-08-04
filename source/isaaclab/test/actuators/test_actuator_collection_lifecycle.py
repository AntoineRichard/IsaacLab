# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the transactional simulation-scoped actuator lifecycle.

Setup:
    - ./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py
Tests:
    - ./isaaclab.sh -p -m pytest
      source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py
      -> verify failed completion rolls every registration back.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from isaaclab.actuators import ActuatorCollection
from isaaclab.assets.asset_base import AssetBase
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.sim.service_locator import ServiceLocator
from isaaclab.sim.simulation_context import SimulationContext


@dataclass
class _Data:
    is_primed: bool = False


class _Control:
    def __init__(
        self,
        *,
        prepare_error: Exception | None = None,
        complete_error: Exception | None = None,
    ) -> None:
        self.prepare_error = prepare_error
        self.complete_error = complete_error
        self.bind_count = 0
        self.complete_count = 0
        self.invalidate_count = 0
        self.asset = type("_Asset", (), {"_is_initialized": False, "data": _Data()})()

    @property
    def num_joints(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return "cpu"

    def discover_native_actuators(self, cfgs) -> set[str]:
        return set()

    def prepare_actuator_binding(self, binding) -> None:
        del binding
        if self.prepare_error is not None:
            raise self.prepare_error

    def bind_actuator_view(self, view) -> None:
        del view
        self.bind_count += 1

    def complete_articulation_initialization(self) -> None:
        self.complete_count += 1
        if self.complete_error is not None:
            raise self.complete_error
        self.asset._is_initialized = True
        self.asset.data.is_primed = True

    def invalidate_actuator_view(self) -> None:
        self.invalidate_count += 1
        self.asset._is_initialized = False
        self.asset.data.is_primed = False


class _Simulation:
    def __init__(self) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 1), dtype=torch.bool),
            cfg_rows={1: (0,), 2: (0,)},
        )

    def get_clone_plan(self) -> ClonePlan:
        return self._clone_plan


class _DeferredAsset(AssetBase):
    @property
    def num_instances(self) -> int:
        return 0

    @property
    def data(self):
        return None

    def reset(self, env_ids=None) -> None:
        del env_ids

    def write_data_to_sim(self) -> None:
        pass

    def update(self, dt: float) -> None:
        del dt

    def _initialize_impl(self) -> None:
        pass


def _register(collection: ActuatorCollection, key: object, control: _Control):
    return collection.register_articulation(
        key=key,
        cfgs={},
        control=control,
        replication_cfg_id=1 if key == "first" else 2,
        debug_validation=False,
        debug_value_resolution=False,
    )


def test_pending_view_rejects_runtime_access_before_publication() -> None:
    collection = ActuatorCollection(_Simulation())
    view = _register(collection, "first", _Control())

    with pytest.raises(RuntimeError, match="pending actuator view"):
        _ = view.command


def test_failed_finalization_publishes_no_partial_generation() -> None:
    collection = ActuatorCollection(_Simulation())
    good = _register(collection, "first", _Control())
    bad = _register(collection, "second", _Control(prepare_error=ValueError("bad IdealPDActuator wheel")))

    with pytest.raises(ValueError, match="bad IdealPDActuator wheel"):
        collection.finalize()

    assert collection.generation is None
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = good.command
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = bad.command


def test_second_completion_failure_rolls_back_first_completed_asset() -> None:
    collection = ActuatorCollection(_Simulation())
    first_control = _Control()
    second_control = _Control(complete_error=RuntimeError("second completion"))
    first = _register(collection, "first", first_control)
    second = _register(collection, "second", second_control)

    with pytest.raises(RuntimeError, match="second completion"):
        collection.finalize()

    assert collection.generation is None
    assert not first_control.asset._is_initialized and not first_control.asset.data.is_primed
    assert not second_control.asset._is_initialized and not second_control.asset.data.is_primed
    assert first_control.invalidate_count == second_control.invalidate_count == 1
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = first.command
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = second.command


def test_late_registration_marks_active_generation_dirty_until_replay() -> None:
    collection = ActuatorCollection(_Simulation())
    first = _register(collection, "first", _Control())
    collection.finalize()
    _register(collection, "second", _Control())

    with pytest.raises(RuntimeError, match="late registration.*rebuild"):
        first.compute()


def test_clear_generation_invalidates_old_facade_before_next_generation() -> None:
    collection = ActuatorCollection(_Simulation())
    first = _register(collection, "first", _Control())
    collection.finalize()
    old_generation = first.generation
    collection.clear_generation()

    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = first.command

    replacement = _register(collection, "first", _Control())
    collection.finalize()
    assert replacement.generation == old_generation + 1


def test_stop_replay_invalidates_old_facade_and_builds_new_generation() -> None:
    collection = ActuatorCollection(_Simulation())
    old = _register(collection, "first", _Control())
    collection.finalize()
    old_generation = old.generation

    collection.clear_generation()
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = old.command

    replacement = _register(collection, "first", _Control())
    collection.finalize()
    assert replacement.generation == old_generation + 1
    assert replacement is not old


def test_close_invalidates_views_and_permanently_rejects_registration() -> None:
    collection = ActuatorCollection(_Simulation())
    view = _register(collection, "first", _Control())
    collection.finalize()
    collection.close()
    collection.close()

    with pytest.raises(RuntimeError, match="closed"):
        _ = view.command
    with pytest.raises(RuntimeError, match="closed"):
        _register(collection, "second", _Control())


def test_context_lifecycle_bridges_keep_the_service_lazy() -> None:
    context = object.__new__(SimulationContext)
    context._services = ServiceLocator()

    SimulationContext._finalize_actuator_collection(context, None)
    SimulationContext._clear_actuator_collection_generation(context, None)
    assert context.services[ActuatorCollection] is None

    collection = SimulationContext._get_actuator_collection(context)
    assert context.services[ActuatorCollection] is collection


def test_deferred_asset_completion_requires_an_explicit_deferral() -> None:
    asset = object.__new__(_DeferredAsset)
    asset._is_initialized = False
    asset._initialization_deferred = False
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None

    with pytest.raises(RuntimeError, match="not deferred"):
        asset._complete_deferred_initialization()
    asset._defer_initialization()
    asset._complete_deferred_initialization()
    assert asset._is_initialized
    assert not asset._initialization_deferred
