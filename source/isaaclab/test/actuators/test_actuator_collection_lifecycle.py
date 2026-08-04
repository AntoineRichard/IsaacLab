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

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isaaclab.actuators import ActuatorCollection, actuator_collection
from isaaclab.actuators.actuator_pd import IdealPDActuator, ImplicitActuator
from isaaclab.assets.asset_base import AssetBase
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.physics import PhysicsEvent
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
        num_joints: int = 0,
    ) -> None:
        self.prepare_error = prepare_error
        self.complete_error = complete_error
        self.bind_count = 0
        self.complete_count = 0
        self.invalidate_count = 0
        self._num_joints = num_joints
        self.asset = type("_Asset", (), {"_is_initialized": False, "data": _Data()})()

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def device(self) -> str:
        return "cpu"

    def discover_native_actuators(self, cfgs) -> set[str]:
        return set()

    def find_joints(self, name_keys):
        del name_keys
        return list(range(self.num_joints)), ["wheel"] * self.num_joints

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


class _FakeCallbackHandle:
    def __init__(self, callbacks, entry) -> None:
        self._callbacks = callbacks
        self._entry = entry

    def deregister(self) -> None:
        self._callbacks.remove(self._entry)


class _FakePhysicsManager:
    callbacks = []

    @classmethod
    def reset(cls) -> None:
        cls.callbacks = []

    @classmethod
    def _prepare_stage_creation(cls) -> None:
        pass

    @classmethod
    def initialize(cls, context) -> None:
        del context

    @classmethod
    def get_scene_data_backend(cls):
        return object()

    @classmethod
    def register_callback(cls, callback, event, order=0, name=None, wrap_weak_ref=True):
        del wrap_weak_ref
        entry = SimpleNamespace(callback=callback, event=event, order=order, name=name)
        cls.callbacks.append(entry)
        return _FakeCallbackHandle(cls.callbacks, entry)

    @classmethod
    def dispatch(cls, event) -> None:
        snapshot = sorted((entry for entry in cls.callbacks if entry.event == event), key=lambda entry: entry.order)
        for entry in snapshot:
            entry.callback(None)

    @classmethod
    def close(cls) -> None:
        cls.callbacks = []


def _make_fake_context(monkeypatch) -> SimulationContext:
    class _StageCache:
        @staticmethod
        def Get():
            return _StageCache()

        def GetId(self, stage):
            del stage
            return SimpleNamespace(ToLongInt=lambda: 1)

        def Insert(self, stage):
            del stage

        def Size(self):
            return 0

    class _Settings:
        def get(self, name):
            return 0

        def set_bool(self, name, value):
            del name, value

        def set_int(self, name, value):
            del name, value

        def set_float(self, name, value):
            del name, value

        def set_string(self, name, value):
            del name, value

        def set(self, name, value):
            del name, value

    pxr = ModuleType("pxr")
    pxr.UsdUtils = SimpleNamespace(StageCache=_StageCache)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setattr("isaaclab.sim.simulation_context.has_kit", lambda: False)
    monkeypatch.setattr("isaaclab.sim.simulation_context.create_new_stage", lambda: object())
    monkeypatch.setattr("isaaclab.sim.simulation_context.SimulationContext._init_usd_physics_scene", lambda self: None)
    monkeypatch.setattr("isaaclab.sim.simulation_context.SceneDataProvider", lambda backend: object())
    monkeypatch.setattr(
        "isaaclab.sim.simulation_context.RenderContext", lambda: SimpleNamespace(ensure_initialize=lambda: None)
    )
    monkeypatch.setattr("isaaclab.sim.simulation_context.VisMarkerRegistry", lambda: SimpleNamespace())
    monkeypatch.setattr("isaaclab.sim.simulation_context.SettingsManager.instance", lambda: _Settings())
    monkeypatch.setattr("isaaclab.sim.simulation_context._resolve_physics_cfg", lambda physics, use_isaac_sim: physics)
    monkeypatch.setattr("isaaclab.sim.simulation_context.stage_utils.close_stage", lambda: None)
    monkeypatch.setattr("isaaclab.sim.simulation_context.clear_resolve_matching_names_cache", lambda: None)
    _FakePhysicsManager.reset()
    SimulationContext._instance = None
    cfg = SimpleNamespace(
        physics=SimpleNamespace(class_type=_FakePhysicsManager, dt=0.01),
        create_stage_in_memory=True,
        device="cpu",
        dt=0.01,
        render_interval=1,
    )
    return SimulationContext(cfg)


def _register(collection: ActuatorCollection, key: object, control: _Control):
    return collection.register_articulation(
        key=key,
        cfgs={},
        control=control,
        replication_cfg_id=1 if key == "first" else 2,
        debug_validation=False,
        debug_value_resolution=False,
    )


def _managed_cfgs(actuator_type=IdealPDActuator) -> dict[str, SimpleNamespace]:
    return {"wheel": SimpleNamespace(class_type=actuator_type, joint_names_expr=["wheel"])}


def _register_managed(collection: ActuatorCollection, key: object, control: _Control, cfgs=None):
    return collection.register_articulation(
        key=key,
        cfgs=_managed_cfgs() if cfgs is None else cfgs,
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


def test_second_completion_failure_rolls_back_first_completed_asset(monkeypatch) -> None:
    collection = ActuatorCollection(_Simulation())
    first_control = _Control(num_joints=1)
    second_control = _Control(complete_error=RuntimeError("second completion"), num_joints=1)
    first = _register_managed(collection, "first", first_control)
    second = _register_managed(collection, "second", second_control)
    allocated_stores = []
    allocate = actuator_collection._TypedStore.allocate

    def record_allocation(store, layouts, *, device):
        allocated_stores.append(store)
        allocate(store, layouts, device=device)

    monkeypatch.setattr(actuator_collection._TypedStore, "allocate", record_allocation)

    with pytest.raises(RuntimeError, match="wheel.*IdealPDActuator.*second completion"):
        collection.finalize()

    assert collection.generation is None
    assert not first_control.asset._is_initialized and not first_control.asset.data.is_primed
    assert not second_control.asset._is_initialized and not second_control.asset.data.is_primed
    assert first_control.invalidate_count == second_control.invalidate_count == 1
    assert allocated_stores and all(not store._fields for store in allocated_stores)
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = first.command
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = second.command


def test_same_type_registrations_use_disjoint_global_storage() -> None:
    collection = ActuatorCollection(_Simulation())
    first = _register_managed(collection, "first", _Control(num_joints=1))
    second = _register_managed(collection, "second", _Control(num_joints=1))

    collection.finalize()

    assert first.is_ready and second.is_ready
    bindings = collection._active_generation.bindings
    first_slice = bindings[0].layout.type_layouts[IdealPDActuator].global_slice
    second_slice = bindings[1].layout.type_layouts[IdealPDActuator].global_slice
    assert first_slice.stop <= second_slice.start


def test_failed_retry_then_stop_invalidates_every_registration() -> None:
    collection = ActuatorCollection(_Simulation())
    first_control = _Control(complete_error=RuntimeError("retry completion"))
    second_control = _Control()
    _register(collection, "first", first_control)
    _register(collection, "second", second_control)

    with pytest.raises(RuntimeError, match="retry completion"):
        collection.finalize()
    first_control.complete_error = None
    collection.finalize()
    collection.clear_generation()

    assert first_control.invalidate_count == second_control.invalidate_count == 2
    assert not first_control.asset._is_initialized and not first_control.asset.data.is_primed
    assert not second_control.asset._is_initialized and not second_control.asset.data.is_primed


def test_partial_store_allocation_failure_releases_earlier_store(monkeypatch) -> None:
    collection = ActuatorCollection(_Simulation())
    _register_managed(collection, "first", _Control(num_joints=1), _managed_cfgs(IdealPDActuator))
    _register_managed(collection, "second", _Control(num_joints=1), _managed_cfgs(ImplicitActuator))
    created_stores = []
    allocate = actuator_collection._TypedStore.allocate

    def fail_after_first_allocation(store, layouts, *, device):
        created_stores.append(store)
        if len(created_stores) == 2:
            raise RuntimeError("second store allocation")
        allocate(store, layouts, device=device)

    monkeypatch.setattr(actuator_collection._TypedStore, "allocate", fail_after_first_allocation)

    with pytest.raises(RuntimeError, match="second store allocation"):
        collection.finalize()

    assert len(created_stores) == 2
    assert all(not store._fields for store in created_stores)


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


def test_context_eagerly_registers_named_actuator_bridges_before_ready_dispatch(monkeypatch) -> None:
    context = _make_fake_context(monkeypatch)
    ready_callbacks = sorted(
        (entry for entry in _FakePhysicsManager.callbacks if entry.event == PhysicsEvent.PHYSICS_READY),
        key=lambda entry: entry.order,
    )
    stop_callbacks = [entry for entry in _FakePhysicsManager.callbacks if entry.event == PhysicsEvent.STOP]

    assert [(entry.order, entry.name) for entry in ready_callbacks] == [
        (5, "render_context_initialize"),
        (20, "actuator_collection_finalize"),
    ]
    assert [(entry.order, entry.name) for entry in stop_callbacks] == [(20, "actuator_collection_stop")]
    assert context.services[ActuatorCollection] is None


def test_context_ready_snapshot_finalizes_order_ten_registration_and_stop_replays(monkeypatch) -> None:
    context = _make_fake_context(monkeypatch)
    context.set_clone_plan(
        ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 1), dtype=torch.bool),
            cfg_rows={1: (0,), 2: (0,)},
        )
    )
    registered = []

    def register_order_ten(_payload) -> None:
        collection = context._get_actuator_collection()
        control = _Control()
        registered.append(_register(collection, "first" if not registered else "second", control))

    _FakePhysicsManager.register_callback(
        register_order_ten, PhysicsEvent.PHYSICS_READY, order=10, name="asset_initialize"
    )
    _FakePhysicsManager.dispatch(PhysicsEvent.PHYSICS_READY)
    collection = context.services[ActuatorCollection]
    assert collection.is_finalized
    assert registered[0].is_ready

    old = registered[0]
    _FakePhysicsManager.dispatch(PhysicsEvent.STOP)
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = old.command
    _FakePhysicsManager.dispatch(PhysicsEvent.PHYSICS_READY)
    assert registered[1].is_ready
    assert registered[1] is not old


def test_context_clear_instance_closes_collection_before_a_second_context(monkeypatch) -> None:
    first = _make_fake_context(monkeypatch)
    collection = first._get_actuator_collection()
    view = _register(collection, "first", _Control())

    SimulationContext.clear_instance()
    with pytest.raises(RuntimeError, match="closed"):
        _ = view.command
    second = _make_fake_context(monkeypatch)
    assert second.services[ActuatorCollection] is None
    SimulationContext.clear_instance()


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
