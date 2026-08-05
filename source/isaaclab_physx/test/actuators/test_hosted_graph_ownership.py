# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused ownership tests for hosted-Newton actuator CUDA graphs."""

from __future__ import annotations

import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from isaaclab_physx.assets.articulation.actuator_control import PhysxActuatorControl


def test_hosted_graph_cleanup_retains_capture_owners_until_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed hosted teardown pins capture owners until its graph handles are released."""
    import isaaclab_newton.actuators._graph as graph_module
    from isaaclab_newton.actuators._graph import _CapturedGraphLease

    class Owner:
        pass

    class Adapter(Owner):
        def unregister_articulation_ranges(self, _ranges) -> None:
            pass

    raw_command_effort = Owner()
    raw_computed_effort = Owner()
    raw_applied_effort = Owner()
    collection = SimpleNamespace(
        generation=17,
        joint_command=SimpleNamespace(effort=SimpleNamespace(warp=raw_command_effort)),
        computed_effort=SimpleNamespace(warp=raw_computed_effort),
        applied_effort=SimpleNamespace(warp=raw_applied_effort),
    )
    adapter = Adapter()
    adapter._states_a = Owner()
    adapter._states_b = Owner()
    wrapper = Owner()
    ranges = [Owner()]
    owner_refs = [
        weakref.ref(raw_command_effort),
        weakref.ref(raw_computed_effort),
        weakref.ref(raw_applied_effort),
        weakref.ref(adapter),
        weakref.ref(adapter._states_a),
        weakref.ref(adapter._states_b),
        weakref.ref(wrapper),
        weakref.ref(ranges[0]),
    ]
    graph = SimpleNamespace(
        device=SimpleNamespace(context=object(), context_guard=nullcontext()),
        graph=object(),
        graph_exec=object(),
    )
    attempts = 0

    def synchronize(_device) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected hosted graph synchronization failure")

    monkeypatch.setattr(graph_module.wp, "synchronize_device", synchronize)
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=lambda *_args: True,
                wp_cuda_graph_exec_destroy=lambda *_args: True,
            )
        ),
    )
    control = object.__new__(PhysxActuatorControl)
    control._native_active = True
    control._physx_actuator_wrapper = wrapper
    control._native_actuator_graphs = (
        _CapturedGraphLease(
            graph,
            generation=control._make_native_actuator_graph_generation(collection, adapter, wrapper, ranges),
            label="hosted owner lifetime graph",
        ),
    )
    control._native_actuator_graph_index = 0
    control._native_actuator_graph_dt = 0.01
    articulation = SimpleNamespace(
        newton_actuator_adapter=adapter,
        _newton_native_ranges=ranges,
        _physx_actuator_wrapper=wrapper,
        _implicit_dof_mask=None,
        _implicit_dof_mask_owner=None,
        _native_dof_mask=None,
        _native_dof_mask_owner=None,
        _native_dof_masks=None,
        _native_dof_mask_owners=None,
        _has_newton_actuators=True,
        _data=SimpleNamespace(_sim_bind_joint_computed_effort=Owner()),
    )
    control._articulation = articulation
    lease = control._native_actuator_graphs[0]
    collection.joint_command = None
    collection.computed_effort = None
    collection.applied_effort = None
    del collection, adapter, wrapper, ranges
    del raw_command_effort, raw_computed_effort, raw_applied_effort

    with pytest.raises(RuntimeError, match="injected hosted graph synchronization failure"):
        control._clear_native_actuator_state()

    gc.collect()
    assert control._native_actuator_graphs == (lease,)
    assert lease.is_live is False
    assert control._physx_actuator_wrapper is None
    assert articulation.newton_actuator_adapter is None
    assert articulation._newton_native_ranges is None
    assert all(owner_ref() is not None for owner_ref in owner_refs)
    with pytest.raises(RuntimeError, match="hosted owner lifetime graph.*revoked"):
        lease.launch()

    control.invalidate_actuator_graphs()

    gc.collect()
    assert control._native_actuator_graphs is None
    assert all(owner_ref() is None for owner_ref in owner_refs)


def test_hosted_partial_capture_retains_failed_cleanup_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed partial-capture cleanup remains reachable instead of falling back with freed buffers."""
    import isaaclab_physx.assets.articulation.actuator_control as control_module

    class FailingLease:
        is_live = False

        def revoke(self) -> None:
            raise RuntimeError("injected partial graph cleanup failure")

    class PartialCapture:
        count = 0

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.graph = None

        def __enter__(self):
            type(self).count += 1
            if self.count == 2:
                raise RuntimeError("injected second capture failure")
            self.graph = object()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            del exc_type, exc_value, traceback
            return False

    lease = FailingLease()
    adapter = SimpleNamespace(_states_a=object(), _states_b=object())
    control = object.__new__(PhysxActuatorControl)
    control._articulation = SimpleNamespace(
        device="cuda:0",
        newton_actuator_adapter=adapter,
        _newton_native_ranges=(),
    )
    control._physx_actuator_wrapper = object()
    control._native_actuator_graphs = None
    control._native_actuator_graph_index = 0
    control._native_actuator_graph_dt = None
    control._run_native_actuator_kernels = lambda _collection, _dt: None
    monkeypatch.setattr(control_module.wp, "ScopedCapture", PartialCapture)
    monkeypatch.setattr(control_module, "_make_captured_graph_lease", lambda *_args, **_kwargs: lease)

    field = SimpleNamespace(warp=object())
    collection = SimpleNamespace(
        generation=object(),
        joint_command=SimpleNamespace(effort=field),
        computed_effort=field,
        applied_effort=field,
    )
    with pytest.raises(RuntimeError, match="injected second capture failure") as raised:
        control._capture_native_actuator_graphs(collection, 0.01)

    assert any("injected partial graph cleanup failure" in note for note in raised.value.__notes__)
    assert control._native_actuator_graphs == (lease,)
