# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused pointer-lifetime tests for the native Newton actuator adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter


class _PointerOwner:
    """Small Newton-shaped component that can fail a selected pointer write."""

    def __init__(self, events: list[tuple[str, object]], **arrays: wp.array) -> None:
        object.__setattr__(self, "_events", events)
        object.__setattr__(self, "_failure", None)
        object.__setattr__(self, "_fail_after_assignment", False)
        object.__setattr__(self, "_writes", 0)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    def fail_after(self, write_count: int, *, after_assignment: bool = False) -> None:
        """Raise on the selected mutation write, after recording it."""
        object.__setattr__(self, "_failure", write_count)
        object.__setattr__(self, "_fail_after_assignment", after_assignment)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_") and name not in {"_computed_forces", "_applied_forces"}:
            object.__setattr__(self, name, value)
            return
        write_count = self._writes + 1
        object.__setattr__(self, "_writes", write_count)
        self._events.append((name, value))
        if self._failure == write_count and not self._fail_after_assignment:
            raise RuntimeError(f"injected {name} pointer failure")
        object.__setattr__(self, name, value)
        if self._failure == write_count:
            raise RuntimeError(f"injected {name} pointer failure")


class _ActuatorOwner(_PointerOwner):
    """Newton-shaped actuator owner that records output pointer replacements."""

    def __init__(
        self, events: list[tuple[str, object]], controller: object, clamping: tuple[object, ...], **arrays
    ) -> None:
        super().__init__(events, **arrays)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "clamping", clamping)
        object.__setattr__(self, "joint_f", object())
        object.__setattr__(self, "control", object())


def _array(value: float) -> wp.array:
    """Return a genuine Warp array used by Newton controller fields."""
    return wp.full(2, value, dtype=wp.float32, device="cpu")


def _adapter(*, owns_actuators: bool = False) -> NewtonActuatorAdapter:
    """Build only the private state required by the pointer transaction."""
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter._direct_pointer_bindings = {}
    adapter._owns_actuators = owns_actuators
    return adapter


def _actuator(events: list[tuple[str, object]]) -> tuple[SimpleNamespace, dict[str, wp.array]]:
    """Build a genuine-shape controller/clamping/output pointer graph."""
    originals = {
        "kp": _array(1.0),
        "kd": _array(2.0),
        "limit": _array(3.0),
        "computed": _array(4.0),
        "applied": _array(5.0),
    }
    controller = _PointerOwner(events, kp=originals["kp"], kd=originals["kd"])
    clamping = _PointerOwner(events, max_effort=originals["limit"])
    return (
        _ActuatorOwner(
            events,
            controller,
            (clamping,),
            _computed_forces=originals["computed"],
            _applied_forces=originals["applied"],
        ),
        originals,
    )


def _parameters() -> dict[str, wp.array]:
    """Return canonical arrays with the same Warp ABI as direct binding."""
    return {"stiffness": _array(11.0), "damping": _array(12.0), "effort_limit": _array(13.0)}


def test_direct_pointer_install_rolls_back_every_mutation_in_reverse_order() -> None:
    """A mid-install write failure restores controller, clamping, and output identities."""
    events: list[tuple[str, object]] = []
    actuator, originals = _actuator(events)
    actuator.clamping[0].fail_after(1)
    adapter = _adapter()
    computed, applied = _array(21.0), _array(22.0)

    mutations = adapter._prepare_direct_pointer_mutations(actuator, _parameters(), computed, applied)

    with pytest.raises(RuntimeError, match="injected max_effort"):
        adapter._install_direct_pointer_binding(actuator, object(), mutations)

    assert actuator.controller.kp is originals["kp"]
    assert actuator.controller.kd is originals["kd"]
    assert actuator.clamping[0].max_effort is originals["limit"]
    assert actuator._computed_forces is originals["computed"]
    assert actuator._applied_forces is originals["applied"]
    assert adapter._direct_pointer_bindings == {}
    # The restoring writes follow the successfully installed writes backwards.
    assert [name for name, _ in events] == ["kp", "kd", "max_effort", "kd", "kp"]


def test_direct_pointer_teardown_preserves_external_rebind_and_retries_failures() -> None:
    """Last release only restores still-owned pointers and retains a failed inverse write for retry."""
    events: list[tuple[str, object]] = []
    actuator, originals = _actuator(events)
    adapter = _adapter()
    handle = object()
    mutations = adapter._prepare_direct_pointer_mutations(actuator, _parameters(), _array(21.0), _array(22.0))
    adapter._install_direct_pointer_binding(actuator, handle, mutations)

    external_kp = _array(30.0)
    actuator.controller.kp = external_kp
    actuator.controller.fail_after(actuator.controller._writes + 1)

    with pytest.raises(RuntimeError, match="injected kd") as raised:
        adapter._release_direct_pointer_binding(actuator, handle)

    assert actuator.controller.kp is external_kp
    assert actuator.controller.kd is not originals["kd"]
    assert actuator.clamping[0].max_effort is originals["limit"]
    assert actuator._computed_forces is originals["computed"]
    assert actuator._applied_forces is originals["applied"]
    assert "idempotent" not in str(raised.value)
    assert id(actuator) in adapter._direct_pointer_bindings

    actuator.controller.fail_after(-1)
    adapter._release_direct_pointer_binding(actuator, handle)

    assert actuator.controller.kp is external_kp
    assert actuator.controller.kd is originals["kd"]
    assert adapter._direct_pointer_bindings == {}


def test_direct_pointer_shared_registration_keeps_alias_until_final_release() -> None:
    """Identical installations share one reverse log instead of restoring the other user's alias."""
    events: list[tuple[str, object]] = []
    actuator, originals = _actuator(events)
    adapter = _adapter()
    parameters = _parameters()
    computed, applied = _array(21.0), _array(22.0)
    first, second = object(), object()
    mutations = adapter._prepare_direct_pointer_mutations(actuator, parameters, computed, applied)
    adapter._install_direct_pointer_binding(actuator, first, mutations)
    adapter._install_direct_pointer_binding(
        actuator,
        second,
        adapter._prepare_direct_pointer_mutations(actuator, parameters, computed, applied),
    )

    adapter._release_direct_pointer_binding(actuator, first)
    assert actuator.controller.kp is not originals["kp"]
    adapter._release_direct_pointer_binding(actuator, second)
    assert actuator.controller.kp is originals["kp"]
    assert actuator._computed_forces is originals["computed"]
    assert actuator._applied_forces is originals["applied"]


def test_direct_pointer_rebind_reinstalls_alias_after_successful_restore() -> None:
    """A completed restore removes its transaction log before the next direct bind."""
    events: list[tuple[str, object]] = []
    actuator, originals = _actuator(events)
    adapter = _adapter()
    first, second = object(), object()
    first_parameters = _parameters()
    adapter._install_direct_pointer_binding(
        actuator,
        first,
        adapter._prepare_direct_pointer_mutations(actuator, first_parameters, _array(21.0), _array(22.0)),
    )
    adapter._release_direct_pointer_binding(actuator, first)
    assert adapter._direct_pointer_bindings == {}
    assert actuator.controller.kp is originals["kp"]

    second_parameters = {**_parameters(), "stiffness": _array(31.0)}
    adapter._install_direct_pointer_binding(
        actuator,
        second,
        adapter._prepare_direct_pointer_mutations(actuator, second_parameters, _array(41.0), _array(42.0)),
    )
    assert actuator.controller.kp is second_parameters["stiffness"]
    adapter._release_direct_pointer_binding(actuator, second)
    assert actuator.controller.kp is originals["kp"]


def test_direct_pointer_restore_retries_derived_refresh_after_inverse_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A derived refresh failure retains an empty inverse log for a later retry."""
    events: list[tuple[str, object]] = []
    actuator, originals = _actuator(events)
    adapter = _adapter()
    handle = object()
    adapter._install_direct_pointer_binding(
        actuator,
        handle,
        adapter._prepare_direct_pointer_mutations(actuator, _parameters(), _array(21.0), _array(22.0)),
    )
    refresh_calls = 0

    def refresh_once(_actuator) -> None:
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            raise RuntimeError("derived refresh failed")

    monkeypatch.setattr(adapter, "_refresh_dc_motor_corner_velocity", refresh_once)
    with pytest.raises(RuntimeError, match="derived refresh failed"):
        adapter._release_direct_pointer_binding(actuator, handle)
    binding = adapter._direct_pointer_bindings[id(actuator)]
    assert binding.mutations == []
    assert actuator.controller.kp is originals["kp"]

    adapter._release_direct_pointer_binding(actuator, handle)
    assert refresh_calls == 2
    assert adapter._direct_pointer_bindings == {}


@pytest.mark.parametrize("output_name", ["_computed_forces", "_applied_forces"])
def test_direct_pointer_output_write_failure_reverses_every_earlier_alias(output_name: str) -> None:
    """A failing output replacement restores all controller and clamping aliases in reverse order."""
    events: list[tuple[str, object]] = []
    actuator, originals = _actuator(events)
    adapter = _adapter()
    joint_f, control = actuator.joint_f, actuator.control
    output_write = 1 if output_name == "_computed_forces" else 2
    actuator.fail_after(output_write, after_assignment=True)

    with pytest.raises(RuntimeError, match=output_name):
        adapter._install_direct_pointer_binding(
            actuator,
            object(),
            adapter._prepare_direct_pointer_mutations(actuator, _parameters(), _array(21.0), _array(22.0)),
        )

    assert actuator.controller.kp is originals["kp"]
    assert actuator.controller.kd is originals["kd"]
    assert actuator.clamping[0].max_effort is originals["limit"]
    assert actuator._computed_forces is originals["computed"]
    assert actuator._applied_forces is originals["applied"]
    assert actuator.joint_f is joint_f
    assert actuator.control is control


def test_direct_pointer_prevalidation_rejects_bad_alias_before_any_write() -> None:
    """Invalid canonical arrays fail during descriptor construction, before a Newton pointer is touched."""
    events: list[tuple[str, object]] = []
    actuator, _ = _actuator(events)
    invalid = _parameters()
    invalid["damping"] = wp.zeros(3, dtype=wp.float32, device="cpu")

    with pytest.raises(ValueError, match="identical flat sizes"):
        NewtonActuatorAdapter._prepare_direct_pointer_mutations(actuator, invalid, _array(21.0), _array(22.0))

    assert events == []


def test_direct_pointer_rejects_incompatible_live_alias_without_mutation() -> None:
    """A second user cannot replace another registration with a different canonical range."""
    events: list[tuple[str, object]] = []
    actuator, _ = _actuator(events)
    adapter = _adapter()
    first = object()
    adapter._install_direct_pointer_binding(
        actuator,
        first,
        adapter._prepare_direct_pointer_mutations(actuator, _parameters(), _array(21.0), _array(22.0)),
    )
    before = len(events)

    with pytest.raises(RuntimeError, match="already directly bound"):
        adapter._install_direct_pointer_binding(
            actuator,
            object(),
            adapter._prepare_direct_pointer_mutations(actuator, _parameters(), _array(31.0), _array(32.0)),
        )

    assert len(events) == before
    adapter._release_direct_pointer_binding(actuator, first)


def test_direct_pointer_teardown_collects_failures_from_multiple_owners() -> None:
    """Reverse cleanup continues across owners and retains the primary exception with later notes."""
    events: list[tuple[str, object]] = []
    actuator, _ = _actuator(events)
    adapter = _adapter()
    handle = object()
    adapter._install_direct_pointer_binding(
        actuator,
        handle,
        adapter._prepare_direct_pointer_mutations(actuator, _parameters(), _array(21.0), _array(22.0)),
    )
    actuator.clamping[0].fail_after(actuator.clamping[0]._writes + 1)
    actuator.controller.fail_after(actuator.controller._writes + 1)

    with pytest.raises(RuntimeError, match="max_effort") as raised:
        adapter._release_direct_pointer_binding(actuator, handle)

    assert any("kd" in note for note in getattr(raised.value, "__notes__", ()))
    actuator.clamping[0].fail_after(-1)
    actuator.controller.fail_after(-1)
    adapter._release_direct_pointer_binding(actuator, handle)


def test_owned_direct_pointer_log_discards_successful_originals() -> None:
    """Hosted success retains aliases only; its temporary original arrays leave the adapter log."""
    events: list[tuple[str, object]] = []
    actuator, _ = _actuator(events)
    adapter = _adapter(owns_actuators=True)
    handle = object()
    adapter._install_direct_pointer_binding(
        actuator,
        handle,
        adapter._prepare_direct_pointer_mutations(actuator, _parameters(), _array(21.0), _array(22.0)),
    )

    adapter._discard_owned_direct_pointer_binding(actuator, handle)

    assert adapter._direct_pointer_bindings == {}


def test_real_direct_dc_pointer_install_and_restore_refreshes_corner_velocity() -> None:
    """Direct DC aliases refresh derived corner velocity on install and final restore."""
    import newton
    from newton.actuators import ClampingDCMotor, ClampingMaxEffort, ControllerPD

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.add_actuator(
        ControllerPD,
        index=0,
        kp=1_000.0,
        kd=0.0,
        clamping=[
            (ClampingDCMotor, {"max_motor_effort": 100.0, "velocity_limit": 1.0, "saturation_effort": 1.0}),
            (ClampingMaxEffort, {"max_effort": 100.0}),
        ],
    )
    model = builder.finalize(device="cpu")
    (signature,) = builder.actuator_entries
    actuator = model.actuators[0]
    dc = next(component for component in actuator.clamping if hasattr(component, "corner_velocity"))
    original_corner = dc.corner_velocity.numpy().copy()
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=1,
        dof_offset=0,
        device="cpu",
        actuator_keys=(signature,),
        actuator_dof_indices={signature: (0,)},
    )
    parameters = {
        "stiffness": wp.full(1, 1_000.0, dtype=wp.float32, device="cpu"),
        "damping": wp.zeros(1, dtype=wp.float32, device="cpu"),
        "effort_limit": wp.full(1, 5.0, dtype=wp.float32, device="cpu"),
        "velocity_limit": wp.full(1, 2.0, dtype=wp.float32, device="cpu"),
        "saturation_effort": wp.full(1, 20.0, dtype=wp.float32, device="cpu"),
    }
    handle = object()
    adapter._install_direct_pointer_binding(
        actuator,
        handle,
        adapter._prepare_direct_pointer_mutations(
            actuator,
            parameters,
            wp.zeros(1, dtype=wp.float32, device="cpu"),
            wp.zeros(1, dtype=wp.float32, device="cpu"),
        ),
    )

    assert dc.corner_velocity.numpy().tolist() == pytest.approx([2.5])
    source = wp.full(1, 1_000.0, dtype=wp.float32, device="cpu")
    clipped = wp.zeros(1, dtype=wp.float32, device="cpu")
    dc.modify_forces(
        source,
        clipped,
        wp.zeros(1, dtype=wp.float32, device="cpu"),
        wp.zeros(1, dtype=wp.float32, device="cpu"),
        actuator.indices,
        actuator.indices,
        device=wp.get_device("cpu"),
    )
    assert clipped.numpy().tolist() == pytest.approx([5.0])

    adapter._release_direct_pointer_binding(actuator, handle)
    assert dc.corner_velocity.numpy() == pytest.approx(original_corner)
