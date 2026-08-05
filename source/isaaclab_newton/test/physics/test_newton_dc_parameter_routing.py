# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-Newton regressions for DC motor runtime parameter routing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray


def _make_runtime_control(adapter, *, num_envs: int, num_joints: int):
    """Return a minimal control owner around a real Newton actuator adapter."""
    from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl

    backend_to_user = wp.array(list(range(num_joints)), dtype=wp.int32, device="cpu")
    articulation = SimpleNamespace(
        num_instances=num_envs,
        num_joints=num_joints,
        num_fixed_tendons=0,
        device="cpu",
        data=SimpleNamespace(has_joint_ordering=False),
        newton_actuator_adapter=adapter,
        _joint_backend_to_user_map=lambda: backend_to_user,
    )
    control = NewtonActuatorControl(articulation)
    control._native_dof_offset = 0
    return control


def _write_parameter(control, name: str, canonical: ProxyArray, owner_slots: torch.Tensor) -> None:
    """Route one canonical DC parameter update through the public backend hook."""
    from isaaclab.actuators import DCMotor
    from isaaclab.actuators.actuator_control import _ActuatorParameterWrite

    control.write_actuator_parameter(
        name,
        _ActuatorParameterWrite(
            value=canonical.torch,
            actuator_type=DCMotor,
            canonical=canonical,
            backend_owner_slots=owner_slots,
        ),
    )


def _dc_components(actuators) -> tuple[list[object], list[object]]:
    """Return the real DC and absolute-effort clamping components."""
    dc_components = [
        next(component for component in actuator.clamping if hasattr(component, "max_motor_effort"))
        for actuator in actuators
    ]
    max_components = [
        next(component for component in actuator.clamping if hasattr(component, "max_effort")) for actuator in actuators
    ]
    return dc_components, max_components


def _source_snapshot(actuators) -> dict[str, np.ndarray]:
    """Copy every routed Newton source and the derived DC corner."""
    dc_components, max_components = _dc_components(actuators)
    return {
        "saturation_effort": np.concatenate([component.saturation_effort.numpy() for component in dc_components]),
        "velocity_limit": np.concatenate([component.velocity_limit.numpy() for component in dc_components]),
        "max_motor_effort": np.concatenate([component.max_motor_effort.numpy() for component in dc_components]),
        "max_effort": np.concatenate([component.max_effort.numpy() for component in max_components]),
        "corner_velocity": np.concatenate([component.corner_velocity.numpy() for component in dc_components]),
    }


def _clip_positive_effort(actuators, source_velocities: np.ndarray) -> np.ndarray:
    """Apply real Newton DC clamping at one velocity per flattened actuator source."""
    dc_components, _ = _dc_components(actuators)
    clipped = []
    source_offset = 0
    for actuator, component in zip(actuators, dc_components, strict=True):
        count = component.velocity_limit.shape[0]
        source = wp.full(count, 1_000.0, dtype=wp.float32, device="cpu")
        destination = wp.zeros(count, dtype=wp.float32, device="cpu")
        positions = wp.zeros(6, dtype=wp.float32, device="cpu")
        indices = actuator.indices.numpy().copy()
        velocity_values = np.zeros(6, dtype=np.float32)
        velocity_values[indices] = source_velocities[source_offset : source_offset + count]
        velocities = wp.array(velocity_values, dtype=wp.float32, device="cpu")
        component.modify_forces(
            source,
            destination,
            positions,
            velocities,
            actuator.indices,
            actuator.indices,
            device=wp.get_device("cpu"),
        )
        clipped.append(destination.numpy().copy())
        source_offset += count
    return np.concatenate(clipped)


def _write_and_assert(
    control,
    actuators,
    name: str,
    canonical: ProxyArray,
    values: list[list[float]],
    owner_slots: torch.Tensor,
    target_source_indices: tuple[int, ...],
) -> None:
    """Route one source, proving isolation, corner refresh, and changed real clipping."""
    before = _source_snapshot(actuators)
    target_values = np.asarray(values, dtype=np.float32).reshape(-1)
    target_indices = np.asarray(target_source_indices, dtype=np.int64)
    if target_values.shape != target_indices.shape:
        raise AssertionError(f"Expected {target_indices.size} routed values, got {target_values.size}.")

    if name == "velocity_limit":
        boundary_velocities = before["velocity_limit"].copy()
        boundary_velocities[target_indices] = target_values
        clipping_before = _clip_positive_effort(actuators, boundary_velocities)
    else:
        boundary_velocities = np.zeros_like(before["velocity_limit"])
        clipping_before = _clip_positive_effort(actuators, boundary_velocities)

    canonical.torch.copy_(torch.as_tensor(values, dtype=canonical.torch.dtype))
    _write_parameter(control, name, canonical, owner_slots)

    after = _source_snapshot(actuators)
    target_attributes = {
        "saturation_effort": ("saturation_effort",),
        "velocity_limit": ("velocity_limit",),
        "effort_limit": ("max_motor_effort", "max_effort"),
    }[name]
    for attribute in ("saturation_effort", "velocity_limit", "max_motor_effort", "max_effort"):
        expected = before[attribute].copy()
        if attribute in target_attributes:
            expected[target_indices] = target_values
        np.testing.assert_allclose(after[attribute], expected)

    expected_corner = after["velocity_limit"] * (1.0 + after["max_motor_effort"] / after["saturation_effort"])
    np.testing.assert_allclose(after["corner_velocity"], expected_corner)

    clipping_after = _clip_positive_effort(actuators, boundary_velocities)
    if name == "velocity_limit":
        np.testing.assert_allclose(clipping_after[target_indices], np.zeros(target_indices.size), atol=1.0e-6)
    else:
        expected_zero_clip = np.minimum(after["saturation_effort"], after["max_motor_effort"])
        np.testing.assert_allclose(clipping_after[target_indices], expected_zero_clip[target_indices])
    assert np.all(clipping_after[target_indices] != clipping_before[target_indices])


def _hosted_adapter(monkeypatch: pytest.MonkeyPatch):
    """Build a USD-hosted direct DC actuator with real Newton components."""
    import newton.actuators as newton_actuators
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ClampingDCMotor, ClampingMaxEffort, ControllerPD

    from pxr import Usd

    from isaaclab.actuators import DCMotor

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot")
    parsed_by_path = {}
    for joint_index in range(3):
        prim = stage.DefinePrim(f"/Robot/Actuator_{joint_index}")
        parsed_by_path[str(prim.GetPath())] = SimpleNamespace(
            target_path=f"/Robot/joint_{joint_index}",
            controller_class=ControllerPD,
            controller_kwargs={"kp": 1_000.0, "kd": 0.0},
            component_specs=[
                (
                    ClampingDCMotor,
                    {"max_motor_effort": 1.0, "velocity_limit": 1.0, "saturation_effort": 1.0},
                ),
                (ClampingMaxEffort, {"max_effort": 1.0}),
            ],
        )
    monkeypatch.setattr(
        newton_actuators,
        "parse_actuator_prim",
        lambda prim: parsed_by_path.get(str(prim.GetPath())),
    )
    arrays = {
        "stiffness": ProxyArray(wp.full((2, 3), 1_000.0, dtype=wp.float32, device="cpu")),
        "damping": ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu")),
        "effort_limit": ProxyArray(wp.full((2, 3), 100.0, dtype=wp.float32, device="cpu")),
        "velocity_limit": ProxyArray(wp.full((2, 3), 1.0, dtype=wp.float32, device="cpu")),
        "saturation_effort": ProxyArray(wp.full((2, 3), 1.0, dtype=wp.float32, device="cpu")),
        "computed_effort": ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu")),
        "applied_effort": ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu")),
    }
    binding = SimpleNamespace(
        groups={"native": SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays))},
        native_group_names=frozenset({"native"}),
        layout=SimpleNamespace(
            num_joints=3,
            type_layouts={DCMotor: SimpleNamespace(num_worlds=2, num_dofs=3, compact_joint_indices=(0, 1, 2))},
            group_layouts=(
                SimpleNamespace(
                    name="native",
                    actuator_type=DCMotor,
                    joint_indices=(0, 1, 2),
                    joint_names=("joint_0", "joint_1", "joint_2"),
                    type_slice=slice(0, 3),
                ),
            ),
        ),
    )
    adapter = NewtonActuatorAdapter._from_usd_binding(
        binding,
        stage=stage,
        joint_names=["joint_0", "joint_1", "joint_2"],
        num_envs=2,
        num_joints=3,
        device="cpu",
        articulation_prim_path="/Robot",
    )
    return adapter, arrays


def test_hosted_dc_canonical_writes_update_real_newton_clamping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted canonical writes update each DC field and its real torque-speed curve."""
    adapter, arrays = _hosted_adapter(monkeypatch)
    control = _make_runtime_control(adapter, num_envs=2, num_joints=3)
    owner_slots = torch.tensor([0, 1, 2], dtype=torch.int32)
    values = {
        "saturation_effort": [[20.0, 25.0, 30.0], [35.0, 40.0, 45.0]],
        "velocity_limit": [[2.0, 2.5, 3.0], [3.5, 4.0, 4.5]],
        "effort_limit": [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
    }

    for name, value in values.items():
        _write_and_assert(control, adapter.actuators, name, arrays[name], value, owner_slots, tuple(range(6)))


def _native_staged_adapter():
    """Build real native A/B/A DC motors whose setter paths must remain disjoint."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ClampingDCMotor, ClampingMaxEffort, ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(6)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation(joints)
    for index in (0, 2, 3, 5):
        builder.add_actuator(
            ControllerPD,
            index=index,
            kp=1_000.0,
            kd=0.0,
            clamping=[
                (ClampingDCMotor, {"max_motor_effort": 100.0, "velocity_limit": 1.0, "saturation_effort": 1.0}),
                (ClampingMaxEffort, {"max_effort": 100.0}),
            ],
        )
    for index in (1, 4):
        builder.add_actuator(
            ControllerPD,
            index=index,
            kp=1_000.0,
            kd=0.0,
            delay_steps=1,
            clamping=[
                (ClampingDCMotor, {"max_motor_effort": 100.0, "velocity_limit": 1.0, "saturation_effort": 1.0}),
                (ClampingMaxEffort, {"max_effort": 100.0}),
            ],
        )
    keys = tuple(builder.actuator_entries)
    model = builder.finalize(device="cpu")
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=2,
        num_joints=3,
        dof_offset=0,
        device="cpu",
        actuator_keys=keys,
        actuator_dof_indices={keys[0]: (0, 2), keys[1]: (1,)},
    )
    return adapter


def test_native_dc_staged_a_b_a_canonical_writes_route_each_field() -> None:
    """Native A/B/A setters route each DC parameter without cross-group contamination."""
    adapter = _native_staged_adapter()
    control = _make_runtime_control(adapter, num_envs=2, num_joints=3)
    owner_a = torch.tensor([0, -1, 1], dtype=torch.int32)
    owner_b = torch.tensor([-1, 0, -1], dtype=torch.int32)
    writes = {
        "saturation_effort": (
            [[20.0, 30.0], [40.0, 50.0]],
            [[55.0], [65.0]],
            [[50.0, 80.0], [60.0, 100.0]],
        ),
        "velocity_limit": (
            [[2.0, 2.5], [3.0, 3.5]],
            [[2.2], [2.8]],
            [[3.0, 4.0], [5.0, 6.0]],
        ),
        "effort_limit": (
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0], [10.0]],
            [[11.0, 12.0], [13.0, 14.0]],
        ),
    }

    for name, (first_a, only_b, final_a) in writes.items():
        for values, owners, target_indices in (
            (first_a, owner_a, (0, 1, 2, 3)),
            (only_b, owner_b, (4, 5)),
            (final_a, owner_a, (0, 1, 2, 3)),
        ):
            canonical = ProxyArray(wp.array(values, dtype=wp.float32, device="cpu"))
            _write_and_assert(control, adapter.actuators, name, canonical, values, owners, target_indices)


def test_native_parameter_ranges_keep_shared_controller_exact_types_isolated() -> None:
    """An A/B/A exact-type write patches only its owned shared-controller occurrence."""
    from isaaclab.actuators.actuator_control import _ActuatorParameterWrite

    class _TypeA:
        pass

    class _TypeB:
        pass

    adapter = _native_staged_adapter()
    control = _make_runtime_control(adapter, num_envs=2, num_joints=3)
    shared = adapter.actuators[0]
    articulation = control._articulation
    range_base = {
        "actuator": shared,
        "direct": False,
        "compact_joint_ids": wp.array([0], dtype=wp.int32, device="cpu"),
        "canonical_slots": wp.array([0], dtype=wp.int32, device="cpu"),
        "controller_stride": 2,
    }
    articulation._newton_native_ranges = (
        SimpleNamespace(
            **range_base,
            actuator_type=_TypeA,
            controller_local_slots=wp.array([0], dtype=wp.int32, device="cpu"),
        ),
        SimpleNamespace(
            **range_base,
            actuator_type=_TypeB,
            controller_local_slots=wp.array([1], dtype=wp.int32, device="cpu"),
        ),
    )
    owner_slots = torch.tensor([0, -1, -1], dtype=torch.int32)
    before = shared.controller.kp.numpy().copy()
    first = ProxyArray(wp.array([[9.0], [10.0]], dtype=wp.float32, device="cpu"))
    control.write_actuator_parameter(
        "stiffness",
        _ActuatorParameterWrite(
            value=first.torch,
            actuator_type=_TypeA,
            canonical=first,
            backend_owner_slots=owner_slots,
        ),
    )
    after_a = shared.controller.kp.numpy().copy()
    np.testing.assert_allclose(after_a[[0, 2]], [9.0, 10.0])
    np.testing.assert_allclose(after_a[[1, 3]], before[[1, 3]])

    second = ProxyArray(wp.array([[20.0], [30.0]], dtype=wp.float32, device="cpu"))
    control.write_actuator_parameter(
        "stiffness",
        _ActuatorParameterWrite(
            value=second.torch,
            actuator_type=_TypeB,
            canonical=second,
            backend_owner_slots=owner_slots,
        ),
    )
    after_b = shared.controller.kp.numpy()
    np.testing.assert_allclose(after_b[[0, 2]], [9.0, 10.0])
    np.testing.assert_allclose(after_b[[1, 3]], [20.0, 30.0])


def test_physx_range_parameter_write_keeps_duplicate_controller_occurrences_isolated() -> None:
    """PhysX production ranges patch one selected exact-type duplicate occurrence."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from isaaclab_physx.assets.articulation.actuator_control import PhysxActuatorControl
    from newton.actuators import ControllerPD

    class _TypeA:
        pass

    class _TypeB:
        pass

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.0)
    builder.add_actuator(ControllerPD, index=0, kp=7.0, kd=0.0)
    model = builder.finalize(device="cpu")
    (signature,) = builder.actuator_entries
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=1,
        dof_offset=0,
        device="cpu",
        actuator_keys=(signature,),
        actuator_dof_indices={signature: (0, 0)},
    )
    actuator = model.actuators[0]
    range_base = {
        "actuator": actuator,
        "direct": False,
        "compact_joint_ids": wp.array([0], dtype=wp.int32, device="cpu"),
        "canonical_slots": wp.array([0], dtype=wp.int32, device="cpu"),
        "controller_stride": 2,
    }
    control = object.__new__(PhysxActuatorControl)
    control._articulation = SimpleNamespace(
        num_instances=1,
        num_joints=1,
        device="cpu",
        newton_actuator_adapter=adapter,
        _newton_native_ranges=(
            SimpleNamespace(
                **range_base,
                actuator_type=_TypeA,
                controller_local_slots=wp.array([0], dtype=wp.int32, device="cpu"),
            ),
            SimpleNamespace(
                **range_base,
                actuator_type=_TypeB,
                controller_local_slots=wp.array([1], dtype=wp.int32, device="cpu"),
            ),
        ),
    )
    control._all_env_mask = wp.ones(1, dtype=wp.bool, device="cpu")
    control._all_joint_mask = wp.ones(1, dtype=wp.bool, device="cpu")

    def write(value: float, actuator_type: type, env_selected: bool) -> None:
        canonical = ProxyArray(wp.array([[value]], dtype=wp.float32, device="cpu"))
        control._write_native_actuator_parameter(
            "stiffness",
            SimpleNamespace(
                canonical=canonical,
                backend_owner_slots=wp.array([0], dtype=wp.int32, device="cpu"),
                actuator_type=actuator_type,
                env_mask=wp.array([env_selected], dtype=wp.bool, device="cpu"),
                joint_mask=wp.array([True], dtype=wp.bool, device="cpu"),
                env_ids=None,
                joint_ids=None,
            ),
        )

    write(9.0, _TypeA, False)
    np.testing.assert_allclose(actuator.controller.kp.numpy(), [2.0, 7.0])
    write(9.0, _TypeA, True)
    np.testing.assert_allclose(actuator.controller.kp.numpy(), [9.0, 7.0])
    write(20.0, _TypeB, True)
    np.testing.assert_allclose(actuator.controller.kp.numpy(), [9.0, 20.0])
