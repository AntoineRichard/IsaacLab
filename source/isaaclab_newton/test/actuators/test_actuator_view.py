# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the standalone Newton actuator parameter view."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.actuators.actuator_view import ActuatorView
from newton.actuators import Actuator, ClampingMaxEffort, ControllerPD, Delay


def _make_actuator(
    kp: list[float],
    *,
    delay_steps: list[int] | None = None,
    max_effort: list[float] | None = None,
) -> Actuator:
    """Create a CPU PD actuator with one parameter entry per index."""
    count = len(kp)
    return Actuator(
        indices=wp.array(range(count), dtype=wp.uint32, device="cpu"),
        controller=ControllerPD(
            kp=wp.array(kp, dtype=wp.float32, device="cpu"),
            kd=wp.zeros(count, dtype=wp.float32, device="cpu"),
        ),
        delay=(
            Delay(wp.array(delay_steps, dtype=wp.int32, device="cpu"), max(delay_steps))
            if delay_steps is not None
            else None
        ),
        clamping=(
            [ClampingMaxEffort(wp.array(max_effort, dtype=wp.float32, device="cpu"))]
            if max_effort is not None
            else None
        ),
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )


def _mapping() -> wp.array:
    return wp.array([[0, -1, 1], [2, -1, 3]], dtype=wp.int32, device="cpu")


def test_get_gathers_only_the_requested_actuator():
    first = _make_actuator([10.0, 20.0, 30.0, 40.0])
    second = _make_actuator([50.0, 60.0, 70.0, 80.0])
    view = ActuatorView(
        {
            first: _mapping(),
            second: wp.array([[-1, 0, -1], [-1, 2, -1]], dtype=wp.int32, device="cpu"),
        }
    )

    values = view.get_actuator_parameter(first, "controller", "kp")

    np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])


def test_set_scatters_only_mapped_values():
    actuator = _make_actuator([10.0, 20.0, 30.0, 40.0])
    view = ActuatorView({actuator: _mapping()})
    values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device="cpu")

    view.set_actuator_parameter(actuator, "controller", "kp", values)

    np.testing.assert_array_equal(actuator.controller.kp.numpy(), [11.0, 21.0, 31.0, 41.0])


def test_set_honors_world_mask():
    actuator = _make_actuator([10.0, 20.0, 30.0, 40.0])
    view = ActuatorView({actuator: _mapping()})
    values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device="cpu")

    view.set_actuator_parameter(
        actuator,
        "controller",
        "kp",
        values,
        mask=wp.array([False, True], dtype=wp.bool, device="cpu"),
    )

    np.testing.assert_array_equal(actuator.controller.kp.numpy(), [10.0, 20.0, 31.0, 41.0])


def test_component_and_parameter_names_are_strings():
    actuator = _make_actuator(
        [10.0, 20.0, 30.0, 40.0],
        delay_steps=[1, 2, 1, 2],
        max_effort=[100.0, 200.0, 300.0, 400.0],
    )
    view = ActuatorView({actuator: _mapping()})

    delays = view.get_actuator_parameter(actuator, "delay", "delay_steps")
    efforts = view.get_actuator_parameter(actuator, "clamping.0", "max_effort")

    np.testing.assert_array_equal(delays.numpy(), [[1, 0, 2], [1, 0, 2]])
    np.testing.assert_array_equal(efforts.numpy(), [[100.0, 0.0, 200.0], [300.0, 0.0, 400.0]])


class _ArticulationView:
    def __init__(self, actuators: list[Actuator], mappings: dict[Actuator, wp.array]):
        self.world_count = 2
        self.model = SimpleNamespace(actuators=actuators)
        self._mappings = mappings

    def _get_actuator_dof_mapping(self, actuator: Actuator) -> wp.array:
        return self._mappings[actuator]


def test_from_articulation_view_extracts_default_mappings():
    actuator = _make_actuator([10.0, 20.0, 30.0, 40.0])
    source = _ArticulationView(
        [actuator],
        {actuator: wp.array([0, -1, 1, 2, -1, 3], dtype=wp.int32, device="cpu")},
    )

    view = ActuatorView.from_articulation_view(source)

    values = view.get_actuator_parameter(actuator, "controller", "kp")
    np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])


def test_from_articulation_view_accepts_actuator_subset():
    included = _make_actuator([10.0, 20.0, 30.0, 40.0])
    excluded = _make_actuator([50.0, 60.0, 70.0, 80.0])
    source = _ArticulationView(
        [included, excluded],
        {
            included: wp.array([0, -1, 1, 2, -1, 3], dtype=wp.int32, device="cpu"),
            excluded: wp.array([-1, 0, -1, -1, 2, -1], dtype=wp.int32, device="cpu"),
        },
    )

    view = ActuatorView.from_articulation_view(source, [included])

    with pytest.raises(KeyError):
        view.get_actuator_parameter(excluded, "controller", "kp")
