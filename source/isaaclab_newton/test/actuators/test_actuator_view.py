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


def _make_pd_actuator(
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
            Delay(
                delay_steps=wp.array(delay_steps, dtype=wp.int32, device="cpu"),
                max_delay=max(delay_steps),
            )
            if delay_steps is not None
            else None
        ),
        clamping=(
            [ClampingMaxEffort(wp.array(max_effort, dtype=wp.float32, device="cpu"))]
            if max_effort is not None
            else None
        ),
    )


def test_get_actuator_parameter_gathers_one_actuator_into_view_layout():
    """A getter must gather mapped values and leave unowned DOFs at zero."""
    actuator = _make_pd_actuator([10.0, 20.0, 30.0, 40.0])
    mapping = wp.array([[0, -1, 1], [2, -1, 3]], dtype=wp.int32, device="cpu")
    view = ActuatorView([(actuator, mapping)])

    values = view.get_actuator_parameter(actuator, "controller", "kp")

    np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])


def test_set_actuator_parameter_scatters_only_mapped_values():
    """A setter must ignore values at mapping entries marked with -1."""
    actuator = _make_pd_actuator([10.0, 20.0, 30.0, 40.0])
    mapping = wp.array([[0, -1, 1], [2, -1, 3]], dtype=wp.int32, device="cpu")
    view = ActuatorView([(actuator, mapping)])
    values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device="cpu")

    view.set_actuator_parameter(actuator, "controller", "kp", values)

    np.testing.assert_array_equal(actuator.controller.kp.numpy(), [11.0, 21.0, 31.0, 41.0])


def test_set_actuator_parameter_honors_world_mask():
    """A world mask must prevent writes to every DOF in unselected worlds."""
    actuator = _make_pd_actuator([10.0, 20.0, 30.0, 40.0])
    mapping = wp.array([[0, -1, 1], [2, -1, 3]], dtype=wp.int32, device="cpu")
    view = ActuatorView([(actuator, mapping)])
    values = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device="cpu")
    mask = wp.array([False, True], dtype=wp.bool, device="cpu")

    view.set_actuator_parameter(actuator, "controller", "kp", values, mask=mask)

    np.testing.assert_array_equal(actuator.controller.kp.numpy(), [10.0, 20.0, 31.0, 41.0])


def test_calls_use_only_the_requested_actuator_binding():
    """Selecting one binding must neither read nor write another actuator."""
    first = _make_pd_actuator([10.0, 20.0, 30.0, 40.0])
    second = _make_pd_actuator([50.0, 60.0, 70.0, 80.0])
    first_mapping = wp.array([[0, -1, 1], [2, -1, 3]], dtype=wp.int32, device="cpu")
    second_mapping = wp.array([[-1, 0, -1], [-1, 2, -1]], dtype=wp.int32, device="cpu")
    view = ActuatorView([(first, first_mapping), (second, second_mapping)])

    values = view.get_actuator_parameter(first, "controller", "kp")
    replacement = wp.array([[11.0, 999.0, 21.0], [31.0, 999.0, 41.0]], dtype=wp.float32, device="cpu")
    view.set_actuator_parameter(first, "controller", "kp", replacement)

    np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])
    np.testing.assert_array_equal(second.controller.kp.numpy(), [50.0, 60.0, 70.0, 80.0])


def test_get_actuator_parameter_resolves_delay_and_indexed_clamping():
    """Component paths must expose delay and a selected clamping component."""
    actuator = _make_pd_actuator(
        [10.0, 20.0, 30.0, 40.0],
        delay_steps=[1, 2, 1, 2],
        max_effort=[100.0, 200.0, 300.0, 400.0],
    )
    mapping = wp.array([[0, -1, 1], [2, -1, 3]], dtype=wp.int32, device="cpu")
    view = ActuatorView([(actuator, mapping)])

    delays = view.get_actuator_parameter(actuator, "delay", "delay_steps")
    efforts = view.get_actuator_parameter(actuator, "clamping.0", "max_effort")

    np.testing.assert_array_equal(delays.numpy(), [[1, 0, 2], [1, 0, 2]])
    np.testing.assert_array_equal(efforts.numpy(), [[100.0, 0.0, 200.0], [300.0, 0.0, 400.0]])


@pytest.mark.parametrize("component_name", ["clamping", "clamping.x", "unknown"])
def test_get_actuator_parameter_rejects_malformed_component_paths(component_name: str):
    """Malformed component strings must fail before launching a kernel."""
    actuator = _make_pd_actuator([10.0])
    view = ActuatorView([(actuator, wp.array([[0]], dtype=wp.int32, device="cpu"))])

    with pytest.raises(ValueError):
        view.get_actuator_parameter(actuator, component_name, "kp")


def test_get_actuator_parameter_rejects_missing_delay():
    """Requesting a delay from an actuator without one must fail explicitly."""
    actuator = _make_pd_actuator([10.0])
    view = ActuatorView([(actuator, wp.array([[0]], dtype=wp.int32, device="cpu"))])

    with pytest.raises(ValueError, match="does not have a delay"):
        view.get_actuator_parameter(actuator, "delay", "delay_steps")


def test_get_actuator_parameter_rejects_out_of_range_clamping_index():
    """A clamping index outside the actuator's component list must fail explicitly."""
    actuator = _make_pd_actuator([10.0], max_effort=[100.0])
    view = ActuatorView([(actuator, wp.array([[0]], dtype=wp.int32, device="cpu"))])

    with pytest.raises(IndexError, match="clamping index"):
        view.get_actuator_parameter(actuator, "clamping.1", "max_effort")


def test_get_actuator_parameter_rejects_missing_parameter():
    """A missing parameter name must identify the requested component and parameter."""
    actuator = _make_pd_actuator([10.0])
    view = ActuatorView([(actuator, wp.array([[0]], dtype=wp.int32, device="cpu"))])

    with pytest.raises(AttributeError, match="missing"):
        view.get_actuator_parameter(actuator, "controller", "missing")


class _FakeNewtonArticulationView:
    """Minimal provider for Newton's existing flat actuator mappings."""

    def __init__(self, actuators: list[Actuator], mappings: dict[Actuator, wp.array], world_count: int = 2):
        self.world_count = world_count
        self.model = SimpleNamespace(actuators=actuators)
        self._mappings = mappings

    def _get_actuator_dof_mapping(self, actuator: Actuator) -> wp.array:
        return self._mappings[actuator]


def test_from_articulation_view_uses_model_actuators_by_default():
    """The convenience factory must turn Newton's flat mapping into a working view."""
    actuator = _make_pd_actuator([10.0, 20.0, 30.0, 40.0])
    flat_mapping = wp.array([0, -1, 1, 2, -1, 3], dtype=wp.int32, device="cpu")
    source_view = _FakeNewtonArticulationView([actuator], {actuator: flat_mapping})

    view = ActuatorView.from_articulation_view(source_view)
    values = view.get_actuator_parameter(actuator, "controller", "kp")

    assert view.shape == (2, 3)
    np.testing.assert_array_equal(values.numpy(), [[10.0, 0.0, 20.0], [30.0, 0.0, 40.0]])


def test_from_articulation_view_accepts_an_explicit_actuator_subset():
    """An explicit actuator list must exclude other model actuator mappings."""
    included = _make_pd_actuator([10.0, 20.0, 30.0, 40.0])
    excluded = _make_pd_actuator([50.0, 60.0, 70.0, 80.0])
    mappings = {
        included: wp.array([0, -1, 1, 2, -1, 3], dtype=wp.int32, device="cpu"),
        excluded: wp.array([-1, 0, -1, -1, 2, -1], dtype=wp.int32, device="cpu"),
    }
    source_view = _FakeNewtonArticulationView([included, excluded], mappings)

    view = ActuatorView.from_articulation_view(source_view, [included])

    with pytest.raises(KeyError, match="not bound"):
        view.get_actuator_parameter(excluded, "controller", "kp")


def test_constructor_rejects_empty_and_duplicate_bindings():
    """A view must have an unambiguous mapping for every bound actuator."""
    actuator = _make_pd_actuator([10.0])
    mapping = wp.array([[0]], dtype=wp.int32, device="cpu")

    with pytest.raises(ValueError, match="At least one"):
        ActuatorView([])
    with pytest.raises(ValueError, match="only once"):
        ActuatorView([(actuator, mapping), (actuator, mapping)])


@pytest.mark.parametrize(
    "mapping",
    [
        [[0]],
        wp.array([0], dtype=wp.int32, device="cpu"),
        wp.array([[0]], dtype=wp.uint32, device="cpu"),
    ],
)
def test_constructor_rejects_invalid_mapping_type_rank_or_dtype(mapping):
    """Mappings must use the exact two-dimensional signed-index contract."""
    actuator = _make_pd_actuator([10.0])

    with pytest.raises(ValueError, match="two-dimensional wp.int32"):
        ActuatorView([(actuator, mapping)])


def test_constructor_rejects_inconsistent_mapping_shapes():
    """Every binding must describe the same output view layout."""
    first = _make_pd_actuator([10.0])
    second = _make_pd_actuator([20.0, 30.0])

    with pytest.raises(ValueError, match="Expected mapping shape"):
        ActuatorView(
            [
                (first, wp.array([[0]], dtype=wp.int32, device="cpu")),
                (second, wp.array([[0, 1]], dtype=wp.int32, device="cpu")),
            ]
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (wp.array([[11.0, 12.0]], dtype=wp.float32, device="cpu"), "values shape"),
        (wp.array([[11]], dtype=wp.int32, device="cpu"), "values dtype"),
    ],
)
def test_set_actuator_parameter_rejects_invalid_values(values: wp.array, message: str):
    """Writes must reject arrays that do not match the view or parameter storage."""
    actuator = _make_pd_actuator([10.0])
    view = ActuatorView([(actuator, wp.array([[0]], dtype=wp.int32, device="cpu"))])

    with pytest.raises(ValueError, match=message):
        view.set_actuator_parameter(actuator, "controller", "kp", values)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (wp.array([1], dtype=wp.int32, device="cpu"), "mask dtype"),
        (wp.array([True, False], dtype=wp.bool, device="cpu"), "mask shape"),
    ],
)
def test_set_actuator_parameter_rejects_invalid_masks(mask: wp.array, message: str):
    """World masks must match the view's world axis and Boolean dtype."""
    actuator = _make_pd_actuator([10.0])
    view = ActuatorView([(actuator, wp.array([[0]], dtype=wp.int32, device="cpu"))])
    values = wp.array([[11.0]], dtype=wp.float32, device="cpu")

    with pytest.raises(ValueError, match=message):
        view.set_actuator_parameter(actuator, "controller", "kp", values, mask=mask)


@pytest.mark.parametrize(
    "parameter",
    [
        10.0,
        wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
    ],
)
def test_get_actuator_parameter_rejects_non_array_or_non_flat_parameters(parameter):
    """Only flat per-actuator Warp arrays are valid mapped parameters."""
    actuator = _make_pd_actuator([10.0])
    actuator.controller.invalid = parameter
    view = ActuatorView([(actuator, wp.array([[0]], dtype=wp.int32, device="cpu"))])

    with pytest.raises(ValueError, match="one-dimensional Warp array"):
        view.get_actuator_parameter(actuator, "controller", "invalid")


@pytest.mark.parametrize(
    ("world_count", "mapping", "message"),
    [
        (0, wp.array([], dtype=wp.int32, device="cpu"), "positive world count"),
        (2, wp.array([0, 1, 2], dtype=wp.int32, device="cpu"), "not divisible"),
        (2, wp.array([[0], [1]], dtype=wp.int32, device="cpu"), "one-dimensional"),
    ],
)
def test_from_articulation_view_rejects_invalid_source_layouts(
    world_count: int,
    mapping: wp.array,
    message: str,
):
    """The factory must reject source layouts it cannot reshape unambiguously."""
    actuator = _make_pd_actuator([10.0, 20.0, 30.0])
    source_view = _FakeNewtonArticulationView([actuator], {actuator: mapping}, world_count=world_count)

    with pytest.raises(ValueError, match=message):
        ActuatorView.from_articulation_view(source_view)
