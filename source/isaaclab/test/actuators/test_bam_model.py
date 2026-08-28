# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the BAM voltage-domain servo math core.

The reference outputs in ``data/bam_xl330_m6_goldens.npz`` come from the upstream
`BAM <https://github.com/Rhoban/bam>`_ package (``xl330``/``m6`` at commit
``62bd8ce``); see ``scripts/tools/generate_bam_goldens.py``. The port under test runs in
``float32`` while the fixture is ``float64``, so the comparisons use a ``float32``-sized
relative tolerance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import isaaclab.actuators as actuators
from isaaclab.actuators import (
    BamMotorParams,
    apply_stiction_clip,
    battery_sag,
    compute_duty,
    compute_friction_budget,
    compute_motor_torque,
    compute_stribeck_coeff,
)

pytestmark = pytest.mark.unit

GOLDEN_FILE = Path(__file__).parent / "data" / "bam_xl330_m6_goldens.npz"
"""Reference outputs generated from the upstream BAM package."""

PARAMS_FILE = Path(actuators.__file__).parent / "data" / "bam_xl330_m6.json"
"""Vendored copy of the BAM ``xl330``/``m6`` parameters shipped with Isaac Lab."""

# float32 round-off of the ported pipeline against the float64 fixture. The absolute term
# is what carries the comparison near zero crossings of the DC-motor equation, where
# ``kt * V / R - kt^2 * dq / R`` cancels two terms of order 1 and leaves ~1e-7 of float32
# noise; a parameter error large enough to matter (1e-4 relative) still fails by ~600x.
RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture(scope="module")
def goldens() -> dict[str, np.ndarray]:
    """Golden inputs, reference outputs and reference scalars."""
    with np.load(GOLDEN_FILE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def params() -> BamMotorParams:
    """Motor parameters loaded from the vendored JSON."""
    return BamMotorParams.from_json(PARAMS_FILE)


def _row(array: np.ndarray) -> torch.Tensor:
    """Shape a golden sample vector as a ``(1, num_samples)`` float32 tensor.

    The math core is written for ``(num_envs, num_joints)`` tensors, so the 1024 golden
    samples are laid out as a single environment with 1024 independent joints.
    """
    return torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)


def _scalar(goldens: dict[str, np.ndarray], key: str) -> torch.Tensor:
    """Shape a golden scalar as the ``(num_envs, 1)`` per-environment tensor the API takes."""
    return torch.full((1, 1), float(goldens[f"attr_{key}"]), dtype=torch.float32)


def test_vendored_params_match_reference(params: BamMotorParams, goldens: dict[str, np.ndarray]) -> None:
    """The vendored JSON carries the same constants the goldens were generated with."""
    for name in (
        "kt",
        "R",
        "armature",
        "error_gain",
        "max_pwm",
        "max_current",
        "friction_base",
        "friction_viscous",
        "friction_stribeck",
        "dtheta_stribeck",
        "alpha",
        "load_friction_motor",
        "load_friction_external",
        "load_friction_motor_stribeck",
        "load_friction_external_stribeck",
        "load_friction_motor_quad",
        "load_friction_external_quad",
    ):
        assert getattr(params, name) == float(goldens[f"attr_{name}"]), name
    # m6 = Stribeck + directional load friction + quadratic load coupling.
    assert (params.stribeck, params.load_dependent, params.directional, params.quadratic) == (True, True, True, True)


def test_compute_duty_matches_reference(params: BamMotorParams, goldens: dict[str, np.ndarray]) -> None:
    """Firmware P-law, current limiter and PWM clamp reproduce the reference duty cycle."""
    duty = compute_duty(
        _row(goldens["q_target"]),
        _row(goldens["q"]),
        _row(goldens["dq"]),
        _scalar(goldens, "kp"),
        _scalar(goldens, "vin"),
        params,
    )
    torch.testing.assert_close(duty, _row(goldens["duty"]), rtol=RTOL, atol=ATOL)


def test_compute_motor_torque_matches_reference(params: BamMotorParams, goldens: dict[str, np.ndarray]) -> None:
    """The DC-motor equation with back-EMF reproduces the reference torque."""
    torque = compute_motor_torque(
        _row(goldens["duty"]),
        _row(goldens["dq"]),
        _scalar(goldens, "vin"),
        params,
    )
    torch.testing.assert_close(torque, _row(goldens["motor_torque"]), rtol=RTOL, atol=ATOL)


def test_compute_stribeck_coeff_matches_reference(params: BamMotorParams, goldens: dict[str, np.ndarray]) -> None:
    """The Stribeck coefficient reproduces the reference decay curve."""
    coeff = compute_stribeck_coeff(_row(goldens["dq"]), params)
    torch.testing.assert_close(coeff, _row(goldens["stribeck_coeff"]), rtol=RTOL, atol=ATOL)


def test_compute_friction_budget_matches_reference(params: BamMotorParams, goldens: dict[str, np.ndarray]) -> None:
    """The m6 friction budget reproduces the reference ``dof_frictionloss`` values."""
    budget = compute_friction_budget(
        _row(goldens["prev_tau"]),
        _row(goldens["ext_tau"]),
        _row(goldens["stribeck_coeff"]),
        params,
    )
    torch.testing.assert_close(budget, _row(goldens["frictionloss_budget"]), rtol=RTOL, atol=ATOL)


def test_compute_friction_budget_applies_scale(params: BamMotorParams, goldens: dict[str, np.ndarray]) -> None:
    """``friction_scale`` scales the whole budget, so it can randomize friction per joint."""
    scale = torch.full((1, int(goldens["attr_num_samples"])), 2.5)
    budget = compute_friction_budget(
        _row(goldens["prev_tau"]),
        _row(goldens["ext_tau"]),
        _row(goldens["stribeck_coeff"]),
        params,
        scale,
    )
    torch.testing.assert_close(budget, 2.5 * _row(goldens["frictionloss_budget"]), rtol=RTOL, atol=ATOL)


def test_compute_duty_saturates_at_max_pwm(params: BamMotorParams) -> None:
    """Without the current limiter, a large position error saturates the PWM duty cycle."""
    unlimited = replace(params, max_current=None)
    error = torch.tensor([[1.0e3, -1.0e3]])
    duty = compute_duty(
        error,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.full((1, 1), 200.0),
        torch.full((1, 1), 7.4),
        unlimited,
    )
    torch.testing.assert_close(duty, torch.tensor([[params.max_pwm, -params.max_pwm]]))


def test_compute_duty_follows_current_limiter_window(params: BamMotorParams) -> None:
    """At high back-EMF the duty is pinned to the current-limiter window, not to the P-law."""
    vin, dq = 7.4, 20.0
    # Window centre (back-EMF feed-forward) and half-width (R * I_max) in duty units.
    center = params.kt * dq / vin
    span = params.R * params.max_current / vin
    duty = compute_duty(
        torch.tensor([[1.0e3, -1.0e3]]),
        torch.zeros(1, 2),
        torch.full((1, 2), dq),
        torch.full((1, 1), 200.0),
        torch.full((1, 1), vin),
        params,
    )
    # Upper edge of the window is above max_pwm, so the PWM clamp wins there; the lower
    # edge sits inside the physical range and the limiter binds.
    expected = torch.tensor([[min(center + span, params.max_pwm), center - span]], dtype=torch.float32)
    torch.testing.assert_close(duty, expected, rtol=RTOL, atol=ATOL)
    # The bound sample draws exactly the firmware current limit.
    torque = compute_motor_torque(duty, torch.full((1, 2), dq), torch.full((1, 1), vin), params)
    assert torque[0, 1].item() == pytest.approx(-params.kt * params.max_current, rel=1e-5)


def test_battery_sag_couples_joints_of_one_actuator_group() -> None:
    """The supply drop is driven by the summed load of the whole bus and clamped from below."""
    vin = torch.full((2, 1), 7.4)
    prev_tau = torch.tensor([[0.2, -0.4, 0.1], [2.0, 2.0, 2.0]])
    sagged = battery_sag(vin, prev_tau, sag_gain=torch.full((2, 1), 2.0), vin_min=6.0)
    assert sagged.shape == (2, 1)
    # env 0: 7.4 - 2.0 * 0.7 = 6.0 ; env 1: 7.4 - 2.0 * 6.0 < 6.0 -> floored.
    torch.testing.assert_close(sagged, torch.tensor([[6.0], [6.0]]), rtol=RTOL, atol=ATOL)


def test_battery_sag_without_lower_bound() -> None:
    """``vin_min=None`` leaves the drop unclamped."""
    sagged = battery_sag(torch.full((1, 1), 7.4), torch.tensor([[1.0, 1.0]]), sag_gain=1.0, vin_min=None)
    torch.testing.assert_close(sagged, torch.tensor([[5.4]]), rtol=RTOL, atol=ATOL)


def test_apply_stiction_clip_holds_static_joint() -> None:
    """A joint at rest whose net torque fits in the budget is held: the net torque cancels."""
    dt, frictionloss, viscous = 0.005, 0.005, 0.004
    effort = apply_stiction_clip(
        motor_tau=torch.tensor([[0.001, 0.0]]),
        ext_tau=torch.tensor([[0.0, 0.003]]),
        dq=torch.zeros(1, 2),
        frictionloss=torch.full((1, 2), frictionloss),
        viscous=viscous,
        dt=dt,
    )
    # The returned effort excludes the external load, so cancelling the net torque means
    # applying exactly the opposite of it.
    torch.testing.assert_close(effort, torch.tensor([[0.0, -0.003]]), rtol=RTOL, atol=ATOL)


def test_apply_stiction_clip_breaks_away_and_opposes_motion() -> None:
    """Above the budget (or while moving) friction saturates and opposes the motion."""
    dt, frictionloss, viscous = 0.005, 0.005, 0.004
    effort = apply_stiction_clip(
        motor_tau=torch.tensor([[0.02, 0.02]]),
        ext_tau=torch.zeros(1, 2),
        dq=torch.tensor([[0.0, 1.0]]),
        frictionloss=torch.full((1, 2), frictionloss),
        viscous=viscous,
        dt=dt,
    )
    # At rest: only the Coulomb budget is subtracted. Moving: the viscous term adds to it.
    expected = torch.tensor([[0.02 - frictionloss, 0.02 - (frictionloss + viscous * 1.0)]])
    torch.testing.assert_close(effort, expected, rtol=RTOL, atol=ATOL)
