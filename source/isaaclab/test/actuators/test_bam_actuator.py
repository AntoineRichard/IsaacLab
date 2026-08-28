# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the Lab-executed BAM servo actuator.

The math core itself is covered against upstream reference outputs in
``test_bam_model.py``; these tests check what the actuator adds on top of it: the
configuration resolution, the per-environment randomization state, the command delay and
the wiring of the pipeline (which is checked by re-deriving the expected effort from the
core functions rather than by hard-coded numbers).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

import isaaclab.actuators as actuators
from isaaclab.actuators import BAM_XL330_M6_PARAMS_FILE, BamActuatorCfg
from isaaclab.actuators.bam_model import (
    BamMotorParams,
    apply_stiction_clip,
    compute_duty,
    compute_friction_budget,
    compute_motor_torque,
    compute_stribeck_coeff,
)
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.unit

DT = 0.005
"""Physics timestep [s] used by the tests, matching the reference deployment."""

JOINT_NAMES = ["joint_0", "joint_1"]

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

PARAMS = BamMotorParams.from_json(BAM_XL330_M6_PARAMS_FILE)


def make_actuator(num_envs: int = 2, device: str = "cpu", **cfg_kwargs) -> actuators.BamActuator:
    """Build a BAM actuator on ``JOINT_NAMES`` the way the actuator collection would."""
    cfg = BamActuatorCfg(joint_names_expr=[".*"], dt=DT, **cfg_kwargs)
    return cfg.class_type(
        cfg,
        joint_names=JOINT_NAMES,
        joint_ids=slice(None),
        num_envs=num_envs,
        device=device,
        stiffness=0.0,
        damping=0.0,
    )


def step(actuator, target: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Run one actuator step and return the applied effort."""
    action = ArticulationActions(
        joint_positions=target.clone(),
        joint_velocities=torch.zeros_like(target),
        joint_efforts=torch.zeros_like(target),
    )
    out = actuator.compute(action, joint_pos, joint_vel)
    return out.joint_efforts


"""
Configuration resolution.
"""


def test_default_params_file_is_the_vendored_xl330():
    """The default parameter file is the vendored XL330 ``m6`` fit."""
    assert Path(BAM_XL330_M6_PARAMS_FILE) == Path(actuators.__file__).parent / "data" / "bam_xl330_m6.json"
    assert Path(BAM_XL330_M6_PARAMS_FILE).is_file()

    actuator = make_actuator()
    assert actuator.params == PARAMS


@pytest.mark.parametrize("device", DEVICES)
def test_buffers_and_limits_resolve_per_joint(device):
    """Configuration is resolved into ``(num_envs, num_joints)`` buffers."""
    actuator = make_actuator(num_envs=3, device=device, actuator_effort_limit={"joint_0": 1.0, "joint_1": 2.0})

    assert actuator.is_implicit_model is False
    assert actuator.num_joints == 2
    for tensor in (actuator.computed_effort, actuator.applied_effort):
        assert tensor.shape == (3, 2)
        assert tensor.device.type == torch.device(device).type
    for tensor in (actuator.vin, actuator.sag_gain, actuator.friction_scale, actuator.kp_scale, actuator.kd_scale):
        assert tensor.shape == (3, 1)

    expected = torch.tensor([[1.0, 2.0]], device=device).expand(3, 2)
    torch.testing.assert_close(actuator.actuator_effort_limit, expected.clone())


def test_firmware_gain_and_voltage_fallbacks():
    """``kp_fw``/``vin`` default to the deployment values and fall back to the fit."""
    assert BamActuatorCfg(joint_names_expr=[".*"]).kp_fw == 200.0
    assert make_actuator().firmware_kp == 200.0
    # None means "use the value identified in the parameter file".
    assert make_actuator(kp_fw=None).firmware_kp == PARAMS.kp == 400.0

    torch.testing.assert_close(make_actuator().vin, torch.full((2, 1), PARAMS.vin))
    torch.testing.assert_close(make_actuator(vin=7.4).vin, torch.full((2, 1), 7.4))


def test_invalid_delay_configuration_is_rejected():
    """Out-of-range delay settings fail at construction rather than at the first step."""
    with pytest.raises(ValueError, match="min_delay must not be negative"):
        make_actuator(min_delay=-1)
    with pytest.raises(ValueError, match="max_delay"):
        make_actuator(min_delay=3, max_delay=2)
    with pytest.raises(ValueError, match="delay_hold_prob"):
        make_actuator(delay_hold_prob=1.5)


def test_torque_domain_gains_warn_when_set():
    """Configuring the unused PD gains warns instead of silently doing nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_actuator()  # gains left unset: nothing to warn about
    with pytest.warns(UserWarning, match="stiffness and damping is ignored"):
        make_actuator(stiffness=10.0, damping=1.0)


def test_vin_range_overrides_vin_and_is_sampled_per_env():
    """``vin_range`` takes precedence over ``vin`` and is sampled per environment."""
    actuator = make_actuator(num_envs=256, vin=7.4, vin_range=(6.5, 8.2))
    assert actuator.vin.min() >= 6.5 and actuator.vin.max() <= 8.2
    assert actuator.vin.std() > 0.0


def test_startup_samples_are_not_resampled_on_reset():
    """Supply voltage, sag gain and friction scale are startup quantities."""
    actuator = make_actuator(
        num_envs=64, vin_range=(6.5, 8.2), vin_drop_gain_range=(0.0, 0.2), friction_scale_range=(0.9, 1.1)
    )
    before = (actuator.vin.clone(), actuator.sag_gain.clone(), actuator.friction_scale.clone())
    actuator.reset(slice(None))
    for old, new in zip(before, (actuator.vin, actuator.sag_gain, actuator.friction_scale)):
        torch.testing.assert_close(old, new)
    assert actuator.sag_gain.min() >= 0.0 and actuator.sag_gain.max() <= 0.2
    assert actuator.friction_scale.min() >= 0.9 and actuator.friction_scale.max() <= 1.1


"""
Torque pipeline.
"""


@pytest.mark.parametrize("device", DEVICES)
def test_first_step_effort_matches_the_math_core(device):
    """A held pose at zero velocity reproduces the shared core, term by term."""
    actuator = make_actuator(num_envs=2, device=device)
    joint_pos = torch.tensor([[0.35, -0.46], [0.0, 0.1]], device=device)
    joint_vel = torch.zeros_like(joint_pos)
    target = joint_pos + torch.tensor([[0.02, -0.05], [0.1, 0.0]], device=device)

    effort = step(actuator, target, joint_pos, joint_vel)

    kp = torch.full((2, 1), 200.0, device=device)
    vin = torch.full((2, 1), PARAMS.vin, device=device)
    duty = compute_duty(target, joint_pos, joint_vel, kp, vin, PARAMS)
    motor_tau = compute_motor_torque(duty, joint_vel, vin, PARAMS)
    # First step: the previous-effort caches are zero, so the external-torque estimate and
    # the motor-side load of the friction budget are both zero.
    zeros = torch.zeros_like(joint_pos)
    budget = compute_friction_budget(zeros, zeros, compute_stribeck_coeff(joint_vel, PARAMS), PARAMS)
    expected = apply_stiction_clip(
        motor_tau, zeros, joint_vel, budget, PARAMS.friction_viscous, DT, inertia=PARAMS.armature
    )

    torch.testing.assert_close(effort, expected)
    torch.testing.assert_close(actuator.computed_effort, expected)
    torch.testing.assert_close(actuator.applied_effort, expected)


def test_stiction_uses_the_reflected_armature_not_the_unit_default():
    """The stopping torque is sized with the motor armature, not ``inertia=1``.

    With ``inertia=1`` the static window shrinks by ``1 / armature`` (~550x), which would
    make a slowly creeping joint break away instead of being held.
    """
    device = "cpu"
    actuator = make_actuator(num_envs=1, device=device)
    joint_pos = torch.zeros(1, 2, device=device)
    joint_vel = torch.full((1, 2), 0.01, device=device)

    effort = step(actuator, joint_pos.clone(), joint_pos, joint_vel)

    zeros = torch.zeros_like(joint_pos)
    vin = torch.full((1, 1), PARAMS.vin, device=device)
    duty = compute_duty(joint_pos, joint_pos, joint_vel, torch.full((1, 1), 200.0, device=device), vin, PARAMS)
    motor_tau = compute_motor_torque(duty, joint_vel, vin, PARAMS)
    budget = compute_friction_budget(zeros, zeros, compute_stribeck_coeff(joint_vel, PARAMS), PARAMS)
    kwargs = dict(dq=joint_vel, frictionloss=budget, viscous=PARAMS.friction_viscous, dt=DT)

    with_armature = apply_stiction_clip(motor_tau, zeros, inertia=PARAMS.armature, **kwargs)
    with_unit_inertia = apply_stiction_clip(motor_tau, zeros, inertia=1.0, **kwargs)

    torch.testing.assert_close(effort, with_armature)
    assert not torch.allclose(with_armature, with_unit_inertia)


def test_external_torque_estimate_is_the_rotor_momentum_residual():
    """The estimator returns the load that the previous step's effort had to balance."""
    device = "cpu"
    actuator = make_actuator(num_envs=1, device=device)
    joint_pos = torch.zeros(1, 2, device=device)
    joint_vel = torch.zeros(1, 2, device=device)

    # Statically holding a load: the joint does not move, so the estimate is the negated
    # effort applied on the previous step.
    effort = step(actuator, torch.tensor([[0.1, -0.1]], device=device), joint_pos, joint_vel)
    torch.testing.assert_close(actuator._estimate_external_torque(joint_vel), -effort)

    # Accelerating: the rotor inertia times the measured acceleration is added on top.
    next_vel = torch.tensor([[0.5, -0.25]], device=device)
    expected = PARAMS.armature * (next_vel - joint_vel) / DT - effort
    torch.testing.assert_close(actuator._estimate_external_torque(next_vel), expected)


def test_second_step_effort_matches_the_math_core_under_load():
    """A statically held load pins the whole pipeline, external-torque sign included.

    On the second step the joint has not moved but the previous one applied a torque, so the
    external-torque estimate is a non-zero ``-tau_prev``: the load the actuator had to
    balance. It enters both the friction budget and the stopping torque of the stiction clip,
    so the expected effort below only matches if the estimate is used with that sign.
    """
    device = "cpu"
    actuator = make_actuator(num_envs=1, device=device)
    joint_pos = torch.zeros(1, 2, device=device)
    joint_vel = torch.zeros(1, 2, device=device)
    target = torch.tensor([[0.1, -0.1]], device=device)

    kp = torch.full((1, 1), 200.0, device=device)
    vin = torch.full((1, 1), PARAMS.vin, device=device)
    stribeck = compute_stribeck_coeff(joint_vel, PARAMS)
    zeros = torch.zeros_like(joint_pos)
    clip_kwargs = dict(dq=joint_vel, viscous=PARAMS.friction_viscous, dt=DT, inertia=PARAMS.armature)

    # Step 1 runs unloaded: both previous-step caches are zero.
    duty = compute_duty(target, joint_pos, joint_vel, kp, vin, PARAMS)
    motor_tau = compute_motor_torque(duty, joint_vel, vin, PARAMS)
    budget = compute_friction_budget(zeros, zeros, stribeck, PARAMS)
    first = apply_stiction_clip(motor_tau, zeros, frictionloss=budget, **clip_kwargs)
    torch.testing.assert_close(step(actuator, target, joint_pos, joint_vel), first)
    assert first.abs().min() > 0.0, "the first step must apply a torque for the load estimate to be non-zero"

    # Step 2: the joint did not move, so the estimate is exactly the negated applied effort.
    external = -first
    budget = compute_friction_budget(motor_tau, external, stribeck, PARAMS)
    expected = apply_stiction_clip(motor_tau, external, frictionloss=budget, **clip_kwargs)
    torch.testing.assert_close(step(actuator, target, joint_pos, joint_vel), expected)


def test_first_step_after_a_reset_sees_no_acceleration():
    """A joint reset while moving must not produce an acceleration spike.

    The velocity cache is seeded with the first velocity observed after a reset, so the
    external-torque estimate falls back to its quasi-static form instead of reading the
    whole joint velocity as a one-step acceleration.
    """
    device = "cpu"
    actuator = make_actuator(num_envs=1, device=device)
    joint_pos = torch.zeros(1, 2, device=device)
    fast = torch.full((1, 2), 3.0, device=device)
    target = torch.tensor([[0.3, -0.3]], device=device)

    step(actuator, target, joint_pos, torch.zeros_like(joint_pos))
    actuator.reset(slice(None))
    effort = step(actuator, target, joint_pos, fast)

    vin = torch.full((1, 1), PARAMS.vin, device=device)
    duty = compute_duty(target, joint_pos, fast, torch.full((1, 1), 200.0, device=device), vin, PARAMS)
    motor_tau = compute_motor_torque(duty, fast, vin, PARAMS)
    zeros = torch.zeros_like(joint_pos)
    budget = compute_friction_budget(zeros, zeros, compute_stribeck_coeff(fast, PARAMS), PARAMS)
    expected = apply_stiction_clip(motor_tau, zeros, fast, budget, PARAMS.friction_viscous, DT, PARAMS.armature)
    torch.testing.assert_close(effort, expected)


def test_battery_sag_lowers_the_supply_after_a_loaded_step():
    """The supply voltage sags with the previous step's summed motor torque."""
    device = "cpu"
    actuator = make_actuator(num_envs=1, device=device, vin_drop_gain_range=(0.2, 0.2), vin_min=6.0)
    joint_pos = torch.zeros(1, 2, device=device)
    joint_vel = torch.zeros_like(joint_pos)
    target = torch.tensor([[1.0, 1.0]], device=device)

    step(actuator, target, joint_pos, joint_vel)
    # The first step runs unloaded, so the supply is still nominal.
    torch.testing.assert_close(actuator.effective_vin, actuator.vin)

    vin = torch.full((1, 1), PARAMS.vin, device=device)
    kp = torch.full((1, 1), 200.0, device=device)
    motor_tau = compute_motor_torque(
        compute_duty(target, joint_pos, joint_vel, kp, vin, PARAMS), joint_vel, vin, PARAMS
    )
    expected_vin = torch.clamp(vin - 0.2 * motor_tau.abs().sum(dim=-1, keepdim=True), min=6.0)
    assert expected_vin.item() < PARAMS.vin

    step(actuator, target, joint_pos, joint_vel)
    torch.testing.assert_close(actuator.effective_vin, expected_vin)


def test_reset_clears_the_previous_step_caches():
    """After a reset the actuator behaves as it did on the very first step."""
    device = "cpu"
    actuator = make_actuator(num_envs=2, device=device, vin_drop_gain_range=(0.2, 0.2))
    joint_pos = torch.zeros(2, 2, device=device)
    target = torch.tensor([[0.5, -0.5], [0.2, 0.2]], device=device)

    first = step(actuator, target, joint_pos, torch.zeros_like(joint_pos))
    loaded = step(actuator, target, joint_pos, torch.full_like(joint_pos, 0.3))
    assert not torch.allclose(loaded, first)

    actuator.reset(slice(None))
    torch.testing.assert_close(step(actuator, target, joint_pos, torch.zeros_like(joint_pos)), first)


def test_feed_forward_efforts_are_ignored():
    """The modelled firmware only accepts a position setpoint."""
    actuator = make_actuator(num_envs=1)
    joint_pos = torch.zeros(1, 2)
    target = torch.tensor([[0.1, 0.1]])
    baseline = step(actuator, target, joint_pos, torch.zeros_like(joint_pos))

    actuator.reset(slice(None))
    action = ArticulationActions(
        joint_positions=target.clone(),
        joint_velocities=torch.zeros_like(target),
        joint_efforts=torch.full_like(target, 5.0),
    )
    torch.testing.assert_close(actuator.compute(action, joint_pos, torch.zeros_like(joint_pos)).joint_efforts, baseline)


"""
Domain-randomization hooks.
"""


def test_dr_hooks_only_touch_the_given_envs():
    """``set_*``/``reset_*`` write only the selected environments."""
    actuator = make_actuator(num_envs=4, friction_scale_range=(0.5, 0.5))
    env_ids = torch.tensor([0, 2])

    actuator.set_friction_scale(env_ids, torch.tensor([[1.5], [2.5]]))
    torch.testing.assert_close(actuator.friction_scale.squeeze(-1), torch.tensor([1.5, 0.5, 2.5, 0.5]))
    actuator.reset_friction_scale(env_ids)
    torch.testing.assert_close(actuator.friction_scale.squeeze(-1), torch.full((4,), 0.5))

    actuator.set_gains(env_ids, kp_scale=torch.tensor([[1.2], [0.8]]), kd_scale=torch.tensor([[0.5], [0.5]]))
    torch.testing.assert_close(actuator.kp_scale.squeeze(-1), torch.tensor([1.2, 1.0, 0.8, 1.0]))
    torch.testing.assert_close(actuator.kd_scale.squeeze(-1), torch.tensor([0.5, 1.0, 0.5, 1.0]))
    actuator.reset_gains(env_ids)
    torch.testing.assert_close(actuator.kp_scale.squeeze(-1), torch.ones(4))
    torch.testing.assert_close(actuator.kd_scale.squeeze(-1), torch.ones(4))


def test_friction_scale_multiplies_the_budget():
    """A larger friction scale opposes motion more strongly."""
    joint_pos = torch.zeros(1, 2)
    joint_vel = torch.full((1, 2), 0.5)
    target = torch.zeros(1, 2)

    nominal = step(make_actuator(num_envs=1), target, joint_pos, joint_vel)
    scaled = make_actuator(num_envs=1, friction_scale_range=(2.0, 2.0))
    doubled = step(scaled, target, joint_pos, joint_vel)
    # The joint is moving positive with a zero-error command, so friction is negative and
    # doubling the budget doubles it.
    assert (doubled < nominal).all()


def test_gain_scales_enter_the_control_law():
    """``kp_scale`` scales the firmware gain and ``kd_scale`` the velocity it sees."""
    device = "cpu"
    actuator = make_actuator(num_envs=1, device=device)
    joint_pos = torch.zeros(1, 2, device=device)
    joint_vel = torch.full((1, 2), 0.4, device=device)
    target = torch.tensor([[0.1, 0.2]], device=device)

    actuator.set_gains(slice(None), kp_scale=torch.tensor([[1.3]]), kd_scale=torch.tensor([[0.7]]))
    effort = step(actuator, target, joint_pos, joint_vel)

    kp = torch.full((1, 1), 200.0 * 1.3, device=device)
    vin = torch.full((1, 1), PARAMS.vin, device=device)
    scaled_vel = joint_vel * 0.7
    duty = compute_duty(target, joint_pos, scaled_vel, kp, vin, PARAMS)
    motor_tau = compute_motor_torque(duty, scaled_vel, vin, PARAMS)
    zeros = torch.zeros_like(joint_pos)
    budget = compute_friction_budget(zeros, zeros, compute_stribeck_coeff(joint_vel, PARAMS), PARAMS)
    expected = apply_stiction_clip(
        motor_tau, zeros, joint_vel, budget, PARAMS.friction_viscous, DT, inertia=PARAMS.armature
    )
    torch.testing.assert_close(effort, expected)


"""
Command delay.
"""


def test_constant_lag_returns_the_target_from_k_steps_ago():
    """With ``min_delay == max_delay == k`` the command is the one from ``k`` steps ago."""
    lag = 2
    delayed = make_actuator(num_envs=1, min_delay=lag, max_delay=lag)
    reference = make_actuator(num_envs=1)
    joint_pos = torch.zeros(1, 2)
    joint_vel = torch.zeros(1, 2)
    targets = [torch.full((1, 2), 0.1 * (index + 1)) for index in range(6)]

    for index, target in enumerate(targets):
        # Before the ring has filled, the freshest available command is returned.
        expected_target = targets[max(0, index - lag)]
        torch.testing.assert_close(
            step(delayed, target, joint_pos, joint_vel),
            step(reference, expected_target, joint_pos, joint_vel),
        )


def test_hold_prob_one_freezes_the_lag():
    """``delay_hold_prob = 1`` never resamples, so the lag stays at its post-reset zero."""
    actuator = make_actuator(num_envs=4, min_delay=1, max_delay=3, delay_hold_prob=1.0)
    reference = make_actuator(num_envs=4)
    joint_pos = torch.zeros(4, 2)
    joint_vel = torch.zeros(4, 2)

    for index in range(5):
        target = torch.full((4, 2), 0.1 * (index + 1))
        torch.testing.assert_close(
            step(actuator, target, joint_pos, joint_vel), step(reference, target, joint_pos, joint_vel)
        )
        assert int(actuator.delay_time_lags.max()) == 0


def test_lag_is_resampled_only_on_the_per_env_update_period():
    """Lags change only on an environment's own ``delay_update_period`` phase."""
    period = 4
    actuator = make_actuator(num_envs=64, min_delay=0, max_delay=3, delay_update_period=period)
    joint_pos = torch.zeros(64, 2)
    joint_vel = torch.zeros(64, 2)

    lags = []
    for index in range(24):
        step(actuator, torch.full((64, 2), 0.1 * index), joint_pos, joint_vel)
        lags.append(actuator.delay_time_lags.clone())
    history = torch.stack(lags)
    assert history.min() >= 0 and history.max() <= 3

    phases = set()
    for env in range(64):
        changed = (history[1:, env] != history[:-1, env]).nonzero().flatten() + 1
        residues = {int(index) % period for index in changed}
        assert len(residues) <= 1, f"env {env} changed its lag on more than one phase: {residues}"
        phases |= residues
    # Per-env phase offsets stagger the refresh across environments.
    assert len(phases) > 1


def test_reset_clears_the_delay_state():
    """A reset drops the buffered commands and the lag of the reset environments."""
    actuator = make_actuator(num_envs=2, min_delay=2, max_delay=2)
    joint_pos = torch.zeros(2, 2)
    joint_vel = torch.zeros(2, 2)
    for index in range(4):
        step(actuator, torch.full((2, 2), 0.1 * (index + 1)), joint_pos, joint_vel)

    actuator.reset([0])
    assert int(actuator.delay_time_lags[0]) == 0

    fresh = torch.full((2, 2), 9.0)
    reference = make_actuator(num_envs=2)
    effort = step(actuator, fresh, joint_pos, joint_vel)
    # Env 0 restarted, so it sees the fresh command; env 1 still lags by two steps.
    torch.testing.assert_close(effort[0], step(reference, fresh, joint_pos, joint_vel)[0])
    assert not torch.allclose(effort[1], effort[0])


@pytest.mark.parametrize("env_ids", [[0], slice(0, 1)])
def test_partial_reset_keeps_the_other_envs_delayed_history(env_ids):
    """Resetting one environment must not clear the buffered commands of the others.

    The delayed command of the untouched environment is what proves its ring survived: a
    slice selection that leaked through as "all environments" would clear its history and
    hand it the fresh command instead of the one from two steps ago.
    """
    lag = 2
    actuator = make_actuator(num_envs=2, min_delay=lag, max_delay=lag)
    untouched = make_actuator(num_envs=2, min_delay=lag, max_delay=lag)
    joint_pos = torch.zeros(2, 2)
    joint_vel = torch.zeros(2, 2)
    fresh = torch.full((2, 2), 9.0)
    for index in range(4):
        target = torch.full((2, 2), 0.1 * (index + 1))
        step(actuator, target, joint_pos, joint_vel)
        step(untouched, target, joint_pos, joint_vel)

    actuator.reset(env_ids)
    effort = step(actuator, fresh, joint_pos, joint_vel)
    expected = step(untouched, fresh, joint_pos, joint_vel)

    # Env 1 was not reset, so it behaves exactly like the actuator that was never reset.
    torch.testing.assert_close(effort[1], expected[1])
    # ... while env 0 restarted and now runs on the fresh command.
    assert not torch.allclose(effort[0], expected[0])
