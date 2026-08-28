# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lab-executed BAM servo actuator.

The model is the PyTorch port of `BAM (Better Actuator Models)
<https://github.com/Rhoban/bam>`_ at commit ``62bd8ce`` of its ``mjlab_frictionloss``
branch. :mod:`isaaclab.actuators.bam_model` holds the stateless math; this module owns the
per-environment state around it (supply voltage, gain and friction randomization, the
command delay and the previous-step caches) and drives the whole pipeline from Isaac Lab,
returning a pure effort command.

External-torque estimate
------------------------

BAM's gearbox friction is *load dependent*: the budget grows with the torque flowing
through the gears, which the model splits into a motor-side and an external-side
contribution. The reference implementation reads the external side straight out of MuJoCo
(``-qfrc_bias + qfrc_constraint``, minus its own friction constraint). Isaac Lab's actuator
interface exposes only the joint positions and velocities, so this class estimates it from
the rotor's momentum balance over the previous step:

.. math::

    \\tau_{ext} \\approx I_{a} \\frac{\\dot{q} - \\dot{q}_{prev}}{\\Delta t} - \\tau_{applied, prev}

where :math:`I_a` is the reflected rotor inertia (``params.armature``). This is exactly the
reaction torque the output shaft sees in a free-body diagram of the rotor alone: everything
past the shaft -- the link inertia, gravity, contacts -- counts as external load, which is
the decomposition BAM's gearbox model asks for. It differs from the reference quantity by
the link-side inertial term :math:`I_{link} \\ddot{q}`, which no actuator-local signal can
supply; the estimate is therefore exact for a statically loaded joint and degrades as the
link accelerates. :meth:`BamActuator._estimate_external_torque` is the single place this
approximation lives, so a backend that can read the true generalized forces overrides just
that method.

Command delay
-------------

The delay reproduces the reference semantics (see ``artifacts/microduck/
upstream_reference.md`` section 8, citing ``mjlab/utils/buffers/delay_buffer.py``): the lag
is drawn inclusively from ``[min_delay, max_delay]``, a draw is only attempted every
``delay_update_period`` steps on a per-environment phase, ``delay_hold_prob`` may keep the
previous lag anyway, and a reset clears the ring, the lag and the step counter of the reset
environments. :class:`~isaaclab.utils.buffers.DelayBuffer` resamples per-environment lags
only on reset and has no notion of an update period or a phase, so the lag policy is owned
here and only the ring buffer is reused.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.physics import PhysicsManager
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase
from .bam_model import (
    BamMotorParams,
    apply_stiction_clip,
    battery_sag,
    compute_duty,
    compute_friction_budget,
    compute_motor_torque,
    compute_stribeck_coeff,
)

if TYPE_CHECKING:
    from .actuator_bam_cfg import BamActuatorCfg


class BamActuator(ActuatorBase):
    """BAM voltage-domain servo model, executed by Isaac Lab.

    The pipeline of one step is: delay the position command, sag the supply voltage with the
    previous step's load, run the firmware proportional controller to a PWM duty cycle,
    convert that to a motor torque through the DC-motor equation, size the gearbox friction
    budget from the previous motor torque and the estimated external load, and finally clip
    the torque with that budget so a joint whose net torque fits inside it is held static.

    The model consumes only the position target of the control action; the modelled firmware
    has no torque input, so feed-forward efforts and velocity targets are ignored.

    Per-environment randomization is exposed as public tensors that events write into:
    :attr:`vin` and :attr:`sag_gain` are startup quantities (a robot keeps its battery across
    episodes), while :attr:`friction_scale`, :attr:`kp_scale` and :attr:`kd_scale` are meant
    to be re-drawn per episode through :meth:`set_friction_scale` and :meth:`set_gains`.
    """

    cfg: BamActuatorCfg
    """The configuration for the actuator model."""

    params: BamMotorParams
    """Identified motor, firmware and friction parameters of the modelled servo."""

    vin: torch.Tensor
    """Nominal per-environment supply voltage [V]. Shape is (num_envs, 1)."""

    sag_gain: torch.Tensor
    """Per-environment supply sag gain [V/(N.m)]. Shape is (num_envs, 1)."""

    effective_vin: torch.Tensor
    """Supply voltage after the sag of the last :meth:`compute` [V]. Shape is (num_envs, 1)."""

    friction_scale: torch.Tensor
    """Per-environment multiplier of the friction budget [-]. Shape is (num_envs, 1)."""

    kp_scale: torch.Tensor
    """Per-environment multiplier of the firmware gain [-]. Shape is (num_envs, 1)."""

    kd_scale: torch.Tensor
    """Per-environment multiplier of the velocity the motor model sees [-]. Shape is (num_envs, 1)."""

    def __init__(
        self,
        cfg: BamActuatorCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        actuator_effort_limit: torch.Tensor | float | None = None,
        actuator_velocity_limit: torch.Tensor | float | None = None,
    ):
        """Initialize the actuator.

        Args:
            cfg: The configuration of the actuator model.
            joint_names: The joint names in the articulation.
            joint_ids: The joint indices in the articulation.
            num_envs: Number of articulations in the view.
            device: Device used for processing.
            stiffness: Unused. The position loop runs in the firmware domain, parameterized
                by :attr:`~isaaclab.actuators.BamActuatorCfg.kp_fw`.
            damping: Unused. The model's damping is the motor's back-EMF.
            actuator_effort_limit: Effort clipping limit of the model [N.m]. It is a safety
                clamp on top of the electrical limit the duty-cycle model already enforces.
            actuator_velocity_limit: Actuator velocity limit [rad/s].
        """
        del stiffness, damping
        super().__init__(
            cfg, joint_names, joint_ids, num_envs, device, actuator_effort_limit, actuator_velocity_limit
        )

        self.params = BamMotorParams.from_json(cfg.params_file)
        self._firmware_kp = float(cfg.kp_fw) if cfg.kp_fw is not None else self.params.kp
        self._dt = float(cfg.dt) if cfg.dt is not None else PhysicsManager.get_physics_dt()

        if cfg.max_delay < cfg.min_delay:
            raise ValueError(f"BamActuatorCfg.max_delay ({cfg.max_delay}) must not be below min_delay.")
        if not 0.0 <= cfg.delay_hold_prob <= 1.0:
            raise ValueError(f"BamActuatorCfg.delay_hold_prob must lie in [0, 1]. Received: {cfg.delay_hold_prob}.")

        # startup-sampled per-environment quantities, held constant across resets.
        self.vin = self._sample_per_env(cfg.vin_range, cfg.vin if cfg.vin is not None else self.params.vin)
        self.sag_gain = self._sample_per_env(cfg.vin_drop_gain_range, 0.0)
        self.effective_vin = self.vin.clone()
        self.friction_scale = self._sample_per_env(cfg.friction_scale_range, 1.0)
        self._default_friction_scale = self.friction_scale.clone()

        # per-episode gain randomization.
        self.kp_scale = torch.ones(self._num_envs, 1, device=self._device)
        self.kd_scale = torch.ones_like(self.kp_scale)
        self._default_kp_scale = self.kp_scale.clone()
        self._default_kd_scale = self.kd_scale.clone()

        # previous-step caches. The friction budget and the supply sag consume the motor
        # torque, the external-torque estimate the effort that was actually applied.
        self._prev_motor_effort = torch.zeros_like(self.computed_effort)
        self._prev_applied_effort = torch.zeros_like(self.computed_effort)
        self._prev_joint_vel = torch.zeros_like(self.computed_effort)
        # A reset leaves no previous velocity to differentiate against. Seeding it with the
        # first observed one on the next step reports zero acceleration instead of the spike
        # a zeroed cache would produce for a joint that is reset while moving.
        self._needs_velocity_seed = torch.ones(self._num_envs, 1, dtype=torch.bool, device=self._device)

        # command-delay state; see the module docstring for the semantics.
        self._delay_buffer: CircularBuffer | None = None
        if cfg.max_delay > 0:
            self._delay_buffer = CircularBuffer(cfg.max_delay + 1, self._num_envs, self._device)
        self._delay_lags = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._delay_step_count = torch.zeros_like(self._delay_lags)
        self._delay_phase = self._sample_delay_phase()

    """
    Properties.
    """

    @property
    def firmware_kp(self) -> float:
        """Firmware proportional gain the control law runs at, before :attr:`kp_scale` [-]."""
        return self._firmware_kp

    @property
    def delay_time_lags(self) -> torch.Tensor:
        """Command lag currently applied to each environment [physics steps]. Shape is (num_envs,)."""
        return self._delay_lags

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        """Clear the previous-step caches and the delay state of the given environments.

        The startup-sampled quantities (:attr:`vin`, :attr:`sag_gain` and the default of
        :attr:`friction_scale`) are deliberately left untouched, matching the reference
        implementation: they describe the hardware, not the episode.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        self._prev_motor_effort[env_ids] = 0.0
        self._prev_applied_effort[env_ids] = 0.0
        self._prev_joint_vel[env_ids] = 0.0
        self._needs_velocity_seed[env_ids] = True
        self._delay_lags[env_ids] = 0
        self._delay_step_count[env_ids] = 0
        if self.cfg.delay_update_period > 0:
            self._delay_phase[env_ids] = self._sample_delay_phase()[env_ids]
        if self._delay_buffer is not None:
            self._delay_buffer.reset(None if isinstance(env_ids, slice) else env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Run the BAM pipeline and return an effort-only control action.

        Args:
            control_action: Desired joint positions [rad], velocities and feed-forward
                efforts. Only the positions are used.
            joint_pos: Current joint positions [rad], shape ``(num_envs, num_joints)``.
            joint_vel: Current joint velocities [rad/s], shape ``(num_envs, num_joints)``.

        Returns:
            The joint efforts to apply [N.m], with the position and velocity targets cleared.
        """
        params = self.params
        target = self._apply_delay(control_action.joint_positions)

        # freshly reset environments start with zero measured acceleration.
        torch.where(self._needs_velocity_seed, joint_vel, self._prev_joint_vel, out=self._prev_joint_vel)
        self._needs_velocity_seed.fill_(False)

        # The joints of one group share a supply, so its sag is driven by their total load.
        self.effective_vin = battery_sag(self.vin, self._prev_motor_effort, self.sag_gain, self.cfg.vin_min)
        # kd_scale scales the electrical damping only: the velocity enters the control law
        # and the torque equation solely through the back-EMF term.
        scaled_vel = joint_vel * self.kd_scale

        duty = compute_duty(
            target, joint_pos, scaled_vel, self._firmware_kp * self.kp_scale, self.effective_vin, params
        )
        motor_effort = compute_motor_torque(duty, scaled_vel, self.effective_vin, params)

        external_effort = self._estimate_external_torque(joint_vel)
        budget = compute_friction_budget(
            self._prev_motor_effort,
            external_effort,
            compute_stribeck_coeff(joint_vel, params),
            params,
            self.friction_scale,
        )
        self.computed_effort = apply_stiction_clip(
            motor_effort,
            external_effort,
            joint_vel,
            budget,
            params.friction_viscous,
            self._dt,
            inertia=params.armature,
        )
        self.applied_effort = self._clip_effort(self.computed_effort)

        # copied rather than aliased: the caches must survive a caller mutating the returned
        # efforts, and a reset must not write through into the effort telemetry.
        self._prev_motor_effort.copy_(motor_effort)
        self._prev_applied_effort.copy_(self.applied_effort)
        self._prev_joint_vel.copy_(joint_vel)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    """
    Domain-randomization hooks.
    """

    def set_friction_scale(self, env_ids: Sequence[int] | slice | torch.Tensor, friction_scale: torch.Tensor):
        """Set the friction-budget scale of the given environments.

        Args:
            env_ids: Environments to write.
            friction_scale: Scale of the whole velocity-independent friction budget [-],
                broadcastable to ``(len(env_ids), 1)``.
        """
        self.friction_scale[env_ids] = friction_scale

    def reset_friction_scale(self, env_ids: Sequence[int] | slice | torch.Tensor):
        """Restore the startup friction-budget scale of the given environments."""
        self.friction_scale[env_ids] = self._default_friction_scale[env_ids]

    def set_gains(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor,
        kp_scale: torch.Tensor | None = None,
        kd_scale: torch.Tensor | None = None,
    ):
        """Set the firmware gain scales of the given environments.

        Args:
            env_ids: Environments to write.
            kp_scale: Multiplier of the firmware proportional gain [-], broadcastable to
                ``(len(env_ids), 1)``. If None, the gain is left unchanged.
            kd_scale: Multiplier of the velocity the motor model sees [-], broadcastable to
                ``(len(env_ids), 1)``. If None, the damping is left unchanged.
        """
        if kp_scale is not None:
            self.kp_scale[env_ids] = kp_scale
        if kd_scale is not None:
            self.kd_scale[env_ids] = kd_scale

    def reset_gains(self, env_ids: Sequence[int] | slice | torch.Tensor):
        """Restore the default firmware gain scales of the given environments."""
        self.kp_scale[env_ids] = self._default_kp_scale[env_ids]
        self.kd_scale[env_ids] = self._default_kd_scale[env_ids]

    """
    Helper functions.
    """

    def _estimate_external_torque(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Estimate the external torque the gearbox works against.

        The rotor's momentum balance over the previous step, ``armature * ddq - tau_prev``,
        which is the load reaction seen at the output shaft. See the module docstring for
        what this approximates and what it misses.

        Args:
            joint_vel: Current joint velocities [rad/s], shape ``(num_envs, num_joints)``.

        Returns:
            External torque [N.m], shape ``(num_envs, num_joints)``.
        """
        joint_acc = (joint_vel - self._prev_joint_vel) / self._dt
        return self.params.armature * joint_acc - self._prev_applied_effort

    def _sample_per_env(self, value_range: tuple[float, float] | None, default: float) -> torch.Tensor:
        """Draw one uniform per-environment value, or fill with ``default`` when unset."""
        buffer = torch.empty(self._num_envs, 1, device=self._device)
        if value_range is None:
            return buffer.fill_(default)
        return buffer.uniform_(*value_range)

    def _sample_delay_phase(self) -> torch.Tensor:
        """Draw a per-environment phase offset for the lag resampling."""
        if self.cfg.delay_update_period <= 0:
            return torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        return torch.randint(
            0, self.cfg.delay_update_period, (self._num_envs,), dtype=torch.long, device=self._device
        )

    def _apply_delay(self, target: torch.Tensor) -> torch.Tensor:
        """Buffer the position command and return the lagged one."""
        if self._delay_buffer is None:
            return target
        self._delay_buffer.append(target)
        self._update_delay_lags()
        return self._delay_buffer[self._delay_lags]

    def _update_delay_lags(self):
        """Resample the per-environment lags following the reference update policy."""
        if self.cfg.delay_update_period > 0:
            should_update = (self._delay_step_count + self._delay_phase) % self.cfg.delay_update_period == 0
        else:
            should_update = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        if self.cfg.delay_hold_prob > 0.0:
            should_update &= torch.rand(self._num_envs, device=self._device) >= self.cfg.delay_hold_prob
        candidate = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (self._num_envs,),
            dtype=torch.long,
            device=self._device,
        )
        self._delay_lags = torch.where(should_update, candidate, self._delay_lags)
        self._delay_step_count += 1
