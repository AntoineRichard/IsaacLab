# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg
from .bam_model import BAM_XL330_M6_PARAMS_FILE

if TYPE_CHECKING:
    from .actuator_bam import BamActuator


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the BAM voltage-domain servo actuator.

    The identified motor and friction parameters come from the file named by
    :attr:`params_file`; everything configured here is either a deployment setting that the
    identification does not capture (the firmware gain and the supply voltage of the robot
    the model is used on) or a domain-randomization range.

    Note:
        :attr:`~isaaclab.actuators.ActuatorBaseCfg.stiffness` and
        :attr:`~isaaclab.actuators.ActuatorBaseCfg.damping` are unused by this model. Its
        position loop runs in the firmware domain, parameterized by :attr:`kp_fw`, and its
        damping is the physical back-EMF of the motor.
    """

    class_type: type["BamActuator"] | str = "{DIR}.actuator_bam:BamActuator"

    params_file: str = BAM_XL330_M6_PARAMS_FILE
    """Path of the BAM parameter file to load. Defaults to the vendored Dynamixel XL330 ``m6`` fit."""

    kp_fw: float | None = 200.0
    """Firmware proportional gain [-].

    This is a servo setting rather than an identified constant, so it is configured
    per deployment. If None, the value identified in :attr:`params_file` is used.
    """

    vin: float | None = None
    """Nominal supply voltage [V].

    If None, the value identified in :attr:`params_file` is used. Overridden by
    :attr:`vin_range` when that is set.
    """

    vin_range: tuple[float, float] | None = None
    """Range to sample the per-environment supply voltage from [V].

    Sampled once at construction and held constant across resets, because a robot's battery
    does not change between episodes. Takes precedence over :attr:`vin`.
    """

    vin_drop_gain_range: tuple[float, float] | None = None
    """Range to sample the per-environment supply sag gain from [V/(N.m)].

    The gain models the voltage drop across the battery and wiring resistance under load,
    ``vin_eff = vin - gain * sum_j |tau_j|``. Sampled once at construction and held constant
    across resets. If None, the gain is zero and the supply does not sag.
    """

    vin_min: float | None = None
    """Lower bound on the supply voltage after the load-induced sag [V], or None for no bound."""

    friction_scale_range: tuple[float, float] | None = None
    """Range to sample the per-environment friction-budget scale from [-].

    The scale multiplies the whole velocity-independent friction budget (Coulomb, Stribeck
    and load-dependent terms). Sampled once at construction; the sample is the value that
    :meth:`~isaaclab.actuators.BamActuator.reset_friction_scale` restores. Per-episode
    friction randomization is applied by an event calling
    :meth:`~isaaclab.actuators.BamActuator.set_friction_scale`. If None, the scale is 1.
    """

    min_delay: int = 0
    """Minimum command delay [physics steps]. Defaults to 0."""

    max_delay: int = 0
    """Maximum command delay [physics steps]. Defaults to 0, which disables the delay."""

    delay_hold_prob: float = 0.0
    """Probability of keeping the current lag instead of resampling it [-]. Defaults to 0."""

    delay_update_period: int = 0
    """Number of physics steps between lag resamples. Defaults to 0, which resamples every step.

    When positive, each environment gets its own phase offset in ``[0, delay_update_period)``
    so that the resamples are staggered rather than synchronized across environments.
    """

    dt: float | None = None
    """Physics timestep the actuator is stepped at [s].

    If None, it is read from the running simulation at construction. The model needs it to
    size the stopping torque of its static-friction clip and to differentiate the joint
    velocities, neither of which the base actuator interface provides.
    """
