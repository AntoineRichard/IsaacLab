# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .actuator_base_cfg import ActuatorBaseCfg
from .bam_model import BAM_XL330_M6_PARAMS_FILE


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

    This model requires ``use_newton_actuators=True`` with the Newton backend. On MuJoCo Warp,
    the controller publishes its friction budget and viscous damping into the solver and reads
    the external load from its generalized forces. Other Newton solvers use the controller's
    torque-level friction approximation.
    """

    class_type: type | None = None
    """No Isaac Lab-executed model; Newton constructs the controller from its USD schema."""

    stiffness: dict[str, float] | float | None = None
    """Unused by this model. Defaults to None so that a configuration validates unset.

    Configuration validation rejects an object that still holds the inherited ``MISSING``
    sentinel, so the field is defaulted here rather than left required.
    Leave it unset; the firmware gain is configured with :attr:`kp_fw`.
    """

    damping: dict[str, float] | float | None = None
    """Unused by this model. Defaults to None so that a configuration validates unset.

    See :attr:`stiffness`.
    """

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
    and load-dependent terms). Sampled once at construction. Per-episode friction randomization
    writes the controller's ``friction_scale`` through
    :func:`~isaaclab.actuators.newton.write_group_parameter`. If None, the scale is 1.
    """

    min_delay: int = 0
    """Minimum command delay [physics steps]. Defaults to 0."""

    max_delay: int = 0
    """Maximum command delay [physics steps]. Defaults to 0, which disables the delay."""

    delay_hold_prob: float = 0.0
    """Probability of keeping the current lag instead of resampling it [-]. Defaults to 0."""

    delay_update_period: int = 0
    """Number of physics steps between lag resamples. Defaults to 0, which resamples every step.

    When positive, a phase offset in ``[0, delay_update_period)`` staggers the resamples rather
    than synchronizing them. The Newton controller draws the phase and lag per driven joint.
    """

    stiff_frictionloss: bool = True
    """Stiffen the joint friction constraint on a solver that applies the friction itself [-].

    MuJoCo Warp has no noslip solver: its friction-loss constraint stays soft and a statically
    held joint creeps. Setting this replaces the constraint's solver reference with the stiff,
    timestep-independent form the reference implementation uses.
    """


BACKLASH_JOINT_TEMPLATE: str = "passive_{joint}_backlash"
"""Name of the play hinge a :class:`BamBacklashActuatorCfg` servo reads its encoder through.

Formatted with the driven joint's name, so ``head_pitch`` is read through
``passive_head_pitch_backlash``. This is the convention the reference implementation's backlash
generator emits and the one the converted assets carry; it is a naming contract rather than a
configured value, because the two joints have to be paired up by a plant the asset already
describes.
"""


@configclass
class BamBacklashActuatorCfg(BamActuatorCfg):
    """Configuration for a BAM servo whose gearbox backlash is modelled as a second joint.

    A gearbox with play is modelled by splitting the servo in two: the configured joint is the
    motor output, and a second, unactuated hinge in series with it carries the play. The link
    angle is the two summed. The real servo's magnetic encoder sits on the *output* side of that
    play, so the firmware closes its position loop on the sum -- while the rotor winds through
    the dead zone the measured position, and hence the proportional error, does not move.

    This configuration is :class:`BamActuatorCfg` in every parameter; what it adds is that
    contract. The Newton backend pairs each driven joint with
    ``BACKLASH_JOINT_TEMPLATE.format(joint=<joint name>)`` at articulation initialization and
    binds the pair through
    :meth:`~isaaclab.actuators.newton.ControllerBam.bind_backlash_indices`. A driven joint whose
    plant carries no such hinge keeps the plain servo's behaviour, bit for bit, so one
    configuration covers a robot whose joints are only partly modelled with play.

    The group's joint expression selects the *servos*, never the play hinges: the hinges are
    joints nothing drives, and the asset's own expression (``^(?!passive_).*``) already leaves
    them out.

    Note:
        This model runs on the Newton-native actuator path only, and refuses to run anywhere
        else rather than degrading. Its encoder view is an index into the whole articulation's
        joint-position array, which only the Newton controller is handed; Isaac Lab's actuator
        loop is given one group's joints and cannot read a joint outside it. Running the plain
        servo instead would silently drop the modelled play, so a backend that steps native
        actuators through the host adapter (PhysX, OVPhysX) and a simulation configured with
        ``use_newton_actuators=False`` both raise.
    """
