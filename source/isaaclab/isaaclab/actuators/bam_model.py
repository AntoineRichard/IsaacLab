# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic math core of the BAM voltage-domain servo model.

The functions here are a PyTorch port of the `BAM (Better Actuator Models)
<https://github.com/Rhoban/bam>`_ project, at commit ``62bd8ce`` of its
``mjlab_frictionloss`` branch. BAM models a hobby servo in the *voltage* domain: the
firmware runs a proportional position controller that outputs a PWM duty cycle, the duty
cycle drives a DC motor through its winding resistance and back-EMF, and the resulting
gearbox friction is load-dependent (it grows with the torque flowing through the gears).

Every function is pure and stateless, operates elementwise on ``(num_envs, num_joints)``
tensors, and is device- and dtype-agnostic, so it can be shared by the different Isaac Lab
physics backends. Per-environment quantities (supply voltage, firmware gain) are passed as
``(num_envs, 1)`` tensors and broadcast over the joints of one actuator group.

References:
    * ``bam/actuator.py`` (``VoltageControlledActuator``) -- firmware control law and the
      DC-motor torque equation.
    * ``bam/mjlab.py`` (``BamActuator``) -- vectorized friction budget and supply sag.
    * ``bam/simulate.py`` (``Simulator.step``) -- static-friction (stiction) clipping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

BAM_XL330_M6_PARAMS_FILE: str = str(Path(__file__).parent / "data" / "bam_xl330_m6.json")
"""Path of the BAM parameters vendored with Isaac Lab (Dynamixel XL330, ``m6`` model)."""

# Friction-model flags of the BAM model family, mirroring ``bam/model.py`` (``models``).
# Only the models Isaac Lab supports are listed; see :meth:`BamMotorParams.from_json`.
_BAM_MODEL_FLAGS: dict[str, dict[str, bool]] = {
    "m1": {"stribeck": False, "load_dependent": False, "directional": False, "quadratic": False},
    "m2": {"stribeck": True, "load_dependent": False, "directional": False, "quadratic": False},
    "m5": {"stribeck": True, "load_dependent": True, "directional": True, "quadratic": False},
    "m6": {"stribeck": True, "load_dependent": True, "directional": True, "quadratic": True},
}


@dataclass(frozen=True)
class BamMotorParams:
    """Identified motor, firmware and friction parameters of one BAM actuator model.

    The values are per actuator *type* (not per joint or per environment): they come from
    fitting the BAM model to bench measurements of a specific servo. Isaac Lab ships the
    identified parameters of the Dynamixel XL330 in
    ``isaaclab/actuators/data/bam_xl330_m6.json``; see the ``ATTRIBUTION.md`` next to it.

    The friction-model flags select which terms of the budget are active and follow the
    BAM ``m1``--``m6`` model family (``m6``, the model Isaac Lab uses, enables all of them).

    Attributes:
        kt: Motor torque constant [N.m/A], equivalently the back-EMF constant [V.s/rad].
        R: Motor winding resistance [Ohm].
        armature: Rotor inertia reflected through the gearbox [kg.m^2].
        error_gain: Converts ``kp`` times the position error [rad] into a duty cycle [-].
        max_pwm: Largest duty-cycle magnitude the firmware can command [-].
        max_current: Firmware current limit [A], or None to disable the current limiter.
        kp: Nominal firmware proportional gain [-].
        vin: Nominal supply voltage [V].
        friction_base: Load-independent Coulomb friction [N.m].
        friction_viscous: Viscous friction coefficient [N.m.s/rad].
        friction_stribeck: Extra Coulomb friction at rest, from the Stribeck effect [N.m].
        dtheta_stribeck: Velocity scale over which the Stribeck effect decays [rad/s].
        alpha: Exponent shaping the Stribeck decay [-].
        load_friction_motor: Gearbox friction per unit of motor-side torque [N.m/N.m].
        load_friction_external: Gearbox friction per unit of external torque [N.m/N.m].
        load_friction_motor_stribeck: Stribeck part of ``load_friction_motor`` [N.m/N.m].
        load_friction_external_stribeck: Stribeck part of ``load_friction_external`` [N.m/N.m].
        load_friction_motor_quad: Quadratic back-driving friction coefficient [N.m/(N.m)^2].
        load_friction_external_quad: Quadratic driving friction coefficient [N.m/(N.m)^2].
        stribeck: Whether the Stribeck (near-zero-velocity) friction terms are active.
        load_dependent: Whether the gearbox friction grows with the transmitted torque.
        directional: Whether the load-dependent friction distinguishes the motor side from
            the external side. Requires ``load_dependent``.
        quadratic: Whether the quadratic load-coupling term is active. Requires
            ``directional`` and ``stribeck``.
    """

    kt: float
    R: float
    armature: float
    error_gain: float
    max_pwm: float
    max_current: float | None
    kp: float
    vin: float
    friction_base: float
    friction_viscous: float
    friction_stribeck: float = 0.0
    dtheta_stribeck: float = 1.0
    alpha: float = 1.0
    load_friction_motor: float = 0.0
    load_friction_external: float = 0.0
    load_friction_motor_stribeck: float = 0.0
    load_friction_external_stribeck: float = 0.0
    load_friction_motor_quad: float = 0.0
    load_friction_external_quad: float = 0.0
    stribeck: bool = False
    load_dependent: bool = False
    directional: bool = False
    quadratic: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> BamMotorParams:
        """Load parameters from a BAM parameter file.

        The file layout is the one BAM writes in ``bam/params/<motor>/<model>.json``,
        extended with the firmware constants that BAM keeps in code
        (``error_gain``, ``max_pwm``, ``max_current``, ``kp``, ``vin``). Keys that do not
        name a field of this class (such as ``q_offset`` or ``actuator``) are ignored, so
        an upstream parameter file can be vendored verbatim.

        Args:
            path: Path of the JSON file to read.

        Returns:
            The parsed parameters, with the friction-model flags set from the ``model`` key.

        Raises:
            KeyError: If the file does not declare a ``model``, or declares a BAM friction
                model that this port does not implement (the non-directional load-dependent
                models ``m3`` and ``m4``).
            TypeError: If a parameter required by the declared model is missing.
        """
        content = json.loads(Path(path).read_text())
        model_name = content.get("model")
        if model_name not in _BAM_MODEL_FLAGS:
            raise KeyError(
                f"BAM parameter file '{path}' declares model {model_name!r}, which is not supported."
                f" Supported models: {sorted(_BAM_MODEL_FLAGS)}."
            )
        field_names = {field.name for field in fields(cls)}
        values = {key: value for key, value in content.items() if key in field_names}
        return cls(**values, **_BAM_MODEL_FLAGS[model_name])


def compute_duty(
    q_target: torch.Tensor,
    q: torch.Tensor,
    dq: torch.Tensor,
    kp: torch.Tensor,
    vin: torch.Tensor,
    params: BamMotorParams,
) -> torch.Tensor:
    """Compute the PWM duty cycle commanded by the servo firmware.

    The firmware runs a pure proportional position controller, ``duty = error_gain * kp *
    (q_target - q)``, then applies its current limiter and finally the physical PWM clamp.
    The current limiter can only act on the duty cycle, so it is expressed as the duty
    window that keeps ``|I| = |duty * vin - kt * dq| / R`` below ``max_current``. Because
    the PWM clamp is applied last, a large back-EMF can push that window outside the
    achievable range and the current limit is then not actually reached -- which is how
    the real firmware behaves. Ported from ``VoltageControlledActuator.compute_control``
    (``bam/actuator.py:251-269``).

    Args:
        q_target: Target joint positions [rad], shape ``(num_envs, num_joints)``.
        q: Current joint positions [rad], shape ``(num_envs, num_joints)``.
        dq: Current joint velocities [rad/s], shape ``(num_envs, num_joints)``.
        kp: Firmware proportional gain [-], shape ``(num_envs, 1)``.
        vin: Supply voltage [V], shape ``(num_envs, 1)``.
        params: Motor parameters of the actuator.

    Returns:
        Duty cycle in ``[-max_pwm, max_pwm]`` [-], shape ``(num_envs, num_joints)``.
    """
    duty = (q_target - q) * kp * params.error_gain
    if params.max_current is not None:
        # Window centre is the duty that exactly cancels the back-EMF; half-width is the
        # duty that drives ``max_current`` through the winding resistance.
        duty_center = params.kt * dq / vin
        duty_span = params.R * params.max_current / vin
        duty = torch.clamp(duty, duty_center - duty_span, duty_center + duty_span)
    return torch.clamp(duty, -params.max_pwm, params.max_pwm)


def compute_motor_torque(
    duty: torch.Tensor,
    dq: torch.Tensor,
    vin: torch.Tensor,
    params: BamMotorParams,
) -> torch.Tensor:
    """Compute the motor torque produced by a duty cycle.

    Applies the DC-motor equation with back-EMF, ``tau = kt * V / R - kt^2 * dq / R``,
    to the terminal voltage ``V = vin * duty``. The firmware current limit is not applied
    here: :func:`compute_duty` already models it as a duty-cycle constraint. Ported from
    ``VoltageControlledActuator.compute_torque`` (``bam/actuator.py:289-292``).

    Args:
        duty: Commanded duty cycle [-], shape ``(num_envs, num_joints)``.
        dq: Current joint velocities [rad/s], shape ``(num_envs, num_joints)``.
        vin: Supply voltage [V], shape ``(num_envs, 1)``.
        params: Motor parameters of the actuator.

    Returns:
        Motor-side torque [N.m], shape ``(num_envs, num_joints)``.
    """
    volts = vin * duty
    return params.kt * volts / params.R - (params.kt**2) * dq / params.R


def compute_stribeck_coeff(dq: torch.Tensor, params: BamMotorParams) -> torch.Tensor:
    """Compute the Stribeck blending coefficient.

    The coefficient ``exp(-(|dq| / dtheta_stribeck)^alpha)`` is 1 at rest and decays to 0
    once the joint moves, and it weights the friction terms that only act near zero
    velocity. Ported from ``BamActuator.compute`` (``bam/mjlab.py:657``).

    Args:
        dq: Current joint velocities [rad/s], shape ``(num_envs, num_joints)``.
        params: Motor parameters of the actuator.

    Returns:
        Blending coefficient in ``[0, 1]`` [-], shape ``(num_envs, num_joints)``.
    """
    return torch.exp(-torch.pow(torch.abs(dq) / params.dtheta_stribeck, params.alpha))


def compute_friction_budget(
    prev_tau: torch.Tensor,
    ext_tau: torch.Tensor,
    stribeck_coeff: torch.Tensor,
    params: BamMotorParams,
    friction_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Compute the velocity-independent friction budget of the gearbox.

    The budget is the largest friction torque the joint can oppose to motion before it
    breaks away; the viscous part (``params.friction_viscous``) is not included. On top of
    the load-independent Coulomb and Stribeck terms, the load-dependent terms grow with the
    torque flowing through the gearbox, which BAM splits into a motor-side contribution
    (``motor_tau``, the torque the motor pushes into the gears) and an external one
    (``ext_tau``, the torque the load pushes back). The quadratic term of the ``m6`` model
    then picks the driving or back-driving coefficient depending on which side dominates.

    Ported from ``BamActuator._compute_friction_budget`` (``bam/mjlab.py:442-496``), which
    is the vectorized path the reference GPU consumer runs. It differs from the CPU path
    (``bam.model.Model.compute_frictions``) in how the quadratic term is gated; the two
    agree for ``m1``--``m5``.

    Args:
        prev_tau: Motor-side torque [N.m], shape ``(num_envs, num_joints)``. This is the
            torque applied on the previous step, matching the reference implementation.
        ext_tau: External (gravity, contact, constraint) torque on the gearbox [N.m],
            shape ``(num_envs, num_joints)``.
        stribeck_coeff: Stribeck coefficient from :func:`compute_stribeck_coeff` [-],
            shape ``(num_envs, num_joints)``.
        params: Motor parameters of the actuator.
        friction_scale: Multiplier applied to the whole budget [-], for randomizing
            friction per environment or per joint. Broadcastable to
            ``(num_envs, num_joints)``.

    Returns:
        Friction budget [N.m], shape ``(num_envs, num_joints)``.
    """
    budget = torch.full_like(prev_tau, params.friction_base)
    if params.stribeck:
        budget = budget + stribeck_coeff * params.friction_stribeck
    if params.load_dependent:
        budget = budget + torch.abs(ext_tau * params.load_friction_external - prev_tau * params.load_friction_motor)
        if params.stribeck:
            gearbox_tau_stribeck = torch.abs(
                ext_tau * params.load_friction_external_stribeck - prev_tau * params.load_friction_motor_stribeck
            )
            budget = budget + stribeck_coeff * gearbox_tau_stribeck
            if params.quadratic:
                # Driving (motor wins) loads the gearbox through the external torque;
                # back-driving (load wins) loads it through the motor torque.
                abs_ext, abs_motor = torch.abs(ext_tau), torch.abs(prev_tau)
                drive_mask = (abs_motor > abs_ext).to(prev_tau.dtype)
                quad_term = (
                    drive_mask * params.load_friction_external_quad * abs_ext**2
                    + (1.0 - drive_mask) * params.load_friction_motor_quad * abs_motor**2
                )
                budget = budget + stribeck_coeff * quad_term
    return budget * friction_scale


def apply_stiction_clip(
    motor_tau: torch.Tensor,
    ext_tau: torch.Tensor,
    dq: torch.Tensor,
    frictionloss: torch.Tensor,
    viscous: float,
    dt: float,
    inertia: torch.Tensor | float,
) -> torch.Tensor:
    """Apply the friction budget to the motor torque at the torque level.

    Friction is capped by the torque that would bring the joint to rest within ``dt``, so a
    joint whose net torque fits in its budget is held static instead of being pushed
    backwards. Ported from ``Simulator.step`` (``bam/simulate.py:67-72``).

    This is the approximation used when the physics backend cannot represent joint dry
    friction itself: the reference GPU pipeline instead writes the budget into MuJoCo's
    ``dof_frictionloss`` and lets the constraint solver do the clipping, which resolves the
    stiction jointly with the other constraints rather than one joint at a time.

    The returned torque excludes ``ext_tau``, which enters only through the stopping-torque
    test: the simulator applies the external load itself, so the actuator must apply only
    the motor and friction torques. Holding a static joint therefore means returning
    ``-ext_tau``, the torque that cancels the external load.

    Args:
        motor_tau: Motor-side torque [N.m], shape ``(num_envs, num_joints)``.
        ext_tau: External torque on the joint [N.m], shape ``(num_envs, num_joints)``.
        dq: Current joint velocities [rad/s], shape ``(num_envs, num_joints)``.
        frictionloss: Friction budget from :func:`compute_friction_budget` [N.m], shape
            ``(num_envs, num_joints)``.
        viscous: Viscous friction coefficient [N.m.s/rad].
        dt: Control timestep [s].
        inertia: Joint inertia used to size the stopping torque [kg.m^2], broadcastable to
            ``(num_envs, num_joints)``. Required, and deliberately so: it sets how close to
            rest a joint must be to count as static, and a value that is wrong by a factor of
            ``k`` widens or narrows that window by the same factor. Pass the reflected rotor
            inertia (``BamMotorParams.armature``) when the actuator only knows its own gearbox,
            or the full joint inertia when the caller has it -- the reference simulator uses the
            latter.

    Returns:
        Actuator torque to apply [N.m], shape ``(num_envs, num_joints)``.
    """
    net_tau = motor_tau + ext_tau
    # Torque that would bring the joint to a stop in one timestep.
    tau_stop = (inertia / dt) * dq + net_tau
    budget = frictionloss + viscous * torch.abs(dq)
    friction_tau = -torch.sign(tau_stop) * torch.minimum(torch.abs(tau_stop), budget)
    return motor_tau + friction_tau


def battery_sag(
    vin: torch.Tensor,
    prev_tau: torch.Tensor,
    sag_gain: torch.Tensor | float,
    vin_min: float | None = None,
) -> torch.Tensor:
    """Compute the supply voltage left after the load-induced battery drop.

    All joints of one actuator group share a supply, so the voltage drop is driven by the
    summed magnitude of the torques they drew on the previous step:
    ``vin_eff = max(vin - sag_gain * sum_j |prev_tau_j|, vin_min)``. Ported from
    ``BamActuator.compute`` (``bam/mjlab.py:602-607``).

    Args:
        vin: Nominal supply voltage [V], shape ``(num_envs, 1)``.
        prev_tau: Motor torques of the previous step [N.m], shape
            ``(num_envs, num_joints)``. Zero these on reset so an episode starts unloaded.
        sag_gain: Effective source resistance of the supply [V/(N.m)], broadcastable to
            ``(num_envs, 1)``.
        vin_min: Lower bound on the sagged voltage [V], or None to leave it unbounded.

    Returns:
        Effective supply voltage [V], shape ``(num_envs, 1)``.
    """
    load = prev_tau.abs().sum(dim=-1, keepdim=True)
    vin_eff = vin - sag_gain * load
    if vin_min is not None:
        vin_eff = torch.clamp(vin_eff, min=vin_min)
    return vin_eff
