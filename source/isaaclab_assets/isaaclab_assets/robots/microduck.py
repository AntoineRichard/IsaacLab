# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pollen Robotics MicroDuck open-source biped.

MicroDuck is a 0.74 kg, 14-DOF biped driven by Dynamixel XL330 servos: five joints per leg
(hip yaw/roll/pitch, knee, ankle) plus a four-DOF head (neck pitch, head pitch/yaw/roll).

The following configuration is available:

* :data:`MICRODUCK_CFG`: MicroDuck on the BAM servo model, in the upstream stand pose.

The asset it spawns is converted from the upstream MJCF and is generated rather than committed;
see ``ATTRIBUTION.md`` next to :data:`MICRODUCK_USD_PATH` for its provenance and
``scripts/tools/convert_microduck.py`` for the conversion.
"""

import os

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import BAM_XL330_M6_PARAMS_FILE, BamActuatorCfg, BamMotorParams
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

MICRODUCK_USD_PATH = os.path.join(
    ISAACLAB_ASSETS_DATA_DIR, "Robots", "PollenRobotics", "MicroDuck", "microduck_walk.usd"
)
"""Path of the converted MicroDuck asset."""

MICRODUCK_REGENERATE_COMMAND = "uv run --extra importers python scripts/tools/convert_microduck.py"
"""Command that regenerates :data:`MICRODUCK_USD_PATH` from the pinned upstream MJCF."""


def _spawn_microduck(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the MicroDuck asset, reporting its absence with the command that regenerates it.

    USD files are excluded from the repository, so this asset is produced on demand. Checking here
    rather than at import time keeps the configuration inspectable in a tree without the asset,
    which the fidelity tests rely on.

    Args:
        prim_path: Prim path or pattern to spawn the asset at.
        cfg: Spawner configuration.
        translation: Translation w.r.t. the parent prim. Defaults to the one in the USD file.
        orientation: Orientation as (w, x, y, z) w.r.t. the parent prim. Defaults to the one in
            the USD file.
        **kwargs: Forwarded to :meth:`~isaaclab.sim.spawn_from_usd`.

    Returns:
        The spawned prim.

    Raises:
        FileNotFoundError: If the converted asset has not been generated.
    """
    if not os.path.isfile(cfg.usd_path):
        raise FileNotFoundError(
            f"The MicroDuck asset is missing: {cfg.usd_path}. It is generated rather than committed;"
            f" create it with '{MICRODUCK_REGENERATE_COMMAND}'."
        )
    return sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)


##
# Servo model.
#
# Upstream drives MicroDuck with the BAM model of the Dynamixel XL330: a voltage-controlled motor
# whose firmware closes the position loop in the duty-cycle domain, behind a load-dependent gearbox
# friction model identified on a testbench. :class:`~isaaclab.actuators.BamActuatorCfg` is that
# model, reading the same vendored ``xl330/m6`` fit, so the settings below are upstream's
# ``_BAM_ACTUATOR_KWARGS`` (reference section 6) rather than a derived equivalent.
##

_XL330_PARAMS = BamMotorParams.from_json(BAM_XL330_M6_PARAMS_FILE)
"""The identified XL330 ``m6`` parameters, read here so the derived limit below cannot drift."""

MICRODUCK_SERVO_VIN_RANGE = (6.5, 8.2)
"""Per-robot bus voltage range [V] the battery is drawn from (upstream ``_BAM_ACTUATOR_KWARGS``)."""

MICRODUCK_SERVO_EFFORT_LIMIT = max(MICRODUCK_SERVO_VIN_RANGE) * _XL330_PARAMS.kt / _XL330_PARAMS.R
"""Electrical stall torque [N·m] at the top of the battery range, ~1.068.

Upstream's own bound on the BAM output (``bam/mjlab.py``: ``force_limit`` is derived from
``max(vin_range)``). The model's firmware current limiter already shapes the duty cycle, but it
sizes a *current* window that a large back-EMF can push outside the achievable PWM range, so the
torque is not bounded by ``kt * i_max``; this is. It sits above the MJCF's ``forcerange`` of
0.96 N·m, which survives conversion as the solver-side clamp and therefore binds first.
"""

MICRODUCK_JOINT_DAMPING = 0.053
"""Passive joint viscous damping [N·m·s/rad], from the MJCF ``chosen_actuator`` class.

MuJoCo's ``dof_damping``, which the MJCF-to-USD conversion does not carry (it is written only as
``mjc:damping``, outside the schema resolvers Isaac Lab passes to Newton).
"""

MICRODUCK_JOINT_FRICTION = 0.0048
"""Passive joint dry friction [N·m], the MJCF ``frictionloss``, lost in conversion for the same
reason as :data:`MICRODUCK_JOINT_DAMPING`."""


MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=_spawn_microduck,
        usd_path=MICRODUCK_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.NewtonArticulationRootPropertiesCfg(self_collision_enabled=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # midpoint of the (0.12, 0.13) base height upstream resets to. The conversion clears the
        # MJCF home height the importer bakes into the root transform, so this sets the spawn
        # height rather than offsetting it.
        pos=(0.0, 0.0, 0.125),
        # upstream HOME_FRAME (STAND2): the trunk leans forward so the CoM sits over the ankle axis
        joint_pos={
            ".*hip_yaw": 0.0,
            "left_hip_roll": -0.0873,
            "right_hip_roll": 0.0873,
            "left_hip_pitch": -0.4579,
            "right_hip_pitch": 0.4579,
            "left_knee": -0.0049,
            "right_knee": 0.0049,
            "left_ankle": 0.4530,
            "right_ankle": -0.4530,
            "neck_pitch": 0.3491,
            "head_pitch": 0.3491,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "servos": BamActuatorCfg(
            # every joint of the walk model is driven; the expression also excludes the passive
            # backlash and roller hinges of the upstream variants, as upstream does
            joint_names_expr=["^(?!passive_).*"],
            # the vendored xl330 ``m6`` fit, which is the file upstream identifies MicroDuck against
            params_file=BAM_XL330_M6_PARAMS_FILE,
            # firmware P-gain, not a Lab stiffness: it scales a position error into a duty cycle
            kp_fw=200.0,
            # per-robot battery voltage [V], its sag under load [V/(N·m)] and the floor after the
            # sag [V]. All three are drawn once per environment at construction and held across
            # resets, because a robot does not swap its battery between episodes.
            vin_range=MICRODUCK_SERVO_VIN_RANGE,
            vin_drop_gain_range=(0.0, 0.2),
            vin_min=6.0,
            # per-robot gearbox friction spread. The ``randomize_joint_friction`` reset event
            # overwrites this every episode; the draw here is what a task without that event gets,
            # and what ``reset_friction_scale`` restores.
            friction_scale_range=(0.9, 1.1),
            actuator_effort_limit=MICRODUCK_SERVO_EFFORT_LIMIT,
            # restored here because the conversion drops them; armature is left to the USD, which
            # does carry the MJCF value
            viscous_friction=MICRODUCK_JOINT_DAMPING,
            friction=MICRODUCK_JOINT_FRICTION,
            # upstream delay_min_lag / delay_max_lag, in physics steps
            min_delay=3,
            max_delay=6,
        ),
    },
)
"""Configuration for the Pollen Robotics MicroDuck biped.

No ``stiffness``/``damping``: the BAM model ignores both (its position loop is ``kp_fw`` and its
damping is the motor's back-EMF) and warns if they are set.

:data:`MICRODUCK_JOINT_DAMPING` and :data:`MICRODUCK_JOINT_FRICTION` are the MJCF's
``chosen_actuator`` joint dynamics, and what they mean depends on the execution path. With
``use_newton_actuators=True`` on MJWarp they are seeds only: the native controller republishes the
live friction budget and viscous coefficient into the solver's ``dof_frictionloss`` /
``dof_damping`` every physics step, which is what the reference implementation does. With
``use_newton_actuators=False`` the Isaac Lab-executed model clips the torque against its own budget
instead and never writes to the solver, so these are the *only* joint-level dissipation the solver
has -- and it needs them: without any, this 14-DOF biped diverges under an untrained policy within
twenty control steps at 2048 environments (the joint velocities run away while the applied torque
stays at its clamp), and the divergence is what a reward NaN then reports. That makes the dry
friction slightly double-counted on this path, by 0.0048 N·m against a ~1 N·m stall torque, which
is the price of a stable integration.

The armature is left to the USD, which carries the MJCF's 0.0018 -- the same value the BAM fit
identifies.
"""
