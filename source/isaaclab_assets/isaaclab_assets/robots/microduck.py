# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pollen Robotics MicroDuck open-source biped.

MicroDuck is a 0.74 kg, 14-DOF biped driven by Dynamixel XL330 servos: five joints per leg
(hip yaw/roll/pitch, knee, ankle) plus a four-DOF head (neck pitch, head pitch/yaw/roll).

The following configuration is available:

* :data:`MICRODUCK_CFG`: MicroDuck on explicit delayed PD servos, in the upstream stand pose.

The asset it spawns is converted from the upstream MJCF and is generated rather than committed;
see ``ATTRIBUTION.md`` next to :data:`MICRODUCK_USD_PATH` for its provenance and
``scripts/tools/convert_microduck.py`` for the conversion.
"""

import math
import os

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
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
# Upstream drives MicroDuck with a calibrated BAM model of the XL330 (a voltage-controlled motor
# with a load-dependent friction model), which Isaac Lab has no equivalent of. The gains below are
# that model's small-signal equivalent at the nominal operating point, computed from the same
# published parameters so they carry their provenance rather than being tuned. They are meant to be
# replaced by a BAM actuator, not adjusted.
##

_XL330_KT = 0.36601349688984386
"""Torque constant [N·m/A]. BAM ``params/xl330/m6.json``, the model upstream fits MicroDuck to."""

_XL330_RESISTANCE = 2.8113923539223227
"""Winding resistance [ohm], from the same BAM fit."""

_XL330_MAX_CURRENT = 1.75
"""Firmware current limit [A] of the XL330, from BAM's ``XL330Actuator``."""

_XL330_ERROR_GAIN = (4096.0 / (2.0 * math.pi)) / (256.0 * 885.0)
"""Duty cycle per radian of position error, per unit of firmware P-gain [1/rad].

BAM's ``XL330Actuator``: encoder counts per *radian* -- the 4096 counts per revolution divided by
2 pi -- over the firmware's P-gain divisor times its PWM limit. It converts a firmware gain into the
fraction of the bus voltage the servo applies.
"""

_FIRMWARE_KP = 200.0
"""Firmware position P-gain MicroDuck ships with (upstream ``_BAM_ACTUATOR_KWARGS``)."""

_NOMINAL_VOLTAGE = 7.35
"""Nominal bus voltage [V]: the midpoint of upstream's (6.5, 8.2) battery randomization range."""

MICRODUCK_SERVO_STIFFNESS = _FIRMWARE_KP * _XL330_ERROR_GAIN * _NOMINAL_VOLTAGE * _XL330_KT / _XL330_RESISTANCE
"""Servo position gain [N·m/rad], ~0.5507.

The firmware error gain turns a position error into a duty cycle, the bus voltage turns that into a
winding voltage, and ``kt / R`` turns that into a stall torque. Cross-check: the fallback
``position`` actuator in the upstream MJCF declares ``kp = 0.55``.
"""

MICRODUCK_SERVO_DAMPING = _XL330_KT * _XL330_KT / _XL330_RESISTANCE
"""Servo velocity gain [N·m·s/rad], ~0.0477.

Back-EMF, which is the motor's own velocity feedback: the firmware runs no derivative term
(the MJCF's ``position`` actuator has ``kv = 0``).
"""

MICRODUCK_SERVO_EFFORT_LIMIT = _XL330_KT * _XL330_MAX_CURRENT
"""Servo rated torque [N·m], ~0.6405.

The firmware current limiter binds before the winding voltage does: at the top of the battery range
the voltage bound is ``8.2 * kt / R`` = 1.07 N·m, well above ``kt * i_max``. It is also below the
MJCF's ``forcerange`` of 0.96 N·m, which stays as the solver clamp.
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
        "servos": DelayedPDActuatorCfg(
            # every joint of the walk model is driven; the expression also excludes the passive
            # backlash and roller hinges of the upstream variants, as upstream does
            joint_names_expr=["^(?!passive_).*"],
            stiffness=MICRODUCK_SERVO_STIFFNESS,
            damping=MICRODUCK_SERVO_DAMPING,
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
"""Configuration for the Pollen Robotics MicroDuck biped."""
