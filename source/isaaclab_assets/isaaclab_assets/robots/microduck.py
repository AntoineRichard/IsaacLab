# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pollen Robotics MicroDuck open-source biped.

MicroDuck is a 0.74 kg biped driven by 14 Dynamixel XL330 servos: five joints per leg (hip
yaw/roll/pitch, knee, ankle) plus a four-DOF head (neck pitch, head pitch/yaw/roll).

Upstream ships three flat-task robot models. All three are driven by those same 14 servos, so the
action space is 14-dimensional throughout; they differ in geometry, and one of them adds hinges that
nothing drives:

* :data:`MICRODUCK_CFG`: MicroDuck on the BAM servo model, in the upstream stand pose. The walking
  model, whose only ground contact is through the two foot soles. 14 joints.
* :data:`MICRODUCK_ALLCOLLISIONS_CFG`: the same robot with the trunk, hip, shin and head colliders
  upstream's stand-up and roulade tasks need to reach the ground. 14 joints.
* :data:`MICRODUCK_ROLLERS_CFG`: the all-collisions robot with each foot replaced by two passively
  rolling wheels. **18 joints**, of which the four wheel hinges are undriven -- and it stands 2.4 cm
  taller than the other two, which its configuration does not know (see its own documentation).

The assets they spawn are converted from the upstream MJCFs and are generated rather than committed;
see ``ATTRIBUTION.md`` next to :data:`MICRODUCK_USD_PATH` for their provenance and
``scripts/tools/convert_microduck.py`` for the conversion.

One non-robot prop lives here too, next to the robots it is kicked by:
:data:`MICRODUCK_BALL_CFG`, the 70 mm hollow ball of upstream's ball-kick task. It is authored
directly rather than converted, because its MJCF is a single analytic sphere.
"""

import copy
import math
import os

from pxr import Gf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import BAM_XL330_M6_PARAMS_FILE, BamActuatorCfg, BamMotorParams, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.sim.utils import clone

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

_MICRODUCK_DATA_DIR = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "PollenRobotics", "MicroDuck")
"""Directory the converted MicroDuck assets are written to."""

MICRODUCK_USD_PATH = os.path.join(_MICRODUCK_DATA_DIR, "microduck_walk.usd")
"""Path of the converted MicroDuck walking asset."""

MICRODUCK_ALLCOLLISIONS_USD_PATH = os.path.join(_MICRODUCK_DATA_DIR, "microduck_allcollisions.usd")
"""Path of the converted MicroDuck all-collisions asset."""

MICRODUCK_ROLLERS_USD_PATH = os.path.join(_MICRODUCK_DATA_DIR, "microduck_rollers.usd")
"""Path of the converted MicroDuck roller asset."""

MICRODUCK_BEAK_USD_PATH = os.path.join(_MICRODUCK_DATA_DIR, "microduck_beak.usd")
"""Path of the converted MicroDuck asset whose beak opens."""


def _regenerate_command(usd_path: str) -> str:
    """Return the conversion command that produces a given MicroDuck asset.

    The converter names its output after the upstream model it converts, so the model selector the
    command needs is recoverable from the path rather than tracked next to it.

    Args:
        usd_path: Path of the converted asset.

    Returns:
        The command to run from the repository root.
    """
    command = "uv run --extra importers python scripts/tools/convert_microduck.py"
    model = os.path.splitext(os.path.basename(usd_path))[0].removeprefix("microduck_")
    return command if model == "walk" else f"{command} --model {model}"


MICRODUCK_REGENERATE_COMMAND = _regenerate_command(MICRODUCK_USD_PATH)
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
            f" create it with '{_regenerate_command(cfg.usd_path)}'."
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
"""The identified XL330 ``m6`` parameters, read here so the derived limit below cannot drift.

The read happens at module import rather than behind a cached accessor because
:data:`MICRODUCK_SERVO_EFFORT_LIMIT` and :data:`MICRODUCK_CFG` both consume it at module scope, so
deferring it would buy nothing. The module is lazily exported (``isaaclab_assets.robots`` calls
:func:`~isaaclab.utils.module.lazy_export`), so the file is only imported when something reaches for
:data:`MICRODUCK_CFG` -- which needs these parameters anyway.
"""

MICRODUCK_SERVO_VIN_RANGE = (6.5, 8.2)
"""Per-robot bus voltage range [V] the battery is drawn from (upstream ``_BAM_ACTUATOR_KWARGS``)."""

MICRODUCK_SERVO_EFFORT_LIMIT = max(MICRODUCK_SERVO_VIN_RANGE) * _XL330_PARAMS.kt / _XL330_PARAMS.R
"""Electrical stall torque [N·m] at the top of the battery range, ~1.068.

Upstream's own bound on the BAM output (``bam/mjlab.py``: ``force_limit`` is derived from
``max(vin_range)``). The model's firmware current limiter already shapes the duty cycle, but it
sizes a *current* window that a large back-EMF can push outside the achievable PWM range, so the
torque is not bounded by ``kt * i_max``; this is.

**Known parity gap, ~11 % in peak torque.** Upstream does not merely *add* this bound: its
``edit_spec`` (``bam/mjlab.py:263-265``) overwrites the MJCF's ``forcerange`` with
``±1.0676 N·m``, so 1.0676 is the only ceiling its actuators ever see. This port leaves the
authored ``forcerange`` of ``±0.96 N·m`` in place, and the conversion carries it through as the
solver-side joint effort limit -- so the two bounds bind on opposite sides:

* **upstream:** the 1.0676 N·m actuator bound binds, and there is no tighter solver clamp;
* **here:** the 0.96 N·m solver clamp binds first, and this 1.0676 N·m actuator limit is a
  backstop that only shows up in the actuator's own telemetry.

MicroDuck therefore trains against a peak joint torque about 11 % below upstream's, which matters
for a robot whose walking gait is torque-limited at the ankles.

The 0.96 is kept deliberately. It is the value the MJCF authors, and reproducing the MJCF is what
the asset-fidelity tests assert of the conversion (``test_joint_effort_limits_match_mjcf``);
overriding it from an actuator configuration would make the spawned articulation stop matching its
source while the conversion tests still passed, which is the failure mode those tests exist to
prevent. Closing the gap belongs in the converter or in an explicit, tested override of
``joint_effort_limit``, not as a silent side effect of the servo model.
"""

MICRODUCK_JOINT_DAMPING = 0.00536
"""Passive joint viscous damping [N·m·s/rad], the value upstream actually deploys.

This is the vendored ``xl330`` ``m6`` fit's ``friction_viscous`` (``0.005359668274599504`` in
:data:`~isaaclab.actuators.BAM_XL330_M6_PARAMS_FILE`), rounded to three significant digits.
Upstream's BAM binding republishes that coefficient into MuJoCo's ``dof_damping`` every
step, so ~0.0054 -- not the MJCF's ``chosen_actuator`` value of ``0.053`` -- is what the deployed
robot is trained and identified against.

MuJoCo's ``dof_damping`` is not carried by the MJCF-to-USD conversion (it is written only as
``mjc:damping``, outside the schema resolvers Isaac Lab passes to Newton), so the configuration has
to restore it either way; the only question was which value.

.. warning::

    This value requires MJWarp's MuJoCo-parity joint limits to be enabled, i.e.
    :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_default_joint_limit_solref` left at
    its default of ``True``. Turning that escape hatch off without also restoring ``0.053`` here
    reproduces the divergence described below.

**Why it used to be 0.053.** The 10x inflated MJCF value was a workaround for a solver defect, not a
plant property. Newton's unauthored joint-limit gains (``limit_ke = 1e4`` / ``limit_kd = 1e1``)
convert to an *underdamped* MuJoCo limit constraint -- ``jnt_solref ~ (0.0072, 0.242)`` on this
robot instead of MuJoCo's critically damped ``(0.02, 1.0)`` -- so every limit contact of this
0.74 kg, limit-bounded biped pumped energy that only passive damping could remove. At the true
0.0054 the state went non-finite within a few hundred steps; ten times more damping masked it. The
bisection that isolated the limit constraint as necessary and sufficient is in
``artifacts/microduck/mjlab_repro/report.md`` (E4).

The defect is fixed at the backend layer, by the flag named in the warning above: unauthored joint
limits now resolve to MuJoCo's default ``solreflimit`` of ``(0.02, 1.0)``. With that in place,
dropping the damping to upstream's
value cuts the golden-trajectory joint RMSE against upstream mjlab by **82%** (0.0440 -> 0.0079 rad)
and the attitude error by 87%, and makes MicroDuck fall on exactly upstream's step -- see
``artifacts/microduck/golden_trajectories/comparison_report.md``.
"""

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
            # restored here because the conversion drops it; armature is left to the USD, which does
            # carry the MJCF value, and the dry friction belongs to the BAM model rather than to the
            # solver on either execution path
            viscous_friction=MICRODUCK_JOINT_DAMPING,
            # upstream delay_min_lag / delay_max_lag, in physics steps
            min_delay=3,
            max_delay=6,
        ),
    },
)
"""Configuration for the Pollen Robotics MicroDuck biped.

No ``stiffness``/``damping``: the BAM model ignores both (its position loop is ``kp_fw`` and its
damping is the motor's back-EMF) and warns if they are set.

**No ``friction`` either, and that is deliberate.** The dry friction is the BAM model's, on both
execution paths, exactly as in the reference: upstream's binding zeroes the MJCF's ``frictionloss``
on every joint it drives and applies the load-dependent budget itself. Configuring the MJCF's
0.0048 N·m here would be ignored on the Isaac Lab-executed path -- :class:`BamActuator` declares
:attr:`~isaaclab.actuators.ActuatorBase.applies_joint_friction`, so the collection zeroes the
group's solver friction and warns -- and overwritten every physics step on the Newton-native one.

:data:`MICRODUCK_JOINT_DAMPING` is the one joint dynamic the conversion drops that the
configuration still restores, and what it means depends on the execution path. With
``use_newton_actuators=True`` on MJWarp it is a seed only: the native controller republishes the
live friction budget and viscous coefficient into the solver's ``dof_frictionloss`` /
``dof_damping`` every physics step, which is what the reference implementation does. With
``use_newton_actuators=False`` the Isaac Lab-executed model clips the torque against its own budget
instead and never writes to the solver, so this viscous term is the *only* joint-level dissipation
the solver has -- and it is what keeps the integration stable there.

**The joint damping now matches upstream's deployment.** Both paths integrate MicroDuck at the
``m6`` fit's ``friction_viscous``, which is what upstream's BAM binding republishes into
``dof_damping`` every step. The earlier 10x inflated value (the MJCF's ``0.053``) was a workaround
for the underdamped joint-limit conversion, now fixed on the backend -- see
:data:`MICRODUCK_JOINT_DAMPING`, and note that this configuration is only stable while
:attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_default_joint_limit_solref` stays
enabled. The Task-11 reading of the divergence as a plant-level instability of the
MJCF -> USD -> Newton -> MJWarp path was correct about the path and wrong about the cause: the
``DelayedPDActuatorCfg`` configuration diverged at ~0.0054 for the same solver reason, not because
of the servo model.

The armature is left to the USD, which carries the MJCF's 0.0018 -- the same value the BAM fit
identifies.
"""


def _microduck_variant_cfg(usd_path: str) -> ArticulationCfg:
    """Return :data:`MICRODUCK_CFG` respawned from another converted MicroDuck model.

    The upstream models share one skeleton, one home pose, one servo group and one set of joint
    dynamics, and differ only in which geoms collide and -- for the roller model -- in the passive
    wheel hinges the servo expression already excludes. Deriving them from one configuration is what
    keeps a change to the servo deployment from reaching only some of the robots.

    Args:
        usd_path: Path of the converted asset the configuration spawns.

    Returns:
        A configuration that is :data:`MICRODUCK_CFG` in everything but the asset it spawns.
    """
    cfg = copy.deepcopy(MICRODUCK_CFG)
    cfg.spawn.usd_path = usd_path
    return cfg


MICRODUCK_ALLCOLLISIONS_CFG = _microduck_variant_cfg(MICRODUCK_ALLCOLLISIONS_USD_PATH)
"""Configuration for the MicroDuck biped on upstream's all-collisions model.

Upstream's stand-up and roulade tasks run on this model. It has the walking model's joints, bodies,
sites and sensors, and six more colliders: a second trunk shell, the two hip cheeks, and the three
head shells on ``jaw_soft``. All six reach world contact in the MJCF, and the conversion keeps them
there -- the head shells in particular, because a robot that rolls over its head needs a head that
touches the ground.

Everything else, including the servo group and the joint dynamics the conversion drops, is
:data:`MICRODUCK_CFG`'s.
"""

MICRODUCK_BEAK_JOINT_NAME = "mouth"
"""The fifteenth servo, which opens the beak.

Named for upstream's own wire order, where ``mouth`` sits at index 9 of 15 between ``head_roll`` and
``right_hip_yaw`` (``scripts/bake-duck-mesh.py`` in ``pollen-robotics/microduck``).
"""

MICRODUCK_BEAK_CLOSED = -0.08726646259971647
"""Jaw angle [rad] with the beak shut, upstream's ``MOUTH_CLOSED`` of -5 degrees.

Reproduced independently from the meshes: sweeping the jaw about its measured hinge puts the minimum
gap to the upper mouth surface at 0.06 mm at exactly this angle. See
``artifacts/microduck/pickplace/BEAK.md``.
"""

MICRODUCK_BEAK_OPEN = 0.5235987755982988
"""Jaw angle [rad] with the beak fully open, upstream's ``MOUTH_OPEN`` of +30 degrees.

The aperture at the mouth line there is **31.1 mm**, which is the bound on what this robot can
pick up. See ``artifacts/microduck/pickplace/BEAK.md``; note that a *minimum* gap of 17.4 mm also
falls out of the same sweep and is not the aperture.
"""

MICRODUCK_ROLLERS_CFG = _microduck_variant_cfg(MICRODUCK_ROLLERS_USD_PATH)
"""Configuration for the MicroDuck biped on upstream's roller model.

The all-collisions robot with each foot replaced by a two-wheel bogie: the ``ankle_left`` and
``ankle_right`` bodies become ``ankle_l_v1`` and ``ankle_r_v1``, each carrying two ``tire`` bodies on
a ``passive_*_wheel`` hinge, and the two soles give way to the four tires as the ground contact.

The 18 hinges are **not** 18 degrees of freedom to drive. The four wheels are passive: the MJCF
gives them no actuator, no limits and no joint damping or friction, and the servo group's
``^(?!passive_).*`` expression -- upstream's own, carried by :data:`MICRODUCK_CFG` for exactly this
model -- leaves them out. The action space is therefore the same 14 servos as on the other models,
and the wheels roll on the properties the USD carries from the MJCF (an armature of 1e-4 and nothing
else). Their home position is zero, which is the MJCF's, so they are not listed in the initial state.

.. warning::

    **A roller task must override this configuration's** ``init_state.pos``.
    The spawn height inherited from :data:`MICRODUCK_CFG` is 0.125 m, which is the walking model's:
    the midpoint of the ``(0.12, 0.13)`` base height upstream resets a *legged* MicroDuck to. Wheels
    are taller than soles. With this model's joints at the home pose and its lowest collider resting
    on the ground, ``trunk_base`` sits at a measured **0.14070 m**, so spawning at 0.125 m puts the
    tires roughly **1.6 cm below the floor**. Upstream's own roller task resets into
    ``(0.1335, 0.1435)``.

    The height is deliberately *not* corrected here. It is one of a group of roller-specific
    quantities that upstream sizes against the wheel-less model it used to load by mistake -- the
    reset band, the ``com_height_target`` band and the reward's ``wheel_radius`` default of 0.0175
    against a measured tire radius of 0.0150 -- and whether this port reproduces upstream verbatim
    or re-measures is one decision, taken once, at the task level. This configuration's job is to
    carry the MJCF faithfully; it deliberately does not guess at the answer by fixing one member of
    that group in isolation.
"""


MICRODUCK_BEAK_CFG = _microduck_variant_cfg(MICRODUCK_BEAK_USD_PATH)
MICRODUCK_BEAK_CFG.init_state.joint_pos = {
    **MICRODUCK_CFG.init_state.joint_pos,
    # a duck at rest has its beak shut, and the mesh's own baked pose is 5 degrees open
    MICRODUCK_BEAK_JOINT_NAME: MICRODUCK_BEAK_CLOSED,
}
MICRODUCK_BEAK_CFG.actuators = {
    # The fourteen policy servos, named rather than matched. The other variants select them with
    # upstream's ``^(?!passive_).*``, which would sweep the fifteenth joint into the same group and
    # silently widen the action space; this model is the one where that expression stops being safe.
    "servos": copy.deepcopy(MICRODUCK_CFG.actuators["servos"]).replace(
        joint_names_expr=[
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        ],
    ),
    # The beak, on its own group and its own controller. It is deliberately **not** a policy output:
    # upstream's networks are fourteen actions with this joint skipped, and the real runtime drives
    # the mouth from higher-level control -- "beak to the floor, one button". A task opens and shuts
    # it by writing this joint's target, not by handing the policy a fifteenth action.
    "beak": ImplicitActuatorCfg(
        joint_names_expr=[MICRODUCK_BEAK_JOINT_NAME],
        effort_limit=MICRODUCK_SERVO_EFFORT_LIMIT,
        velocity_limit=10.0,
        stiffness=2.0,
        damping=0.05,
    ),
}
"""Configuration for the MicroDuck biped with a beak that opens.

**The one MicroDuck model with no upstream MJCF.** The real robot has fifteen servos and the
fifteenth, ``mouth``, drives a grasping beak; every upstream RL model welds that jaw on as a fixed
geom and says so outright -- ``scripts/bake-duck-mesh.py`` notes that "``mouth`` is a servo without
an MJCF joint (the jaw is a fixed geom), so it never appears in a bake". This variant is patched from
``robot_allcollisions.xml`` by
:func:`~scripts.tools.convert_microduck.split_beak_into_hinged_body`, which moves both ``jaw`` geoms
onto a hinged child body at a pivot **measured from the pinned meshes** and re-splits the head's
inertial so the composite mass, centre of mass and inertia are unchanged.

Two consequences worth stating before anyone trains on it:

* **It cannot be accuracy-gated against upstream.** A fifteenth joint changes the state vector, so
  the golden trajectories diverge by construction. Every other task keeps
  :data:`MICRODUCK_ALLCOLLISIONS_USD_PATH` and its gates untouched.
* **The action space is still fourteen.** The beak has its own actuator group precisely so that a
  task's action term, which selects the servos by name, cannot pick it up.

See ``artifacts/microduck/pickplace/BEAK.md`` for the hinge measurement and its cross-checks.
"""


##
# Ball prop.
#
# Upstream's ball-kick task adds one free body to the scene, described by a 15-line MJCF
# (``robot/microduck/ball.xml``) holding a single analytic sphere. There is nothing for the mesh
# importer to carry, so it is authored here rather than converted -- which also means it needs no
# generated asset and is available in a tree that has never run the converter.
##

MICRODUCK_BALL_RADIUS = 0.035
"""Radius [m] of the ball prop: a 70 mm-diameter floorball, as upstream's MJCF comment describes."""

MICRODUCK_BALL_MASS = 0.015
"""Mass [kg] of the ball prop, 2.0 % of the 0.737 kg robot."""

MICRODUCK_BALL_SLIDING_FRICTION = 0.5
"""Sliding friction coefficient the ball's MJCF geom authors.

It is the one contact coefficient upstream customizes; the torsional and rolling ones are left at
MuJoCo's defaults of 0.005 and 1e-4, which are also Newton's shape defaults, so they are not
restated here.

**This coefficient is masked in every contact the ball actually makes**, on both stacks, and that is
upstream's behaviour rather than a port artefact. MuJoCo -- and Newton's MuJoCo Warp solver, which
reproduces the rule -- mixes contact friction as the element-wise *maximum* of the two shapes unless
one carries a higher ``priority``. The ground plane, the robot's shells and the two soles all sit at
1.0, so every ball contact resolves to 1.0 and this 0.5 never binds. It is carried anyway: it is
what the MJCF authors, and a surface slipperier than the ball would use it.
"""


@clone
def _spawn_microduck_ball(
    prim_path: str,
    cfg: sim_utils.SphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the ball prop and give it the inertia of a hollow shell rather than of a solid sphere.

    Upstream's MJCF states the ball's inertia outright -- ``diaginertia="1.225e-5 ..."``, which is
    exactly ``(2/3) m r^2``, a thin spherical shell. Nothing derives that from the geometry: a
    uniform-density sphere of the same mass and radius has ``(2/5) m r^2 = 7.35e-6``, 40 % less, and
    that is what a sphere prim carrying only a mass resolves to. A ball 40 % easier to spin up rolls
    and slides differently off the same kick, so the shell tensor is authored here.

    The tensor is derived from the configuration's own mass and radius rather than restated as a
    constant, so the three numbers cannot drift apart.

    Args:
        prim_path: Prim path or pattern to spawn the ball at.
        cfg: Sphere spawner configuration. Its ``mass_props.mass`` is required.
        translation: Translation w.r.t. the parent prim. Defaults to the origin.
        orientation: Orientation as (w, x, y, z) w.r.t. the parent prim. Defaults to identity.
        **kwargs: Forwarded to :func:`~isaaclab.sim.spawn_sphere`.

    Returns:
        The spawned prim.

    Raises:
        ValueError: If the configuration carries no explicit mass to derive the inertia from.
    """
    if cfg.mass_props is None or cfg.mass_props.mass is None:
        raise ValueError(
            "The MicroDuck ball spawns with an explicit hollow-shell inertia derived from its mass,"
            f" so 'mass_props.mass' must be set. Received mass_props={cfg.mass_props}."
        )
    # ``spawn_sphere`` is itself clone-decorated, but this wrapper has already resolved the pattern
    # down to a single path, so the inner decorator is inert and the inertia below is authored on
    # the prototype prim the outer one copies into every environment.
    prim = sim_utils.spawn_sphere(prim_path, cfg, translation, orientation, **kwargs)
    inertia = 2.0 / 3.0 * cfg.mass_props.mass * cfg.radius**2
    UsdPhysics.MassAPI.Apply(prim).CreateDiagonalInertiaAttr().Set(Gf.Vec3f(inertia, inertia, inertia))
    return prim


##
# Marble prop.
#
# The ball above is upstream's floorball, sized for kicking. Nothing about it is sized for the beak:
# it is 70 mm across and the beak opens 31 mm, so it cannot be picked up even in principle. This is
# the prop for tasks where the robot is meant to *hold* something.
##

MICRODUCK_MARBLE_RADIUS = 0.010
"""Radius [m] of the marble prop: 20 mm across.

Sized against the **beak**, not against the robot. The measured aperture at the mouth line is
**31.1 mm** at full open (``artifacts/microduck/pickplace/BEAK.md``), so 20 mm leaves 11 mm of
clearance: enough that the marble seats between the mandibles rather than being pinched at the very
tip, and large enough to read on a recording of a 25 cm robot.

.. note::

    An earlier version of this constant was 12 mm, sized against a **17.4 mm** figure that was not
    the aperture at all -- it was the minimum distance from any front-half jaw vertex to the upper
    mouth surface, which is dominated by vertices near the hinge where the mandibles barely separate.
    The aperture is measured on the vertices that actually touch when the beak is shut, all of which
    lie at the tip. The correct figure is nearly twice the wrong one.
"""

MICRODUCK_MARBLE_DENSITY = 2500.0
"""Density [kg/m^3] of the marble, soda-lime glass."""

MICRODUCK_MARBLE_MASS = 4.0 / 3.0 * math.pi * MICRODUCK_MARBLE_RADIUS**3 * MICRODUCK_MARBLE_DENSITY
"""Mass [kg] of the marble, ~2.26 g -- derived from its own radius and density rather than stated.

1.4 % of the robot's 0.74 kg, against the ball's 2.0 %. The latch spring is sized from this number
rather than around it; see
:data:`~isaaclab_tasks.contrib.microduck.pickplace.pickplace_env_cfg.MICRODUCK_LATCH_STIFFNESS`.
"""

MICRODUCK_MARBLE_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        radius=MICRODUCK_MARBLE_RADIUS,
        # solver-common schemas rather than the PhysX ones, as the ball uses: this prop has no
        # backend-specific property and the tasks that spawn it run on MJWarp
        rigid_props=sim_utils.RigidBodyBaseCfg(),
        collision_props=sim_utils.CollisionBaseCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=MICRODUCK_MARBLE_MASS),
        physics_material=UsdPhysicsRigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.6,
            # MuJoCo has no restitution coefficient -- its bounce comes out of the contact solver
            # reference -- and a marble that bounced away from the beak would be a different task
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.95)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, MICRODUCK_MARBLE_RADIUS)),
)
"""A 12 mm glass marble, sized so the MicroDuck's beak can actually close on it.

Unlike :data:`MICRODUCK_BALL_CFG` this carries **no custom spawner**. The ball needs one because its
MJCF states a hollow-shell inertia that the geometry does not imply; a marble is solid, and a sphere
prim carrying a mass already resolves to the ``(2/5) m r^2`` a solid sphere has. Its inertia works
out at 3.3e-8 kg m^2, comfortably above the 1e-10 floor Newton silently inflates.

Blue rather than the ball's orange, so a frame with both in it is unambiguous.
"""


MICRODUCK_BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        func=_spawn_microduck_ball,
        radius=MICRODUCK_BALL_RADIUS,
        # solver-common schemas rather than the PhysX ones: the ball has no backend-specific
        # property to set, and this task runs on MJWarp
        rigid_props=sim_utils.RigidBodyBaseCfg(),
        collision_props=sim_utils.CollisionBaseCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=MICRODUCK_BALL_MASS),
        physics_material=UsdPhysicsRigidBodyMaterialCfg(
            static_friction=MICRODUCK_BALL_SLIDING_FRICTION,
            dynamic_friction=MICRODUCK_BALL_SLIDING_FRICTION,
            # MuJoCo has no restitution coefficient -- its bounce comes out of the contact solver
            # reference, which the MJCF leaves at the default. Zero is the matching UsdPhysics value.
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, MICRODUCK_BALL_RADIUS)),
)
"""Configuration for the ball prop of upstream's MicroDuck ball-kick task.

A free-floating, non-articulated 70 mm / 15 g hollow plastic sphere -- a floorball -- resting on the
ground 0.3 m in front of the robot. That initial position only decides where it sits before the
first reset: the ball-kick task places it in front of the kicking foot every episode, in the robot's
own yaw frame.

The MJCF gives the geom no collision masks, so it collides with everything, and upstream applies
none of the robot's collision editing to it. The mass, the radius and the sliding friction are the
MJCF's; the inertia is a hollow shell rather than the solid sphere the geometry would imply, and is
authored by :func:`_spawn_microduck_ball`.
"""
