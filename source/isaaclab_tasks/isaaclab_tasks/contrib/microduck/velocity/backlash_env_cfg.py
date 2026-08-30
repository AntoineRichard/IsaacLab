# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gear-backlash twin of the MicroDuck flat velocity environment.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
whose ``make_backlash_variant`` (``tasks/backlash.py``) derives a ``-Backlash-`` twin of every
MicroDuck task. The analysis this port is written against is
``artifacts/microduck/backlash_investigation.md``; sections cited below as "report section N" are
its.

The plant is :data:`~isaaclab_assets.MICRODUCK_BACKLASH_CFG`: an unactuated
``passive_<servo>_backlash`` hinge in series with each of the 14 servos, carrying plus or minus one
degree of free travel. Nothing drives those hinges -- their range *is* the gear teeth -- so the
robot has 28 joints and still 14 actions, and the deployed 61-wide observation layout is unchanged.
What changes is what the numbers in the two joint blocks *mean*: the servo's magnetic encoder sits
on the far side of the gearbox, so the policy reads ``qpos[servo] + qpos[backlash]`` rather than the
motor angle, and the BAM firmware closes its position loop on the same sum.

This is an A/B experiment on the plant, not a new task. It is therefore derived from
:class:`~isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg.MicroDuckVelocityFlatEnvCfg` and
changes as little as possible, so that a difference in a training curve is attributable to the gear
play and to nothing else.

Upstream makes four edits; this port makes two of them, plus a solver-budget change that is ours
(report section 1.4):

.. list-table::
    :header-rows: 1

    * - Upstream edit
      - Here
    * - 1. Swap in the matching backlash robot
      - Same, :data:`~isaaclab_assets.MICRODUCK_BACKLASH_CFG` (the *walking* collision model, so an
        A/B against the base task is unconfounded).
    * - 2. Remap ``joint_pos`` / ``joint_vel`` in both observation groups, injecting a servo-only
        selection where a term had none
      - Remap only. Every selection in this package is already spelled out as 14 exact joint names,
        and name resolution matches in full, so none of them can pick up a ``passive_`` joint --
        upstream's injected ``^(?!passive_).*`` has nothing left to do.
    * - 3. Scope ``dof_pos_limits`` to the servos
      - Same. Its default selection is every joint, and the play hinges spend their life pinned
        against their limits -- that is what backlash *is* -- so leaving it would charge a permanent
        soft-limit penalty no policy can avoid.
    * - 4. Disambiguate the posture reward's standard-deviation patterns
      - **Not needed.** Upstream's patterns are resolved against a selection that includes the play
        hinges, where ``passive_left_hip_yaw_backlash`` matches both ``.*hip_yaw.*`` and the roller
        recipe's ``.*passive_.*``. Here the posture reward selects the 10 leg joints by exact name,
        so no play hinge is ever in the selection to be ambiguous about.
    * - --
      - **Added:** ``njmax`` is raised, because the play hinges are 14 always-active limit rows per
        environment that the base task's budget was measured without. See :meth:`__post_init__`.

Two properties of the play hinges are load-bearing and are asserted rather than assumed by
``source/isaaclab_tasks/test/test_microduck_backlash_env.py``: the observation widths stay 61 and 14,
and no term selection in the whole configuration resolves a ``passive_`` joint -- except one, which
is deliberate and named below.

**Audit: every term that reads joint state, and which side of the play it reads.** Upstream's
consistency invariant (report section 1.3) is that a reward tracking a quantity the policy observes
through the play must measure the same view, or the robot is paid twice for the same degree of
freedom. The audit is term by term, and it is the reason this module is short:

.. list-table::
    :header-rows: 1

    * - Term
      - What it reads
      - View
    * - ``observations.policy.joint_pos`` / ``joint_vel``
      - the 14 servos
      - **Encoder.** Remapped here; this is the deployed contract.
    * - ``observations.critic.joint_pos`` / ``joint_vel``
      - the 14 servos
      - **Encoder.** Remapped here too: the critic must value the state the actor acts on.
    * - ``rewards.head_pose_tracking``, ``rewards.head_pose_bias``
      - the 4 head servos, against a command
      - **Encoder**, resolved inside the terms -- which is where upstream resolves it, so there is
        no configuration edit here. These two are exactly the invariant's case: they price a
        quantity the actor observes.
    * - ``rewards.pose``
      - the 10 leg servos, against the stand pose
      - **Motor**, as upstream leaves it. The play is 1 degree against tolerances of 2.9 to 23
        degrees, and the reward holds the *robot's* posture rather than scoring something the policy
        was told to track.
    * - ``rewards.dof_pos_limits``
      - joint positions against the soft limits
      - **Motor**, and scoped to the servos -- see edit 3 above.
    * - ``terminations.nan_state``
      - every joint's position and velocity
      - **Both, deliberately.** It is a health check on the articulation, so the play hinges are
        exactly what it must not be blind to.
    * - ``actions.joint_pos``, ``rewards.action_rate_l2``
      - the action buffers
      - Neither: they read commands, not state. The action term writes targets to the 14 servos.
    * - ``events.reset_robot_joints``
      - writes every joint
      - Its offsets are zero-width, so the play hinges reset to their centred default of 0.
    * - ``events.encoder_bias`` / ``randomize_encoder_bias``
      - allocates and fills one bias per joint
      - Only the servo slots are ever read: one encoder per servo means one calibration error per
        servo, and the play summand stays raw (upstream ``mdp.py:6191-6193``).
    * - ``events.randomize_armature``
      - scales every joint's armature
      - **Includes the play hinges, deliberately** -- see :class:`EventsCfg` note in
        :meth:`__post_init__`.

Everything else -- the two velocity-tracking rewards, ``upright``, ``body_ang_vel``,
``angular_momentum``, the four foot terms, ``self_collisions``, ``body_pose_tracking``, and every
termination but ``nan_state`` -- reads root, body or sensor state and never a joint.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.mdp as mdp

from isaaclab_assets import MICRODUCK_BACKLASH_CFG

from .flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from .velocity_env_cfg import MICRODUCK_JOINT_NAMES

MICRODUCK_BACKLASH_NJMAX = 96
"""Per-environment MuJoCo Warp constraint budget [rows] on the played plant.

Measured rather than inherited, and the measurement is the reason this module touches the solver at
all. The base flat task ships ``njmax = 64`` against a structural peak of 54 -- 4 pyramidal rows per
contact times 10 contacts, plus the 14 servo limits. A play hinge lives *on* its limits, because the
limits are the gear teeth, so this plant adds 14 permanently active rows and nothing else: the
structural bound is 68 with the same 10 contacts.

Profiled the way the base task's budget was, under random actions with the tilt termination dropped
so the robots sprawl: **peak 65 constraints and 10 contacts per environment at 256 environments, 66
and 10 at 2048** -- agreeing to one across the two scales, which is what says the tail has been
sampled. Logs: ``artifacts/microduck/backlash/profile_backlash_{256,2048}envs_worstcase.log``, from
``artifacts/microduck/profile_microduck_contacts.py``.

The shipped 64 therefore does not merely run tight on this plant, the peak overflows it; the
feasibility probe measured the overshoot degrading by 3.2x when it does. ``nconmax`` is left at the
base task's 10: the play hinges are joints, not colliders, and the contact peak did not move.

Both profiles were taken at or below 2048 environments, so the value is sized on a sample that stops
short of the training scale this task is meant to run at.
"""


@configclass
class MicroDuckVelocityBacklashFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """MicroDuck flat velocity-tracking environment on the gear-backlash plant.

    The base task's recipe verbatim, on a robot whose servos have a degree of gear play and whose
    encoders read through it. See the module docstring for the edit-by-edit comparison against
    upstream and for the term-by-term audit of which side of the play each term reads.
    """

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # 1. the plant. The walking collision model, so that an A/B against the base task differs
        # in the gear play and in nothing else.
        self.scene.robot = MICRODUCK_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. the encoder view, in both groups. The actor because that is what the deployed runtime
        # feeds it, and the critic because a value function that saw the motor angle would be
        # valuing a state the actor cannot observe or act on. The selections are untouched: they
        # already name the 14 servos, and the widths therefore stay 14 and 14.
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.func = mdp.joint_pos_rel_backlash
            # the actor's velocity block is wrapped in the bus-latency term, which holds the term it
            # delays in its own parameters
            if group.joint_vel.func is mdp.delayed_observation:
                group.joint_vel.params["term_func"] = mdp.joint_vel_rel_backlash
            else:
                group.joint_vel.func = mdp.joint_vel_rel_backlash

        # 3. the soft-limit penalty, scoped to the servos. The term's default selection is every
        # joint, and a play hinge rides its limits by construction -- that is the gear teeth
        # touching -- so an unscoped penalty is a constant tax of 14 saturated rows that no policy
        # can escape and that would drown the signal the term exists for.
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True
        )

        # The armature randomization is deliberately left at ``.*`` and therefore *does* scale the
        # play hinges' 0.001 armature, which upstream also does: its own backlash variant edits the
        # reward set and never the event set. That armature is solver conditioning rather than a
        # physical rotor inertia, so scaling it by plus or minus ten percent randomizes how stiffly
        # the teeth are modelled rather than a property of the robot. It is reproduced rather than
        # narrowed because the deployed comparison is against upstream's plant, and narrowing it is
        # a retune with its own training run.

        # The solver budget, which the play hinges change and the base task's profiling did not see.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = MICRODUCK_BACKLASH_NJMAX
        self.sim.physics.default = newton_mjwarp
