# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place on the MicroDuck model whose beak opens.

The flat task spawns :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG`, whose jaw is welded shut
because every upstream RL model welds it shut. The real robot has a fifteenth servo that opens a
grasping beak, so this variant spawns :data:`~isaaclab_assets.MICRODUCK_BEAK_CFG` instead and drives
that servo from the latch.

What changes, and it is deliberately only three things:

* the robot, and with it a fifteenth joint;
* an interval event that opens the beak on approach and shuts it on the object;
* nothing else. The observations, the actions, the rewards, the terminations and the curriculum are
  the flat task's, inherited rather than restated.

**The action space is still fourteen.** The beak is not a policy output on the real robot -- its
networks are fourteen actions with the mouth skipped, driven from higher-level control -- so it is
not one here either. :data:`~isaaclab_assets.MICRODUCK_BEAK_CFG` puts it in its own actuator group so
that the action term, which selects the servos by name, cannot pick it up; a checkpoint trained on
the flat task therefore loads and runs against this variant unchanged, which is what makes it usable
for looking at the beak.

.. warning::

    This variant **cannot be accuracy-gated against upstream**. A fifteenth joint changes the state
    vector, so the golden trajectories diverge by construction. It is the port's first deliberate
    divergence from the pinned models, and it is confined to this file and to its own asset -- every
    other task keeps ``microduck_allcollisions.usd`` and its gates untouched.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.mdp as mdp

from isaaclab_assets.robots.microduck import (
    MICRODUCK_BEAK_CFG,
    MICRODUCK_BEAK_CLOSED,
    MICRODUCK_BEAK_JOINT_NAME,
    MICRODUCK_BEAK_OPEN,
)

from .pickplace_env_cfg import (
    MICRODUCK_LATCH_RADIUS,
    MICRODUCK_MOUTH_TIP_OFFSET,
    MicroDuckPickPlaceFlatEnvCfg,
)

MICRODUCK_BEAK_OPEN_DISTANCE = 0.12
"""Mouth-tip-to-object distance [m] at which the beak starts to gape.

Comfortably wider than :data:`~isaaclab_tasks.contrib.microduck.pickplace.pickplace_env_cfg.
MICRODUCK_LATCH_RADIUS`, and it has to be: a beak that only opened once the object was already
inside the latch radius would still be shut at the moment it is supposed to be closing on something.
Roughly two head-lengths, so the gape is visibly a reach rather than a twitch.
"""


@configclass
class MicroDuckPickPlaceBeakFlatEnvCfg(MicroDuckPickPlaceFlatEnvCfg):
    """The flat pick-and-place task on the robot whose beak opens."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.scene.robot = MICRODUCK_BEAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Every control step, for every environment, as the latch itself is. Declared after the
        # latch update so the beak reads the state that step's transitions wrote rather than the
        # previous step's -- the two are one event seen from two sides, and a beak that shut a step
        # late would be visibly wrong on exactly the frame anyone looks at.
        self.events.drive_beak = EventTerm(
            func=mdp.drive_beak_from_latch,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["jaw_soft"],
                    joint_names=[MICRODUCK_BEAK_JOINT_NAME],
                    preserve_order=True,
                ),
                "object_cfg": SceneEntityCfg("object"),
                "mouth_offset_b": MICRODUCK_MOUTH_TIP_OFFSET,
                "open_distance": MICRODUCK_BEAK_OPEN_DISTANCE,
                "closed_angle": MICRODUCK_BEAK_CLOSED,
                "open_angle": MICRODUCK_BEAK_OPEN,
            },
        )
        assert MICRODUCK_BEAK_OPEN_DISTANCE > MICRODUCK_LATCH_RADIUS

        # Beak scale rather than body scale: the gape is 17 mm on a 25 cm robot.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.45, 0.45, 0.25), lookat=(0.0, 0.0, 0.10))
