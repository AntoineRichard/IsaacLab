# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files import UsdFileCfg

LEG_JOINT_NAMES = [
    ".*_hip_roll",
    ".*_hip_yaw",
    ".*_hip_pitch",
    ".*_knee",
    ".*_toe_a",
    ".*_toe_b",
]

ARM_JOINT_NAMES = [".*_arm_.*"]


DIGIT_V4_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=".*",
            stiffness=None,
            damping=None,
        ),
    },
)
