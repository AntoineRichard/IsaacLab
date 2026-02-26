# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers.manager_term_cfg import ObservationGroupCfg as ObsGroup
from isaaclab.managers.manager_term_cfg import ObservationTermCfg as ObsTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.envs.mdp.observations import (
    base_ang_vel,
    base_lin_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)


@configclass
class AgileTeacherPolicyObservationsCfg(ObsGroup):
    """Observation specifications for the Agile lower body policy.

    Note: This configuration defines only part of the observation input to the Agile lower body policy.
    The lower body command portion is appended to the observation tensor in the action term, as that
    is where the environment has access to those commands.
    """

    base_lin_vel = ObsTerm(
        func=base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    base_ang_vel = ObsTerm(
        func=base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    projected_gravity = ObsTerm(
        func=projected_gravity,
        scale=1.0,
    )

    joint_pos = ObsTerm(
        func=joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
            ),
        },
    )

    joint_vel = ObsTerm(
        func=joint_vel_rel,
        scale=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
            ),
        },
    )

    actions = ObsTerm(
        func=last_action,
        scale=1.0,
        params={
            "action_name": "lower_body_joint_pos",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
