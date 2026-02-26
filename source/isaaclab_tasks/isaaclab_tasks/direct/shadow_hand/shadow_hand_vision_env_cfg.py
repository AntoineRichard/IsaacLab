# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.utils.configclass import configclass

from .feature_extractor import FeatureExtractorCfg
from .shadow_hand_env_cfg import ShadowHandEnvCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg


@configclass
class ShadowHandVisionEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1225, env_spacing=2.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, -0.35, 1.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="world"),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=120,
        height=120,
    )
    feature_extractor = FeatureExtractorCfg()

    # env
    observation_space = 164 + 27  # state observation + vision CNN embedding
    state_space = 187 + 27  # asymettric states + vision CNN embedding


@configclass
class ShadowHandVisionEnvPlayCfg(ShadowHandVisionEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)
