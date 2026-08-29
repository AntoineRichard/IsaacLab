# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-commanded crouch-glide trick task for the roller-skating MicroDuck biped."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-RollerCrouch-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rollercrouch_env_cfg:MicroDuckRollerCrouchFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MicroDuckRollerCrouchPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
