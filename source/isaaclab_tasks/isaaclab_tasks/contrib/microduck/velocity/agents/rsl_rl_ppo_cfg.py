# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck velocity tasks.

Every hyper-parameter is upstream's; see ``artifacts/microduck/upstream_reference.md`` section 6,
"RL agent cfg".
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from ..velocity_env_cfg import MICRODUCK_STEPS_PER_ITERATION


@configclass
class MicroDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the MicroDuck velocity-tracking tasks."""

    # tied to the curriculum stage tables' step math, not restated as its own literal
    num_steps_per_env = MICRODUCK_STEPS_PER_ITERATION
    max_iterations = 50000
    save_interval = 250
    experiment_name = "microduck_velocity"
    # Asymmetric actor-critic, as upstream trains it: the actor reads the corrupted 61-wide deploy
    # vector and the critic reads the privileged group, which adds the base linear velocity and the
    # foot terms the robot has no sensor for.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
