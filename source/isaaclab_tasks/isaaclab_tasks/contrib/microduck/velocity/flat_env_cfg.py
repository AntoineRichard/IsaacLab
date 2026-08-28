# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-terrain variant of the MicroDuck velocity-tracking environment."""

from isaaclab.utils.configclass import configclass

from .velocity_env_cfg import MicroDuckVelocityRoughEnvCfg


@configclass
class MicroDuckVelocityFlatEnvCfg(MicroDuckVelocityRoughEnvCfg):
    """MicroDuck velocity-tracking environment on flat ground."""

    def __post_init__(self):
        super().__post_init__()

        # physics -- a plane and a two-footed 0.74 kg robot need far fewer constraint and contact
        # slots than rough terrain. Profiling 2048 and 4096 environments under random actions --
        # including a worst case with the tilt termination removed so the robots sprawl -- peaks at
        # 10 contacts and 54 constraints per environment. Only three geometries collide (the ground
        # plane and the two foot soles), so that peak is structural: 4 pyramidal rows x 10 contacts
        # + 14 joint limits = 54. ``nconmax`` is a per-environment share of one shared contact
        # buffer and so cannot overflow at the structural peak; ``njmax`` is a hard per-environment
        # cap and carries margin above it.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 64
        newton_mjwarp.solver_cfg.nconmax = 10
        # upstream's flat solver profile (reference section 1); its rough profile raises them
        newton_mjwarp.solver_cfg.iterations = 10
        newton_mjwarp.solver_cfg.ls_iterations = 20
        self.sim.physics.default = newton_mjwarp

        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # a plane has no difficulty levels to progress through, and the term reads the terrain
        # generator's configuration, so it cannot merely sit inert here. Upstream deletes it on
        # flat terrain for the same reason (reference section 2.8).
        self.curriculum.terrain_levels = None
