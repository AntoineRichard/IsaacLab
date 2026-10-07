.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

:orphan:

MicroDuck Walking
=================

``IsaacContrib-Velocity-Flat-MicroDuck`` trains the walking MicroDuck on a plane with Newton MJWarp and
:ref:`BAM servos <actuators-bam>`, porting the recipe from
`microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_.

Train with RSL-RL:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 4096

Play an RSL-RL checkpoint:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab play --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 16 \
     --checkpoint /path/to/model.pt --visualizer newton_gl

Policy interface
----------------

The policy runs at 50 Hz (four 0.005 s physics steps). Its 14 actions are joint-position offsets [rad]
from the standing pose, ordered left leg, neck and head, then right leg, independent of the USD joint order.

The actor reads 61 values: base angular velocity (3), projected gravity (3), joint-position offsets (14),
joint velocities (14), previous actions (14), velocity commands (3), head-pose commands (4), and body-pose
commands (6). The critic reads 76 values, including privileged foot state.

Training randomizes encoder bias, IMU misalignment, observation delay, BAM friction, mass, center of mass,
and armature. Playback disables observation noise and pushes.

Rough terrain
-------------

``IsaacContrib-Velocity-Rough-MicroDuck`` uses
`microduck_rl's gentle terrain mix <https://github.com/pollen-robotics/microduck_rl/blob/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py>`_:
flat ground, stairs up to 1.5 cm, random grids up to 1 cm, and gentle slopes in ten difficulty levels, advanced by
successful walking episodes. The actor stays blind to terrain; two downward rays per foot give the critic and the foot-clearance
rewards the ground height.

Terrain promotion requires a full episode without a fall, at least 1 m of commanded and actual walking,
and at least 5 s of translation commands (also at least 25% of the episode). Integrated XY tracking error
must be at most 50% of commanded path length, and mean yaw-rate error at most 0.5 rad/s. Command changes
and reversals are accounted for throughout the episode. Falls demote; eligible episodes also demote when
XY error exceeds 80% or yaw-rate error exceeds 0.8 rad/s. Other episodes retain their level.

These experimental thresholds are parameters of ``curriculum.terrain_levels``. Logs under
``Curriculum/terrain_levels/`` report mean level, promotion/demotion fractions, walking eligibility,
and actual walking distance. This replaces the previous 4 m net-displacement gate; existing checkpoints
can still load, but resumed training follows the new progression rule.

AGILE combined
--------------

Select ``presets=agile_combined`` on either the flat or rough task to use the combined tracking recipe:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Rough-MicroDuck --num_envs 4096 \
     --max_iterations 50000 --run_name agile_combined presets=agile_combined

This preset tracks horizontal velocity in the heading frame and yaw rate around world Z, with
tolerances of 0.15 m/s and 0.35 rad/s and reward weights of 4.0 each. The leg-pose reward applies
only to standing commands. The action-rate penalty starts at -0.1 and reaches -0.2, -0.4, -0.6,
-0.8, and -1.0 after 3,000, 4,000, 5,000, 6,000, and 7,000 PPO iterations, respectively, assuming
the default 24 rollout steps per iteration.

The rough task retains the tracking-and-survival terrain curriculum described above. The preset
adds no body-height reward or policy input. Combine it with gearbox play using
``presets=agile_combined,backlash``; use the same presets when playing the resulting checkpoint.

Newton model-update batching is enabled by the backend. The MJWarp Cholesky launch size is tunable
with ``env.sim.physics.solver_cfg.cholesky_solve_block_dim=64``, the setting used for the L40S
experiments; the best value can vary with GPU and environment count.

Backlash
--------

Add ``presets=backlash`` to either task to train the robot with ±1° of gearbox play in series with each servo:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 4096 presets=backlash

Encoders and head-pose rewards then measure servo plus play angle, only servo joints incur the soft-limit penalty,
and the policy interface is unchanged, so existing walking policies load.
