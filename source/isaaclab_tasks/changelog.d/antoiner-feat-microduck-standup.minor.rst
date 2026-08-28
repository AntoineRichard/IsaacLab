Added
^^^^^

* Added the contributed MicroDuck stand-up task ``IsaacContrib-StandUp-Flat-MicroDuck`` under
  ``contrib/microduck/standup``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. An episode starts the robot face down, face up, folded into the sitting
  keyframe a sit policy hands off, or already standing, and asks it to reach and hold the standing
  keyframe. It shares the velocity tasks' 61-wide deployed observation contract, so one runtime can
  feed either policy from the same buffer, and it spawns
  :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG` -- a robot that pushes itself off the floor
  needs the trunk, hip, shin and head colliders the walking model does not have.

  Its self-collision sensor reproduces upstream's, which filters the whole trunk subtree against
  itself, through the Newton backend's shape-level ``sensor_shape_prim_expr`` and
  ``filter_shape_prim_expr``: the model's ten enabled colliders against each other, many-to-many.
  Body-level filtering cannot express that, because Isaac Lab resolves a per-partner force matrix
  only for a sensor whose ``prim_path`` matches a single prim per environment. The sensor therefore
  sees limb-against-limb contacts with no trunk at either end -- a deep knee fold drives each shin
  into the hip shell on its own side, and a crossed pose brings sole against sole. The reward
  saturates the result, because sensing both sides of a contact reports it twice.
* Added ``reset_ground_state`` under ``contrib/microduck/mdp``, the reset event that *is* the
  stand-up task's episode distribution: it samples one of four ground keyframes -- face down, face
  up, the sitting keyframe with joint and tilt noise, or standing -- and writes the root height,
  orientation and, for the sitting bucket, the joint pose. Being a mixture of named keyframes rather
  than one continuous band, it has no stock counterpart.
* Added the MicroDuck stand-up reward terms under ``contrib/microduck/mdp``:
  ``joint_pose_gaussian``, ``joint_pose_l1``, ``root_height_gaussian``, ``root_height_l1``,
  ``com_upward_velocity``, ``trunk_vertical_accel_penalty``, ``body_ang_vel_at_height``,
  ``body_upright_linear``, ``upright_gaussian_at_height``, ``standing_composite_score`` and
  ``joint_torque_rate_l2``. Each documents why the closest stock ``isaaclab.envs.mdp`` term could
  not be reused; the recurring reasons are that the rise recipe stacks a wide and a narrow layer of
  the same attractor, that its L1 bootstraps have to keep a constant gradient where the Gaussians
  are flat, and that its motion penalties are gated on trunk height and tilt so they cannot tax the
  recovery they are meant to smooth.
* Added ``event_param_stages`` under ``contrib/microduck/mdp``, a staged curriculum that
  shallow-merges named parameters into a live event term. It drives the stand-up task's
  reset-distribution ramp and its push-magnitude ramp, and reproduces upstream's two different step
  comparison operators through its ``inclusive`` flag.
* Added ``test/test_microduck_standup_env.py``, which compares the assembled configuration term by
  term against the upstream reward, event, curriculum, command and observation tables without
  launching the simulator, pins the reset-event order the spawn distribution depends on, and builds,
  resets and steps the task -- including a check that every ground-pose bucket actually spawns. It
  skips the simulator tests when the generated all-collisions MicroDuck USD is absent.

Changed
^^^^^^^

* ``head_pose_bias_penalty`` under ``contrib/microduck/mdp`` gained an optional upright gate
  (``gate_height_low``, ``gate_height_high``, ``gate_tilt_full_deg``, ``gate_tilt_zero_deg``), which
  suppresses both the accumulated error and the returned penalty while the robot is low or tilted.
  The gate is off by default, so the velocity tasks are unaffected.
* ``body_pose_tracking_6d`` under ``contrib/microduck/mdp`` gained an ``axis_weights`` parameter and
  now returns the weighted rather than the plain mean of its six per-axis Gaussians. It defaults to
  equal weights, so the velocity tasks are unaffected.
* ``UniformPoseDeltaCommandCfg`` under ``contrib/microduck/mdp`` gained ``zero_command_prob``, the
  probability that a resample yields the exact all-zero command, which keeps the deployment idle
  case in the training distribution. It defaults to 0.0, so the velocity tasks are unaffected.
