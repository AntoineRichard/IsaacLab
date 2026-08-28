Added
^^^^^

* Added the MicroDuck reward terms under ``contrib/microduck/mdp``: ``track_linear_velocity``,
  ``track_angular_velocity``, ``upright``, ``pose_mode_switch``, ``head_pose_tracking``,
  ``head_pose_bias_penalty``, ``body_pose_tracking_6d``, ``feet_air_time_windowed``,
  ``foot_clearance``, ``foot_swing_height``, ``foot_slip``, ``body_ang_vel_xy_l2``,
  ``angular_momentum_l2`` and ``self_collision_cost``. They are ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and the mjlab
  velocity template it builds on. Each one documents why the closest stock
  ``isaaclab.envs.mdp`` term could not be reused; the recurring reasons are that upstream folds the
  vertical-velocity and roll/pitch-rate penalties inside the velocity-tracking exponents, and that
  it measures the base in the root link frame rather than the centre-of-mass frame.
* Added the MicroDuck termination term ``robot_state_is_nan`` under ``contrib/microduck/mdp``, which
  terminates an environment whose joint state, root state or named contact-sensor forces have
  stopped being finite.
* Added ``test/test_microduck_rewards.py``, which unit-tests every MicroDuck reward and termination
  term against an environment double with hand-computed expected values. The contact-sensor double
  returns the landing mask as float32, matching what
  ``ContactSensor.compute_first_contact`` produces, so the tests exercise the dtype the Newton
  backend actually returns.
