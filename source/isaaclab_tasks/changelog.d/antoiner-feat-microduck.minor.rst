Added
^^^^^

* Added the contributed MicroDuck task family under ``contrib/microduck``, with the velocity-tracking
  tasks ``IsaacContrib-Velocity-Flat-MicroDuck`` and ``IsaacContrib-Velocity-Rough-MicroDuck`` and an
  RSL-RL PPO configuration carrying the upstream hyper-parameters. The tasks are ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and run on the
  Newton MJWarp backend.
* Added ``test/test_microduck_env.py``, which builds, resets and steps both tasks. It skips when the
  generated MicroDuck USD is absent, as the asset tests do.
* Added the MicroDuck command terms ``UniformPoseDeltaCommand`` and ``MicroDuckVelocityCommand``
  under ``contrib/microduck/mdp``. ``UniformPoseDeltaCommand`` is an N-dimensional uniform
  pose-delta command whose width follows its per-dimension range tuple, used for the 4-D head-joint
  and 6-D body-pose commands. ``MicroDuckVelocityCommand`` extends the stock uniform velocity
  command with a
  forward-only bucket (``rel_forward_envs``) and a turn-in-place bucket
  (``rel_turn_in_place_envs``), and the velocity tasks now use it for their ``base_velocity`` term.
* Added the MicroDuck observation terms under ``contrib/microduck/mdp``, which model the sensor
  imperfections the deployed policy has to tolerate: ``joint_pos_rel_biased`` reports the joint
  positions through a constant per-robot encoder bias, ``base_ang_vel_imu_misaligned`` and
  ``projected_gravity_imu_misaligned`` read the IMU through a constant per-robot mounting
  misalignment, ``delayed_observation`` wraps another term in a per-environment bus latency whose
  lag is re-drawn on a configurable period, and ``foot_contact_forces_safe``, ``foot_air_time_safe``
  and ``foot_height_safe`` guard the critic's sensor reads against non-finite values.
* Added ``BiasedJointPositionAction`` and ``randomize_encoder_bias`` under
  ``contrib/microduck/mdp``. The action term subtracts the encoder bias the biased observation
  adds, closing the loop the way the servo's own position loop does; both read the shared bias the
  ``randomize_encoder_bias`` startup event samples, which ``encoder_bias`` resolves.
* Added ``test/test_microduck_mdp.py``, which unit-tests the MicroDuck command, observation and
  action terms against an environment double, including the precedence between the standing,
  forward-only and turn-in-place buckets and the delay term's lag-resampling cadence.
