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
* Added the MicroDuck staged curriculum terms ``reward_weight_stages``, ``standing_envs_stages``,
  ``command_range_stages`` and ``event_range_stages`` under ``contrib/microduck/mdp``. Each applies
  the payload of the last stage of a schedule the global environment-step count has passed, which
  the single-stage ``modify_reward_weight`` cannot express.
* Completed the MicroDuck velocity recipe. The two velocity tasks now carry upstream's full reward
  set (velocity and head-pose tracking, the posture reward, the foot clearance, swing-height,
  slip and air-time terms, the self-collision cost and the trunk regularizers), its
  domain-randomization suite (foot friction, encoder bias, trunk mass and inertia, trunk and head
  centre of mass, joint armature and planar pushes), its NaN-state and terrain-bounds
  terminations, its head-pose and body-pose commands, and its seven staged curricula.
  ``IsaacContrib-Velocity-Rough-MicroDuck`` now generates upstream's gentle rough terrain --
  flat ground, sub-centimetre steps, low random grids and shallow slopes -- and progresses through
  its levels; the flat task keeps a plain ground plane.
* Added a dedicated filtered contact sensor to the MicroDuck scene so the self-collision reward can
  read a per-partner force matrix, which the unfiltered ``contact_forces`` sensor does not report.
* Added recipe-parity tests to ``test/test_microduck_env.py``, which compare the assembled
  configuration term by term against the upstream reward, randomization, curriculum, command and
  terrain tables without launching the simulator.
* Completed the MicroDuck observation contract. The actor group is now the 61-wide vector the
  deployed policy expects -- base angular velocity, projected gravity, joint positions, joint
  velocities, the previous action, then the ``[twist(3), head_pose(4), body_pose(6)]`` command
  block -- read through the encoder bias, the IMU misalignment and the bus latency the robot's own
  sensors have, with the joint blocks pinned to upstream's servo order. The velocity tasks also
  gained a privileged ``critic`` group carrying the same terms uncorrupted plus the base linear
  velocity and the four foot terms, which the RSL-RL configuration now feeds to the value function.
  The action term is ``BiasedJointPositionActionCfg``, which closes the encoder-bias loop the
  biased observation opens.
* Added observation-contract tests to ``test/test_microduck_env.py``, which measure both group
  widths term by term against the deploy layout and check that each block of the flat actor vector
  carries the signal the layout names, including the servo order inside the joint block.
