Added
^^^^^

* Added the contributed MicroDuck task family under ``contrib/microduck``, with the velocity-tracking
  tasks ``IsaacContrib-Velocity-Flat-MicroDuck`` and ``IsaacContrib-Velocity-Rough-MicroDuck`` and an
  RSL-RL PPO configuration carrying the upstream hyper-parameters. The tasks are ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and run on the
  Newton MJWarp backend.
* Added ``test/test_microduck_env.py``, which builds, resets and steps both tasks. It skips when the
  generated MicroDuck USD is absent, as the asset tests do.
* Added the MicroDuck command terms ``JointPoseCommand`` and ``MicroDuckVelocityCommand`` under
  ``contrib/microduck/mdp``. ``JointPoseCommand`` is an N-dimensional uniform pose-delta command
  whose width follows its per-dimension range tuple, used for the 4-D head-joint and 6-D body-pose
  commands. ``MicroDuckVelocityCommand`` extends the stock uniform velocity command with a
  forward-only bucket (``rel_forward_envs``) and a turn-in-place bucket
  (``rel_turn_in_place_envs``), and the velocity tasks now use it for their ``base_velocity`` term.
* Added ``test/test_microduck_mdp.py``, which unit-tests the MicroDuck command terms against an
  environment double, including the precedence between the standing, forward-only and
  turn-in-place buckets.
