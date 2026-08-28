Added
^^^^^

* Added the contributed MicroDuck task family under ``contrib/microduck``, with the velocity-tracking
  tasks ``IsaacContrib-Velocity-Flat-MicroDuck`` and ``IsaacContrib-Velocity-Rough-MicroDuck`` and an
  RSL-RL PPO configuration carrying the upstream hyper-parameters. The tasks are ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and run on the
  Newton MJWarp backend.
* Added ``test/test_microduck_env.py``, which builds, resets and steps both tasks. It skips when the
  generated MicroDuck USD is absent, as the asset tests do.
