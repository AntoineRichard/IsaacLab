Added
^^^^^

* Added the contributed MicroDuck swizzle task ``IsaacContrib-Velocity-Swizzle-MicroDuck`` under
  ``contrib/microduck/velocity``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The *swizzle* is the beginner's skating gait -- both blades stay down and
  the legs open and close in mirror -- which is precisely the degenerate waddle the roller-skating
  task spends six reward terms suppressing. It is that recipe with the anti-swizzle half deleted and
  the symmetry paid for instead, so it is registered next to it in the velocity family, as upstream
  registers it.
* Added ``leg_symmetry_reward``, ``grounded_reward`` and ``heading_tracking_reward`` under
  ``contrib/microduck/mdp``. The symmetry condition is ``q_left + q_right ~= 0`` rather than
  ``q_left ~= q_right``, because the model uses mirrored left/right sign conventions; the two legs
  are selected as a pair of order-preserving joint selections, so a joint is always compared against
  its own mirror.
* Added ``test/test_microduck_swizzle_env.py``. The task is a delta on the roller recipe, so the
  tests transcribe the delta -- the reward swap, the two re-opened command clamps and the four new
  curricula -- and assert that the scene, the sensors, the events, the terminations, the action space
  and the physics preset are compared object-for-object against the roller task rather than
  re-transcribed. The simulator-backed tests assert both observation widths and evaluate the
  symmetry reward under physics against a mirrored and a scissored leg pose; they skip when the
  generated roller MicroDuck USD is absent.

Changed
^^^^^^^

* Reproduced upstream's narrowing of the ``pose`` reward from all eighteen joints to the ten leg
  joints, which changes the *value* of that reward and not only its scope: the kernel is a mean over
  the selection, so the same joint errors now divide by ten instead of by eighteen. Upstream narrows
  it so the neck and head are free to follow the head-pose command, and the roller task's ``999.0``
  tolerance entries -- its way of neutralizing the wheels inside an eighteen-joint mean -- go with
  the wheels rather than being left dead in the dictionaries.
* Reproduced upstream's overlap between the two heading rewards. ``heading_hold`` pays for staying at
  the spawn heading and ``heading_tracking`` for reaching a resampled one, and between iterations
  1750 and 2500 both weights are non-zero, so the policy is paid for both at once. Nothing upstream
  says so; the deployed policy was trained through the crossover, and the port keeps it and records
  it in a test rather than squaring the schedules off.
* Kept the family's NaN-guard norm, which is a deviation from upstream on this task as on its
  siblings: upstream registers ``nan_state`` with an empty sensor list here and this port names the
  foot contact sensor and uses the two NaN-safe critic terms, inherited from the roller task.
* Inherited the roller task's *measured* MuJoCo Warp contact budget rather than re-profiling. That is
  a bound rather than an assumption: it was profiled under random actions with the tilt termination
  removed, so the robots sprawl and every collider reaches the floor, and a swizzle keeps four tires
  down and nothing else.
* Reproduced upstream's inherited ``max_iterations`` of 50 000 and its disabled symmetry
  augmentation, both of which arrive through a two-field copy of the roller runner. The second is
  worth naming: this is the one task in the family whose defining property *is* left-right symmetry,
  and upstream trains it without the augmentation that would exploit that.
