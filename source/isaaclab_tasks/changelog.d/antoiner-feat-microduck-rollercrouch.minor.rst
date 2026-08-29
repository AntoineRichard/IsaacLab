Added
^^^^^

* Added the contributed MicroDuck crouch-glide task ``IsaacContrib-RollerCrouch-Flat-MicroDuck``
  under ``contrib/microduck/rollercrouch``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The robot arrives rolling on its skates, folds into a deep crouch, holds it
  for two seconds while it glides on the momentum it brought, and stands back up -- four segments on
  a five-second clock whose phase the policy is told, so a twenty-second episode is four complete
  cycles. Unlike the ground-pick gesture the clock always starts at phase 0, which is what the
  deployed runtime does and what stops the policy learning "stay low" from spawns that are already
  low.
* Added ``crouch_glide_pose_gaussian``, ``crouch_glide_pose_l1``, ``forward_speed_reward`` and
  ``crouch_forward_lean`` under ``contrib/microduck/mdp``. The two pose terms score all fourteen
  servos, head included, against a phase-blended interpolation between a standing and a crouched
  keyframe; ``forward_speed_reward`` is deliberately command-independent, because on a phase command
  there is no speed to gate on and the glide has to survive the fold.
* Added ``test/test_microduck_rollercrouch_env.py``. Besides the term-for-term recipe parity, two of
  its tests record known upstream defects rather than assert correct behavior -- see below -- and the
  simulator-backed ones assert both observation widths, evaluate the pose reward across the cycle it
  is defined on, and check the per-environment friction storage with the wheel event removed. They
  skip when the generated roller MicroDuck USD is absent.

Changed
^^^^^^^

* Reproduced upstream's ``CROUCH_POSE`` verbatim although two of its fourteen targets lie outside the
  compiled model's hard joint limits -- ``neck_pitch`` 2.7 degrees past its upper stop and
  ``right_knee`` 1.1 degrees past its lower one (upstream issue draft 018). The pose was read off the
  physical robot and never checked against the simulation model, so both pose rewards charge a
  residual on those joints that no policy can zero and the "hold the crouch" reward can never
  saturate. It is reproduced because the deployed policy was trained against these targets, and it is
  **recorded by two tests** rather than left implicit: one reads the compiled model's joint ranges
  and pins the violation to exactly those two joints and those two magnitudes, and one scores the
  keyframe clamped into the stops -- the best a policy can physically reach -- and shows it strictly
  below the unreachable target.
* Reproduced upstream's frictionless-bearing randomization verbatim: the wheel-friction event ships a
  degenerate ``(0.0, 0.0)`` range and, unlike the skating task's, no curriculum ramps it, so this
  environment trains on perfectly frictionless bearings for its whole run (upstream issue draft 017).
  That is a sim-to-real optimism in exactly the degree of freedom a glide depends on, and it is kept
  because the deployed policy was trained with it; correcting it is a retune with its own run.
  The event is **not** removed as dead code even though it randomizes nothing: upstream separately
  omits the startup event that registers the BAM actuator's friction fields for per-world expansion,
  and this degenerate event is upstream's sole declarer of one of those fields, so deleting it there
  breaks the actuator at the first multi-environment step. Isaac Lab's BAM actuator owns
  per-environment friction storage unconditionally, so that interlock does not exist here -- which is
  asserted, with the event removed, rather than assumed.
* Did not carry upstream's stale claims about ``STAND_POSE``. Its two comments say the pose is the
  simulator's HOME and ask that it be kept close to HOME for a clean hand-off to the roller policy;
  measured on the pinned model it is 55 and 56 degrees off HOME at the knees and stands the trunk
  27 mm lower. The values are reproduced unchanged and only the claims about them are dropped.
* Injected the entry momentum through the root reset rather than through a reset-mode push, as
  upstream does and locks with its own regression test: a push *adds* to the current root velocity,
  which on an environment that has already diverged sends the free joint to NaN.
* Collapsed the ``action_rate_l2`` weight to the single value that was ever live. Upstream declares
  ``-1.0`` on the term and ``-0.5`` at stage 0 of the curriculum that owns it, and the curriculum
  manager runs before the first reward evaluation, so the declared literal was dead.
* Sized the MuJoCo Warp contact budget from profiling rather than inheriting the skating task's:
  ``njmax`` is 176 and ``nconmax`` is 36, against a measured peak of 90 constraints and 29 contacts
  per environment under random actions with the tilt termination dropped and the pushes forced to
  full magnitude, profiled at 256, 2048 and 4096 environments. The fold is a heavier contact set than
  the stride's 83 and 26 on the same model. The **solver iteration counts are upstream's** template
  values, 10 and 20.
* Kept the family's NaN-guard norm, which is a deviation from upstream on this task: upstream
  registers ``nan_state`` with an empty sensor list and this port names the foot contact sensor and
  uses the two NaN-safe critic terms, inherited from the roller task.
* Built the environment as a delta on the roller-skating one, where upstream rebuilds it from the raw
  mjlab template and states the sensors, the randomization suite and both observation groups again by
  hand. The two constructions agree in value -- what upstream states again is the roller recipe term
  for term -- and the parity tests transcribe upstream's tables independently, so an inherited value
  that drifted from upstream's fails rather than agrees with itself.
