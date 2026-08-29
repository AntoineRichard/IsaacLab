Added
^^^^^

* Added the contributed MicroDuck spin task ``IsaacContrib-Spin-Flat-MicroDuck`` under
  ``contrib/microduck/spin``, with an RSL-RL PPO configuration carrying upstream's hyper-parameters.
  It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The trick is one counter-clockwise turn on the spot -- launch, hold three
  radians a second, brake, rest -- on a four-second clock whose phase the policy is told, so a
  twenty-second episode is five turns. The area under that yaw-rate trapezoid is 6.30 rad, which is
  one turn per cycle to a quarter of a percent.
* Added ``spin_rate_track``, ``spin_rate_l1``, ``spin_stay_in_place``, ``spin_wheel_differential``,
  ``spin_grounded`` and ``leg_antisymmetry`` under ``contrib/microduck/mdp``. The yaw rate is read in
  the **body** frame, because that is what the robot's own gyro reports and therefore what the policy
  observes. ``leg_antisymmetry`` is the mirror image of the swizzle task's ``leg_symmetry_reward``:
  the model uses mirrored left/right sign conventions, so a symmetric pose reads as
  ``q_left + q_right ~= 0`` and the scissor that drives a rotation reads as ``q_left ~= q_right``.
* Added ``test/test_microduck_spin_env.py``. The task is structurally the crouch-glide trick with a
  rate objective in place of a pose one, so the tests pin the differences: the dropped
  angular-momentum term, the envelope arithmetic derived rather than transcribed, the neck penalty
  scoped to leave the head yaw free, and the decaying scissor curriculum. The simulator-backed tests
  read the rate reward at three points of the cycle and check that both mechanism hints are gated to
  zero across the standing rest; they skip when the generated roller MicroDuck USD is absent.

Changed
^^^^^^^

* Dropped ``angular_momentum``, as upstream does and as no other MicroDuck task does. It charges the
  norm of the whole angular-momentum vector and would fight a spin head-on, where ``body_ang_vel``
  charges roll and pitch only and therefore damps the wobble without touching the rotation.
* Reproduced ``SPIN_WHEEL_OMEGA_SCALE`` verbatim although its stated derivation does not reproduce.
  Upstream derives 17.0 rad/s from a half-track it states as 0.0499 m and a 0.0175 m tire radius;
  measured on the pinned roller model the half-track is 0.03925 m at the foot sites and the tire
  radius is 0.0150 m, so neither input holds and neither does the result. The constant is what the
  deployed policy trained against and is kept unchanged; only the arithmetic behind it is dropped,
  and a test reproduces that arithmetic to show it does not land on 17.0.
* Named the four wheel hinges rather than reproducing upstream's ``^passive_.*`` selector. Upstream
  uses the looser pattern on this task alone, where its own conventions and every sibling use
  ``^passive_.*wheel``; on a model carrying backlash hinges the looser one would pick those up and
  widen the critic. On the plain roller model the two are equivalent, so there is no live defect to
  reproduce, and naming the hinges -- which is what this port does everywhere -- makes the divergence
  unexpressible rather than merely harmless.
* Reproduced upstream's frictionless-bearing randomization verbatim, as on the crouch-glide task: the
  wheel-friction event ships a degenerate ``(0.0, 0.0)`` range and no curriculum ramps it, so this
  environment trains on free bearings for its whole run (upstream issue draft 017). The event is kept
  rather than removed as dead code, for the reason recorded on the crouch-glide task.
* Collapsed the ``action_rate_l2`` weight to the single value that was ever live. Upstream declares
  ``-1.0`` on the term and ``-0.5`` at stage 0 of the curriculum that owns it, and the curriculum
  manager runs before the first reward evaluation, so the declared literal was dead.
* Sized the MuJoCo Warp contact budget from profiling rather than inheriting the skating task's:
  ``njmax`` is 176 and ``nconmax`` is 36, against a measured peak of 90 constraints and 30 contacts
  per environment under random actions with the tilt termination dropped and the pushes forced to
  full magnitude, profiled at 256, 2048 and 4096 environments. Three radians a second about a 39 mm
  half-track drags four small tire patches with continuously rotating contact normals, which is the
  regime where these budgets matter most and the one upstream leaves at the template's unexamined 35.
  The **solver iteration counts are upstream's** template values, 10 and 20.
* Kept the family's NaN-guard norm, which is a deviation from upstream on this task: upstream
  registers ``nan_state`` with an empty sensor list and this port names the foot contact sensor and
  uses the two NaN-safe critic terms, inherited from the roller task.
