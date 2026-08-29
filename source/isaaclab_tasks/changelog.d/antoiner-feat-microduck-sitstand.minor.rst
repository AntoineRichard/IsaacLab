Added
^^^^^

* Added the contributed MicroDuck sit-stand task ``IsaacContrib-SitStand-Flat-MicroDuck`` under
  ``contrib/microduck/sitstand``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. One policy performs both posture transitions, driven by a binary request in
  the twist slot of the shared 61-wide deployed observation: 0 asks the robot to stand and 1 asks it
  to sit, flipping every 3.5 to 6.5 seconds so a single 12-second episode trains a descent, a seated
  rest, a rise and a standing rest. The reset draws the starting posture independently of the first
  request, which is what also trains holding each posture.

  There is no trajectory, no waypoint and no phase clock. What replaces them is the command term's
  **slewed target**: every posture reward tracks a blend that moves toward the requested flag at a
  fixed rate over two seconds, while the policy observes the raw flag. Upstream established the
  difference across two runs -- against the raw flag, dropping instantly collects the whole
  goal-state payout for every step saved and beat a one-second descent by about sevenfold, and the
  trained policy crash-sat.
* Added ``SitStandCommand`` and ``SitStandCommandCfg`` under ``contrib/microduck/mdp``, the posture
  command described above. It exposes the raw flag as its command and the slewed blend as
  ``alpha``; the blend is re-seeded from the trunk height the robot actually spawned at, so a seated
  spawn under a stand request starts its ramp at the sit end rather than being dragged upward.
* Added ten posture reward terms under ``contrib/microduck/mdp``: ``posture_pose_gaussian``,
  ``posture_pose_l1``, ``posture_height_gaussian``, ``posture_height_l1``,
  ``posture_rise_bootstrap``, ``posture_stillness``, ``posture_composite``,
  ``trunk_downward_velocity_penalty``, ``trunk_upward_velocity_penalty`` and
  ``upright_linear_at_height``. Each is the commanded-posture counterpart of a stand-up term that
  tracks one fixed goal, so the target -- joint pose and trunk height together -- is selected from
  the command instead of being a constant. ``posture_rise_bootstrap`` is the one that reads the raw
  flag rather than the blend, deliberately: it switches off the instant a sit is requested and can
  never bid against the descent.

  Five of these kernels return values at or below zero and are therefore configured with **positive**
  weights. Upstream lost a full run to getting that backwards, which turned its three speed and shock
  penalties into the largest positive terms in the stack, so the sign convention is documented on
  every one of them and asserted in the tests.
* Added ``test/test_microduck_sitstand_env.py`` and twenty posture-kernel cases to
  ``test/test_microduck_rewards.py``. The recipe is compared term by term against the upstream tables
  without launching the simulator, including both observation groups' entity selections, the reward
  sign convention and the four curriculum stage tables. The simulator-backed tests assert the
  observation widths, that the reset seeds the posture blend from the pose the robot actually spawned
  in, and that the blend slews at the configured rate while the observation carries the unslewed
  flag; they skip when the generated all-collisions MicroDuck USD is absent.

Changed
^^^^^^^

* Changed ``posture_stillness`` and the late-phase penalty gates to keep upstream's **two** tilt-gate
  conventions apart rather than unifying them. Upstream interpolates its stillness gate in
  ``cos(tilt)`` and its penalty gates in the angle; the two agree at their bounds and nowhere in
  between, so a trunk at the angular midpoint of a 25-to-60-degree window scores 0.5 through one and
  0.625 through the other. Both are now available as ``_tilt_gate`` and ``_cos_tilt_gate``.
* Kept the family's NaN-guard norm, which is a deviation from upstream on this task. Upstream
  registers ``nan_state`` with an empty sensor list and reads the raw foot observations; the port
  names the foot contact sensor and uses the two NaN-safe critic terms, as it does on every sibling.
  The extraction reads upstream's empty list as drift rather than design and recommends closing it
  everywhere in the port. The guard only changes behaviour in states that are already broken.
* Sized the MuJoCo Warp contact budget of this task from profiling rather than transcribing
  upstream's: ``njmax`` is 128 and ``nconmax`` is 32, against a measured peak of 82 constraints and
  28 contacts per environment under random actions with the pushes forced to full magnitude.
  Upstream raises ``nconmax`` to 200 here, which is the whole of its answer to a seated contact
  divergence it observed; the measurement says the buffer was not what was binding. The **solver
  iteration counts are transcribed** -- 30 and 50 against the template's 10 and 20 -- because those
  are what upstream's dated, causal note attributes the fix to, and this is the one task in the
  family that raises them.
* Seeded the posture blend from ``reset`` where upstream does it inside ``compute``, guarded on the
  episode counter. Upstream's own comment records that as a workaround: its command manager resets
  before the event that teleports the robot into its spawn pose, so a reset hook would read the
  pre-teleport height. Isaac Lab fires reset-mode events before it resets the command manager, so the
  hook reads the spawn pose and the workaround is not needed.
* Dropped the velocity-tracking metrics from the posture command, as
  ``RelativeHeadingVelocityCommand`` already does for the roller task: the inherited error and
  success-rate metrics would compare a command against a quantity it does not name. Upstream leaves
  them registered and logs a constant zero.
* Made the posture rewards raise when the command term they name exposes no blend, where upstream
  silently falls back to the raw flag. The fallback is unreachable on every shipped configuration,
  and a posture reward quietly reading an un-slewed flag is the exact failure the slew exists to
  prevent.
* Reused the stand-up task's sitting keyframe and its two rest heights rather than restating them.
  Upstream keeps its sit, stand-up and sit-stand environments' keyframes in sync by hand and says so
  in each of them; a second copy is the drift that instruction guards against. As elsewhere in the
  port the keyframe is keyed by joint name, because the converted asset resolves joints in Newton's
  order rather than the MJCF's, and a keyframe joint that is not among the scored joints now raises
  instead of being silently rewarded at the stand pose.
