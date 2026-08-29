Added
^^^^^

* Added the contributed MicroDuck ground-pick task ``IsaacContrib-GroundPick-Flat-MicroDuck`` under
  ``contrib/microduck/groundpick``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The robot brings its mouth to the floor as close as possible *without
  touching*, correctly oriented, holds it there, stands back up cleanly and rests -- a four-segment
  gesture on a four-second clock whose phase the policy is told, so a twenty-second episode is five
  complete cycles. The "without touching" is an **equilibrium rather than a constraint**:
  ``mouth_ground_proximity`` pulls the mouth down and ``head_impact_penalty`` charges the contact
  force it would arrive with, and the hover is where the two balance.
* Added ``GroundPickPhaseCommand`` and ``GroundPickPhaseCommandCfg`` under ``contrib/microduck/mdp``.
  The three-wide twist slot of the shared 61-wide deployed observation carries the cycle phase as
  ``(cos, sin, 0)`` instead of a velocity, so the wrap from the end of a cycle to its start is
  continuous rather than the largest jump in the input. The clock is open loop -- nothing the robot
  does can move it -- and each episode starts at a uniformly drawn phase, so environments do not
  bend in lockstep.
* Added seven ground-pick reward terms under ``contrib/microduck/mdp``:
  ``mouth_ground_proximity_phased``, ``mouth_perpendicular_phased``,
  ``ground_pick_return_pose_phased``, ``ground_pick_return_upright_phased``,
  ``neck_vel_descent_penalty``, ``feet_grounded_reward`` and ``body_impact_cost``. The first five are
  gated on the cycle phase through two gates that are deliberately **not** complements: they sum to
  one across the return and nowhere else, which is what leaves the approach unpriced by the return
  terms.
* Added ``sample_mouth_payload`` and ``apply_mouth_payload_force`` under ``contrib/microduck/mdp``, a
  10-40 g point mass drawn per episode and hung off the mouth tip from the moment the mouth closes,
  so the return is a lift rather than an unweighted extension. The robot is never told what it
  weighs.
* Added ``test/test_microduck_groundpick_env.py``. The recipe is compared term by term against the
  upstream tables without launching the simulator, including both observation groups' entity
  selections, the cycle segment durations, the two phase gates' overlap and the curriculum stage
  tables. The simulator-backed tests assert the observation widths, that the phase clock advances
  exactly one cycle per period, that the payload wrench appears only after the mouth closes, and the
  two mouth-sensing checks described below; they skip when the generated all-collisions MicroDuck USD
  is absent.

Changed
^^^^^^^

* Measured the ``mouth_tip`` **site** as a fixed offset in the ``jaw_soft`` body frame, because Isaac
  Lab has no site concept and this is the family's most site-dependent task -- both mouth rewards are
  defined on that site upstream. The offset and the mouth's pointing axis are read off the pinned
  ``robot_allcollisions.xml``, and the mapping is asserted from two independent directions: against
  ``mujoco.mj_forward``'s own site kinematics on that MJCF for three joint configurations, and
  against the physical event the channel exists to describe -- a scripted fold to the floor, where
  the terrain-filtered head-impact sensor stays silent through the whole approach and fires only once
  the derived mouth height is within millimetres of the ground.
* Registered the mouth payload as an **interval event** on a zero-width interval rather than as
  upstream's weight-zero reward term. Upstream's function writes an external wrench every step and
  returns zeros, so a pass that pruned zero-weight rewards would delete the payload physics
  silently. The consequence of the move is that the wrench is written after the command manager
  advances the clock rather than before, so it lags upstream by one control step out of the two
  hundred in a cycle.
* Sensed upstream's ``neck``-subtree head-impact sensor as the three ``jaw_soft`` collision shells.
  Measured on the pinned MJCF, that subtree is ``neck``, ``neck_pitch``, ``yaw_roll_motion`` and
  ``jaw_soft``, and only ``jaw_soft`` carries geometry -- so the two are the same sensor rather than
  an approximation. It is filtered against the terrain, as upstream's is, so a knee brushing a head
  shell is not charged as a face-plant.
* Collapsed the ``action_rate_l2`` weight to the single value that was ever live. Upstream declares
  ``-2.0`` on the term and ``-0.8`` at stage 0 of the curriculum that owns it, and the curriculum
  manager runs before the first reward evaluation, so the declared literal was dead.
* Measured the mouth height above the environment origin rather than as a raw world ``z``. Upstream
  reads a raw world ``z`` here alone among its height terms, which is equivalent on a ground plane
  and silently wrong on its rough variant.
* Kept the family's NaN-guard norm, which is a deviation from upstream on this task, and carried the
  longest sensor list in the family: upstream registers ``nan_state`` with an empty list, while the
  port names the foot contact sensor, the terrain-filtered sole sensor and the head-impact sensor,
  and uses the two NaN-safe critic terms. This is the one task in the family that reads a contact
  force into a *reward*, where a single non-finite value poisons the episode sum rather than one
  observation column.
* Sized the MuJoCo Warp contact budget from profiling rather than transcribing upstream's, which
  leaves the mjlab template's ``nconmax = 35`` unexamined: ``njmax`` is 128 and ``nconmax`` is 32,
  against a measured peak of 86 constraints and 27 contacts per environment under random actions with
  the tilt termination dropped and the pushes forced to full magnitude, profiled at 256, 2048 and
  4096 environments. The **solver iteration counts are upstream's** template values, 10 and 20.
* Transcribed upstream's pre-audit IMU latency bound of three control steps, which this task is
  alone in the family in still carrying -- every sibling uses one, the value a 2026-07 audit settled
  on against a measured hardware envelope, and this task's own comment claims a parity it no longer
  has. Keeping it trains a policy that tolerates a wider latency than the hardware has, which is the
  safe direction, and the divergence is asserted in the tests so it stays deliberate.
* Made ``feet_flat_penalty``'s contact gate optional. Upstream's roller recipes pass a sensor and its
  ground-pick recipe does not, and that argument is the only difference between the two uses: with
  the gate off, both feet are asked to lie flat at every phase, which is what a gesture with no swing
  phase wants.
* Normalized ``feet_grounded_reward`` by the number of selected sensing objects, where upstream sums
  its per-foot contact flags and divides by a hard-coded two. Identical with two feet selected, and
  it does not silently mis-scale if the selection changes.
* Did not carry upstream's ``soft_landing`` slot. The term is live upstream at ``-1e-5`` -- what is a
  no-op there is the *assignment*, because the mjlab base template already ships that weight -- and
  this port's shared base carries no such term, so dropping it is a real deviation of order ``1e-5``
  against a stack summing to order 10. It is worth naming for what it demonstrates rather than what
  it weighs: it is gated on the *magnitude* of the twist command, and this task's twist is a unit
  vector on the circle, so an inherited "only while moving" gate is permanently open on any
  phase-command environment.
