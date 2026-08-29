Added
^^^^^

* Added the contributed MicroDuck ball-kick task ``IsaacContrib-BallKick-Flat-MicroDuck`` under
  ``contrib/microduck/ballkick``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The robot starts standing at a random heading with
  :data:`~isaaclab_assets.MICRODUCK_BALL_CFG` on the ground just in front of one foot, and has five
  seconds to kick it away at about a walking pace and stay upright. There is no command, no phase
  clock and no curriculum on the kick itself.

  It is the **only two-entity environment** in the family, and the only one whose critic is wider
  than the stand-up task's: the actor keeps the shared 61-wide deployed contract with the head-pose
  and body-pose slots zero-padded and **no ball observation at all** -- the real robot has no ball
  sensing -- while the critic gains a privileged three-wide ball position and velocity in the base
  frame, at 80. Left-right symmetry stays off, because the kick is inherently one-footed. The reset
  chain is load-bearing: the ball placement runs after the ground-state reset and is measured in the
  robot's *reset* yaw frame, so moving it earlier would aim it at a heading the robot no longer has.
* Added the MicroDuck ball-kick reward terms under ``contrib/microduck/mdp``:
  ``ball_forward_velocity``, ``ball_speed_overshoot_penalty`` and ``single_foot_grounded_reward``.
  The first two project the ball's velocity onto a per-episode direction frozen at reset -- so the
  policy cannot redefine "forward" by turning after the kick -- and together form a one-sided
  plateau around the target speed. ``single_foot_grounded_reward`` reads a terrain-filtered contact
  sensor rather than a net contact force, which on this task is the difference between "the support
  foot is on the floor" and "the support foot is touching the ball".
* Added ``ball_pos_in_base`` and ``ball_vel_in_base`` under ``contrib/microduck/mdp``, the two
  privileged critic observations. ``ball_vel_in_base`` rotates the ball's *world* velocity into the
  base frame without subtracting the robot's own, as upstream does, so a robot walking at a
  stationary ball reads zero rather than a closing velocity.
* Added ``ball_kick_direction`` and ``reset_ball_in_front_of_foot`` under ``contrib/microduck/mdp``.
  The event places the ball in the robot's yaw frame at rest, exactly touching the ground, and
  freezes the episode's kick direction; the accessor is the shared per-environment state the two
  kick rewards read, defaulting to ``+x`` before the first reset as upstream's does.
* Sized the MuJoCo Warp solver budget of the ball-kick task from profiling rather than from
  upstream's value, as the velocity task's budget was: ``njmax`` is 128 and ``nconmax`` is 36,
  against a measured peak of 86 constraints and 30 contacts per environment under random actions
  with the tilt termination dropped and the pushes at full magnitude. Upstream raises ``nconmax``
  from its template's 35 to 50 for the ball's own contacts and leaves the constraint budget
  untouched; the measured peak was identical at 256 and at 2048 environments, and the parity test
  asserts a floor under both numbers so a later retune cannot drop below the measurement.
* Added ``test/test_microduck_ballkick_env.py``, which compares the assembled configuration term by
  term against the upstream reward, event, curriculum, command and observation tables without
  launching the simulator, and then runs the acceptance tests for what makes this task different:
  that the ball lands in front of the kicking foot at every reset heading and rests on the ground at
  zero velocity, that the frozen kick direction is the reset heading, that the kick rewards pay a
  ball pushed along that direction and nothing for one pushed back, and that the support-foot reward
  reads the floor rather than the ball pressed against the same sole. It skips the simulator tests
  when the generated all-collisions MicroDuck USD is absent.

Changed
^^^^^^^

* Changed ``reset_ground_state``'s per-bucket parameters -- ``prone_z_range``, ``sitting_z_range``,
  ``standing_z_range`` and ``sitting_joint_pos`` -- to be optional, and to raise when a bucket with a
  positive probability is missing the parameters it spawns from. The ball-kick task is standing-only
  and would otherwise have to invent height bands and a seated keyframe for three buckets it never
  draws. The stand-up task passes all four and is unaffected.
* Reproduced upstream's kick weights **verbatim although its own commentary contradicts them**.
  ``BALL_TARGET_SPEED`` is 1.0 while the comment block above the two kick rewards describes a
  landscape calibrated for 0.25 -- an at-target payoff of about +3 per step and a zero crossing at
  1.0 m/s. At the shipped constants the kick pays ``12*min(v, 1.0) - 4*max(v - 1.0, 0)``: +12 per
  step at target, zero at 4.0 m/s and a floor of -8. That roughly doubles the task's reward mass, so
  the shared regularizers act at about half their intended relative strength while the ball rolls.
  The code is ported rather than the prose, because the shipped weights are what upstream's runs
  were trained against; the discrepancy is documented where the constant is defined, and rescaling
  the two weights to 3.0 and 1.0 restores the documented landscape.
* Changed the ball-kick task's NaN guarding to follow the rest of the ported family rather than
  upstream's. Upstream leaves ``robot_state_is_nan``'s sensor list empty here and reads its critic
  contact terms unguarded, which its own extraction reads as drift rather than design; the foot
  contact sensor is named in the termination and the two NaN-safe critic terms are used. The two
  ball observations and both kick rewards additionally guard themselves, because **nothing on either
  stack NaN-checks the ball**: the termination reads the robot only, so a free body the solver
  ejected would reach the learner through those four terms and nothing else. The guards only change
  behaviour in states that are already broken.
* Filtered the support-foot contact sensor against the terrain, where the family's other foot
  sensors read an unfiltered net contact force. On this task the narrowing is required rather than
  preferred: the ball rolls along the ground at exactly sole height, so an unfiltered sole would
  report "grounded" while the foot was airborne and merely touching the ball.
