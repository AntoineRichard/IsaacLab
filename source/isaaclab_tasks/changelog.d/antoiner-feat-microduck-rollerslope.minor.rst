Added
^^^^^

* Added the contributed MicroDuck slope-descent task ``IsaacContrib-RollerSlope-Flat-MicroDuck``
  under ``contrib/microduck/rollerslope``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The roller-skating MicroDuck is placed on a ramp with a commanded velocity
  of exactly zero and has to ride it down: it cannot pedal, so gravity supplies the motion and the
  policy supplies only the balance. A terrain-level curriculum promotes an environment that rode its
  ramp out onto a steeper one, from 2 degrees to 20.
* Added ``FlatRampTerrainCfg`` and ``flat_ramp_terrain`` under ``contrib/microduck/rollerslope``, a
  task-local sub-terrain plugged into the stock terrain generator: a starting platform, a ramp whose
  angle the difficulty interpolates, and a run-out platform at its foot. The spawn origin it returns
  sits **on the incline**, which is what places the robot on the slope through the inherited
  origin-relative reset events. Nothing in ``isaaclab.terrains`` changes.
* Added ``wheel_glide_reward``, ``reset_rolling_entry``, ``terrain_levels_slope`` and
  ``slope_move_masks`` under ``contrib/microduck/mdp``. The glide reward is capped, one-sided and
  read off the **wheels**, so nothing pays for dropping faster, rolling back up the hill is free
  rather than charged, and running down on the blades earns nothing. The rolling entry writes the
  root velocity and the four wheel rates together, so an episode opens rolling instead of skidding.
  The curriculum scores the **signed** distance down the slope rather than a planar norm, which is
  what demotes a robot that slid backwards instead of promoting it.
* Added ``test/test_microduck_rollerslope_env.py`` and ``test/test_microduck_rollerslope_terrain.py``.
  The terrain tests measure the built mesh with downward ray casts against an independently written
  piecewise-linear profile, rather than re-checking the formulas that built it. The environment
  tests pin the *deletion* list as well as the recipe -- upstream keeps one inherited reward and
  rebuilds the rest, which is the failure mode a derived configuration has and a standalone one does
  not -- and two simulator-backed tests cover what the kernel tests cannot: the wheel velocities are
  **read back off the articulation** after a reset, and the environment origins are checked against
  the surface of the terrain that was actually generated.

Known issue
^^^^^^^^^^^

* Training ``IsaacContrib-RollerSlope-Flat-MicroDuck`` is blocked on a Newton contact gap on
  generated mesh terrain, which the environment configuration does not cause and cannot work around.
  Once the wheels are turning the roller model's tires stop carrying the robot on a generated mesh:
  it rides for about 0.1 s, sinks 45 mm and stops, where the same robot in the same pose at the same
  entry speed rolls normally on the skating task's analytic ground plane. Four controls localise it
  -- it reproduces at a **zero-degree** ramp angle, so it is not the incline; it does not reproduce
  on the plane, so it is not the robot; it does not reproduce at rest, so static contact is fine; and
  neither the collision margin nor contact reduction moves it. The terrain geometry is verified
  correct: the environment origins are bit-identical to upstream's and a downward ray cast onto the
  built mesh lands on them to 1e-6 m on all ten rows. The configuration, the registration and the
  parity tests are unaffected, and the environment builds, resets and steps NaN-free.

Changed
^^^^^^^

* Converted upstream's upright standard deviation instead of copying it. Upstream scores trunk tilt
  as ``exp(-(1 - cos t) / 0.2^2)`` where this family's ``upright`` scores ``exp(-sin^2(t) / std^2)``,
  and matching the two exponents at vertical gives ``sqrt(2) * 0.2 = sqrt(0.08)``. The two agree
  exactly upright and this one decays slightly faster off it -- 0.0439 against 0.0351 at 30 degrees
  -- and by the 57.3 degree tilt termination both are four orders of magnitude down. The exactly
  equivalent kernel in this package carries a height gate the task must not have, so using it would
  have meant adding an ungated variant for one term.
* Raised ``neck_joint_pos_l2`` to -0.75, which is the one weight that breaks a copy-the-block port
  from the skating recipe: the other three terms this task restates from it -- ``feet_flat``,
  ``neck_action_rate_l2`` and ``joint_torques_l2`` -- are identical, and a test compares them against
  the skating task rather than against a transcribed literal so the two cannot drift apart.
* Sized the MuJoCo Warp contact budget from profiling rather than inheriting the skating task's:
  ``njmax`` is 192 and ``nconmax`` is 42, against a measured peak of 98 constraints and 35 contacts
  per environment under random actions with the tilt termination dropped and the pushes forced to
  full magnitude, profiled at 256, 2048 and 4096 environments. The inherited ``nconmax`` of 32 sits
  below that peak: four tires on a triangle mesh, and on the joins between three boxes where one
  tire can touch two faces, demand more than four tires on a plane. The **solver iteration counts
  are upstream's** template values, 10 and 20.
* Turned the forced-forward command bucket off on this task, where upstream leaves the template's
  ``rel_forward_envs = 0.2``. The bucket clamps the surge slot to at least 0.3 at resample time, and
  upstream's relative-heading command override replaces the base per-step update and so never
  applies the standing-environment zeroing that its own ``rel_standing_envs = 1.0`` was set to
  trigger -- so on upstream's stack a fifth of the environments carry a 0.3 throttle for a whole
  resampling interval, in an observation slot that no reward on this task reads. Reproducing that
  would mean dropping the zeroing from a command class shared with the skating and swizzle tasks;
  turning the bucket off instead makes the three twist slots identically zero at every step, which
  is what neutralizing the command was for.
* Reproduced upstream's ``wheel_radius`` of 0.0175 m in both the glide reward and the rolling entry,
  although the model measures 0.0150 m. In the reward it only mislabels the cap, which is reached at
  a true ground speed of 0.300 m/s rather than 0.350; in the entry it means the wheels start about
  14 percent slower than rolling without slipping would need, so upstream's stated "zero slip" entry
  carries a few centimetres a second of it. Both are what the reference policy trained against, and
  correcting the radius is a retune with its own training run rather than a port fix.
* Reproduced the inherited ``+/-0.5 m`` horizontal spawn jitter rather than narrowing it for the
  slope. On an origin that sits on the incline it can start the trunk up to 109 mm inside the
  starting platform or 182 mm above the ramp at 20 degrees, against a standing height of 138 mm, so
  the first fraction of a second of an episode is a settle. It is upstream's live behaviour and part
  of what the reference policy trained against; the accuracy-gate regimes pin it away instead of the
  environment doing so.
* Did not rebuild upstream's ``nan_policy = "sanitize"``, an mjlab observation-group field with no
  Isaac Lab equivalent. It guards a contact divergence upstream measures at roughly one
  step-environment in 25 million, where the free joint goes to NaN and reaches the observation one
  step before the ``nan_state`` termination recycles the episode. This family already carries the
  NaN-guarded critic terms and that termination, so the residual exposure is the actor's own terms
  on a single step; closing it would mean touching every inherited observation term or the
  observation manager itself. The gap is documented where the environment is configured.
* Did not port upstream's ``reset_action_history`` event. Isaac Lab's action manager already zeroes
  both the previous and the current action for the environments being reset and then resets every
  action term, and this package's stateful action-rate penalty is a class term the reward manager
  resets on the same boundary, so a port would have been a no-op event.
* Placed the sub-terrain on the documented ``(0, 0)`` to ``size`` tile, where upstream places both
  its geometry and its origin at local ``y = 0`` and so straddles the tile boundary. The offset is
  shared by the geometry and the origin, so no robot-relative geometry changes and nothing in the
  task reads an absolute ``y``; matching upstream literally would put the sub-terrain outside the
  tile the terrain importer and its border machinery assume.
* Corrected the ``heading_hold_reward`` note about upstream's anchor timing. Both stacks increment
  the episode counter before computing rewards, so upstream's ``episode_length_buf <= 1`` condition
  holds on exactly one reward evaluation per episode and the two implementations anchor at the same
  instant; the note claimed a one-step deviation that does not exist.
