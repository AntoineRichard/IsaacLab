Added
^^^^^

* Added the contributed MicroDuck pick-and-place task ``IsaacContrib-PickPlace-Flat-MicroDuck`` under
  ``contrib/microduck/pickplace``, with an RSL-RL PPO configuration. It runs on the Newton MJWarp
  backend. The robot spawns standing at a random heading with an object on the ground in front of it
  and a drop point commanded somewhere else; it has to walk to the object, take it in its mouth,
  carry it and set it down on the target. **Unlike every other task in this package it has no
  upstream counterpart**, so it is designed rather than ported: its term table, the rulings behind
  each judgment call and its reward-hacking audit live in
  ``artifacts/microduck/pickplace/DESIGN.md``, and acceptance is internal on the roller stand-up
  task's precedent -- staged, physically motivated, and scoring the whole reward stack end-to-end
  under physics.
* Added ``PickPlaceLatchState``, ``pickplace_latch_state``, ``reset_pickplace_latch``,
  ``reset_object_in_reach`` and ``update_pickplace_latch`` under ``contrib/microduck/mdp``. The
  MicroDuck mouth is not a gripper -- there is no actuated jaw close, only neck and head degrees of
  freedom -- so what holds the object is a **compliant virtual weld**: while latched, a
  three-degree-of-freedom spring-damper pulls the object toward an anchor rigidly attached to the
  mouth tip, and the *equal and opposite* wrench is written onto the head. The object stays a fully
  dynamic rigid body throughout, so the robot genuinely carries its weight and its inertia. The
  mechanism writes only wrench buffers and changes no model structure, so it survives CUDA graph
  capture.
* Added ``PickPlaceTargetCommand`` and ``PickPlaceTargetCommandCfg`` under ``contrib/microduck/mdp``.
  The drop point is drawn once per episode in polar coordinates around the object's own placement, in
  the robot's reset yaw frame, and published as the offset from the robot base to that point in the
  base frame -- so it rotates as the robot turns. Its ranges are a tuple of ``(low, high)`` pairs,
  deliberately the shape ``UniformPoseDeltaCommandCfg`` uses, so the existing ``command_range_stages``
  curriculum term widens it unchanged.
* Added nine pick-and-place reward terms under ``contrib/microduck/mdp``:
  ``pickplace_approach_progress``, ``pickplace_mouth_to_object``, ``pickplace_mouth_down``,
  ``pickplace_latch_bonus``, ``pickplace_carry_hold``, ``pickplace_carry_progress``,
  ``pickplace_object_clearance``, ``pickplace_place_success`` and ``pickplace_place_precision``. They
  are gated on the **latch state** rather than on a clock, so the task's phases emerge from what the
  robot has achieved instead of from a cycle it is told the position of.
* Added ``object_pos_in_base``, ``object_vel_in_base``, ``mouth_to_object_in_base``,
  ``pickplace_latched_flag`` and ``pickplace_succeeded_flag`` under ``contrib/microduck/mdp``.
* Added ``test/test_microduck_pickplace_mdp.py`` and ``test/test_microduck_pickplace_env.py``. Since
  there is no upstream to gate against, the design document's term table is transcribed into the
  environment test and *is* the parity table. The simulator-backed acceptance scripts a full pick,
  carry and place and asserts each block's rewards through the real reward manager, that the carried
  object loads the robot with an equal and opposite wrench carrying a moment about the head, and that
  an overloaded grip gives way rather than dragging the object through the scene. They skip when the
  generated all-collisions MicroDuck USD is absent.

Changed
^^^^^^^

* Generalized ``ball_pos_in_base`` and ``ball_vel_in_base`` over the scene entity they read. Both
  remain as named entry points for the ball-kick critic and behave identically; the pick-and-place
  task reads the same kernels for its own prop through ``object_pos_in_base`` and
  ``object_vel_in_base``. No caller needs to change.
* Made the grip **force-limited rather than force-clamped**: above ``2 N`` the latch breaks and the
  object is dropped. A clamped constraint is a winch, and a policy that found one would drag the
  object through the scene rather than carry it. The break is evaluated *before* the latch in the
  same step, so an object that has just been torn loose -- and is therefore still inside the latch
  radius -- cannot immediately re-form its grip, which is what would make the limit a no-op.
* Made the release **geometric rather than an action**: the object is set down when it is inside the
  placement tolerance of the drop point *and* near the floor. The alternative costs a fifteenth
  action dimension, and the 14-servo action vector is a hardware contract on every task in the
  family. The release edge is therefore also the success edge, and the success flag is sticky for the
  rest of the episode, so the one-shot placement bonuses cannot be farmed by picking the object back
  up.
* Made both distance rewards **potential-based rather than Gaussian on the distance**. A closed path
  sums to exactly zero and standing anywhere pays exactly zero, so there is no range at which
  loitering is profitable -- which a level-based term would create at every range.
* Sized ``carry_hold`` from the reward-hacking audit rather than by taste. ``mouth_to_object`` pays
  its full weight for holding the mouth on the object without picking it up and is gated off the
  moment the object is latched, so any carry bonus at or below it makes *refusing to pick the object
  up* strictly dominant. The inequality is asserted in the environment test, from the weights and
  again from measured rewards under physics.
* Reused the existing ``MICRODUCK_BALL_CFG`` prop rather than authoring a new one. At 15 g it sits
  inside the 10-40 g mouth-payload band the ground-pick task already validates at the same attachment
  point, and its mass, hollow-shell inertia, collision and material are pinned by that asset's own
  fidelity tests. ``isaaclab_assets`` is therefore untouched.
* Gated the fall termination on tilt **and** height, as two separate stock terms. The family's
  velocity-stand investigation found that tilt-only gating opened a lie-flat reward-hacking basin: a
  robot folded flat stays nominally inside a 70-degree bound while doing nothing the task asks for.
* Did not carry the ground-pick task's ungated ``feet_flat`` penalty, and weakened ``feet_grounded``
  against it. That task's gesture has no swing phase; this one has to walk, and an ungated flat-foot
  penalty charges every step the robot takes.
* Did not pad the actor observation to the walking family's 61-wide deployed contract. The actor is
  55 wide and the critic 75; this is a new task with a new runtime, and padding with zeros would
  advertise a hot-swap compatibility that does not exist. What serves the planned camera version
  instead is **term separation**: the object's position, the drop point and the latch flag are each
  their own observation term and none is folded into a robot-state term, so replacing the object
  position with a camera-derived one is the whole of that migration.
