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
* Made the grip **force-limited rather than force-clamped**: above ``6 N`` the latch breaks and the
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
* Made both distance rewards pay only for arriving rather than for being near, so there is no range
  at which loitering is profitable -- which a Gaussian on the distance would create at every range.
* Re-derived the grip spring against the rate it is **actually integrated at**. The wrench is written
  once per control step by an interval event and the wrench composer is permanent, so the force is
  held constant across all four physics substeps: the spring is a zero-order hold at 50 Hz, not at
  200 Hz. Deriving the stiffness against the physics step understated ``omega * dt`` by the
  decimation factor -- 1.03 rather than the 0.26 claimed -- at a damping ratio of 0.32. It rang, the
  object lagged its anchor by up to 91 mm, and the spring broke its own grip limit roughly every
  control step. At the shipped ``25 N/m`` and ``0.9 N.s/m`` the figures are 0.82 and 0.73, and the
  measured mean grip life goes from 20 control steps to 138. The break force was raised from 2 N,
  which had been set against the object's weight and sat *below* the transient a normal carry
  produces; a limit below the working transient is not a grip limit.
* Sized the whole reward stack in **episode-return units** rather than mixing units across the
  sparse and shaped terms. Reward weights are *rates* -- the reward manager multiplies every term by
  the control step -- so a one-shot bonus contributes ``weight * dt`` to the return, not ``weight``.
  Reading the shaped terms per step and the one-shot terms in absolute units left the entire task
  objective worth 4.26 against 80 for holding the object and standing still, which a first long
  training run duly exploited: it spent 2255 iterations farming a one-control-step grip that paid the
  latch bonus ~444 times an episode, never delivered the object once, and drew 88 % of its return
  from that single term. Two tests now assert the budget instead of reasoning about it -- delivering
  must beat every shortcut with margin, and no single term may exceed half the positive budget.
* Capped the latch bonus at **one payment per episode** with a sticky ``has_latched`` flag. An event
  a policy can re-trigger at will needs a per-episode cap, not a "not yet finished" gate; the
  sticky ``succeeded`` flag guarded the placement bonuses and nothing guarded this one.
* Made ``carry_progress`` a **ratchet** rather than a one-step potential. As a potential it masked
  its *negative* half while the object was unlatched, so a policy could carry the object halfway in,
  bank the gain, drop it, let a 70 mm sphere roll back out for free and sell the same ground again.
  The ratchet pays only for ground closer than the closest the object has reached in the episode and
  tightens that mark whether or not the object is held, which bounds an episode's total payout by
  the distance the object started from the drop point. ``approach_progress`` stays a symmetric
  potential: it is not farmable the same way, because the robot cannot move away from the object
  without being charged for it, and a symmetric term gives the denser learning signal.
* Sized the prop against the **beak** rather than against the robot. The task shipped for three
  training runs carrying ``MICRODUCK_BALL_CFG``, upstream's 70 mm floorball, which is more than twice
  as wide as the beak opens -- so it could never be grasped, only leaned on, and the recordings show
  exactly that. ``MICRODUCK_MARBLE_CFG`` is a 20 mm glass marble against a measured 31.1 mm aperture,
  taken on the jaw vertices that actually meet the upper mouth surface when the beak is shut.
* Derived every latch constant from the prop instead of hand-picking them. The stiffness is
  ``m * (omega*dt / dt)^2`` at a chosen dimensionless stiffness, the damping from a damping ratio and
  the break force from a multiple of the object's weight, so a change of prop carries all of them.
  This is the real repair for the earlier stiffness defect: a hand-picked spring is only correct for
  the prop it was picked against, and this task has now changed prop once. The formula reproduces the
  previous hand-picked values for the ball to three significant figures.
* Derived the placement height from the prop as well, which the scripted acceptance test caught on
  the prop change: the literal it replaced was sized against a 35 mm-radius ball, where it meant
  "surface within 25 mm of the floor", and on a 6 mm marble the same number would have counted a
  marble still held at head height as placed.
* Re-measured the contact budget after the prop change rather than assuming the old one covered it.
  The contact peak falls from 32 to 28 and the constraint peak *rises* from 90 to 94 -- a smaller
  prop is not uniformly cheaper.
* Reused the existing ``MICRODUCK_BALL_CFG`` prop rather than authoring a new one. At 15 g it sits
  inside the 10-40 g mouth-payload band the ground-pick task already validates at the same attachment
  point, and its mass, hollow-shell inertia, collision and material are pinned by that asset's own
  fidelity tests. ``isaaclab_assets`` is therefore untouched.
* Priced falling rather than merely terminating on it. A termination is only a penalty when the rest
  of the episode was worth something, and this stack's mass is a one-shot delivery bonus a policy can
  collect in the first second -- so a fall cost nothing, and a training run learned to grab the
  object, dive at the drop point and topple. It satisfied the literal success criterion on 93 % of
  episodes, in 40 control steps, with the uprightness reward at 0.0003. ``fell_penalty`` charges the
  two fall terminations; ``nan_state`` is deliberately excluded, because a diverged solver is not a
  policy decision. Note the coupling this introduces: ``is_terminated_term`` resolves its keys at
  construction, so any configuration that disables the fall terminations must disable this term too.
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
