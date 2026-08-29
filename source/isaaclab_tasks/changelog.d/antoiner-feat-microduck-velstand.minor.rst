Added
^^^^^

* Added the contributed MicroDuck velocity-plus-fall-recovery task
  ``IsaacContrib-VelStand-Flat-MicroDuck`` under ``contrib/microduck/velstand``, with an RSL-RL PPO
  configuration carrying upstream's hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. One policy walks *and* gets back up, on the shared 61-wide deployed
  observation contract.

  It is the **only task in the family that derives from another**:
  :class:`~isaaclab_tasks.contrib.microduck.velocity.flat_env_cfg.MicroDuckVelocityFlatEnvCfg` is
  inherited whole, as upstream's own factory takes the velocity recipe verbatim, so a change to the
  proven walking recipe reaches this task instead of being restated and left to drift. On top of it
  are six reward terms, one reset event, one termination and five curricula. Two structural changes
  come with the recovery layer: the robot is
  :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG` rather than the walking model, because a robot
  that lies down needs colliders on its trunk, hips, shins and head; and swapping it in makes the
  many-to-many self-collision sensor reachable, which the walking scene has no colliders for.

  The three-phase schedule *is* the task. ``fell_over`` ends a fall for the first 500 iterations, so
  the early rollout is clean walking; a curriculum then widens its limit angle to half a turn rather
  than deleting the term, and ``fallen_too_long`` recycles an episode that stays down for eight
  seconds. The recovery economics stay at weight zero until iteration 1200, which buys a tax-free
  window where a get-up attempt costs nothing; the prone reset ramp opens later still, at 1500, and
  is capped at 45 % so at least 55 % of the experience remains clean walking.
* Added the MicroDuck fall-recovery reward terms under ``contrib/microduck/mdp``:
  ``upright_progress``, ``height_progress``, ``fallen_state_penalty`` and ``recovery_success``. The
  first two are potential-based -- they pay the *change* in trunk uprightness and height and exactly
  zero for holding any pose -- and the other two are the recovery economics: a flat tax that a fall
  arms and only a completed stand releases, and a one-shot bounty on that completion. All four are
  stateful and all four clear their state on reset, where upstream leaves the equivalent buffers to
  carry across an episode boundary.
* Added ``fallen_too_long`` under ``contrib/microduck/mdp``, the failed-recovery termination. Its
  gate is trunk height **or** tilt where the recovery rewards gate on tilt alone, which is upstream's
  deliberate asymmetry: a robot sitting low but upright is not paid as fallen, but it is recycled as
  stuck.
* Added ``termination_param_stages`` under ``contrib/microduck/mdp``, a staged schedule over a
  termination term's parameters. Unlike upstream's equivalent it raises when the term it names is
  absent rather than silently never firing, so a task that removes a termination must remove the
  curriculum that drives it -- as the flat walking task already removes ``terrain_levels`` along with
  its terrain generator.
* Added ``test/test_microduck_velstand_env.py``. The recovery layer is compared term by term against
  the upstream tables without launching the simulator, including the observation-group entity
  selections and the three curriculum stage tables; the inherited walking layer is compared against
  the velocity task's own assembled configuration, because "the velocity recipe, verbatim" is the
  contract and restating its values would let the two drift apart while both suites passed. The
  simulator-backed tests assert the observation widths and that the reset actually spawns the prone
  and crouch poses the recovery layer trains on; they skip when the generated all-collisions
  MicroDuck USD is absent.

Changed
^^^^^^^

* Changed ``reset_ground_state`` to offer a fifth, **crouching** bucket, with ``crouch_prob``,
  ``crouch_z_range``, ``crouch_joint_pos``, ``crouch_depth_range``, ``crouch_pitch_max`` and
  ``crouch_joint_noise``. Unlike the other four it is a continuum rather than a keyframe: one depth
  draw per environment sets the forward lean, the trunk height and the leg fold together, so the
  bucket seeds resets *across* the crouch-to-stand stretch instead of at one point on it. Existing
  callers are unaffected -- the probability defaults to 0.0 and the partition stays a single
  exclusive draw. The velocity-plus-recovery task needs it because that stretch is the last mile of
  a recovery, and a policy that only reaches it at the tail of a rare good rollout collects almost no
  data there.
* Changed ``feet_air_time_windowed`` and ``com_upward_velocity`` to accept an optional
  ``gate_tilt_above_deg``, which suppresses the first while the robot is toppled and restricts the
  second to a toppled robot. Both default to None and are unchanged for every existing caller.
  Upstream wraps the air-time kernel in a second function that forwards its parameters as keyword
  arguments, and its own extraction warns that adding an ``asset_cfg`` to the wrapped term would
  then silently redirect the gate; a parameter cannot collide.
* Kept the family's NaN-guard norm, which here costs nothing: this is the one task in its upstream
  batch that already has full coverage. ``robot_state_is_nan`` is inherited from the velocity recipe
  with the foot contact sensor named, the two NaN-safe critic foot terms come with it, and every new
  recovery kernel reads its trunk height and vertical speed through ``torch.nan_to_num`` exactly as
  its upstream counterpart does. There is **no deviation from upstream to flag on this task** --
  unlike its siblings in the same batch, which drop guards the port restores.
* Sized the MuJoCo Warp solver budget of this task from profiling rather than inheriting the walking
  task's: ``njmax`` is 128 and ``nconmax`` is 32, against a measured peak of 82 constraints and 27
  contacts per environment under random actions with the tilt termination dropped and the pushes at
  full magnitude. The peak was identical at 256 and at 2048 environments, and matches the stand-up
  task's exactly -- the same model in the same floor-contact regime. The walking task's budget, sized
  for two soles on a plane, is 54 constraints and 10 contacts.
