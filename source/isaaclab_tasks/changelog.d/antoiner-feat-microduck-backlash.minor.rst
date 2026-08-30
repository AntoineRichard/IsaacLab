Added
^^^^^

* Added the contributed MicroDuck gear-backlash task ``IsaacContrib-Velocity-Flat-MicroDuck-Backlash``
  under ``contrib/microduck/velocity``, ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and running on the
  Newton MJWarp backend. It is the flat velocity task on
  :data:`~isaaclab_assets.MICRODUCK_BACKLASH_CFG`, whose 14 servos each drive an unactuated
  ``passive_<servo>_backlash`` hinge carrying a degree of free travel either way, and whose encoders
  read through it. **The robot has 28 joints and the task still has 14 actions and the family's
  61-wide actor**, because the two joint blocks report the servos' encoders rather than every degree
  of freedom -- so the deployed runtime and every ONNX export are unaffected.

  It is an A/B experiment on the plant rather than a new recipe, and it therefore reuses the base
  velocity PPO configuration, as upstream does, and differs from the base flat task in exactly four
  places: the robot, the encoder view in both observation groups, the soft joint-limit penalty's
  selection, and the solver's constraint budget. That penalty, ``dof_pos_limits``, is scoped to the
  14 servos, since a play hinge rides its limits by construction and leaving it unscoped would be a
  permanent tax; the armature randomization
  deliberately still covers the play hinges, as upstream's does. Upstream's other two edits -- an
  injected servo-only observation selection and a posture-reward pattern disambiguation -- have
  nothing to do here, because every selection in this package is spelled out as exact joint names
  and Isaac Lab matches names in full.
* Sized the MuJoCo Warp constraint budget of the backlash task from profiling rather than inheriting
  the base flat task's: ``njmax`` is 96 against a measured peak of 65 constraints per environment at
  256 environments and 66 at both 2048 and 4096, under random actions with the tilt termination
  dropped. The structural bound is the base task's 54 plus one permanently active limit row per play
  hinge, and the base task's shipped 64 is below the measured peak. ``nconmax`` is unchanged at 10:
  the play hinges are joints, not colliders. The extra rows are not free: at 4096 environments the
  played plant takes 24% more solver iterations and runs 29% fewer environment steps per second than
  the plain one.
* Added ``joint_pos_rel_backlash`` and ``joint_vel_rel_backlash`` under ``contrib/microduck/mdp``,
  the encoder-view joint observations of the MicroDuck gear-backlash plant. A servo's magnetic
  encoder sits on the output side of the gearbox, so the reading is
  ``qpos[servo] + qpos[passive_<servo>_backlash]`` rather than the motor angle, and the position
  term composes the per-robot encoder bias on the **servo** reading only -- one encoder per servo
  means one calibration error per servo, and the play summand stays raw. The pairing is resolved
  once per selection from
  :data:`~isaaclab.actuators.actuator_bam_cfg.BACKLASH_JOINT_TEMPLATE`, the same naming contract the
  Newton BAM binding uses, and a servo with no play hinge is masked off individually, so both terms
  reduce to ``joint_pos_rel_biased`` and ``joint_vel_rel`` exactly on the models that carry none.

Changed
^^^^^^^

* Changed ``head_pose_tracking`` and ``head_pose_bias_penalty`` under ``contrib/microduck/mdp`` to
  measure the head through the gear play where the model has any, matching what the policy observes.
  Both terms track a quantity the actor reads through its encoders, so on the backlash plant they
  have to score the link angle rather than the motor's: measuring the motor would let the head droop
  the play free of charge *and* charge the policy for biasing the servo up to correct the droop it
  can see. On every model without ``passive_<servo>_backlash`` hinges -- which is every MicroDuck
  model but the backlash one -- the lookup finds nothing and both terms are unchanged.
