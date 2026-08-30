Added
^^^^^

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
