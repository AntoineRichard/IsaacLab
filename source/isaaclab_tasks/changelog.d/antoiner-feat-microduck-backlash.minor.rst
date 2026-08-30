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
