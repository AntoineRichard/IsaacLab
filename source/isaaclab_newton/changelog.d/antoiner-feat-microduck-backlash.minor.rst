Added
^^^^^

* Added ``solreflimit``, ``solimplimit`` and ``damping`` to
  :class:`~isaaclab_newton.sim.schemas.MujocoJointCfg`, so an asset can tune a joint's MuJoCo limit
  constraint and its passive DOF damping per joint prim rather than per articulation. The pair
  authored in ``mjc:solreflimit`` reaches ``jnt_solref`` verbatim, which also opts the joint out of
  the default-``solref`` retag of
  :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_default_joint_limit_solref`;
  ``mjc:solimplimit`` reaches ``jnt_solimp``, and ``mjc:damping`` reaches ``dof_damping`` in
  MuJoCo's per-radian units. Damping is the only channel a joint no actuator group owns has, since
  an actuator model cannot republish damping for a joint it does not drive. All three are inert on
  every solver other than MuJoCo Warp.
* Added resolution of :class:`~isaaclab.actuators.BamBacklashActuatorCfg`'s encoder binding at
  articulation initialization. Each joint the group selects is paired with the
  ``passive_<joint>_backlash`` hinge in series with it, and the resulting per-DOF position indices
  and mask are bound on the Newton BAM controller before the first step, so a servo's firmware loop
  closes on the joint's angle *plus* its gear play. A selected joint the articulation carries no
  such hinge for is masked off and keeps the plain servo's behaviour bit for bit. The pairing is by
  name against the group's own joint selection and never against which joints carry a drive: the
  play hinges do carry one, since the converted assets author a zero-gain force drive on every
  hinge no actuator drives. The indices are in the joint *coordinate* layout, which is the one the
  controller dereferences them in and is not the degree-of-freedom layout an actuator's own indices
  use once the articulation has a floating base. The binding is copied into the arrays the
  controller was finalized with, so a captured decimation graph reads it live rather than baking it
  in.
