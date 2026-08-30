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
