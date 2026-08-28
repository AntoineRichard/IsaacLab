Changed
^^^^^^^

* Changed :data:`~isaaclab_assets.MICRODUCK_CFG` -- and with it
  :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG` and
  :data:`~isaaclab_assets.MICRODUCK_ROLLERS_CFG`, which derive from it -- to stop configuring a
  solver joint ``friction`` on its BAM servo group. The dry friction belongs to the servo model on
  both execution paths, as it does upstream, whose binding zeroes the MJCF's ``frictionloss`` on
  every joint it drives: the Isaac Lab-executed model now zeroes the group's solver friction itself
  (see the ``isaaclab`` entry on
  :attr:`~isaaclab.actuators.ActuatorBase.applies_joint_friction`), and the Newton-native
  controller republishes its own budget over any seed every physics step. The configured value was
  therefore either ignored-with-a-warning or overwritten, and the module-level
  ``MICRODUCK_JOINT_FRICTION`` constant it named is removed. The joint *viscous* friction
  :data:`~isaaclab_assets.MICRODUCK_JOINT_DAMPING` supplies is unchanged and still restored on both
  paths, because the reference implementation also damps the joint through MuJoCo's ``dof_damping``.
