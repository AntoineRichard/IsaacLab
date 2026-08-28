Added
^^^^^

* Added a ``saturate`` parameter to ``self_collision_cost`` under ``contrib/microduck/mdp``. It
  reports upstream's 0-or-1 "is the robot touching itself" flag instead of a count, and it defaults
  to ``False``, so existing callers are unaffected. Set it whenever the sensor senses *both* sides of
  a contact: a many-to-many sensor reports one contact once per shape that carries it, so counting
  would scale the penalty with how finely the model is split into colliders rather than with how much
  of the robot is folded onto itself.
* Added ``MICRODUCK_ALLCOLLISIONS_COLLIDER_XFORMS`` and
  ``MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR`` under
  ``contrib/microduck/velocity/velocity_env_cfg``, which name the ten colliders the conversion leaves
  enabled on :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG` and turn them into a shape-level
  prim expression. The stand-up and forward-roll tasks share them.

Changed
^^^^^^^

* Changed the MicroDuck stand-up and forward-roll tasks' ``self_collision`` sensor from the trunk
  against seven bodies to the model's ten enabled colliders against each other, many-to-many, which
  is upstream's trunk-subtree-against-itself sensor. The narrower sensor could only see contacts with
  the trunk at one end; the widened one also sees limb against limb, with no trunk at either end.
  Both the folded and the crossed families are reachable inside the joint limits: a deep knee fold
  puts each shin into the hip shell on its own side, and swinging both legs through the sagittal
  chain -- hip pitch, knee and ankle all range over plus or minus 1.571 rad -- brings the feet across
  the midline, where the modest hip yaw and roll travel is enough to overlap them. Sole against sole,
  sole against shin and shin against shin all register on the widened sensor.

  Both tasks' ``self_collisions`` reward now passes ``saturate=True``, which holds the term on
  upstream's 0/1 scale, so the configured weights keep their meaning and are not rescaled by the
  wider sensor. Anyone who has copied either task and reads the reward's magnitude should expect the
  term to fire in poses it previously missed.
* Changed the MicroDuck forward-roll task's ``robot_ground_contact`` sensor to filter its colliders
  against the terrain, and its support gate to read the resulting force matrix rather than the net
  contact force. The gate now answers "is the robot touching the *floor*" instead of "is the robot
  touching anything", closing the direction in which it was permissive: a self-contacting airborne
  robot previously read as supported and accumulated rotation, which is the ballistic whip the gate
  exists to prevent. ``roulade_progress`` and ``_update_roulade_state`` therefore require a
  terrain-filtered support sensor and raise if given an unfiltered one.
* Changed the MicroDuck velocity and roller tasks not at all: their ``self_collision`` sensor stays
  one sole against the other. That is not a workaround for a missing capability but the whole
  reachable signal on the walking model, whose remaining self-collision geometries the conversion
  disables because the importer cannot represent the MJCF's ``contype``/``conaffinity`` masks. The
  configuration now points at the shape-level recipe to use if the converter ever regains collision
  groups.
