Added
^^^^^

* Added the contributed MicroDuck forward-roll task ``IsaacContrib-Roulade-Flat-MicroDuck`` under
  ``contrib/microduck/roulade``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The robot starts standing, tips forward over the flat top of its head,
  rolls through 360 degrees and lands back on its feet; like the sit and stand-up tasks it is
  triggered at deployment by the policy switch, so there is no command, no phase clock and no
  reference motion. It shares the velocity and stand-up tasks' 61-wide deployed observation
  contract -- the head-pose and body-pose slots are zero padding here, because the head is part of
  the manoeuvre rather than something to steer -- and it spawns
  :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG`, whose three ``jaw_soft`` head shells are the
  surface the whole task pivots on.

  ``robot_ground_contact``, the sensor behind the anti-breakdance support gate, matches the model's
  ten world colliders against the terrain specifically, as upstream's does, using the Newton
  backend's shape-level ``sensor_shape_prim_expr`` and ``filter_shape_prim_expr``; the accumulator
  reads its per-partner force matrix rather than its net force. That distinction is the gate's whole
  job: a net force cannot tell the floor from the robot's own shin, so a deeply tucked airborne robot
  would read as supported and earn the rotation the gate exists to deny.

  ``head_ground_contact`` remains **narrower than upstream's**: it senses the head body and reports a
  non-zero net contact force rather than a terrain contact, so a knee driven into the head shell
  reads as a head plant. That is guarded downstream, because the over-the-head latch it feeds also
  requires the head top to point at the floor inside a rotation window, so a spurious latch needs the
  robot to be mid-roll with its head already aimed at the ground.
* Added ``RouladeRollState``, ``roulade_roll_state`` and ``reset_roulade_state`` under
  ``contrib/microduck/mdp``. The roll is a stateful task -- nothing in a single frame says how far
  around the robot has come -- so the forward pitch rate is integrated into a per-environment
  accumulator that only advances while the robot is supported and while the roll stays in the
  sagittal plane, and the landing rewards are gated on that integral rather than on a clock. The
  reset event samples a standing start or a mid-roll one, and seeds the bookkeeping to match: a
  mid-roll spawn's accumulator, frontier and paid pointer are pre-set to its spawn pitch and it is
  granted the over-the-head latch, because it never had the chance to earn one.
* Added the MicroDuck forward-roll reward terms under ``contrib/microduck/mdp``:
  ``roulade_progress``, ``roulade_head_pivot``, ``roulade_landing_composite``,
  ``roulade_upright_after_roll``, ``roulade_height_after_roll``, ``roulade_landing_sharp``,
  ``roulade_stand_tax``, ``roulade_rise_velocity``, ``roulade_overspeed_penalty``,
  ``roulade_flatness_penalty``, ``roulade_sagittal_penalty`` and
  ``roulade_lateral_velocity_penalty``, plus the ``roulade_completion_gate`` they are gated by.
  ``roulade_progress`` is the term that advances the accumulator, so it is declared first in the
  reward configuration and the others read the frontier it leaves behind.
* Added ``zero_command_padding`` under ``contrib/microduck/mdp``, which fills a command slot the
  task does not have with zeros so the deployed observation layout stays the same width across the
  whole policy family.
* Added ``contrib/microduck/mdp/symmetry.py``, the family's left-right mirror of the 61-wide
  observation and the 14 servos. The forward-roll task is the only one that switches it on, through
  RSL-RL's symmetry-mirror loss: a roll is left-right symmetric, and the loss fights the sideways
  collapse upstream's runs kept converging to. Data augmentation is deliberately left off, because
  the tables mirror the actor observation and repeat the privileged group unchanged.
* Added ``test/test_microduck_roulade_env.py``, which compares the assembled configuration term by
  term against the upstream reward, event, curriculum, command and observation tables without
  launching the simulator, and then runs the two acceptance tests the port's extraction asks for:
  that a head plant produces head-ground contact, and that a *standing* spawn -- which, unlike a
  mid-roll one, is granted nothing -- can earn the over-the-head latch and open the completion gate.
  It skips the simulator tests when the generated all-collisions MicroDuck USD is absent.

Changed
^^^^^^^

* Changed the forward-roll task's NaN guarding to follow the stand-up task's rather than upstream's.
  Upstream leaves ``robot_state_is_nan``'s sensor list empty on this task and reads its critic
  contact terms unguarded, which its own extraction reads as drift rather than design; this is the
  task that deliberately slams a 0.28 kg head assembly into the floor at 3.5 to 5.5 rad/s, so the
  foot contact sensor is named in the termination and the two NaN-safe critic terms are used. The
  guard only changes behaviour in states that are already broken.
