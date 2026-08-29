Added
^^^^^

* Added the contributed MicroDuck stand-up-on-skates task
  ``IsaacContrib-RollerStandUp-Flat-MicroDuck`` under ``contrib/microduck/rollerstandup``, with an
  RSL-RL PPO configuration carrying upstream's hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. An episode starts the robot face down, face up or already standing on its
  skates and asks it to reach and hold the standing station within six seconds; there is no
  trajectory and no waypoint gating, so the policy discovers its own rise path. The nineteen-term
  reward stack is the wheel-less stand-up recipe's rise stack on the roller robot, with the skating
  task's eight smoothness regularizers carried over.
* Added ``test/test_microduck_rollerstandup_env.py``, including the two tests that stand in for an
  accuracy gate against upstream -- see the note on that below.

Changed
^^^^^^^

* Resolved every joint selection **by name**, where upstream hard-codes indices, and this is a
  divergence to *function* rather than a preference. Upstream's ``_LEG_JOINTS`` are the legs'
  positions in the roller model's eighteen-joint layout, and it feeds them to helpers that have
  already collapsed to the fourteen-wide servo view, so the last two are out of bounds and its
  environment raises ``IndexError`` on the first reward evaluation -- **upstream's version of this
  task cannot run at the pinned commit** (upstream issue draft 016). Naming the joints removes the
  class of bug rather than the instance. A test records the defect by indexing the same servo view
  with upstream's indices and asserting it fails.
* Replaced the accuracy gate against upstream with an **internal acceptance test**, because there is
  no runnable upstream to gate against. It spawns the robot fallen on its skates -- face down and
  face up, at their measured rest heights -- and at the standing station, evaluates the task's own
  nineteen-term reward stack end to end under physics through the reward manager, and asserts the
  ordering the task exists to teach: standing scores several times what face-down does, face-down
  scores above face-up, and ``standing_composite`` is what separates them.
* Inverted the wheel-friction curriculum relative to the skating task's, as upstream does: it starts
  with the bearings almost locked, which makes them behave like feet, and relaxes them over four
  thousand iterations toward the same 0.0015 the skating task ramps *up* to. A rolling contact has no
  longitudinal grip, so there is nothing for a fallen robot to push against, and this is how the
  gesture is bootstrapped. It carries a **deployment gate**: only checkpoints from iteration 4000
  onward are candidates for the robot, since before that the policy leans on a rolling friction the
  hardware does not have. The gate is recorded on the runner configuration and asserted reachable
  inside upstream's iteration budget.
* Kept upstream's sign fix on ``gentle_rise``, which takes a **positive** weight. The kernel already
  returns ``-|a_z|``, so the ``-0.02`` inherited from the wheel-less stand-up task was a double
  negative that *rewarded* vertical acceleration; upstream measured it logging as the only penalty
  term with a positive episode return, identified it as the cause of a violent rise, and flipped it.
* Collapsed the ``action_rate_l2`` weight to the single value that was ever live. Upstream declares
  ``-0.6`` on the term and ``-0.4`` at stage 0 of the curriculum that owns it, and the curriculum
  manager runs before the first reward evaluation, so the declared literal was dead. The ramp itself
  is the wheel-less stand-up task's rather than the skating task's, deliberately: the skating ramp
  reaches ``-2.0`` for a calm gait, which is a movement blocker on a task whose point is a fast rise.
* Kept the family's NaN-guard termination with its sensor list, where upstream reaches for a
  different mechanism on this task alone -- an observation-level ``nan_policy = "sanitize"`` that
  zeroes the offending columns and keeps running. An environment quietly training on zeroed
  observations is harder to notice than a spike in resets, so the port applies one policy uniformly.
* Spelled out the event suite rather than inheriting the skating task's, because declaration order is
  behavior here: ``set_ground_state`` overwrites what the root reset wrote, so it has to fire between
  the root reset and the randomizations rather than after them.
* Sized the MuJoCo Warp contact budget from profiling rather than inheriting: ``njmax`` is 192 and
  ``nconmax`` is 34, against a measured peak of 98 constraints and 28 contacts per environment under
  random actions with the pushes forced to full magnitude, profiled at 256, 2048 and 4096
  environments. This robot spends most of every episode with its head, shoulders and hips on the
  floor, which the skating profile does not cover. The **solver iteration counts are upstream's**
  template values, 10 and 20.
