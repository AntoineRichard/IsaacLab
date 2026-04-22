# Newton API gaps blocking Odin environments

**Input:** `tools/odin/config/newton_gap_candidates.yaml`
**Scope:** Gaps blocking PhysX-kept envs from running on Newton
(`physx_envs.yaml` ∩ ¬ `newton_envs.yaml`), plus related observations
surfaced by T2.1 enumeration.

**Status:** Authored — end of T2.1 curation pass. All
`tools/odin/config/newton_gap_candidates.yaml` rows carry a controlled-vocabulary
`suspected_gap` value (see :data:`tools.odin.common.env_list.GAP_VOCABULARY`);
the appendix is rendered directly from that YAML.

---

## 1. Framework-coverage gap (not a Newton issue)

The enumeration script produced 55 `framework: null` rows in
`physx_envs.yaml`. Odin only dispatches RSL-RL and SKRL, so these envs
are automatically excluded from benchmarking — but most of them are not
*truly* frameworkless; they register `rl_games_cfg_entry_point` and
nothing else. This is an IsaacLab modernization gap, not a Newton
physics gap: we are phasing out rl_games, and these envs need migration
to RSL-RL (or SKRL where appropriate) before Odin can cover them.

### What the `has_rl_games` flag surfaces

`EnvEntry.has_rl_games` is `True` whenever `rl_games_cfg_entry_point`
is registered, independent of whether RSL-RL or SKRL are also registered.
When `framework` is `None` and `has_rl_games` is `True`, the
auto-rejection note reads:

> rl_games-only registration — not dispatched by Odin. Migrate to
> rsl_rl or skrl to enable benchmarking.

### Breakdown of frameworkless rows (from current physx_envs.yaml)

Confirmed rl_games-only families (registered `rl_games_cfg_entry_point`
only):

| Group | Envs blocked | Notes |
|---|---|---|
| `direct/automate` | 2 | Nut-and-bolt assembly (`AutoMate-Assembly-Direct`, `AutoMate-Disassembly-Direct`). |
| `direct/factory` | 3 | Classic Factory tasks. |
| `direct/forge` | 3 | Forge insertion tasks. |

Families registered with **no** learning framework at all (need both
modernization AND RSL-RL/SKRL registration):

| Group | Envs blocked | Observation |
|---|---|---|
| `direct/humanoid_amp` | 3 | No learning entry point in `__init__.py`. |
| `manager_based/manipulation/stack` | 21 | The single biggest contributor to the frameworkless count; the whole Stack family needs an RL training pipeline defined. |
| `manager_based/manipulation/pick_place` | 5 | Similar to Stack. |
| `manager_based/manipulation/place` | 2 | Similar. |
| `manager_based/locomanipulation/pick_place` | 2 | Similar. |
| `manager_based/manipulation/cabinet` | 2 | **Partial** — these are the `IK-Abs` / `IK-Rel` variants; the base Franka/OpenArm Cabinet tasks *do* register RSL-RL. The IK variants are the kinematic controllers, not training variants, and may not need RL at all. |
| `manager_based/manipulation/lift` | 3 | Same pattern — `IK-Abs`/`IK-Rel` + `Lift-Teddy-Bear-Franka-IK-Abs-v0`. |
| `manager_based/manipulation/reach` | 2 | Same — `Reach-Franka-IK-Abs/Rel-v0`. |
| `manager_based/manipulation/deploy` | 2 | Mix of IK-variant + Dexsuite scope; see "Deploy-family duplication" below. |
| `manager_based/manipulation/pick_place` | 5 | Also no framework. |
| `manager_based/classic/cartpole` | 4 | Needs investigation — classic cartpole variants should have at least one framework. |
| `direct/shadow_hand` | 1 | One variant of the ShadowHand family is frameworkless; the other 5 are fine. |

### Recommendation

Treat rl_games-only registrations as an explicit technical-debt item,
not a Newton gap. When Odin's dispatcher queue is finalized, these envs
show up with `keep: false` and the migration note; re-running the
enumeration after migration will flip `framework: null` → `framework:
rsl_rl` (or `skrl`) and clear `has_rl_games` is irrelevant at that point.

The **21 Stack tasks** are the highest-value target: an entire task
family currently invisible to Odin. Any unified RL training recipe for
Stack unlocks 21 rows at once.

---

## 2. Deploy-family duplication observation

The `manager_based/manipulation/deploy/` directory mirrors pieces of
`manager_based/manipulation/reach/` and `…/gear_assembly/`-style
manipulation but registered as a parallel family (`Isaac-Deploy-*`).
What Odin enumeration surfaces:

| Original family | Task | Deploy analogue | Differences |
|---|---|---|---|
| `manipulation/reach/config/ur_10` | `Isaac-Reach-UR10-v0` | `manipulation/deploy/reach/config/ur_10e` → `Isaac-Deploy-Reach-UR10e-v0` | Different gripper: UR10 (no suffix) vs UR10e (with wrist cam / alt gripper). Separate runner cfg (`UR10ReachPPORunnerCfg` vs `URReachPPORunnerCfg`). |
| *(none — new robot)* | — | `Isaac-Deploy-Reach-Rizon4s-v0` | Flexiv Rizon 4s robot: only lives in Deploy. |
| *(none — new task)* | — | `Isaac-Deploy-GearAssembly-UR10e-2F{85,140}-v0` | Gear assembly task + two gripper variants (2F85, 2F140). Only in Deploy. |

Plus a `*-ROS-Inference-v0` variant for every Deploy task (separate env
cfg wired to a ROS-2 action/observation bridge; same runner cfg as the
training variant).

### Why the duplication is concerning for benchmarking

1. **Runtime-perf redundancy** — `Isaac-Deploy-Reach-UR10e-v0` and
   `Isaac-Reach-UR10-v0` exercise the same core task (reach training,
   manipulator arm) on slightly different robots. Running both in an
   Odin sweep doubles wall-time without doubling information if our
   goal is a reach-family perf baseline.
2. **Path-level duplication** — `manipulation/deploy/reach/config/ur_10e/`
   parallels `manipulation/reach/config/ur_10/`. The Deploy version
   doesn't reuse the reach-family env cfg; it defines its own. If the
   base reach task changes, Deploy's copy can silently drift.
3. **The `ROS-Inference` variants are not training tasks** — they wire
   the trained policy to a ROS-2 I/O bridge for sim-to-real deployment.
   Benchmarking them like training runs would measure something
   different (single-policy rollout, not PPO convergence). These should
   be `keep: false` in `physx_envs.yaml` regardless of Newton status.

### Recommendation (curation guidance, not a code change)

For T2.1's curated `physx_envs.yaml`:

- **Deploy ROS-Inference rows** (`*-ROS-Inference-v0`): set `keep:
  false`, `notes: "Deployment inference variant — not a training
  task."` (5 such rows.)
- **Deploy training rows** (`Isaac-Deploy-Reach-UR10e-v0`,
  `Isaac-Deploy-Reach-Rizon4s-v0`): keep only if we want the deploy
  robot variants in the benchmark sweep; otherwise `keep: false` with
  a note pointing at the reach-family analogue.
- **Deploy GearAssembly rows**: unique task, keep.

Treat Deploy vs reach as a *benchmark duplication* concern, not a bug:
Deploy intentionally mirrors the upstream families with deploy-specific
robots and ROS hooks; it is not a code-refactor target for T2.1.

---

## 3. Newton API gaps

Every non-`tbd` row in `newton_gap_candidates.yaml` was assigned a
category from `GAP_VOCABULARY`. The sections below are ordered by
bucket size, from largest to smallest.

| Category | Envs blocked | keep=true | Nature |
|---|---:|---:|---|
| `preset_missing` | 41 | 15 | Pure wiring — **not** an API gap |
| `rough_terrain` | 20 | 0 | API gap — heightfield locomotion |
| `manipulation_coverage` | 6 | 0 | Coverage gap — manipulation scenes untested |
| `sdf_collision` | 5 | 1 | API gap — SDF colliders |
| `parallel_joints` | 2 | 0 | API gap — closed-loop kinematics |
| `controller_untested` | 2 | 0 | Coverage gap — low-level controller |
| `tendons` | 1 | 0 | API gap — tendon actuation |
| `deformable` | 0 | — | — |
| `other` | 0 | — | — |

`keep=true` counts show how many of the category's envs are on Odin's
dispatch shortlist (per user curation in `newton_envs.yaml` or
`newton_gap_candidates.yaml`), i.e. rows the user explicitly cares
about. Categories with `keep=true = 0` are still worth closing for
coverage, but they don't block today's benchmark set.

### 3.1 `preset_missing` — 41 envs, 15 keep=true

**Not an API gap.** Newton already supports the physics these envs
need; they simply don't declare a `newton` `PresetCfg` in their env
config, so `enumerate_newton_envs.py` classified them as gap
candidates.

**Unlock value:** high per-env effort-to-impact ratio — closing this
bucket is mechanical preset wiring. The 15 `keep=true` envs in
particular are tasks the user has explicitly opted into; getting them
running on Newton unlocks 15 rows of the Odin benchmark set with no
Newton backlog involvement.

**Highlights of the 15 keep=true rows:**

| Env | Group |
|---|---|
| `Isaac-Velocity-Flat-Anymal-C-Direct-v0` | `direct/anymal_c` |
| `Isaac-Cart-Double-Pendulum-Direct-v0` | `direct/cart_double_pendulum` |
| `Isaac-Cartpole-Depth-Camera-Direct-v0` | `direct/cartpole` |
| `Isaac-Cartpole-RGB-Camera-Direct-v0` | `direct/cartpole` |
| `Isaac-Franka-Cabinet-Direct-v0` | `direct/franka_cabinet` |
| `Isaac-Quadcopter-Direct-v0` | `direct/quadcopter` |
| `Isaac-TrackPositionNoObstacles-ARL-Robot-1-v0` | `manager_based/drone_arl/track_position_state_based` |
| `Isaac-Open-Drawer-Franka-v0` | `manager_based/manipulation/cabinet` |
| `Isaac-Open-Drawer-OpenArm-v0` | `manager_based/manipulation/cabinet` |
| `Isaac-Deploy-Reach-Rizon4s-v0` | `manager_based/manipulation/deploy` |
| `Isaac-Deploy-Reach-UR10e-v0` | `manager_based/manipulation/deploy` |
| `Isaac-Repose-Cube-Allegro-Play-v0` | `manager_based/manipulation/inhand` |
| `Isaac-Repose-Cube-Allegro-v0` | `manager_based/manipulation/inhand` |
| `Isaac-Reach-OpenArm-v0` | `manager_based/manipulation/reach` |
| `Isaac-Navigation-Flat-Anymal-C-v0` | `manager_based/navigation/config` |

**Action:** file preset-wiring tickets per task family; no Newton API
backlog involvement needed.

### 3.2 `rough_terrain` — 20 envs, 0 keep=true

**API gap.** Heightfield / procedural terrain. All 20 envs are the
Rough variants of `manager_based/locomotion/velocity` across every
legged-robot in the suite (Anymal-B/C/D, Cassie, Digit, G1, H1,
Unitree-A1/Go1/Go2) plus their `Play` companions.

**Unlock value:** medium — the user has set `keep=false` on the whole
family for now, so nothing in today's benchmark set hinges on closing
the gap. But rough-terrain locomotion is a flagship IsaacLab
benchmark and will almost certainly come back into scope once Newton
supports it.

**What's missing:** heightfield-based terrain support in Newton (the
`manager_based/locomotion/velocity` rough variants rely on the
heightfield sampler + friction model that PhysX provides today).

**Action:** confirm the exact Newton API surface needed (heightfield
sampler, stochastic-terrain friction, etc.) with the Newton team.
Single API gap unlocks 20 envs at once.

### 3.3 `manipulation_coverage` — 6 envs, 0 keep=true

**Coverage gap, not an API gap.** Newton has the physics primitives
(rigid bodies + contact) to run these envs; what's missing is *testing*
of the manipulation-scene integration. Affected:

- `manager_based/manipulation/lift/Isaac-Lift-Cube-{Franka,OpenArm}-v0`
  (and their `Play` variants).
- `manager_based/locomanipulation/tracking/Isaac-Tracking-LocoManip-Digit-v0`
  (and its `Play` variant).

**Unlock value:** medium — `keep=false` across the category today, but
these are standard manipulation baselines that the user is likely to
re-enable as Newton stabilises.

**Action:** run the four training variants on Newton manually; file
bugs against Newton only if the runs diverge from PhysX. Closure is a
test-matrix item, not an API change.

### 3.4 `sdf_collision` — 5 envs, 1 keep=true

**API gap.** SDF-based colliders. The five envs break into two
subfamilies:

- `direct/anymal_c/Isaac-Velocity-Rough-Anymal-C-Direct-v0` (1, `keep=true`)
  — also requires rough-terrain support (heightfield); dual-categorized
  in `notes`.
- `manager_based/manipulation/deploy/Isaac-Deploy-GearAssembly-UR10e-*`
  (4 envs: 2F140 and 2F85 grippers, each with a `v0` training variant
  and a `ROS-Inference-v0` deployment variant; all `keep=false`).

**Unlock value:** high. The one `keep=true` row is the user's only
path to an on-Newton rough-terrain locomotion benchmark (it doubles as
a `rough_terrain` candidate). Gear assembly is a marquee task family
that would come back into scope once Newton + rl_games-migration work
unblocks it (see §1).

**What's missing:** SDF collider evaluation on Newton. The assembly
family additionally needs precise normal/friction handling during
tight-clearance insertion.

**Action:** prioritise SDF collider support in Newton. Single gap
unlocks the Rough-Anymal training target and the entire gear-assembly
family once they migrate off rl_games.

### 3.5 `parallel_joints` — 2 envs, 0 keep=true

**API gap.** Closed-loop / parallel kinematic constraints.

- `manager_based/locomotion/velocity/Isaac-Velocity-Flat-Digit-v0`
- `manager_based/locomotion/velocity/Isaac-Velocity-Flat-Digit-Play-v0`

Digit's kinematic chain has parallel linkages (passive joints); PhysX
models these directly but Newton currently lacks the constraint type.

**Unlock value:** low in the current benchmark set (`keep=false` on
both rows), but any future humanoid adopting parallel linkages will
surface the same gap.

**Action:** confirm whether Newton has constraint primitives
sufficient to model parallel linkages and file a single upstream
ticket if not.

### 3.6 `controller_untested` — 2 envs, 0 keep=true

**Coverage gap.** Operational-space-controller (OSC) variants of the
Franka reach task:

- `manager_based/manipulation/reach/Isaac-Reach-Franka-OSC-v0`
- `manager_based/manipulation/reach/Isaac-Reach-Franka-OSC-Play-v0`

The underlying physics (rigid-body manipulator + joint torque
commands) is available on Newton; what's untested is the OSC
controller's stability when driven by Newton's integration step.

**Unlock value:** low in scope today (`keep=false`). OSC is a
research-grade controller that will matter for fine-manipulation
benchmarks later.

**Action:** drive the controller on Newton manually; file bugs only
if the solver diverges.

### 3.7 `tendons` — 1 env, 0 keep=true

**API gap.** Tendon actuation.

- `direct/shadow_hand_over/Isaac-Shadow-Hand-Over-Direct-v0`

**Unlock value:** low in the current sweep (`keep=false`), but
dexterous-manipulation research historically leans on the ShadowHand
and any future dexterity benchmark would re-surface this gap.

**What's missing:** tendon-actuation primitive in Newton.

**Action:** low-priority; batch with any broader tendon / soft-body
work when prioritised.

### 3.8 `deformable` — 0 envs

No currently-enumerated envs require deformable / softbody physics
(cloth, rope, FEM). The category is retained in `GAP_VOCABULARY` for
forward compatibility — if the DeformableObject integration adds new
training envs, they'll land here.

### 3.9 `other` — 0 envs

No rows fell outside the other eight categories. The category is
retained as an escape hatch (`notes:` required when used).

---

## Appendix: per-env table

Rendered from `tools/odin/config/newton_gap_candidates.yaml`. Rows
grouped by gap category (in vocabulary order), then by group, then by
task_id. `Keep` reflects the user's curation in the gap-candidates
YAML — `**no**` means the row is excluded from the Odin dispatch set
even if the gap were closed.

Regenerate with the one-liner in
`docs/superpowers/plans/2026-04-22-odin-t2-1-env-lists.md` Task 14 (use
the order `GAP_VOCABULARY → group → task_id`).

| Task | Group | Gap | Keep | Notes |
|------|-------|-----|------|-------|
| `Isaac-Velocity-Flat-Anymal-C-Direct-v0` | `direct/anymal_c` | `preset_missing` | yes |  |
| `Isaac-Cart-Double-Pendulum-Direct-v0` | `direct/cart_double_pendulum` | `preset_missing` | yes |  |
| `Isaac-Cartpole-Albedo-Camera-Direct-v0` | `direct/cartpole` | `preset_missing` | **no** |  |
| `Isaac-Cartpole-Depth-Camera-Direct-v0` | `direct/cartpole` | `preset_missing` | yes |  |
| `Isaac-Cartpole-RGB-Camera-Direct-v0` | `direct/cartpole` | `preset_missing` | yes |  |
| `Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0` | `direct/cartpole` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0` | `direct/cartpole` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0` | `direct/cartpole` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0` | `direct/cartpole_showcase` | `preset_missing` | **no** | Lets not do the showcase for cartpole |
| `Isaac-Franka-Cabinet-Direct-v0` | `direct/franka_cabinet` | `preset_missing` | yes |  |
| `Isaac-Quadcopter-Direct-v0` | `direct/quadcopter` | `preset_missing` | yes |  |
| `Isaac-TrackPositionNoObstacles-ARL-Robot-1-Play-v0` | `manager_based/drone_arl/track_position_state_based` | `preset_missing` | **no** |  |
| `Isaac-TrackPositionNoObstacles-ARL-Robot-1-v0` | `manager_based/drone_arl/track_position_state_based` | `preset_missing` | yes |  |
| `Isaac-Open-Drawer-Franka-Play-v0` | `manager_based/manipulation/cabinet` | `preset_missing` | **no** |  |
| `Isaac-Open-Drawer-Franka-v0` | `manager_based/manipulation/cabinet` | `preset_missing` | yes |  |
| `Isaac-Open-Drawer-OpenArm-Play-v0` | `manager_based/manipulation/cabinet` | `preset_missing` | **no** |  |
| `Isaac-Open-Drawer-OpenArm-v0` | `manager_based/manipulation/cabinet` | `preset_missing` | yes |  |
| `Isaac-Deploy-Reach-Rizon4s-Play-v0` | `manager_based/manipulation/deploy` | `preset_missing` | **no** |  |
| `Isaac-Deploy-Reach-Rizon4s-ROS-Inference-v0` | `manager_based/manipulation/deploy` | `preset_missing` | **no** |  |
| `Isaac-Deploy-Reach-Rizon4s-v0` | `manager_based/manipulation/deploy` | `preset_missing` | yes |  |
| `Isaac-Deploy-Reach-UR10e-Play-v0` | `manager_based/manipulation/deploy` | `preset_missing` | **no** |  |
| `Isaac-Deploy-Reach-UR10e-ROS-Inference-v0` | `manager_based/manipulation/deploy` | `preset_missing` | **no** |  |
| `Isaac-Deploy-Reach-UR10e-v0` | `manager_based/manipulation/deploy` | `preset_missing` | yes |  |
| `Isaac-Repose-Cube-Allegro-NoVelObs-Play-v0` | `manager_based/manipulation/inhand` | `preset_missing` | **no** |  |
| `Isaac-Repose-Cube-Allegro-NoVelObs-v0` | `manager_based/manipulation/inhand` | `preset_missing` | **no** |  |
| `Isaac-Repose-Cube-Allegro-Play-v0` | `manager_based/manipulation/inhand` | `preset_missing` | yes |  |
| `Isaac-Repose-Cube-Allegro-v0` | `manager_based/manipulation/inhand` | `preset_missing` | yes |  |
| `Isaac-Reach-OpenArm-Bi-Play-v0` | `manager_based/manipulation/reach` | `preset_missing` | **no** |  |
| `Isaac-Reach-OpenArm-Bi-v0` | `manager_based/manipulation/reach` | `preset_missing` | **no** |  |
| `Isaac-Reach-OpenArm-Play-v0` | `manager_based/manipulation/reach` | `preset_missing` | **no** |  |
| `Isaac-Reach-OpenArm-v0` | `manager_based/manipulation/reach` | `preset_missing` | yes |  |
| `Isaac-Navigation-Flat-Anymal-C-Play-v0` | `manager_based/navigation/config` | `preset_missing` | **no** |  |
| `Isaac-Navigation-Flat-Anymal-C-v0` | `manager_based/navigation/config` | `preset_missing` | yes |  |
| `Isaac-Velocity-Rough-Anymal-C-Direct-v0` | `direct/anymal_c` | `sdf_collision` | yes | Also requires rough_terrain support (heightfield). |
| `Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0` | `manager_based/manipulation/deploy` | `sdf_collision` | **no** |  |
| `Isaac-Deploy-GearAssembly-UR10e-2F140-v0` | `manager_based/manipulation/deploy` | `sdf_collision` | **no** |  |
| `Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0` | `manager_based/manipulation/deploy` | `sdf_collision` | **no** |  |
| `Isaac-Deploy-GearAssembly-UR10e-2F85-v0` | `manager_based/manipulation/deploy` | `sdf_collision` | **no** |  |
| `Isaac-Shadow-Hand-Over-Direct-v0` | `direct/shadow_hand_over` | `tendons` | **no** |  |
| `Isaac-Velocity-Rough-Anymal-B-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Anymal-B-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Anymal-C-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Anymal-C-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Anymal-D-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Anymal-D-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Cassie-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Cassie-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Digit-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Digit-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-G1-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-G1-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-H1-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-H1-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Unitree-A1-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Unitree-A1-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Unitree-Go1-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Unitree-Go1-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Unitree-Go2-Play-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Velocity-Rough-Unitree-Go2-v0` | `manager_based/locomotion/velocity` | `rough_terrain` | **no** |  |
| `Isaac-Tracking-LocoManip-Digit-Play-v0` | `manager_based/locomanipulation/tracking` | `manipulation_coverage` | **no** |  |
| `Isaac-Tracking-LocoManip-Digit-v0` | `manager_based/locomanipulation/tracking` | `manipulation_coverage` | **no** |  |
| `Isaac-Lift-Cube-Franka-Play-v0` | `manager_based/manipulation/lift` | `manipulation_coverage` | **no** |  |
| `Isaac-Lift-Cube-Franka-v0` | `manager_based/manipulation/lift` | `manipulation_coverage` | **no** |  |
| `Isaac-Lift-Cube-OpenArm-Play-v0` | `manager_based/manipulation/lift` | `manipulation_coverage` | **no** |  |
| `Isaac-Lift-Cube-OpenArm-v0` | `manager_based/manipulation/lift` | `manipulation_coverage` | **no** |  |
| `Isaac-Velocity-Flat-Digit-Play-v0` | `manager_based/locomotion/velocity` | `parallel_joints` | **no** |  |
| `Isaac-Velocity-Flat-Digit-v0` | `manager_based/locomotion/velocity` | `parallel_joints` | **no** |  |
| `Isaac-Reach-Franka-OSC-Play-v0` | `manager_based/manipulation/reach` | `controller_untested` | **no** |  |
| `Isaac-Reach-Franka-OSC-v0` | `manager_based/manipulation/reach` | `controller_untested` | **no** |  |
