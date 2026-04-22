# Newton API gaps blocking Odin environments

**Input:** `tools/odin/config/newton_gap_candidates.yaml`
**Scope:** Gaps blocking PhysX-kept envs from running on Newton
(`physx_envs.yaml` ∩ ¬ `newton_envs.yaml`), plus related observations
surfaced by T2.1 enumeration.

**Status:** Draft skeleton. Framework-coverage and Deploy-duplication
sections are authored; the per-Newton-gap sections and the appendix are
placeholders to fill during T2.1's human curation pass.

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

*To be filled in during the Task 14 curation pass, once every row in
`tools/odin/config/newton_gap_candidates.yaml` has been assigned a
category from the controlled vocabulary (`sdf_collision`, `tendons`,
`rough_terrain`, `manipulation_coverage`, `deformable`, `other`).*

For each gap category, fill in:

- **Envs blocked:** count from `newton_gap_candidates.yaml`.
- **Unlock value:** high / medium / low based on how many envs the gap
  blocks and how commonly they're used for benchmarking.
- **What's missing:** what API / feature Newton would need to provide.
- **Effort estimate:** rough sizing based on the Newton backlog.
- **Upstream link:** Newton issue / discussion URL if one exists.

### 3.1 SDF collisions

*TODO.* Candidates: `direct/anymal_c` rough variant, any nut-and-bolt /
gear-assembly tasks that survive rl_games migration.

### 3.2 Tendons

*TODO.* Candidates: any dexterous-hand task using tendon-actuated
fingers.

### 3.3 Rough terrain (heightfield)

*TODO.* Candidates: rough-locomotion variants across velocity family.

### 3.4 Manipulation coverage

*TODO.* Candidates: most `manager_based/manipulation/{lift, reach,
cabinet, inhand}` envs — Newton hasn't been wired up for manipulation
scenes yet.

### 3.5 Deformable / softbody

*TODO.* Candidates: any task using cloth, rope, softbody assets.

### 3.6 Other

*TODO.* Reserved for envs whose gap doesn't fit the vocabulary; the
`notes:` field in their YAML row explains the specific miss.

---

## Appendix: per-env table

*To be rendered from `tools/odin/config/newton_gap_candidates.yaml`
once every row's `suspected_gap` has been set to a non-`tbd` value. Use
the one-liner in `docs/superpowers/plans/2026-04-22-odin-t2-1-env-lists.md`
Task 14 to regenerate this table as curation progresses.*

| Task | Group | Gap | Notes |
|------|-------|-----|-------|
| *(pending)* | | | |
