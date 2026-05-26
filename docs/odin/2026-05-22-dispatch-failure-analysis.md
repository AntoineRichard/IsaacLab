# Odin Dispatch Failure Analysis — `20260522-093457`

## Run summary

- Submitted: 273 tasks across 11 OSMO workflows (chunk_size=25), pool `isaac-dev-l40s-04`.
- Image: `nvcr.io/nvidian/antoiner-isaac-lab:develop-odin` (pre symlink-fix; first run with the per-task `exec_timeout` via groups).
- Terminal state (queried directly from OSMO; local poller went stale partway through):
  - **190 COMPLETED** (≈70%)
  - **35 FAILED** — training crashes
  - **23 FAILED_QUEUE_TIMEOUT** — pods never scheduled
  - 25 tasks unaccounted for (likely missed during local-poller stall)

## Failures grouped by root cause

### A. NaN in observations or rewards (Newton training instabilities)

Training crashes with `ValueError: The observation group 'policy' returned by the environment contains NaN values.` (or the analogous reward variant). All 3 seeds fail consistently → systematic, not a flake.

| Task | Backend | Framework | Error |
|---|---|---|---|
| `Isaac-Humanoid-v0` | newton | rsl_rl | `ValueError: ... 'policy' ... contains NaN values` |
| `Isaac-Reach-Franka-OSC-v0` | newton | rsl_rl | `ValueError: The rewards ... contain NaN values` |
| `Isaac-Velocity-Rough-Anymal-C-v0` | newton | rsl_rl | `ValueError: ... 'policy' ... contains NaN values` |
| `Isaac-Velocity-Rough-Cassie-v0` | newton | rsl_rl | `ValueError: ... 'policy' ... contains NaN values` |
| `Isaac-Velocity-Rough-G1-v0` | newton | rsl_rl | `ValueError: ... 'policy' ... contains NaN values` |
| `Isaac-Velocity-Rough-Unitree-Go1-v0` | newton | rsl_rl | `ValueError: ... 'policy' ... contains NaN values` (1/3 — flake) |

**Hypothesis**: Newton sim is propagating non-finite state somewhere (collision normal, contact force, joint torque). The flat variants of the same robots completed cleanly on Newton, so the rough-terrain path or the `Reach-Franka-OSC` operational-space controller is the trigger. Worth filing upstream against the Newton sim layer with one reproducible example.

### B. Tensor shape mismatch (6 vs 2) — Digit-specific

```
RuntimeError: The size of tensor a (6) must match the size of tensor b (2) at non-singleton dimension 1
```

| Task | Backend | Framework |
|---|---|---|
| `Isaac-Tracking-LocoManip-Digit-v0` | newton | rsl_rl |
| `Isaac-Velocity-Flat-Digit-v0` | newton | rsl_rl |
| `Isaac-Velocity-Rough-Digit-v0` | newton | rsl_rl |

All three Digit tasks under Newton hit the same shape mismatch. The number `6` vs `2` is consistent — looks like an action-space or joint-mapping mismatch between the Digit asset's articulation layout (Newton side) and the policy/action manager. **All Digit-on-Newton tasks fail**; physx Digit tasks completed. Almost certainly an asset/spec drift between the two backends.

### C. Numerically-degenerate policy distribution

```
RuntimeError: normal expects all elements of std >= 0.0
```

| Task | Backend | Framework |
|---|---|---|
| `Isaac-Open-Drawer-Franka-v0` | newton | rsl_rl (1/3 — flake) |

Policy sampled a Normal distribution with negative std → upstream of this, log_std went to NaN/-inf, almost certainly from an upstream NaN observation. Same root cause as cluster A, surfaced in PPO sampling instead of obs validation.

### D. Gradient-tracking tensor sent to Warp/CUDA boundary

```
RuntimeError: Can't get __cuda_array_interface__ on Variable that requires grad.
If gradients aren't required, use var.detach() to get Variable that doesn't require grad.
```

| Task | Backend | Framework |
|---|---|---|
| `Isaac-Navigation-Flat-Anymal-C-v0` | physx | rsl_rl |

PhysX (or Warp) called `__cuda_array_interface__` on a torch tensor still in the autograd graph. Bug somewhere in the Navigation env's observation pipeline — most likely a `detach()` missing on an action or velocity command tensor.

### E. Multi-agent observation-space key missing (skrl/MARL)

```
KeyError: 'cart'
  File ".../envs/direct_marl_env.py", line 666, in _configure_env_spaces
    agent: spec_to_gym_space(self.cfg.observation_spaces[agent]) for agent in self.cfg.possible_agents
```

| Task | Backend | Framework |
|---|---|---|
| `Isaac-Cart-Double-Pendulum-Direct-v0` | physx | skrl |

The Cart-Double-Pendulum MARL env declares `possible_agents = ['cart', 'pole']` but `observation_spaces` only ships one entry. Either an upstream spec bug or a skrl-side expectation mismatch with the new direct MARL API. The same task on `rl_games` completed — only the `skrl` wrapper hits this path.

### F. `warp.context` AttributeError (camera skrl trainer)

```
AttributeError: module 'warp' has no attribute 'context'. Did you mean: 'constant'?
```

| Task | Backend | Framework |
|---|---|---|
| `Isaac-Cartpole-Depth-Camera-Direct-v0` | physx | skrl |
| `Isaac-Cartpole-RGB-Camera-Direct-v0` | physx | skrl |

`warp.context` was removed from the Warp Python API in a recent release; the skrl camera-task wrappers still reference it. Should be a one-liner fix on the skrl helper (rename to `warp.utils.config` or whatever the current name is), or a Warp pin if upstream skrl hasn't migrated.

### G. omni.datastore exclusive lock contention

```
[Error] [omni.datastore] Failed to acquire exclusive lock to data store (256>=256)
[Error] [omni.datastore] Failed to create local file data store at '/isaac-sim/kit/cache/DerivedDataCache'
```

| Task | Backend | Framework |
|---|---|---|
| `Isaac-Velocity-Flat-G1-v0` | physx | rsl_rl |

The kit `DerivedDataCache` allows max 256 concurrent locks. When many pods land on the same OSMO node simultaneously they exhaust the slot count and one of them fails to launch. Infrastructure, not task code. **Mitigations**: stagger workflow submission, lower per-node pod density (raise CPU/memory ask), or set `KIT_PROTOCOL_CACHE_DIR` to a per-pod tmpfs path so they don't share.

### H. Queue timeout (capacity, not code)

23 tasks hit `FAILED_QUEUE_TIMEOUT` — pods waited longer than `queue_timeout` (2h) without being scheduled. Same pool-oversubscription pattern from earlier dispatches; not a task bug.

Cluster summary:

| Task | Backend × Framework | Notes |
|---|---|---|
| `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` | physx + newton, all 3 seeds | Heaviest (camera tensors) → resource ask largest |
| `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` | physx + newton, all 3 seeds | Big mesh, large memory request |
| `Isaac-Velocity-Flat-Spot-v0` | physx + newton | |
| `Isaac-Velocity-Flat-Unitree-Go1-v0` | physx + newton | |

**Mitigation**: drop chunk concurrency (fewer simultaneous workflows) or raise `queue_timeout`. Doesn't represent a code regression.

## Framework × backend totals

| Framework | Backend | Completed | Failed (training) | Queue timeout |
|---|---|---:|---:|---:|
| rsl_rl | physx | majority | 1 (Navigation-Flat-Anymal-C) | several |
| rsl_rl | newton | majority | 10 (clusters A/B/C/D) | several |
| skrl | physx | majority | 3 (Cart-Double-Pendulum + 2 camera-cartpole) | several |
| skrl | newton | majority | 0 | several |

(Local poller stalled mid-run; exact per-framework counts pulled from OSMO directly are above.)

## Action items

1. **Upstream report**: NaN-on-Newton cluster (A) — pick one minimal repro (e.g. `Isaac-Velocity-Rough-Cassie-v0 rsl_rl seed42`), file against the Newton sim layer.
2. **Digit asset mismatch (cluster B)**: 6-vs-2 shape mismatch needs an audit of `digit` action manager + Newton joint mapping.
3. **skrl camera-task wrappers (cluster F)**: rename `warp.context` references to the current API.
4. **MARL spec for Cart-Double-Pendulum (cluster E)**: investigate whether `possible_agents` and `observation_spaces` keys match in the cfg.
5. **Infrastructure**: stagger workflow submission (or raise `queue_timeout`) to drain cluster H; investigate per-pod `KIT_PROTOCOL_CACHE_DIR` to fix cluster G.
6. **Local sync tool**: build `tools/odin/scripts/osmo_sync.py` to reconcile `dispatch.json` from OSMO when the poller goes stale (the 25 unaccounted tasks above are the symptom).
