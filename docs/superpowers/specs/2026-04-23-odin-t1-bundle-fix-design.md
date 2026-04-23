# Odin T1 Bundle Fixes — Design

**Status:** approved
**Date:** 2026-04-23
**Tasks covered:** Fixes four bugs identified in Odin's T1 dry-run bundles
(captured in the project-memory note `project_odin_t1_bundle_bug.md`).

## 1. Motivation

The four T1 reference bundles under `odin_runs/` on branch
`antoiner/feat/odin` contain corrupted data despite reporting
`status=completed, exit_code=0`:

1. **Stale TB events.** The bundle's `tb/events.out.tfevents.*` file is a
   byte-for-byte copy of an unrelated March run
   (`logs/rsl_rl/unitree_go1_flat/2026-03-09_15-43-27/events...`), not the
   file emitted by the current training subprocess.
2. **Identical reward series across `physx` vs `newton`.** Both physx and
   newton runs used the same default physics backend, so the learning
   curves match by construction.
3. **SKRL per-iter timing is `total / max_iters` replicated** — zero
   variance.
4. **SKRL `series_per_iter` length is 97, not 300** — read from the
   episode-boundary tag rather than a per-iter tag.

Bug 1 is in `tools/odin/{hugin,munin}/run.py`. Bug 2 is a semantic gap in
`benchmark_{rsl_rl,skrl}.py`'s `--backend` flag. Bugs 3–4 are in
`benchmark_skrl.py`. None of the bugs require schema changes.

## 2. Goals

- Bundle's training-data directory contains exactly the TB events and
  checkpoints produced by this run's training subprocess.
- Passing `--backend newton` to Odin actually selects Newton physics.
- SKRL's `iter_time_s.std` is non-zero on a real run.
- SKRL's `learning.reward.series_per_iter` length equals
  `iterations_completed`.

## 3. Non-goals

- Adding new CLI flags beyond what's strictly needed.
- Bumping the `manifest.json` or `training.json` v1 schema (the data
  shape is unchanged; the fixes produce correct values in the same
  fields).
- Changing RSL-RL's output layout. Checkpoints / `params/` / git state
  written by RSL-RL stay exactly where RSL-RL writes them; we just
  direct that whole tree to the bundle via the existing `log_dir`
  mechanism.
- Extending the bundle's manifest `artifacts` list beyond the cosmetic
  `tb` → `training_data` rename.
- Making the benchmark scripts usable outside of Isaac Sim (they still
  require Kit/Newton runtime as today).

## 4. Overview of changes

| Area | Change | File(s) |
|---|---|---|
| Bug 1 | Add `--log_dir PATH` flag; drop the `_copy_tb_events` heuristic | `benchmark_rsl_rl.py`, `benchmark_skrl.py`, `tools/odin/{hugin,munin}/run.py` |
| Bug 2 | When `--backend X` is set, inject `presets=X` into hydra args | `benchmark_rsl_rl.py`, `benchmark_skrl.py` |
| Bugs 3–4 | New `BenchmarkTrainer(SequentialTrainer)`; swap it into SKRL Runner after construction | new `scripts/benchmarks/skrl_benchmark_trainer.py`, `benchmark_skrl.py` |
| Manifest | No code change needed — `artifacts` is derived from `os.listdir(bundle_dir)`; tracks the directory swap automatically | — |

## 5. Bundle layout after this fix

```
odin_runs/<run_id>/
├── manifest.json           (unchanged; `artifacts` reads "training_data")
├── startup.json            (unchanged)
├── training.json           (unchanged fields; SKRL values become correct)
├── training_data/          (new — was `tb/`)
│   ├── events.out.tfevents.*      (TB events from this run only)
│   ├── params/env.yaml            (written by the training framework)
│   ├── params/agent.yaml
│   ├── git/                       (written by RSL-RL's add_git_repo_to_log)
│   └── model_*.pt                 (RSL-RL checkpoints; SKRL analog)
└── logs/                   (unchanged; controller-side ssh-tail.log)
```

## 6. Bug 1 — `--log_dir` flag + drop the glob heuristic

### 6.1 Benchmark-script changes

Both `benchmark_rsl_rl.py` and `benchmark_skrl.py` add:

```python
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Absolute path to write training-framework outputs (TB events, "
         "checkpoints, configs). When set, overrides the default "
         "auto-generated logs/<framework>/<experiment>/<timestamp>/ path. "
         "Used by Odin to collect outputs directly into the bundle.",
)
```

When `args_cli.log_dir` is set:

- **RSL-RL** (`benchmark_rsl_rl.py` around line 365-372): skip the
  `log_root_path = os.path.join("logs", ...)` + timestamped subdir
  construction. Use `log_dir = os.path.abspath(args_cli.log_dir)`
  directly. `os.makedirs(log_dir, exist_ok=True)` before
  `OnPolicyRunner(..., log_dir=log_dir, ...)`.
- **SKRL** (`benchmark_skrl.py` around line 333-340): set
  `agent_cfg["agent"]["experiment"]["directory"] =
  os.path.dirname(os.path.abspath(args_cli.log_dir)) or "."` and
  `agent_cfg["agent"]["experiment"]["experiment_name"] =
  os.path.basename(os.path.abspath(args_cli.log_dir))`. SKRL's
  `BaseAgent.__init__` (`skrl/agents/torch/base.py:82-91`) applies a
  `if not experiment_name:` fallback that synthesizes a
  `<timestamp>_<AgentClass>` subdir — passing an empty string there
  silently routes TB events under that synthetic subdir, not into
  `<log_dir>`. Path decomposition guarantees both parts are truthy
  strings whose `os.path.join` recomposes to `<log_dir>` exactly.
  `os.path.abspath` must run before `dirname`/`basename` so that a
  trailing slash (which would make `basename` empty) is canonicalised
  away.

When `args_cli.log_dir` is unset, both scripts preserve current
auto-generated behavior. No standalone user is affected.

### 6.2 Hugin / Munin changes

`tools/odin/hugin/run.py` and `tools/odin/munin/run.py`:

- Remove the `_copy_tb_events(...)` function and the
  `rsl_rl_logs / skrl_logs` glob+copy block entirely (currently at
  hugin/run.py:157-164 and munin/run.py:152-159).
- Append to `training_cmd`:

  ```python
  training_cmd += ["--log_dir", os.path.join(bundle_dir, "training_data")]
  ```

- `os.makedirs(os.path.join(bundle_dir, "training_data"), exist_ok=True)`
  before subprocess launch.

### 6.3 Manifest artifact list

No code change. `tools/odin/common/manifest.py:63` derives the
`artifacts` list via `sorted(os.listdir(bundle_dir))` — so once the
training subprocess writes to `training_data/` instead of `tb/`, the
emitted manifest records `"training_data"` automatically.

`tools/odin/tests/test_hugin.py` and `tools/odin/tests/test_munin.py`
carry happy-path tests for the bundle; they gain an assertion on the
new `training_data/` directory name.

## 7. Bug 2 — `--backend` drives physics via hydra `presets=`

The current `--backend` flag's help text says *"Physics backend tag
recorded in the Odin bundle."* — it only tags the bundle. The fix:
**when `--backend X` is set, also inject `presets=X` into the hydra
args before Hydra sees them**, so the preset system applies it to
`env_cfg`.

### 7.1 Implementation

Both scripts already do:

```python
args_cli, hydra_args = parser.parse_known_args()
# ...
sys.argv = [sys.argv[0]] + hydra_args
```

Between those two lines, add:

```python
if args_cli.backend is not None:
    existing_presets = [a for a in hydra_args if a.startswith("presets=")]
    if existing_presets:
        print(
            f"[WARNING] --backend={args_cli.backend} ignored because "
            f"{existing_presets[0]} was explicitly passed."
        )
    else:
        hydra_args = [f"presets={args_cli.backend}"] + hydra_args
```

The explicit-override warning preserves the existing escape hatch
(a user passing their own `presets=...` on the CLI wins). When the
`--backend` flag's help text is updated, it now reads: *"Physics backend
to run with. Drives both the bundle tag and hydra `presets=<backend>` if
the preset exists on the env config."*

### 7.2 Odin wrapper change

None. Hugin/Munin already pass `--backend <backend>`. The preset
application lives inside the benchmark script.

### 7.3 Behavior when the env lacks a `newton` preset

The preset system already handles this: if `presets=newton` is applied
and the env config doesn't declare a `newton` PresetCfg, Hydra raises.
That's the desired failure mode for Odin — the T2.1 curated
`newton_envs.yaml` is the authoritative list of envs with a `newton`
preset, so any `--backend newton` dispatch should succeed on curated
tasks and fail fast on uncurated ones.

## 8. Bugs 3 & 4 — SKRL per-iter capture via `BenchmarkTrainer`

### 8.1 New file `scripts/benchmarks/skrl_benchmark_trainer.py`

A thin subclass of `skrl.trainers.torch.SequentialTrainer` that mirrors
the parent `train()` loop and records one sample per PPO rollout
completion:

```python
class BenchmarkTrainer(SequentialTrainer):
    """SequentialTrainer variant that records per-iter timing + reward.

    Mirrors the parent train() loop but accumulates three Python lists
    at the granularity of one rollout-buffer fill (= one PPO iteration):

    - iter_times_s  [s]     wall time from the start of one rollout's
                            first env step to the end of the agent's
                            learning update (inclusive)
    - iter_rewards  [float] mean-across-envs of rewards summed over the
                            rollout's `rollouts` steps
    - iter_ep_lengths [float] mean finished-episode length during the
                              rollout (excluded when no episode terminated)

    Attributes are populated after train() returns and consumed by
    benchmark_skrl.py's v1 bundle builder.
    """
```

Timing boundary: a "benchmark iteration" is the window from just before
the first env step of a rollout to immediately after
`agent.post_interaction` of the rollout's final timestep (when PPO's
`_update()` has run). This aligns with the `iter_time_s` semantic in the
v1 schema.

Reward capture: sum the per-timestep reward tensors across the rollout
(shape `[num_envs]`), then `.mean().item()` at rollout end — one float
per iteration.

Episode length capture: at each iteration boundary, emit the running
average length of *terminated* episodes observed so far (via SKRL's
`agent.tracking_data["Episode / Total timesteps (mean)"]` if present,
else 0.0). This keeps `iter_ep_lengths` length-aligned 1:1 with
`iter_times_s` and `iter_rewards` — no None handling, no
drop-and-realign logic.

### 8.2 Wiring in `benchmark_skrl.py`

Replace the `Runner(env, agent_cfg)` / `runner.run()` pair with:

```python
from scripts.benchmarks.skrl_benchmark_trainer import BenchmarkTrainer

runner = Runner(env, agent_cfg)
# Swap in the per-iter-capturing trainer. Runner._trainer and
# Runner._agent are mutable attributes (Runner has `trainer` / `agent`
# @property accessors reading them).
trainer_cfg = dict(agent_cfg["trainer"])
trainer_cfg.pop("class", None)  # stock Runner deletes it; mirror that.
bt = BenchmarkTrainer(env=env, agents=runner._agent, cfg=trainer_cfg)
runner._trainer = bt
runner.run()
```

Immediately after `runner.run()` returns:

```python
iter_times_s  = bt.iter_times_s     # len == max_iterations
reward_series = bt.iter_rewards     # len == max_iterations
ep_len_series = bt.iter_ep_lengths  # len == max_iterations
```

Delete the old `total_train_s / max_iterations` computation and the
`parse_tf_logs(log_dir)` call (since we now have the values directly).
The `log_data = {}` fallback also goes away.

### 8.3 What about the `parse_tf_logs` call in RSL-RL?

Out of scope for this spec. The RSL-RL per-iter timing and series
already come from TB events written by RSL-RL, which ARE per-iter (the
RSL-RL OnPolicyRunner writes once per `learn()` iteration). The bundle
bug note's "runtime-level numbers look OK" confirms RSL-RL's values are
real. The only RSL-RL-adjacent fix is Bug 1 (log_dir) and Bug 2
(backend).

## 9. Testing strategy

### 9.1 Unit tests (all run without Kit runtime)

New tests in `tools/odin/tests/`:

- `test_hugin_passes_log_dir_and_drops_copy.py` — monkeypatched
  subprocess fake; asserts `--log_dir <bundle>/training_data` is in
  `training_cmd` and the bundle's `training_data/` dir is pre-created.
- `test_munin_passes_log_dir_and_drops_copy.py` — same for Munin.

New tests in `scripts/benchmarks/tests/` (standard IsaacLab test layout):

- `test_benchmark_rsl_rl_log_dir_override.py` — argparse-level test:
  when `--log_dir <path>` is passed, the script's `log_dir` variable
  equals `<path>` (captured via a light monkey-patch of
  `OnPolicyRunner`).
- `test_benchmark_rsl_rl_backend_injects_preset.py` — argparse-level
  test: `--backend newton` inserts `presets=newton` as the first entry
  of `hydra_args`; an explicit `presets=foo` on the CLI wins.
- `test_benchmark_skrl_log_dir_override.py` — analog for SKRL: the
  agent cfg's `experiment.directory` equals `<path>` and
  `experiment.experiment_name` is empty.
- `test_benchmark_skrl_backend_injects_preset.py` — analog for SKRL.
- `test_skrl_benchmark_trainer.py` — fake env + fake agent;
  `BenchmarkTrainer.train()` produces `iter_times_s` of length
  `max_iterations`, non-constant; `iter_rewards` matches the
  synthetic rewards the fake env emitted; `iter_ep_lengths` handles
  the no-termination case.

### 9.2 Regeneration check (manual, after code lands)

Run all four bundles with identical parameters
(`num_envs=4096, max_iterations=300, seed=42`) via Hugin/Munin. Then:

```python
# compare across backends (MUST differ)
assert physx.reward_series[:5] != newton.reward_series[:5]

# SKRL series length (MUST equal max_iterations)
assert len(skrl_bundle.reward_series) == 300

# SKRL iter-time variance (MUST be non-zero)
assert skrl_bundle.iter_time_s["std"] > 0.0

# TB file timestamp (MUST be from the current run)
assert bundle["tb_file_mtime"] >= bundle["run_start_time"]
```

The regeneration script is not committed — it's a one-shot verification
the human runs. The four bundles live under `odin_runs/` (untracked;
binary data) and get overwritten on disk once they pass. The
`project_odin_t1_bundle_bug.md` memory note is updated to flip the
four-bundle status from ⚠ corrupted → ✓ regenerated and record the new
fingerprint values.

### 9.3 Regression guarantee

- The failing bundles live on disk on `antoiner/feat/odin` and are
  referenced in `project_odin_t1_bundle_bug.md`. After the fix lands,
  the memory note is updated with the new bundle signatures.
- Each unit test includes a "this test fails without the fix" assertion
  (verified by temporarily reverting per IsaacLab's `AGENTS.md`
  testing rule).

## 10. Risks & open questions

- **SKRL's `post_interaction` signalling rollout boundary**: Verified by
  reading skrl 1.4.3's PPO agent — `post_interaction` triggers
  `_update()` when
  `timestep + 1 >= self._learning_starts and (timestep + 1) %
  self._rollouts == 0`. The `BenchmarkTrainer` detects the same
  boundary via the same modular-arithmetic check.
- **Checkpoints inflate bundle size**. A 300-iter run with default
  `save_interval=50` produces ~6 checkpoints × ~10-50 MB each. This is
  an accepted trade — captures a resume-able artifact. A future
  `--no_checkpoints` flag can opt out if needed.
- **`--backend`-drives-preset semantic change** is technically a
  behavior change for any pre-existing user of `--backend`. Since the
  flag was added as part of Odin T1 (never shipped to a standalone
  user) and its help text already says "Physics backend", this is a
  clarification not a breaking change.

## 11. Out of scope (explicit non-actions)

- No changes to RSL-RL's log_dir auto-generation semantics — only the
  `--log_dir` override path is added.
- No new `--no_checkpoints` / `--checkpoint_interval` flags.
- No new schema fields in `training.json`.
- No changes to `scripts/benchmarks/benchmark_rlgames.py` or any other
  benchmark script.
- No CI changes. The regeneration step is human-driven.
