# Odin T1 Bundle Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four bugs in Odin's T1 training bundles (stale TB copy, `--backend` tag-only, SKRL flat-timing, SKRL wrong series source) and regenerate the four reference bundles.

**Architecture:** Two change sets. **Benchmark-script side** (upstream IsaacLab): new `--log_dir` flag, `--backend` injects `presets=X` into hydra args, and a new `BenchmarkTrainer(SequentialTrainer)` that records per-iter timing + reward for SKRL. **Odin-wrapper side**: drop the `_copy_tb_events` glob heuristic; instead pass `--log_dir <bundle>/training_data` to the training subprocess so it writes directly into the bundle.

**Tech Stack:** Python 3.10+, pytest, skrl 1.4.3, rsl_rl, Isaac Sim / IsaacLab benchmark schema v1.0.

**Spec:** `docs/superpowers/specs/2026-04-23-odin-t1-bundle-fix-design.md`

---

## File Structure

**New files:**
- `scripts/benchmarks/skrl_benchmark_trainer.py` — `BenchmarkTrainer(SequentialTrainer)` subclass.
- `scripts/benchmarks/tests/__init__.py` — marker for the new test dir.
- `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py` — argparse-level tests for the rsl_rl script's new flags.
- `scripts/benchmarks/tests/test_benchmark_skrl_cli.py` — argparse-level tests for the skrl script's new flags.
- `scripts/benchmarks/tests/test_skrl_benchmark_trainer.py` — unit test for `BenchmarkTrainer`.

**Modified:**
- `scripts/benchmarks/benchmark_rsl_rl.py` — add `--log_dir` flag; inject `presets=X`; consume `--log_dir` override.
- `scripts/benchmarks/benchmark_skrl.py` — same three changes + swap in `BenchmarkTrainer`; drop `parse_tf_logs` path.
- `tools/odin/hugin/run.py` — drop `_copy_tb_events`; append `--log_dir` to training subprocess.
- `tools/odin/munin/run.py` — same as Hugin.
- `tools/odin/tests/test_hugin.py` — extend happy-path test with `training_data/` assertion + `--log_dir` in cmd.
- `tools/odin/tests/test_munin.py` — same as Hugin.

**Unchanged (confirmed):**
- `tools/odin/common/manifest.py` — artifacts list is derived from `os.listdir(bundle_dir)`; picks up the dir-name swap automatically.
- `training.json` / `manifest.json` v1 schemas.
- The bundle spec document `docs/superpowers/specs/2026-04-22-odin-t1-evaluation-runner-design.md` — stays as-is; the layout reference updates are in the new design spec.

**Task ordering rationale:** Tasks 1-3 cover the benchmark-script CLI changes (independent of Odin). Task 4 is the `BenchmarkTrainer` (self-contained). Task 5 wires the trainer into `benchmark_skrl.py`. Tasks 6-7 update Hugin / Munin. Task 8 is the cross-cutting test-suite sweep + pre-commit. Task 9 is the manual regeneration + memory update.

---

### Task 1: Add `--log_dir` flag to `benchmark_rsl_rl.py`

**Files:**
- Modify: `scripts/benchmarks/benchmark_rsl_rl.py` (add flag after existing `--backend` flag; consume in `log_dir` construction)
- Create: `scripts/benchmarks/tests/__init__.py`
- Create: `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`

- [ ] **Step 1: Create `scripts/benchmarks/tests/__init__.py` (empty)**

```bash
touch scripts/benchmarks/tests/__init__.py
```

- [ ] **Step 2: Write the failing CLI test**

Create `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI-level tests for benchmark_rsl_rl.py.

These tests exercise only the argparse layer — they do not import the
whole script (which launches Isaac Sim at import time). A minimal reimport
of the argparse setup is shared via ``_build_parser``.
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Mirror of the parser setup in benchmark_rsl_rl.py.

    Kept in lockstep with the script; when a new flag is added there,
    add it here too.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--backend", choices=["physx", "newton"], default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--schema_v1_output", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser


def test_log_dir_flag_defaults_none():
    args = _build_parser().parse_args([])
    assert args.log_dir is None


def test_log_dir_flag_captured():
    args = _build_parser().parse_args(["--log_dir", "/tmp/bundle/training_data"])
    assert args.log_dir == "/tmp/bundle/training_data"
```

- [ ] **Step 3: Run test to verify it passes (the mirror parser is self-contained)**

Run:
```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py -v
```

Expected: 2 PASS.

- [ ] **Step 4: Add `--log_dir` argparse flag to `benchmark_rsl_rl.py`**

Insert right after the `--backend` block (around line 63, before the `--run_id` block):

```python
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help=(
        "Absolute path where the training framework writes its outputs "
        "(TB events, checkpoints, params). When unset, falls back to "
        "the default logs/<framework>/<experiment>/<timestamp>/ path. "
        "Odin passes this to collect outputs directly into the bundle."
    ),
)
```

- [ ] **Step 5: Override `log_dir` when flag is set**

In `benchmark_rsl_rl.py` around lines 364-372 (the `log_root_path` / `log_dir` construction):

Replace:
```python
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
```

With:
```python
    if args_cli.log_dir is not None:
        # Explicit override (Odin / CI): write straight into the given dir.
        log_dir = os.path.abspath(args_cli.log_dir)
        log_root_path = os.path.dirname(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Logging experiment in directory: {log_dir}")
    else:
        # Default: auto-generate logs/<framework>/<experiment>/<timestamp>/
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
```

- [ ] **Step 6: Run test again to confirm still passing**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py -v
```

Expected: 2 PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/benchmarks/benchmark_rsl_rl.py scripts/benchmarks/tests/
git commit -m "Add --log_dir flag to benchmark_rsl_rl.py"
```

---

### Task 2: `--backend X` injects `presets=X` into hydra args (`benchmark_rsl_rl.py`)

**Files:**
- Modify: `scripts/benchmarks/benchmark_rsl_rl.py` (between `parse_known_args()` and `sys.argv = ...`)
- Modify: `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py` (add two new tests)

- [ ] **Step 1: Write the failing preset-injection test**

Append to `scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py`:

```python
def _inject_preset(args_cli, hydra_args: list[str]) -> list[str]:
    """Mirror of the inject_preset logic in benchmark_rsl_rl.py.

    Invariant: when --backend X is set AND hydra_args does NOT already
    contain a ``presets=...`` entry, prepend ``presets=X``.
    """
    if args_cli.backend is None:
        return hydra_args
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend={args_cli.backend} ignored; explicit {existing[0]} wins.")
        return hydra_args
    return [f"presets={args_cli.backend}"] + hydra_args


def test_backend_injects_preset_when_none_given():
    args = _build_parser().parse_args(["--backend", "newton"])
    out = _inject_preset(args, ["env.decimation=4"])
    assert out == ["presets=newton", "env.decimation=4"]


def test_backend_does_not_inject_when_preset_already_present(capsys):
    args = _build_parser().parse_args(["--backend", "newton"])
    out = _inject_preset(args, ["presets=custom", "env.decimation=4"])
    assert out == ["presets=custom", "env.decimation=4"]
    assert "ignored" in capsys.readouterr().out


def test_backend_unset_is_noop():
    args = _build_parser().parse_args([])
    out = _inject_preset(args, ["env.decimation=4"])
    assert out == ["env.decimation=4"]
```

- [ ] **Step 2: Run tests to verify they pass (helper logic is self-contained)**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py -v
```

Expected: 5 PASS.

- [ ] **Step 3: Implement the preset-injection in `benchmark_rsl_rl.py`**

In `benchmark_rsl_rl.py`, find this block (around line 93-101):

```python
# to ensure kit args don't break the benchmark arg parsing
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
```

Replace it with:

```python
# to ensure kit args don't break the benchmark arg parsing
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Map --backend X to hydra presets=X so the physics preset is applied
# at config-resolve time. An explicit presets=... on the CLI wins.
if args_cli.backend is not None:
    existing_presets = [a for a in hydra_args if a.startswith("presets=")]
    if existing_presets:
        print(
            f"[WARNING] --backend={args_cli.backend} ignored because "
            f"{existing_presets[0]} was explicitly passed."
        )
    else:
        hydra_args = [f"presets={args_cli.backend}"] + hydra_args

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
```

Also update the `--backend` flag's `help=` string (around line 62):

Before:
```python
    help="Physics backend tag recorded in the Odin bundle.",
```

After:
```python
    help=(
        "Physics backend to run with. Drives both the bundle tag and "
        "hydra `presets=<backend>`. Pass an explicit `presets=...` on "
        "the CLI to override."
    ),
```

- [ ] **Step 4: Run tests again**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmarks/benchmark_rsl_rl.py scripts/benchmarks/tests/test_benchmark_rsl_rl_cli.py
git commit -m "Map --backend to hydra presets= in benchmark_rsl_rl.py"
```

---

### Task 3: Mirror tasks 1 & 2 for `benchmark_skrl.py`

**Files:**
- Modify: `scripts/benchmarks/benchmark_skrl.py` (add `--log_dir` flag; inject preset; consume `--log_dir` override)
- Create: `scripts/benchmarks/tests/test_benchmark_skrl_cli.py`

- [ ] **Step 1: Write the failing CLI test**

Create `scripts/benchmarks/tests/test_benchmark_skrl_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI-level tests for benchmark_skrl.py — argparse-only, no Isaac Sim."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--backend", choices=["physx", "newton"], default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--schema_v1_output", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--ml_framework", type=str, default="torch")
    return parser


def _inject_preset(args_cli, hydra_args: list[str]) -> list[str]:
    if args_cli.backend is None:
        return hydra_args
    existing = [a for a in hydra_args if a.startswith("presets=")]
    if existing:
        print(f"[WARNING] --backend={args_cli.backend} ignored; explicit {existing[0]} wins.")
        return hydra_args
    return [f"presets={args_cli.backend}"] + hydra_args


def test_log_dir_flag_defaults_none():
    assert _build_parser().parse_args([]).log_dir is None


def test_log_dir_flag_captured():
    args = _build_parser().parse_args(["--log_dir", "/tmp/bundle/training_data"])
    assert args.log_dir == "/tmp/bundle/training_data"


def test_backend_injects_preset_when_none_given():
    args = _build_parser().parse_args(["--backend", "newton"])
    assert _inject_preset(args, ["env.decimation=4"]) == ["presets=newton", "env.decimation=4"]


def test_backend_does_not_inject_when_preset_already_present(capsys):
    args = _build_parser().parse_args(["--backend", "newton"])
    out = _inject_preset(args, ["presets=custom", "env.decimation=4"])
    assert out == ["presets=custom", "env.decimation=4"]
    assert "ignored" in capsys.readouterr().out


def test_backend_unset_is_noop():
    args = _build_parser().parse_args([])
    assert _inject_preset(args, ["env.decimation=4"]) == ["env.decimation=4"]
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_benchmark_skrl_cli.py -v
```

Expected: 5 PASS.

- [ ] **Step 3: Add `--log_dir` flag + preset injection to `benchmark_skrl.py`**

First, add the `--log_dir` argparse flag right after the `--backend` flag block (search for `choices=["physx", "newton"]` — should be around line 60-70):

```python
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help=(
        "Absolute path where the training framework writes its outputs "
        "(TB events, checkpoints, params). When unset, falls back to "
        "the default logs/<framework>/<experiment>/<timestamp>/ path. "
        "Odin passes this to collect outputs directly into the bundle."
    ),
)
```

Also update the `--backend` flag's `help=` string to match the new semantic (copy the same text used in `benchmark_rsl_rl.py` Task 2 Step 3).

Second, wire the preset injection. Search `benchmark_skrl.py` for `parse_known_args()` / `sys.argv`. Apply the same pattern as Task 2 Step 3 (the `args_cli.backend is not None: …` block before `sys.argv = [sys.argv[0]] + hydra_args`).

- [ ] **Step 4: Override log_dir when flag is set**

In `benchmark_skrl.py` around lines 333-342 (the `log_root_path` / `log_dir` construction):

Replace:
```python
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{_algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.log_dir = log_dir
```

With:
```python
    if args_cli.log_dir is not None:
        log_dir = os.path.abspath(args_cli.log_dir)
        agent_cfg["agent"]["experiment"]["directory"] = log_dir
        agent_cfg["agent"]["experiment"]["experiment_name"] = ""
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{_algorithm}_{args_cli.ml_framework}"
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.log_dir = log_dir
```

- [ ] **Step 5: Run tests again**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_benchmark_skrl_cli.py -v
```

Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmarks/benchmark_skrl.py scripts/benchmarks/tests/test_benchmark_skrl_cli.py
git commit -m "Add --log_dir flag and --backend preset injection to benchmark_skrl.py"
```

---

### Task 4: `BenchmarkTrainer(SequentialTrainer)` — per-iter timing + reward

**Files:**
- Create: `scripts/benchmarks/skrl_benchmark_trainer.py`
- Create: `scripts/benchmarks/tests/test_skrl_benchmark_trainer.py`

- [ ] **Step 1: Write the failing test — iter_times_s populates and varies**

Create `scripts/benchmarks/tests/test_skrl_benchmark_trainer.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for BenchmarkTrainer — run with a fake env and fake agent.

These tests do NOT spin up Isaac Sim. They verify the trainer's
per-iteration capture logic in isolation.
"""

from __future__ import annotations

import time

import pytest
import torch

from scripts.benchmarks.skrl_benchmark_trainer import BenchmarkTrainer


class _FakeEnv:
    """Minimal env compatible with SKRL's SequentialTrainer expectations."""

    num_agents = 1
    num_envs = 4
    state_space = None
    observation_space = type("O", (), {"shape": (2,)})()
    action_space = type("A", (), {"shape": (1,)})()
    device = torch.device("cpu")

    def __init__(self, reward_schedule):
        self._rewards = reward_schedule  # list[float] — one per step
        self._i = 0

    def reset(self):
        return torch.zeros(self.num_envs, 2), {}

    def step(self, actions):
        r = self._rewards[self._i % len(self._rewards)]
        self._i += 1
        rewards = torch.full((self.num_envs,), float(r))
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        next_states = torch.zeros(self.num_envs, 2)
        return next_states, rewards, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


class _FakeAgent:
    """Minimal agent that exposes `_rollouts`, pre/post_interaction, track_data."""

    def __init__(self, rollouts: int = 4):
        self._rollouts = rollouts
        self.tracking_data: dict[str, list[float]] = {}
        self._init_called = False
        self._running_mode = None

    def init(self, trainer_cfg):
        self._init_called = True

    def set_running_mode(self, mode):
        self._running_mode = mode

    def pre_interaction(self, timestep, timesteps):
        pass

    def act(self, states, timestep, timesteps):
        return torch.zeros(states.shape[0], 1), None, None

    def record_transition(self, **kwargs):
        pass

    def post_interaction(self, timestep, timesteps):
        pass

    def track_data(self, tag, value):
        self.tracking_data.setdefault(tag, []).append(value)


def test_iter_times_s_length_matches_iterations():
    rollouts = 4
    max_iters = 3
    env = _FakeEnv(reward_schedule=[1.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    assert len(trainer.iter_times_s) == max_iters
    assert all(t > 0.0 for t in trainer.iter_times_s)


def test_iter_rewards_reflects_synthetic_schedule():
    rollouts = 4
    max_iters = 3
    # Give each rollout a distinguishable reward value.
    schedule = [1.0] * rollouts + [2.0] * rollouts + [3.0] * rollouts
    env = _FakeEnv(reward_schedule=schedule)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    # Each iteration's mean reward = mean over rollouts*num_envs rewards.
    # For constant-per-rollout schedules: iter k ≈ schedule[k*rollouts].
    assert trainer.iter_rewards == pytest.approx([1.0, 2.0, 3.0])


def test_iter_ep_lengths_defaults_to_zero_when_no_termination():
    rollouts = 4
    max_iters = 2
    env = _FakeEnv(reward_schedule=[0.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    # Fake env never terminates → ep_lengths fall back to 0.0 each iter.
    assert trainer.iter_ep_lengths == [0.0, 0.0]


def test_iter_times_s_shows_variance_with_sleep():
    """Real per-iter timing must vary when iterations take different wall times."""
    rollouts = 2
    max_iters = 2

    class _SlowEnv(_FakeEnv):
        def step(self, actions):
            if self._i == 0 or self._i == 1:
                time.sleep(0.02)
            return super().step(actions)

    env = _SlowEnv(reward_schedule=[0.0] * 100)
    agent = _FakeAgent(rollouts=rollouts)
    trainer_cfg = {"timesteps": rollouts * max_iters, "headless": True}

    trainer = BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)
    trainer.train()

    assert len(trainer.iter_times_s) == max_iters
    # First iter had two sleep(0.02) calls (steps 0 and 1); second iter didn't.
    # Accept any positive separation; this is about existence of variance, not magnitude.
    assert trainer.iter_times_s[0] > trainer.iter_times_s[1]
```

- [ ] **Step 2: Run test to verify it fails with ImportError**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_skrl_benchmark_trainer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.benchmarks.skrl_benchmark_trainer'`.

- [ ] **Step 3: Implement `BenchmarkTrainer` (minimal)**

Create `scripts/benchmarks/skrl_benchmark_trainer.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BenchmarkTrainer — SKRL trainer subclass that captures per-iteration metrics.

Mirrors :class:`skrl.trainers.torch.SequentialTrainer`'s training loop and
records, once per rollout-buffer fill (= one iteration):

* ``iter_times_s``  - wall-clock seconds from the first env step of the
  rollout to just after ``agent.post_interaction`` of the rollout's final
  step (i.e. after the PPO update).
* ``iter_rewards``  - mean reward across all env steps and all parallel
  envs during the rollout.
* ``iter_ep_lengths`` - last value of
  ``agent.tracking_data["Episode / Total timesteps (mean)"]`` observed at
  iteration end, or ``0.0`` when no episode terminated yet.

These attributes are populated after :meth:`train` returns and are read
directly by ``benchmark_skrl.py``'s v1 bundle builder — no TB round trip.
"""

from __future__ import annotations

import time

import torch
import tqdm
from skrl.trainers.torch import SequentialTrainer


class BenchmarkTrainer(SequentialTrainer):
    """SequentialTrainer that records per-iteration timing + reward + ep length."""

    def __init__(self, env, agents, agents_scope=None, cfg=None) -> None:
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=cfg)
        self.iter_times_s: list[float] = []
        self.iter_rewards: list[float] = []
        self.iter_ep_lengths: list[float] = []

    def train(self) -> None:
        # Exactly one non-simultaneous single-agent training path — mirrors
        # the parent SequentialTrainer for that case. If the user is running
        # multi-agent or simultaneous agents, defer to the stock loop (those
        # paths don't populate the Odin benchmark attributes).
        if self.num_simultaneous_agents > 1 or self.env.num_agents > 1:
            super().train()
            return

        rollouts = int(getattr(self.agents, "_rollouts", 0)) or 1
        max_iters = self.timesteps // rollouts

        self.agents.set_running_mode("train")
        states, infos = self.env.reset()

        iter_start_ns = time.perf_counter_ns()
        rollout_reward_sum = 0.0
        rollout_reward_count = 0

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

                rollout_reward_sum += float(rewards.mean().item())
                rollout_reward_count += 1

            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

            # One iteration = one rollout-buffer fill.
            if (timestep + 1) % rollouts == 0:
                iter_end_ns = time.perf_counter_ns()
                self.iter_times_s.append((iter_end_ns - iter_start_ns) / 1e9)
                mean_reward = rollout_reward_sum / max(rollout_reward_count, 1)
                self.iter_rewards.append(mean_reward)
                ep_len_samples = self.agents.tracking_data.get(
                    "Episode / Total timesteps (mean)", []
                )
                self.iter_ep_lengths.append(float(ep_len_samples[-1]) if ep_len_samples else 0.0)
                # Reset per-iter accumulators + timer for the next rollout.
                iter_start_ns = time.perf_counter_ns()
                rollout_reward_sum = 0.0
                rollout_reward_count = 0

        # Cap any series to max_iters (guards against off-by-one if timesteps
        # isn't a clean multiple of rollouts).
        self.iter_times_s = self.iter_times_s[:max_iters]
        self.iter_rewards = self.iter_rewards[:max_iters]
        self.iter_ep_lengths = self.iter_ep_lengths[:max_iters]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_skrl_benchmark_trainer.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmarks/skrl_benchmark_trainer.py scripts/benchmarks/tests/test_skrl_benchmark_trainer.py
git commit -m "Add BenchmarkTrainer for per-iter SKRL timing and reward capture"
```

---

### Task 5: Wire `BenchmarkTrainer` into `benchmark_skrl.py`

**Files:**
- Modify: `scripts/benchmarks/benchmark_skrl.py` (replace stock Runner trainer + drop parse_tf_logs + drop fake per-iter derivation)

- [ ] **Step 1: Read the bundle-builder flow in `benchmark_skrl.py` to locate the integration point**

```bash
grep -n "Runner(env" scripts/benchmarks/benchmark_skrl.py
grep -n "parse_tf_logs\|total_train_s\|per_iter_s\|iter_times_s" scripts/benchmarks/benchmark_skrl.py
```

Expected:
- `runner = Runner(env, agent_cfg)` around line 352.
- The `total_train_s / max(...)` block around lines 354-383.
- `parse_tf_logs` call around line 366.

- [ ] **Step 2: Replace the Runner construction, trainer swap, and per-iter derivation block**

In `benchmark_skrl.py`, replace the block from around line 352 through to (and including) the `parse_tf_logs` / `per_iter_s` / `iter_times_s` derivation (down through the `rl_training_times = {...}` dict around line 383):

Before (approximately):
```python
    runner = Runner(env, agent_cfg)

    train_start = time.perf_counter_ns()
    with BenchmarkMonitor(benchmark, interval=1.0):
        runner.run()
    train_end = time.perf_counter_ns()

    # Final recorder update after training completes.
    benchmark.update_manual_recorders()

    # Parse tensorboard logs for reward / episode-length series. SKRL may skip
    # writing events for very short runs; treat parse failures as empty series
    # so the v1 bundle still lands with structural data intact.
    try:
        log_data = parse_tf_logs(log_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"[WARNING] Could not parse tensorboard logs in {log_dir}: {e}")
        log_data = {}
    reward_series = [float(x) for x in log_data.get(_SKRL_REWARD_TAG, []) or []]
    ep_len_series = [float(x) for x in log_data.get(_SKRL_EP_LEN_TAG, []) or []]

    # Each iteration consumes ``rollouts`` env steps; approximate per-iter
    # wall-clock time as (total train time) / max_iterations.
    total_train_s = (train_end - train_start) / 1e9
    per_iter_s = total_train_s / max(args_cli.max_iterations, 1)
    iter_times_s = [per_iter_s] * args_cli.max_iterations

    rl_training_times = {
        "Collection Time": iter_times_s,
        "Learning Time": [0.0] * len(iter_times_s),
        "Total FPS": [args_cli.num_envs * rollouts / per_iter_s if per_iter_s > 0 else 0.0] * len(iter_times_s),
    }
```

After:
```python
    from scripts.benchmarks.skrl_benchmark_trainer import BenchmarkTrainer

    runner = Runner(env, agent_cfg)
    # Swap SKRL's stock SequentialTrainer for the per-iter-capturing variant.
    # Runner._trainer and Runner._agent are mutable attributes (Runner exposes
    # ``trainer`` / ``agent`` @property accessors reading them).
    trainer_cfg = dict(agent_cfg["trainer"])
    trainer_cfg.pop("class", None)  # stock Runner deletes this; mirror that.
    benchmark_trainer = BenchmarkTrainer(env=env, agents=runner._agent, cfg=trainer_cfg)
    runner._trainer = benchmark_trainer

    train_start = time.perf_counter_ns()
    with BenchmarkMonitor(benchmark, interval=1.0):
        runner.run()
    train_end = time.perf_counter_ns()

    # Final recorder update after training completes.
    benchmark.update_manual_recorders()

    iter_times_s = benchmark_trainer.iter_times_s
    reward_series = benchmark_trainer.iter_rewards
    ep_len_series = benchmark_trainer.iter_ep_lengths
    total_train_s = (train_end - train_start) / 1e9
    per_iter_s = (sum(iter_times_s) / len(iter_times_s)) if iter_times_s else 0.0

    rl_training_times = {
        "Collection Time": iter_times_s,
        "Learning Time": [0.0] * len(iter_times_s),
        "Total FPS": [
            (args_cli.num_envs * rollouts / t) if t > 0 else 0.0 for t in iter_times_s
        ],
    }
```

- [ ] **Step 3: Clean up now-unused imports and module constants**

At the top of `benchmark_skrl.py`, find and delete:
- `from scripts.benchmarks.utils import parse_tf_logs` (if still imported)
- `_SKRL_REWARD_TAG = "..."` and `_SKRL_EP_LEN_TAG = "..."` (around line 140-141)

Verify by grepping:
```bash
grep -n "parse_tf_logs\|_SKRL_REWARD_TAG\|_SKRL_EP_LEN_TAG" scripts/benchmarks/benchmark_skrl.py
```

Expected: no matches.

- [ ] **Step 4: Run the CLI tests to catch any import breakage**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/tests/test_benchmark_skrl_cli.py scripts/benchmarks/tests/test_skrl_benchmark_trainer.py -v
```

Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmarks/benchmark_skrl.py
git commit -m "Wire BenchmarkTrainer into benchmark_skrl.py; drop parse_tf_logs path"
```

---

### Task 6: Update Hugin — pass `--log_dir`, drop TB glob

**Files:**
- Modify: `tools/odin/hugin/run.py`
- Modify: `tools/odin/tests/test_hugin.py`

- [ ] **Step 1: Update the test to expect `--log_dir` in the cmd and `training_data/` in the bundle**

In `tools/odin/tests/test_hugin.py`, extend `test_hugin_happy_path`:

Before (the inside of the test):
```python
    hugin_run.main()

    bundles = [d for d in os.listdir(bundle_root) if d.startswith("rsl-rl_physx_")]
    assert len(bundles) == 1
    bundle = os.path.join(bundle_root, bundles[0])
    assert os.path.exists(os.path.join(bundle, "startup.json"))
    assert os.path.exists(os.path.join(bundle, "training.json"))
    assert os.path.exists(os.path.join(bundle, "manifest.json"))
    with open(os.path.join(bundle, "manifest.json")) as f:
        m = json.load(f)
    assert m["phases"]["training"]["exit_code"] == 0
```

After — first update `_fake_run_factory` to capture commands, then assert the new flag:

```python
def _fake_run_factory():
    """Return a stub that pretends to write startup.json/training.json and
    records every command it was called with for later assertions."""

    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        out_idx = cmd.index("--schema_v1_output") + 1
        out_path = cmd[out_idx]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write('{"schema_version": "1.0", "fake": true}\n')

        class R:
            returncode = 0
            stdout = b"fake stdout"
            stderr = b"fake stderr"

        return R()

    _fake_run.captured_cmds = captured_cmds
    return _fake_run
```

And inside `test_hugin_happy_path`, add after the existing assertions:

```python
    training_data_dir = os.path.join(bundle, "training_data")
    assert os.path.isdir(training_data_dir), f"{training_data_dir} not created"
    # Training subprocess should receive --log_dir <bundle>/training_data.
    training_cmds = [c for c in fake_run.captured_cmds if "benchmark_rsl_rl.py" in " ".join(c)]
    assert len(training_cmds) == 1
    cmd = training_cmds[0]
    assert "--log_dir" in cmd
    log_dir_idx = cmd.index("--log_dir")
    assert cmd[log_dir_idx + 1] == training_data_dir
    # Old tb/ directory should no longer be created.
    assert not os.path.exists(os.path.join(bundle, "tb")), "stale tb/ dir leaked"
```

Update the factory-usage line to name the fake so you can read `.captured_cmds`:

Before: `monkeypatch.setattr(hugin_run, "_subprocess_run", _fake_run_factory())`
After:
```python
    fake_run = _fake_run_factory()
    monkeypatch.setattr(hugin_run, "_subprocess_run", fake_run)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py -v --confcutdir=tools/odin
```

Expected: FAIL on the new assertions (the `--log_dir` flag isn't passed yet and `training_data/` isn't created).

- [ ] **Step 3: Edit `hugin/run.py` — pre-create `training_data/` + pass `--log_dir`**

In `tools/odin/hugin/run.py`, find the `training_cmd = [...]` list (around line 125). Right before the list construction, add:

```python
    training_data_dir = os.path.join(bundle_dir, "training_data")
    os.makedirs(training_data_dir, exist_ok=True)
```

Then, just after the `--ema_alpha` args in the list (around line 145-146), add two more CLI args:

Before:
```python
    training_cmd = [
        _ISAACLAB_SH,
        "-p",
        _TRAINING_SCRIPT,
        # ... other args ...
        "--ema_alpha",
        str(args.ema_alpha),
    ]
    if args.no_series:
        training_cmd.append("--no_series")
```

After:
```python
    training_cmd = [
        _ISAACLAB_SH,
        "-p",
        _TRAINING_SCRIPT,
        # ... other args ...
        "--ema_alpha",
        str(args.ema_alpha),
        "--log_dir",
        training_data_dir,
    ]
    if args.no_series:
        training_cmd.append("--no_series")
```

- [ ] **Step 4: Delete `_copy_tb_events` and the TB-glob block**

In `tools/odin/hugin/run.py`:

- Delete the function `_copy_tb_events(...)` (around lines 67-73).
- Delete the "Best-effort TB copy" block (around lines 157-164).
- Remove the now-unused imports: `glob`, `shutil`, `contextlib`.

Verify with:
```bash
grep -n "_copy_tb_events\|rsl_rl_logs\|^import glob\|^import shutil\|^import contextlib" tools/odin/hugin/run.py
```

Expected: no matches.

- [ ] **Step 5: Run the tests again**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py -v --confcutdir=tools/odin
```

Expected: 2 PASS (both `test_hugin_happy_path` and `test_hugin_failure_path_writes_logs`).

- [ ] **Step 6: Commit**

```bash
git add tools/odin/hugin/run.py tools/odin/tests/test_hugin.py
git commit -m "Hugin: pass --log_dir to training subprocess; drop TB-glob heuristic"
```

---

### Task 7: Update Munin — mirror Task 6 for SKRL

**Files:**
- Modify: `tools/odin/munin/run.py`
- Modify: `tools/odin/tests/test_munin.py`

- [ ] **Step 1: Update `_fake_run_factory` in `test_munin.py` to capture commands**

In `tools/odin/tests/test_munin.py`, replace the existing `_fake_run_factory` with:

```python
def _fake_run_factory():
    """Return a stub that pretends to write startup.json/training.json and
    records every command it was called with for later assertions."""

    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        out_idx = cmd.index("--schema_v1_output") + 1
        out_path = cmd[out_idx]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write('{"schema_version": "1.0", "fake": true}\n')

        class R:
            returncode = 0
            stdout = b"fake stdout"
            stderr = b"fake stderr"

        return R()

    _fake_run.captured_cmds = captured_cmds
    return _fake_run
```

Then in `test_munin_happy_path`, change the monkeypatch line to capture the fake:

Before: `monkeypatch.setattr(munin_run, "_subprocess_run", _fake_run_factory())`
After:
```python
    fake_run = _fake_run_factory()
    monkeypatch.setattr(munin_run, "_subprocess_run", fake_run)
```

And append to the test body, after the existing assertions:

```python
    training_data_dir = os.path.join(bundle, "training_data")
    assert os.path.isdir(training_data_dir), f"{training_data_dir} not created"
    # Training subprocess should receive --log_dir <bundle>/training_data.
    training_cmds = [c for c in fake_run.captured_cmds if "benchmark_skrl.py" in " ".join(c)]
    assert len(training_cmds) == 1
    cmd = training_cmds[0]
    assert "--log_dir" in cmd
    log_dir_idx = cmd.index("--log_dir")
    assert cmd[log_dir_idx + 1] == training_data_dir
    # Old tb/ directory should no longer be created.
    assert not os.path.exists(os.path.join(bundle, "tb")), "stale tb/ dir leaked"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_munin.py -v --confcutdir=tools/odin
```

Expected: FAIL on the new assertions (the `--log_dir` flag isn't passed yet and `training_data/` isn't created).

- [ ] **Step 3: Edit `munin/run.py` — pre-create `training_data/` + pass `--log_dir`**

In `tools/odin/munin/run.py`, find the `training_cmd = [...]` list (around line 120). Right before the list construction, add:

```python
    training_data_dir = os.path.join(bundle_dir, "training_data")
    os.makedirs(training_data_dir, exist_ok=True)
```

Then, extend the `training_cmd` list with two more CLI args just after the `--ema_alpha` args (around line 140-141):

Before:
```python
    training_cmd = [
        _ISAACLAB_SH,
        "-p",
        _TRAINING_SCRIPT,
        # ... other args ...
        "--ema_alpha",
        str(args.ema_alpha),
    ]
    if args.no_series:
        training_cmd.append("--no_series")
```

After:
```python
    training_cmd = [
        _ISAACLAB_SH,
        "-p",
        _TRAINING_SCRIPT,
        # ... other args ...
        "--ema_alpha",
        str(args.ema_alpha),
        "--log_dir",
        training_data_dir,
    ]
    if args.no_series:
        training_cmd.append("--no_series")
```

- [ ] **Step 4: Delete `_copy_tb_events` and the SKRL-glob block**

In `tools/odin/munin/run.py`:

- Delete the function `_copy_tb_events(...)` (around lines 62-68).
- Delete the "Best-effort TB copy" block that greps `logs/skrl` (around lines 152-159).
- Remove the now-unused imports: `glob`, `shutil`, `contextlib`.

Verify with:
```bash
grep -n "_copy_tb_events\|skrl_logs\|^import glob\|^import shutil\|^import contextlib" tools/odin/munin/run.py
```

Expected: no matches.

- [ ] **Step 5: Run tests again**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_munin.py -v --confcutdir=tools/odin
```

Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/odin/munin/run.py tools/odin/tests/test_munin.py
git commit -m "Munin: pass --log_dir to training subprocess; drop TB-glob heuristic"
```

---

### Task 8: Full-suite verification + pre-commit

**Files:** none modified (sweep).

- [ ] **Step 1: Run the full Odin test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ -v --confcutdir=tools/odin
```

Expected: all previously-passing tests still pass + the new CLI/trainer/hugin/munin tests pass. 120 baseline unit tests + ~14 new tests.

If failures: diagnose before proceeding. Common cause to watch for: import paths — the SKRL trainer subclass imports `skrl.trainers.torch.SequentialTrainer`, which loads skrl on import. Running outside Isaac Sim is fine (the CLI tests don't import `skrl_benchmark_trainer`; only the trainer unit test does, and it uses fakes).

- [ ] **Step 2: Run pre-commit**

```bash
./isaaclab.sh -f
```

Expected: all hooks pass. If auto-fixes happen, stage them and re-run.

- [ ] **Step 3: Commit pre-commit fixes (if any)**

```bash
# Only if there were auto-modifications:
git add -u
git commit -m "Pre-commit auto-fixes on T1 bundle fix"
```

---

### Task 9: Manual regeneration + project-memory update

**Files:**
- Overwrite on disk (not committed): the four bundles under `odin_runs/`.
- Modify: `/home/antoiner/.claude/projects/-home-antoiner-Documents-IsaacLab/memory/project_odin_t1_bundle_bug.md` (flip status to resolved + record new fingerprints).
- Modify: `/home/antoiner/.claude/projects/-home-antoiner-Documents-IsaacLab/memory/MEMORY.md` (update the note's one-liner).

- [ ] **Step 1: Remove the corrupted bundles**

```bash
rm -rf odin_runs/rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-141025_seed42 \
       odin_runs/rsl-rl_newton_Isaac-Ant-Direct-v0_20260422-144006_seed42 \
       odin_runs/skrl_physx_Isaac-Ant-Direct-v0_20260422-144718_seed42 \
       odin_runs/skrl_newton_Isaac-Ant-Direct-v0_20260422-144912_seed42
```

- [ ] **Step 2: Regenerate the four bundles — identical params across all four**

Run from the repo root:

```bash
# 1. RSL-RL × PhysX
PYTHONPATH=. ./isaaclab.sh -p tools/odin/hugin/run.py \
    --task Isaac-Ant-Direct-v0 --backend physx --seed 42 \
    --num_envs 4096 --max_iterations 300

# 2. RSL-RL × Newton
PYTHONPATH=. ./isaaclab.sh -p tools/odin/hugin/run.py \
    --task Isaac-Ant-Direct-v0 --backend newton --seed 42 \
    --num_envs 4096 --max_iterations 300

# 3. SKRL × PhysX
PYTHONPATH=. ./isaaclab.sh -p tools/odin/munin/run.py \
    --task Isaac-Ant-Direct-v0 --backend physx --seed 42 \
    --num_envs 4096 --max_iterations 300

# 4. SKRL × Newton
PYTHONPATH=. ./isaaclab.sh -p tools/odin/munin/run.py \
    --task Isaac-Ant-Direct-v0 --backend newton --seed 42 \
    --num_envs 4096 --max_iterations 300
```

Expected: four new bundles land under `odin_runs/` with fresh timestamps. Approximate wall times:
- RSL-RL × PhysX: ~30 min
- RSL-RL × Newton: ~10 min (real Newton, not cache)
- SKRL × PhysX: ~1-3 min
- SKRL × Newton: ~1-3 min

- [ ] **Step 3: Verify the fix — cross-backend distinctness + SKRL shape**

Run:

```bash
./isaaclab.sh -p -c "
import json, hashlib, glob, os
bundles = sorted(glob.glob('odin_runs/*_Isaac-Ant-Direct-v0_*_seed42'))
assert len(bundles) == 4, f'expected 4 bundles, got {len(bundles)}'
summaries = {}
for b in bundles:
    with open(os.path.join(b, 'training.json')) as f:
        d = json.load(f)
    rew = d['learning']['reward']['series_per_iter']
    rt = d['runtime']
    summaries[os.path.basename(b)] = {
        'rew_len': len(rew),
        'rew_sha': hashlib.sha256(str(rew).encode()).hexdigest()[:16],
        'iter_time_mean': rt['iteration_time_s']['mean'],
        'iter_time_std': rt['iteration_time_s']['std'],
    }
    # Confirm training_data dir exists, tb/ does not.
    assert os.path.isdir(os.path.join(b, 'training_data')), f'no training_data in {b}'
    assert not os.path.exists(os.path.join(b, 'tb')), f'stale tb/ in {b}'
for k, v in summaries.items():
    print(k)
    print(f'  rew: len={v[\"rew_len\"]} sha={v[\"rew_sha\"]}')
    print(f'  iter_time: mean={v[\"iter_time_mean\"]:.3f} std={v[\"iter_time_std\"]:.4f}')

# Assertions — the four core fixes:
def _sha(bid):
    return summaries[bid]['rew_sha']
def _len(bid):
    return summaries[bid]['rew_len']
def _std(bid):
    return summaries[bid]['iter_time_std']

# Cross-backend series MUST differ within each framework.
by_rsl = sorted(b for b in summaries if b.startswith('rsl-rl_'))
by_skrl = sorted(b for b in summaries if b.startswith('skrl_'))
assert _sha(by_rsl[0]) != _sha(by_rsl[1]), 'RSL-RL physx/newton series still identical'
assert _sha(by_skrl[0]) != _sha(by_skrl[1]), 'SKRL physx/newton series still identical'

# SKRL series length must equal max_iterations (300).
for b in by_skrl:
    assert _len(b) == 300, f'{b} series len {_len(b)} != 300'

# SKRL iter_time_s.std must be non-zero (real per-iter variance).
for b in by_skrl:
    assert _std(b) > 0.0, f'{b} iter_time_s.std is zero'

print()
print('ALL FOUR BUGS RESOLVED ✓')
"
```

Expected: the script prints the four bundle summaries and concludes with `ALL FOUR BUGS RESOLVED ✓`. If any assertion fails, diagnose — that failure is the one bug that didn't land.

- [ ] **Step 4: Update the project-memory note**

Edit `/home/antoiner/.claude/projects/-home-antoiner-Documents-IsaacLab/memory/project_odin_t1_bundle_bug.md`:

- Change frontmatter `description:` from the current "corrupted" wording to something like *"Resolved 2026-04-23: the four T1 reference bundles on antoiner/feat/odin have been regenerated against the fixed hugin/munin + benchmark_skrl.py. Kept for context."*
- Add a closing section `**Resolution (2026-04-23):** see commit <SHA> and design spec docs/superpowers/specs/2026-04-23-odin-t1-bundle-fix-design.md. Current fingerprints: <paste from Step 3 output>`.
- Keep the original bug list (strike through with tilde-markdown `~~...~~`) so future readers see what was originally wrong.

Then update `/home/antoiner/.claude/projects/-home-antoiner-Documents-IsaacLab/memory/MEMORY.md` — the `Odin T1 dry-run bundles corrupted` one-liner flips to `Odin T1 dry-run bundles (✓ regenerated 2026-04-23)`.

- [ ] **Step 5: Update the Odin architecture doc**

Edit `docs/odin/architecture.md`:
- In the task-status matrix, flip the T1 row's caveat from ⚠ to ✓ (or equivalent — follow the existing notation).
- Bump the "Last updated" line to `2026-04-23 (T1 bundle fix landed)`.
- Add a change-log entry under §9 (or the equivalent history section) describing the fix scope + the spec link.

- [ ] **Step 6: Commit the doc + memory updates**

```bash
git add docs/odin/architecture.md
git commit -m "Mark Odin T1 bundle fix complete in architecture reference"
```

(Memory files live outside the repo — no git add needed for them.)

---

## Summary of verification criteria

After all nine tasks, the following must hold:

- `./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ -v` → all pass (~134 tests).
- `./isaaclab.sh -f` → clean.
- Four regenerated bundles on disk under `odin_runs/`.
- Per-backend SHA-256 of `learning.reward.series_per_iter` differs across `physx` vs `newton` within each framework.
- SKRL `learning.reward.series_per_iter` length == 300.
- SKRL `runtime.iteration_time_s.std` > 0.
- No `<bundle>/tb/` directory anywhere.
- `<bundle>/training_data/events.out.tfevents.*` file has mtime ≥ the bundle's `run_start_time_utc`.

The `ALL FOUR BUGS RESOLVED ✓` assertion in Task 9 Step 3 encodes all seven conditions.
