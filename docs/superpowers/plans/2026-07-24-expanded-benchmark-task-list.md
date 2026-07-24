<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Expanded Benchmark Task List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the paired IsaacLab 2.x-versus-3.0 matrix to 13 tasks, including complementary flat and rough Anymal-D and a runtime-only RGB Cartpole workload using the Kit renderer.

**Architecture:** Extend the checked-in TOML task entries with optional capabilities that are parsed into immutable task and attempt models. Matrix expansion filters unsupported task-mode pairs before creating deterministic identities, while executors translate camera and preset metadata into version-specific command lines and preflight checks. Normalization retains a canonical 13-task order, and plotting uses mode-specific task order so it does not invent an RGB training slot.

**Tech Stack:** Python 3.11+, standard-library `dataclasses` and `tomllib`, pytest, Matplotlib, TOML, Git pre-commit hooks.

## Global Constraints

- Work only in `/home/antoiner/benchmarks/isaaclab2-vs-3/lab2-main` on `antoiner/backport-benchmark-harness`.
- Do not run smoke, canary, final, Docker, Isaac Sim, or GPU benchmark jobs; this is a task-list-only change.
- Keep `num_envs = 4096`, seeds 42/43/44, PhysX, RSL-RL, counterbalancing, metric collection, normalization fields, and report semantics unchanged.
- Keep `anymal_d_flat`; add `anymal_d_rough` as a complementary task.
- Add flat Cassie only; do not add rough Cassie.
- Restrict `cartpole_rgb_kit` to `runtime-100` and `runtime-1000`; do not add or backport an IsaacLab 2.x RSL-RL configuration.
- Use `Isaac-Cartpole-RGB-v0` for IsaacLab 2.x and `Isaac-Cartpole-Camera` with the RGB preset and Kit-backed Isaac RTX renderer for IsaacLab 3.0.
- Add no required or optional dependency.
- Use modern type hints, Google-style docstrings, and the existing 2026 headers for new files.
- Do not edit `CHANGELOG.rst` or `config/extension.toml`. No package changelog fragment is required because only `tools/` controller code and tests are touched.
- Preserve `tools/benchmark_comparison/tests/test_actual_report_artifacts.py` unchanged: its 36/108 counts describe retained historical artifacts, not the new matrix.
- Before every commit, run all hooks with `uvx --from pre-commit pre-commit run --all-files`; the worktree has no local Isaac Sim Python, so `./isaaclab.sh -f` cannot install pre-commit here.

---

## File Responsibility Map

- `tools/benchmark_comparison/matrix.toml`: declarative 13-task mapping and RGB task capabilities.
- `tools/benchmark_comparison/models.py`: immutable task/attempt capability fields and helper methods.
- `tools/benchmark_comparison/matrix.py`: TOML parsing, exact validation, sparse expansion, and new deterministic counts.
- `tools/benchmark_comparison/executors.py`: camera/preset command translation and training-aware registration preflight.
- `tools/benchmark_comparison/normalize.py`: canonical 13-task ordering for CSV/report consumers.
- `tools/benchmark_comparison/plot.py`: mode-specific plotting order that omits unsupported RGB training.
- `tools/benchmark_comparison/tests/test_matrix.py`: task mapping, capability parsing, validation, counts, and sparse expansion tests.
- `tools/benchmark_comparison/tests/test_runner.py`: runner expectations for 76 canary and 228 final attempts.
- `tools/benchmark_comparison/tests/test_report_cli.py`: empty-canary audit expectation for 76 missing attempts.
- `tools/benchmark_comparison/tests/test_executors.py`: RGB command and preflight payload tests.
- `tools/benchmark_comparison/tests/test_normalize.py`: canonical ordering and sparse task-mode ordering tests.
- `tools/benchmark_comparison/tests/test_plot.py`: plotting order regression for runtime-only RGB.

---

### Task 1: Expand the immutable matrix and sparse task-mode model

**Files:**
- Modify: `tools/benchmark_comparison/models.py:75-131`
- Modify: `tools/benchmark_comparison/matrix.toml:9-39`
- Modify: `tools/benchmark_comparison/matrix.py:28-282`
- Modify: `tools/benchmark_comparison/tests/test_matrix.py:18-190`
- Modify: `tools/benchmark_comparison/tests/test_runner.py:123-158`
- Modify: `tools/benchmark_comparison/tests/test_report_cli.py:75-88`

**Interfaces:**
- Consumes: existing `Version`, `BenchmarkMode`, `BenchmarkMatrix`, `MatrixExpansion`, and TOML mode IDs.
- Produces: `BenchmarkTask.supported_modes: tuple[str, ...] | None`, `BenchmarkTask.enable_cameras: bool`, `BenchmarkTask.lab3_presets: tuple[str, ...]`, `BenchmarkTask.supports_mode(mode_id: str) -> bool`, `BenchmarkTask.presets_for(version: Version) -> tuple[str, ...]`, `BenchmarkAttempt.enable_cameras: bool`, and `BenchmarkAttempt.extra_presets: tuple[str, ...]`.
- Produces exact counts: `FINAL_LOGICAL_PAIR_COUNT = 114`, `FINAL_ATTEMPT_COUNT = 228`, `CANARY_LOGICAL_PAIR_COUNT = 38`, and `CANARY_ATTEMPT_COUNT = 76`.

- [ ] **Step 1: Write failing matrix mapping and capability tests**

Replace `_EXPECTED_TASK_ALIASES` and extend the checked-in matrix assertions in `test_matrix.py`:

```python
_EXPECTED_TASK_ALIASES = {
    "cartpole": ("Isaac-Cartpole-v0", "Isaac-Cartpole"),
    "cartpole_rgb_kit": ("Isaac-Cartpole-RGB-v0", "Isaac-Cartpole-Camera"),
    "cartpole_direct": ("Isaac-Cartpole-Direct-v0", "Isaac-Cartpole-Direct"),
    "ant": ("Isaac-Ant-v0", "Isaac-Ant"),
    "ant_direct": ("Isaac-Ant-Direct-v0", "Isaac-Ant-Direct"),
    "humanoid_manager": ("Isaac-Humanoid-v0", "Isaac-Humanoid"),
    "humanoid_direct": ("Isaac-Humanoid-Direct-v0", "Isaac-Humanoid-Direct"),
    "anymal_d_flat": ("Isaac-Velocity-Flat-Anymal-D-v0", "Isaac-Velocity-Flat-AnymalD"),
    "anymal_d_rough": ("Isaac-Velocity-Rough-Anymal-D-v0", "Isaac-Velocity-Rough-AnymalD"),
    "g1_flat": ("Isaac-Velocity-Flat-G1-v0", "Isaac-Velocity-Flat-G1"),
    "cassie_flat": ("Isaac-Velocity-Flat-Cassie-v0", "Isaac-Velocity-Flat-Cassie"),
    "allegro_cube": ("Isaac-Repose-Cube-Allegro-v0", "Isaac-Reorient-Cube-Allegro"),
    "franka_reach": ("Isaac-Reach-Franka-v0", "Isaac-Reach-Franka"),
}


def test_load_matrix_parses_explicit_task_aliases_and_run_parameters() -> None:
    matrix = load_matrix()
    tasks = {task.alias: task for task in matrix.tasks}

    assert {alias: (task.lab2_id, task.lab3_id) for alias, task in tasks.items()} == _EXPECTED_TASK_ALIASES
    assert tasks["cartpole_rgb_kit"].supported_modes == ("runtime-100", "runtime-1000")
    assert tasks["cartpole_rgb_kit"].enable_cameras is True
    assert tasks["cartpole_rgb_kit"].lab3_presets == ("rgb",)
    assert all(task.supported_modes is None for alias, task in tasks.items() if alias != "cartpole_rgb_kit")
    assert all(not task.enable_cameras for alias, task in tasks.items() if alias != "cartpole_rgb_kit")
    assert matrix.num_envs == 4096
    assert matrix.seeds == (42, 43, 44)
```

- [ ] **Step 2: Write failing sparse expansion and validation tests**

Add the following tests and update the old count assertions to 114/228 and 38/76:

```python
def test_rgb_cartpole_expands_runtime_only_while_other_tasks_expand_all_modes() -> None:
    expansion = expand_final_matrix(load_matrix())
    task_modes = {(pair.logical_task, pair.mode.id) for pair in expansion.pairs}

    assert ("cartpole_rgb_kit", "runtime-100") in task_modes
    assert ("cartpole_rgb_kit", "runtime-1000") in task_modes
    assert ("cartpole_rgb_kit", "training-100") not in task_modes
    assert {mode for task, mode in task_modes if task == "anymal_d_flat"} == {
        "runtime-100",
        "runtime-1000",
        "training-100",
    }
    assert {mode for task, mode in task_modes if task == "anymal_d_rough"} == {
        "runtime-100",
        "runtime-1000",
        "training-100",
    }


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'supported_modes = ["runtime-100", "runtime-1000"]',
            "supported_modes = []",
            "task supported_modes must not be empty",
        ),
        (
            'supported_modes = ["runtime-100", "runtime-1000"]',
            'supported_modes = ["runtime-100", "runtime-100"]',
            "duplicate task mode ID",
        ),
        (
            'supported_modes = ["runtime-100", "runtime-1000"]',
            'supported_modes = ["runtime-100", "runtime-999"]',
            "unknown task mode ID",
        ),
    ],
)
def test_load_matrix_rejects_invalid_task_mode_subsets(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _write_invalid_matrix(tmp_path, old, new)

    with pytest.raises(ValueError, match=message):
        load_matrix(path)


def test_load_matrix_rejects_incorrect_task_and_mode_shape(tmp_path: Path) -> None:
    path = _write_invalid_matrix(
        tmp_path,
        'id = "training-100"',
        'id = "training-renamed"',
    )

    with pytest.raises(ValueError, match="unexpected mode IDs"):
        load_matrix(path)
```

Delete `_write_count_preserving_invalid_matrix` and its obsolete six-task
shape test. Update runner assertions to 38 attempts per version for canary and
228 total for final. Update the empty report-only canary audit assertion to 76
failed-or-missing attempts. Do not change `test_actual_report_artifacts.py`.

- [ ] **Step 3: Run the focused tests and verify the red state**

Run:

```bash
uvx --from pytest python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py \
  tools/benchmark_comparison/tests/test_runner.py \
  tools/benchmark_comparison/tests/test_report_cli.py -q
```

Expected: FAIL because the new aliases are absent, `BenchmarkTask` has no capability fields, and the expansion still contains 54 final pairs and 36 canary attempts.

- [ ] **Step 4: Add task and attempt capabilities to the immutable models**

Extend `BenchmarkTask` and `BenchmarkAttempt` in `models.py`:

```python
@dataclass(frozen=True)
class BenchmarkTask:
    """Logical task alias, version-specific IDs, and execution capabilities."""

    alias: str
    lab2_id: str
    lab3_id: str
    supported_modes: tuple[str, ...] | None = None
    enable_cameras: bool = False
    lab3_presets: tuple[str, ...] = ()

    def concrete_id(self, version: Version) -> str:
        """Return the configured task identifier for ``version``."""
        if version is Version.LAB2:
            return self.lab2_id
        return self.lab3_id

    def supports_mode(self, mode_id: str) -> bool:
        """Return whether this task participates in ``mode_id``."""
        return self.supported_modes is None or mode_id in self.supported_modes

    def presets_for(self, version: Version) -> tuple[str, ...]:
        """Return task-specific preset additions for ``version``."""
        if version is Version.LAB3:
            return self.lab3_presets
        return ()
```

Add these required fields to `BenchmarkAttempt` immediately after `framework`:

```python
    enable_cameras: bool
    extra_presets: tuple[str, ...]
```

- [ ] **Step 5: Replace the checked-in task list with the exact 13-task matrix**

Keep the existing `[matrix]` and `[[mode]]` tables. Replace only the task tables in `matrix.toml` with:

```toml
[[task]]
alias = "cartpole"
lab2_id = "Isaac-Cartpole-v0"
lab3_id = "Isaac-Cartpole"

[[task]]
alias = "cartpole_rgb_kit"
lab2_id = "Isaac-Cartpole-RGB-v0"
lab3_id = "Isaac-Cartpole-Camera"
supported_modes = ["runtime-100", "runtime-1000"]
enable_cameras = true
lab3_presets = ["rgb"]

[[task]]
alias = "cartpole_direct"
lab2_id = "Isaac-Cartpole-Direct-v0"
lab3_id = "Isaac-Cartpole-Direct"

[[task]]
alias = "ant"
lab2_id = "Isaac-Ant-v0"
lab3_id = "Isaac-Ant"

[[task]]
alias = "ant_direct"
lab2_id = "Isaac-Ant-Direct-v0"
lab3_id = "Isaac-Ant-Direct"

[[task]]
alias = "humanoid_manager"
lab2_id = "Isaac-Humanoid-v0"
lab3_id = "Isaac-Humanoid"

[[task]]
alias = "humanoid_direct"
lab2_id = "Isaac-Humanoid-Direct-v0"
lab3_id = "Isaac-Humanoid-Direct"

[[task]]
alias = "anymal_d_flat"
lab2_id = "Isaac-Velocity-Flat-Anymal-D-v0"
lab3_id = "Isaac-Velocity-Flat-AnymalD"

[[task]]
alias = "anymal_d_rough"
lab2_id = "Isaac-Velocity-Rough-Anymal-D-v0"
lab3_id = "Isaac-Velocity-Rough-AnymalD"

[[task]]
alias = "g1_flat"
lab2_id = "Isaac-Velocity-Flat-G1-v0"
lab3_id = "Isaac-Velocity-Flat-G1"

[[task]]
alias = "cassie_flat"
lab2_id = "Isaac-Velocity-Flat-Cassie-v0"
lab3_id = "Isaac-Velocity-Flat-Cassie"

[[task]]
alias = "allegro_cube"
lab2_id = "Isaac-Repose-Cube-Allegro-v0"
lab3_id = "Isaac-Reorient-Cube-Allegro"

[[task]]
alias = "franka_reach"
lab2_id = "Isaac-Reach-Franka-v0"
lab3_id = "Isaac-Reach-Franka"
```

- [ ] **Step 6: Parse, validate, and expand the sparse matrix**

In `matrix.py`, set the four count constants to 114, 228, 38, and 76; replace `_TASK_IDENTIFIERS` with the exact ordered mappings above; and parse optional task fields:

```python
def _parse_task(data: Any) -> BenchmarkTask:
    task = _as_dict(data, "task entry")
    supported_modes_value = task.get("supported_modes")
    supported_modes = (
        None
        if supported_modes_value is None
        else tuple(
            _as_str(mode_id, "task.supported_modes")
            for mode_id in _as_list(supported_modes_value, "task.supported_modes")
        )
    )
    return BenchmarkTask(
        alias=_as_str(task.get("alias"), "task.alias"),
        lab2_id=_as_str(task.get("lab2_id"), "task.lab2_id"),
        lab3_id=_as_str(task.get("lab3_id"), "task.lab3_id"),
        supported_modes=supported_modes,
        enable_cameras=_as_bool(task.get("enable_cameras", False), "task.enable_cameras"),
        lab3_presets=tuple(
            _as_str(preset, "task.lab3_presets")
            for preset in _as_list(task.get("lab3_presets", []), "task.lab3_presets")
        ),
    )


def _as_bool(value: Any, name: str) -> bool:
    """Return a TOML boolean or raise a focused validation error."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
```

Filter modes before pair creation and propagate task execution metadata:

```python
        for task in matrix.tasks:
            for mode in matrix.modes:
                if not task.supports_mode(mode.id):
                    continue
                pair_order = len(pairs)
```

Inside `_make_attempt`, add:

```python
        enable_cameras=task.enable_cameras,
        extra_presets=task.presets_for(version),
```

Validate every explicit subset before count checks:

```python
    configured_mode_ids = {mode.id for mode in matrix.modes}
    for task in matrix.tasks:
        if task.supported_modes is not None:
            if not task.supported_modes:
                raise ValueError("task supported_modes must not be empty")
            if len(task.supported_modes) != len(set(task.supported_modes)):
                raise ValueError("duplicate task mode ID")
            if not set(task.supported_modes) <= configured_mode_ids:
                raise ValueError("unknown task mode ID")

    rgb_task = next(task for task in matrix.tasks if task.alias == "cartpole_rgb_kit")
    if (
        rgb_task.supported_modes != ("runtime-100", "runtime-1000")
        or not rgb_task.enable_cameras
        or rgb_task.lab3_presets != ("rgb",)
    ):
        raise ValueError("unexpected RGB task capabilities")
    if any(
        task.supported_modes is not None or task.enable_cameras or task.lab3_presets
        for task in matrix.tasks
        if task.alias != "cartpole_rgb_kit"
    ):
        raise ValueError("unexpected non-RGB task capabilities")
```

Compute matrix shape with supported task-mode cells rather than a Cartesian product:

```python
    task_mode_count = sum(
        1 for task in matrix.tasks for mode in matrix.modes if task.supports_mode(mode.id)
    )
    if task_mode_count * len(matrix.seeds) != FINAL_LOGICAL_PAIR_COUNT:
        raise ValueError(f"expected {FINAL_LOGICAL_PAIR_COUNT} logical pairs")
```

Replace the old six-task shape check with this exact 13-task check; retain the
existing `_TASK_IDENTIFIERS`, `_MODE_IDS`, `_MODE_DEFINITIONS`, seed, and
environment comparisons directly after it:

```python
    if len(matrix.tasks) != 13 or len(matrix.modes) != 3:
        raise ValueError("expected 13 tasks and 3 modes")
```

- [ ] **Step 7: Run the focused tests and verify green**

Run:

```bash
uvx --from pytest python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_matrix.py \
  tools/benchmark_comparison/tests/test_runner.py \
  tools/benchmark_comparison/tests/test_report_cli.py -q
```

Expected: all selected tests PASS; final expansion reports 114 pairs/228 attempts and canary reports 38 pairs/76 attempts.

- [ ] **Step 8: Run hooks, stage, rerun hooks, and commit**

Run:

```bash
uvx --from pre-commit pre-commit run --all-files
git add tools/benchmark_comparison/models.py \
  tools/benchmark_comparison/matrix.toml \
  tools/benchmark_comparison/matrix.py \
  tools/benchmark_comparison/tests/test_matrix.py \
  tools/benchmark_comparison/tests/test_runner.py \
  tools/benchmark_comparison/tests/test_report_cli.py
uvx --from pre-commit pre-commit run --all-files
git commit -m "Expand benchmark task matrix"
```

Expected: both hook runs PASS and the commit contains only the six files in the `git add` command.

---

### Task 2: Add camera-aware commands and training-aware preflight

**Files:**
- Modify: `tools/benchmark_comparison/executors.py:48-53,329-349,628-679`
- Modify: `tools/benchmark_comparison/tests/test_executors.py:45-190`

**Interfaces:**
- Consumes: `BenchmarkAttempt.enable_cameras`, `BenchmarkAttempt.extra_presets`, `BenchmarkTask.supports_mode()`, and `BenchmarkTask.presets_for()` from Task 1.
- Produces: one combined `presets=<physics>,<task presets>` argument, optional `--enable_cameras`, and a registration probe carrying `(task_id, supports_training, enable_cameras, presets)` records.

- [ ] **Step 1: Write failing RGB command tests**

Add to `test_executors.py`:

```python
def test_rgb_cartpole_runtime_commands_enable_cameras_and_select_kit_rgb(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lab2 = Lab2DockerExecutor(config).invocation(_attempt(Version.LAB2, task="cartpole_rgb_kit"))
    lab3 = Lab3UvExecutor(config).invocation(_attempt(Version.LAB3, task="cartpole_rgb_kit"))

    assert "--enable_cameras" in lab2.argv
    assert "--enable_cameras" in lab3.argv
    assert "presets=physx" in lab2.argv
    assert "presets=physx,rgb" in lab3.argv
    assert "presets=physx" not in lab3.argv
    assert all("newton_renderer" not in argument for argument in lab3.argv)


def test_non_camera_tasks_keep_camera_flags_and_extra_presets_disabled(tmp_path: Path) -> None:
    config = _config(tmp_path)

    for version, executor in (
        (Version.LAB2, Lab2DockerExecutor(config)),
        (Version.LAB3, Lab3UvExecutor(config)),
    ):
        invocation = executor.invocation(_attempt(version, task="cartpole"))
        assert "--enable_cameras" not in invocation.argv
        assert all(argument != "presets=physx,rgb" for argument in invocation.argv)
```

- [ ] **Step 2: Write failing preflight capability tests**

Replace the “RSL configs for every task” test with:

```python
def test_version_probes_require_rsl_only_for_training_tasks_and_validate_rgb_camera(tmp_path: Path) -> None:
    config = _config(tmp_path)

    lab2_probe = Lab2DockerExecutor(config).version_invocation().argv[-1]
    lab3_probe = Lab3UvExecutor(config).version_invocation().argv[-1]

    assert "AppLauncher(headless=True, enable_cameras=True)" in lab2_probe
    assert "if supports_training" in lab2_probe
    assert "if supports_training" in lab3_probe
    assert "rsl_rl_cfg_entry_point" in lab2_probe
    assert "rsl_rl_cfg_entry_point" in lab3_probe
    assert "Isaac-Cartpole-RGB-v0" in lab2_probe
    assert "Isaac-Cartpole-Camera" in lab3_probe
    assert "('Isaac-Cartpole-Camera', False, True, ('rgb',))" in lab3_probe
    assert "presets={','.join(presets)}" in lab3_probe
    assert "env_cfg.scene.tiled_camera.data_types == ['rgb']" in lab2_probe
    assert "env_cfg.scene.tiled_camera.data_types == ['rgb']" in lab3_probe
    assert "IsaacRtxRendererCfg" in lab3_probe
```

Also update `test_version_probes_use_version_specific_app_startup_and_sentinel` to expect the camera-enabled Lab 2 launcher string.

- [ ] **Step 3: Run executor tests and verify the red state**

Run:

```bash
uvx --from pytest python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_executors.py -q
```

Expected: FAIL because RGB attempts do not yet add camera/preset arguments and preflight still requires RSL-RL for every task.

- [ ] **Step 4: Combine physics and task presets in benchmark commands**

Replace `_benchmark_arguments` preset handling with:

```python
        presets: list[str] = []
        if attempt.version is not Version.LAB3 or attempt.concrete_task not in _LAB3_DEFAULT_PHYSX_TASKS:
            presets.append("physx")
        presets.extend(attempt.extra_presets)
        preset_argument = (f"presets={','.join(presets)}",) if presets else ()
        camera_arguments = ("--enable_cameras",) if attempt.enable_cameras else ()
        return (
            *common,
            *bounded,
            "--benchmark_formatter",
            "schema,json",
            *preset_argument,
            *camera_arguments,
            "--headless",
        )
```

This yields Lab 2 `presets=physx --enable_cameras` and Lab 3 `presets=physx,rgb --enable_cameras` for RGB Cartpole while leaving existing task commands unchanged.

- [ ] **Step 5: Make registration preflight capability-aware**

At the start of `_registration_probe`, build exact version-specific records:

```python
    tasks = load_matrix().tasks
    task_specs = tuple(
        (
            task.concrete_id(version),
            task.supports_mode("training-100"),
            task.enable_cameras,
            task.presets_for(version),
        )
        for task in tasks
    )
```

Generate the Lab 2 probe with a camera-capable app and conditional agent resolution:

```python
        script = (
            "from isaaclab.app import AppLauncher\n"
            "simulation_app = AppLauncher(headless=True, enable_cameras=True).app\n"
            "from isaaclab.test.benchmark.formatters import MetricsFormatter\n"
            "MetricsFormatter.get_instance('schema')\n"
            "MetricsFormatter.get_instance('json')\n"
            "import gymnasium as gym, isaaclab_tasks\n"
            "from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg\n"
            f"task_specs = {task_specs!r}\n"
            "for task_id, supports_training, enable_cameras, _presets in task_specs:\n"
            "    assert task_id in gym.registry\n"
            "    env_cfg = parse_env_cfg(task_id, device='cuda:0', num_envs=4096)\n"
            "    assert env_cfg.scene.num_envs == 4096\n"
            "    assert type(env_cfg.sim.physx).__name__ == 'PhysxCfg'\n"
            "    if enable_cameras:\n"
            "        assert env_cfg.scene.tiled_camera.data_types == ['rgb']\n"
            "    if supports_training:\n"
            "        agent_cfg = load_cfg_from_registry(task_id, 'rsl_rl_cfg_entry_point')\n"
            "        assert agent_cfg is not None\n"
            f"print({_LAB2_PREFLIGHT_SENTINEL!r}, flush=True)\n"
            "simulation_app.close()\n"
        )
```

Generate the Lab 3 loop from the same records, combining presets and resolving an agent only for training tasks:

```python
        f"task_specs = {task_specs!r}\n"
        f"default_physx_tasks = {_LAB3_DEFAULT_PHYSX_TASKS!r}\n"
        "for task_id, supports_training, enable_cameras, extra_presets in task_specs:\n"
        "    assert task_id in gym.registry\n"
        "    presets = [] if task_id in default_physx_tasks else ['physx']\n"
        "    presets.extend(extra_presets)\n"
        "    sys.argv = [sys.argv[0], 'env.scene.num_envs=4096']\n"
        "    if presets:\n"
        "        sys.argv.append(f\"presets={','.join(presets)}\")\n"
        "    agent_key = 'rsl_rl_cfg_entry_point' if supports_training else None\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        env_cfg, agent_cfg = resolve_task_config(task_id, agent_key)\n"
        "    assert env_cfg.scene.num_envs == 4096\n"
        "    assert type(env_cfg.sim.physics).__name__ == 'PhysxCfg'\n"
        "    if enable_cameras:\n"
        "        assert env_cfg.scene.tiled_camera.data_types == ['rgb']\n"
        "        assert type(env_cfg.scene.tiled_camera.renderer_cfg).__name__ == 'IsaacRtxRendererCfg'\n"
        "    if supports_training:\n"
        "        assert agent_cfg is not None\n"
```

The complete replacement must retain these exact terminal lines in their
current branches so existing preflight behavior remains testable:

```python
            f"print({_LAB2_PREFLIGHT_SENTINEL!r}, flush=True)\n"
            "simulation_app.close()\n"
```

```python
        "print('ok', flush=True)\n"
```

- [ ] **Step 6: Run executor tests and verify green**

Run:

```bash
uvx --from pytest python -m pytest --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_executors.py -q
```

Expected: all executor tests PASS, including existing default-PhysX behavior and the new RGB command/preflight assertions.

- [ ] **Step 7: Run hooks, stage, rerun hooks, and commit**

Run:

```bash
uvx --from pre-commit pre-commit run --all-files
git add tools/benchmark_comparison/executors.py \
  tools/benchmark_comparison/tests/test_executors.py
uvx --from pre-commit pre-commit run --all-files
git commit -m "Support RGB camera benchmark tasks"
```

Expected: both hook runs PASS and the commit contains only executor code and its tests.

---

### Task 3: Add canonical expanded ordering and sparse plotting

**Files:**
- Modify: `tools/benchmark_comparison/normalize.py:25-31`
- Modify: `tools/benchmark_comparison/plot.py:14-145`
- Modify: `tools/benchmark_comparison/tests/test_normalize.py:20-35`
- Modify: `tools/benchmark_comparison/tests/test_plot.py:12-73`

**Interfaces:**
- Consumes: the exact aliases and supported modes from Task 1.
- Produces: `TASK_ORDER: tuple[str, ...]`, `TASK_MODES: dict[str, tuple[str, ...]]`, and `task_order_for_mode(mode: str) -> tuple[str, ...]` for deterministic reporting and plotting.

- [ ] **Step 1: Write failing canonical-order tests**

Add imports and assertions:

```python
from tools.benchmark_comparison.normalize import TASK_MODES, TASK_ORDER, task_order_for_mode


def test_expanded_task_order_keeps_rgb_runtime_only_and_both_anymal_terrains() -> None:
    assert TASK_ORDER == (
        "cartpole",
        "cartpole_rgb_kit",
        "cartpole_direct",
        "ant",
        "ant_direct",
        "humanoid_manager",
        "humanoid_direct",
        "anymal_d_flat",
        "anymal_d_rough",
        "g1_flat",
        "cassie_flat",
        "allegro_cube",
        "franka_reach",
    )
    assert task_order_for_mode("runtime-100") == TASK_ORDER
    assert task_order_for_mode("runtime-1000") == TASK_ORDER
    assert "cartpole_rgb_kit" not in task_order_for_mode("training-100")
    assert TASK_MODES["cartpole_rgb_kit"] == ("runtime-100", "runtime-1000")
```

Place this test in `test_normalize.py`. Add this regression to `test_plot.py`:

```python
def test_plot_task_order_does_not_create_an_rgb_training_slot() -> None:
    from tools.benchmark_comparison.plot import _task_order_for_mode
    from tools.benchmark_comparison.normalize import TASK_ORDER

    assert "cartpole_rgb_kit" in _task_order_for_mode("runtime-100")
    assert "cartpole_rgb_kit" not in _task_order_for_mode("training-100")
    assert _task_order_for_mode("training-100") == tuple(
        task for task in TASK_ORDER if task != "cartpole_rgb_kit"
    )
```

- [ ] **Step 2: Run normalization and plot tests and verify the red state**

Run:

```bash
uvx --from pytest --with matplotlib python -m pytest \
  --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_normalize.py \
  tools/benchmark_comparison/tests/test_plot.py -q
```

Expected: FAIL because `TASK_ORDER` still has six entries and sparse task-mode ordering does not exist.

- [ ] **Step 3: Define the canonical task order and supported modes**

Import `load_matrix`, then replace the current `TASK_ORDER` declaration in
`normalize.py` and add the helper. This keeps `matrix.toml` as the only task
capability source:

```python
from .matrix import load_matrix

_MATRIX = load_matrix()
TASK_ORDER = tuple(task.alias for task in _MATRIX.tasks)
MODE_ORDER = tuple(mode.id for mode in _MATRIX.modes)
TASK_MODES = {
    task.alias: tuple(mode.id for mode in _MATRIX.modes if task.supports_mode(mode.id))
    for task in _MATRIX.tasks
}


def task_order_for_mode(mode: str) -> tuple[str, ...]:
    """Return canonical tasks that support ``mode``."""
    if mode not in MODE_ORDER:
        raise ValueError(f"unknown benchmark mode: {mode}")
    return tuple(task for task in TASK_ORDER if mode in TASK_MODES[task])
```

Do not edit the four existing `_order(TASK_ORDER, ...)` sort keys; the new
constant automatically expands their ordering domain.

- [ ] **Step 4: Make plotting consume the mode-specific order**

Import `task_order_for_mode` into `plot.py`, expose the local helper used by tests, and pass the selected tasks to `_draw_mode`:

```python
from .normalize import MODE_ORDER, TASK_ORDER, VERSION_ORDER, NormalizedRun, read_raw_runs_csv, task_order_for_mode


def _task_order_for_mode(mode: str) -> tuple[str, ...]:
    """Return the deterministic plot order for one benchmark mode."""
    return task_order_for_mode(mode)
```

Inside `generate_plots`:

```python
            for axis, mode in zip(axes, MODE_ORDER, strict=True):
                task_order = _task_order_for_mode(mode)
                _draw_mode(axis, runs, mode, attribute, task_order)
                axis.set_title(mode)
                axis.set_ylabel(y_label)
                axis.set_xticks(range(len(task_order)), [task.replace("_", "\n") for task in task_order])
                axis.tick_params(axis="x", labelsize=8)
```

Change `_draw_mode` to accept the order and use it consistently:

```python
def _draw_mode(
    axis,
    runs: tuple[NormalizedRun, ...],
    mode: str,
    attribute: str,
    task_order: tuple[str, ...],
) -> None:
    width = 0.34
    version_offsets = {"lab2": -width / 2, "lab3": width / 2}
    max_value = max(
        (float(getattr(run, attribute)) for run in runs if run.mode == mode),
        default=1.0,
    )
    label_height = max(max_value * 0.025, 0.1)
    for task_index, task in enumerate(task_order):
        for version in VERSION_ORDER:
            values = sorted(
                (
                    (run.seed, float(getattr(run, attribute)))
                    for run in runs
                    if run.mode == mode and run.logical_task == task and run.version == version
                ),
                key=lambda item: item[0],
            )
            x_position = task_index + version_offsets[version]
            if not values:
                axis.text(
                    x_position,
                    label_height,
                    "Missing",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=_VERSION_COLORS[version],
                )
                continue
            measurements = [value for _, value in values]
            mean = statistics.fmean(measurements)
            standard_deviation = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
            axis.bar(
                x_position,
                mean,
                width=width * 0.86,
                color=_VERSION_COLORS[version],
                alpha=0.55,
                yerr=standard_deviation,
                capsize=3,
                error_kw={"linewidth": 1},
            )
            offsets = _repeat_offsets(len(measurements), width * 0.38)
            axis.scatter(
                [x_position + offset for offset in offsets],
                measurements,
                s=13,
                color=_VERSION_COLORS[version],
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
    axis.set_xlim(-0.55, len(task_order) - 0.45)
    axis.set_ylim(bottom=0)
```

- [ ] **Step 5: Run normalization and plot tests and verify green**

Run:

```bash
uvx --from pytest --with matplotlib python -m pytest \
  --confcutdir=tools/benchmark_comparison/tests \
  tools/benchmark_comparison/tests/test_normalize.py \
  tools/benchmark_comparison/tests/test_plot.py -q
```

Expected: all selected tests PASS; deterministic PNG/SVG regeneration still passes, and training plots have no RGB task slot.

- [ ] **Step 6: Run hooks, stage, rerun hooks, and commit**

Run:

```bash
uvx --from pre-commit pre-commit run --all-files
git add tools/benchmark_comparison/normalize.py \
  tools/benchmark_comparison/plot.py \
  tools/benchmark_comparison/tests/test_normalize.py \
  tools/benchmark_comparison/tests/test_plot.py
uvx --from pre-commit pre-commit run --all-files
git commit -m "Handle sparse benchmark task plots"
```

Expected: both hook runs PASS and the commit contains only ordering/plot code and tests.

---

### Task 4: Verify the complete task-list change

**Files:**
- Verify only: `tools/benchmark_comparison/`
- Verify only: `tools/benchmark_comparison/tests/`
- Verify unchanged: `tools/benchmark_comparison/tests/test_actual_report_artifacts.py`

**Interfaces:**
- Consumes: all interfaces and commits from Tasks 1-3.
- Produces: evidence that the simulator-free controller suite, historical artifacts, formatting hooks, staged state, and branch state are clean.

- [ ] **Step 1: Run the complete simulator-free controller suite**

Run:

```bash
uvx --from pytest --with matplotlib python -m pytest \
  --confcutdir=tools/benchmark_comparison/tests tools/benchmark_comparison/tests -q
```

Expected: every controller test PASS, including the retained actual-artifact checks when `/home/antoiner/benchmarks/isaaclab2-vs-3/artifacts/91631f3328` is present.

- [ ] **Step 2: Verify exact expansion and RGB behavior from the public matrix API**

Run:

```bash
/usr/bin/python3 -c "from tools.benchmark_comparison.matrix import expand_canary_matrix, expand_final_matrix, load_matrix; m=load_matrix(); f=expand_final_matrix(m); c=expand_canary_matrix(m); assert (len(m.tasks), len(f.pairs), len(f.attempts), len(c.pairs), len(c.attempts)) == (13, 114, 228, 38, 76); assert not any(a.logical_task == 'cartpole_rgb_kit' and a.mode.id == 'training-100' for a in f.attempts); assert {'anymal_d_flat', 'anymal_d_rough'} <= {a.logical_task for a in f.attempts}; print('matrix-ok')"
```

Expected output: `matrix-ok`.

- [ ] **Step 3: Run all repository hooks**

Run:

```bash
uvx --from pre-commit pre-commit run --all-files
```

Expected: every hook PASS and no file is rewritten.

- [ ] **Step 4: Audit the final diff and worktree**

Run:

```bash
git diff --check HEAD~3..HEAD
git status --short
git log -4 --oneline
```

Expected:

- `git diff --check` emits no output.
- `git status --short` is empty except for this implementation plan if it was intentionally left uncommitted for execution tracking.
- The last three implementation commits are `Expand benchmark task matrix`, `Support RGB camera benchmark tasks`, and `Handle sparse benchmark task plots`, following the committed design-spec commit.

- [ ] **Step 5: Prepare the completion handoff without running benchmarks**

Report:

```text
Expanded matrix: 13 tasks, 38 task-mode cells, 114 final pairs / 228 attempts,
38 canary pairs / 76 attempts. RGB Cartpole is runtime-only with cameras enabled
and Lab 3 presets=physx,rgb; flat and rough Anymal-D are both present. The full
simulator-free controller suite and all pre-commit hooks pass. No smoke, canary,
final, Docker, Isaac Sim, or GPU benchmark job was run.
```
