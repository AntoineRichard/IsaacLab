# Cartpole and ANYmal-D Kamino DVI Benchmark Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tuned Kamino DVI presets for Cartpole Direct and ANYmal-D Flat, run fresh three-seed five-variant benchmarks for both tasks, and regenerate the combined Ant/Cartpole/ANYmal-D report.

**Architecture:** Keep solver tuning in explicit task-local named physics presets and keep experiment dimensions in the immutable benchmark matrix. Reuse the locked current/PR Newton environments and validated Ant artifacts through ignored worktree symlinks, then run all new identities into the same resumable artifact/log roots. Existing parsing, aggregation, plotting, and reporting code produces the combined report once every task/variant has three complete seeds.

**Tech Stack:** Python 3.12, IsaacLab configuration classes, Newton Kamino/MJWarp, PhysX, RSL-RL, pytest, Hydra presets, TensorBoard event parsing, Matplotlib, Git worktrees.

## Global Constraints

- Use RSL-RL for every training run.
- Use 300 training iterations and seeds 42–44.
- Start each task at 4096 environments; lower the common task count only after an explicit capacity failure.
- Compare current Kamino, PR 3570 P-ADMM, tuned PR 3570 DVI, MJWarp, and PhysX.
- Preserve the validated Ant traces; rerun Cartpole and ANYmal-D Flat from scratch.
- Treat missing reward, episode-length, or success series as a benchmark-schema bug, not a training failure.
- Run variants sequentially on one GPU and preserve atomic manifests, logs, bundles, and TensorBoard events.
- Use `./isaaclab.sh -p` for Python and pytest commands.
- Run `./isaaclab.sh -f` before every commit; document the known missing `git-lfs` executable if it is the only failing hook.
- Do not push to `origin`.

---

### Task 1: Add the Cartpole tuned DVI preset

**Files:**
- Create: `source/isaaclab_tasks/test/core/test_cartpole_physics_presets.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env_cfg.py`

**Interfaces:**
- Consumes: `isaaclab_newton.physics.KaminoSolverCfg` fields already exposed by the tuning branch.
- Produces: `CartpolePhysicsCfg.newton_kamino_dvi: NewtonCfg`, selectable as `presets=newton_kamino_dvi`.

- [ ] **Step 1: Write the failing Cartpole preset test**

Create `source/isaaclab_tasks/test/core/test_cartpole_physics_presets.py` with the 2026 SPDX header and:

```python
"""Tests for the Cartpole Direct physics presets."""

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpolePhysicsCfg


def test_cartpole_preserves_kamino_padmm_preset() -> None:
    """The existing Cartpole Kamino preset must remain the P-ADMM control."""
    physics = CartpolePhysicsCfg()

    assert isinstance(physics.newton_kamino, NewtonCfg)
    assert physics.newton_kamino.solver_cfg.dynamics_solver is None


def test_cartpole_exposes_sparse_kamino_dvi_preset() -> None:
    """The Cartpole DVI preset must preserve task settings and select sparse DVI."""
    physics = CartpolePhysicsCfg()
    solver = physics.newton_kamino_dvi.solver_cfg

    assert isinstance(physics.newton_kamino_dvi, NewtonCfg)
    assert isinstance(solver, KaminoSolverCfg)
    assert solver.dynamics_solver == "dvi"
    assert solver.integrator == "moreau"
    assert solver.use_collision_detector is True
    assert solver.sparse_jacobian is True
    assert solver.sparse_dynamics is True
    assert solver.dynamics_preconditioning is False
    assert solver.dynamics_linear_solver_type == "CR"
    assert solver.dynamics_linear_solver_max_iterations == 9
    assert solver.dvi_block_iterations == 16
    assert solver.dvi_contact_iterations == 2
    assert solver.dvi_bilateral_solve_period == 2
    assert solver.dvi_omega == 0.3
    assert solver.dvi_contact_jacobi_omega == 0.45
    assert solver.dvi_contact_jacobi_relaxation == 0.9
    assert solver.collision_detector_pipeline == "unified"
    assert solver.collision_detector_max_contacts_per_pair == 8
    assert physics.newton_kamino_dvi.use_cuda_graph is True
```

- [ ] **Step 2: Run the new test and verify red**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-pr3570 \
  ./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/core/test_cartpole_physics_presets.py -q
```

Expected: one test passes and `test_cartpole_exposes_sparse_kamino_dvi_preset` fails because `newton_kamino_dvi` is absent.

- [ ] **Step 3: Add the minimal named preset**

Add this field immediately after `CartpolePhysicsCfg.newton_kamino`:

```python
    newton_kamino_dvi: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="moreau",
            dynamics_solver="dvi",
            use_collision_detector=True,
            sparse_jacobian=True,
            sparse_dynamics=True,
            constraints_alpha=0.1,
            dynamics_preconditioning=False,
            dynamics_linear_solver_type="CR",
            dynamics_linear_solver_max_iterations=9,
            dvi_omega=0.3,
            dvi_block_iterations=16,
            dvi_contact_iterations=2,
            dvi_bilateral_solve_period=2,
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
            collision_detector_pipeline="unified",
            collision_detector_max_contacts_per_pair=8,
        ),
        debug_mode=False,
        use_cuda_graph=True,
    )
```

- [ ] **Step 4: Run Cartpole preset and Hydra tests**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-pr3570 \
  ./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_cartpole_physics_presets.py \
  source/isaaclab_tasks/test/core/test_hydra.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run hooks and commit**

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current ./isaaclab.sh -f
git add source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env_cfg.py \
  source/isaaclab_tasks/test/core/test_cartpole_physics_presets.py
git commit -m "Add Cartpole Kamino DVI preset"
```

---

### Task 2: Add the ANYmal-D tuned DVI preset

**Files:**
- Modify: `source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/flat_env_cfg.py`

**Interfaces:**
- Consumes: `KaminoSolverCfg` DVI fields and the existing ANYmal-D `max_contacts_per_world=64` limit.
- Produces: `PhysicsCfg.newton_kamino_dvi: NewtonCfg`, selectable as `presets=newton_kamino_dvi`.

- [ ] **Step 1: Extend the ANYmal-D preset test first**

Add:

```python
def test_anymal_d_flat_exposes_sparse_kamino_dvi_preset() -> None:
    """ANYmal-D Flat must expose the tuned DVI preset with its contact capacity."""
    physics = PhysicsCfg()
    solver = physics.newton_kamino_dvi.solver_cfg

    assert isinstance(physics.newton_kamino_dvi, NewtonCfg)
    assert isinstance(solver, KaminoSolverCfg)
    assert solver.max_contacts_per_world == 64
    assert solver.dynamics_solver == "dvi"
    assert solver.integrator == "moreau"
    assert solver.sparse_jacobian is True
    assert solver.sparse_dynamics is True
    assert solver.dynamics_preconditioning is False
    assert solver.dynamics_linear_solver_type == "CR"
    assert solver.dynamics_linear_solver_max_iterations == 9
    assert solver.dvi_block_iterations == 16
    assert solver.dvi_contact_iterations == 2
    assert solver.dvi_bilateral_solve_period == 2
    assert solver.dvi_omega == 0.3
    assert solver.dvi_contact_jacobi_omega == 0.45
    assert solver.dvi_contact_jacobi_relaxation == 0.9
```

- [ ] **Step 2: Verify the test fails for the missing field**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-pr3570 \
  ./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py -q
```

Expected: the new DVI test fails with `AttributeError`; the two existing preset tests pass.

- [ ] **Step 3: Add the task-local ANYmal-D preset**

Add after `PhysicsCfg.newton_kamino`:

```python
    newton_kamino_dvi = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            max_contacts_per_world=64,
            integrator="moreau",
            dynamics_solver="dvi",
            sparse_jacobian=True,
            sparse_dynamics=True,
            dynamics_preconditioning=False,
            dynamics_linear_solver_type="CR",
            dynamics_linear_solver_max_iterations=9,
            dvi_omega=0.3,
            dvi_block_iterations=16,
            dvi_contact_iterations=2,
            dvi_bilateral_solve_period=2,
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        ),
        num_substeps=1,
        debug_mode=False,
    )
```

- [ ] **Step 4: Verify the ANYmal-D tests pass**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-pr3570 \
  ./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py -q
```

Expected: three tests pass.

- [ ] **Step 5: Run hooks and commit**

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current ./isaaclab.sh -f
git add source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/flat_env_cfg.py \
  source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py
git commit -m "Add ANYmal D Kamino DVI preset"
```

---

### Task 3: Expand the immutable matrix to three tasks

**Files:**
- Modify: `benchmarks/kamino_dvi/tests/test_matrix.py`
- Modify: `benchmarks/kamino_dvi/matrix.py`
- Modify: `benchmarks/kamino_dvi/matrix.yaml`
- Modify: `benchmarks/kamino_dvi/README.md`

**Interfaces:**
- Consumes: `TaskName.ANYMAL_D` and the global five variant definitions.
- Produces: a matrix with 15 cells, 45 full identities, and 15 preflight identities.

- [ ] **Step 1: Change matrix expectations first**

Update `EXPECTED_TASKS` to:

```python
EXPECTED_TASKS = (
    TaskName.CARTPOLE,
    TaskName.ANT,
    TaskName.ANYMAL_D,
)
```

Rename the expansion test to `test_matrix_expands_to_15_cells_and_45_unique_full_runs` and assert 15 cells, 45 runs, and 45 unique runs. Assert 15 preflights. Extend the common-task docstring to name Cartpole, Ant, and ANYmal-D. Add:

```python
    assert ordered_variants(matrix, TaskName.ANYMAL_D, 42) == ALL_VARIANTS
```

- [ ] **Step 2: Verify the matrix tests fail against the two-task configuration**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current \
  ./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests/test_matrix.py -q
```

Expected: failures report two tasks, 10 cells, and 30 full identities instead of the required three tasks, 15 cells, and 45 identities.

- [ ] **Step 3: Update the matrix implementation and YAML**

In `load_matrix`, require exactly 15 cells and 45 full runs:

```python
    if len(expand_cells(matrix)) != 15:
        raise ValueError("approved matrix must contain exactly 15 task/variant cells")
    if len(expand_full_runs(matrix)) != 45:
        raise ValueError("approved matrix must contain exactly 45 full runs")
```

Append to `matrix.yaml`:

```yaml
  - name: Isaac-Velocity-Flat-AnymalD
    variants: [kamino_current, kamino_pr_padmm, kamino_pr_dvi, mjwarp, physx]
```

Update the README opening paragraph to say Cartpole, Ant, and ANYmal-D.

- [ ] **Step 4: Run the complete harness suite**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current \
  ./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests -q
```

Expected: all harness/report tests pass and the matrix expands to 45 full identities.

- [ ] **Step 5: Run hooks and commit**

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current ./isaaclab.sh -f
git add benchmarks/kamino_dvi/matrix.py benchmarks/kamino_dvi/matrix.yaml \
  benchmarks/kamino_dvi/README.md benchmarks/kamino_dvi/tests/test_matrix.py
git commit -m "Extend Kamino benchmark to ANYmal D"
```

---

### Task 4: Reject non-finite learning data during report parsing

**Files:**
- Modify: `benchmarks/kamino_dvi/tests/test_parsing.py`
- Modify: `benchmarks/kamino_dvi/parsing.py`

**Interfaces:**
- Consumes: `_series(data: dict[str, Any], path: str, iterations: int) -> tuple[float, ...]`.
- Produces: the same return type for finite data; raises `MissingBenchmarkFieldError` when any series value is non-finite.

- [ ] **Step 1: Add a failing parser regression test**

Import `_series` and add:

```python
def test_series_rejects_non_finite_metric_values():
    """A completed bundle with NaN learning data must not enter aggregation."""
    data = {"learning": {"reward": {"series_per_iter": [1.0, float("nan"), 3.0]}}}

    with pytest.raises(MissingBenchmarkFieldError, match="non-finite"):
        _series(data, "learning.reward.series_per_iter", 3)
```

- [ ] **Step 2: Verify the regression test fails**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current \
  ./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests/test_parsing.py::test_series_rejects_non_finite_metric_values -q
```

Expected: FAIL because `_series` currently returns the NaN.

- [ ] **Step 3: Add the finite-value guard**

Replace `_series`'s return with:

```python
    values = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in values):
        raise MissingBenchmarkFieldError(f"{path} contains non-finite values")
    return values
```

- [ ] **Step 4: Run parsing and analysis tests**

Run:

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current \
  ./isaaclab.sh -p -m pytest \
  benchmarks/kamino_dvi/tests/test_parsing.py \
  benchmarks/kamino_dvi/tests/test_analysis.py \
  benchmarks/kamino_dvi/tests/test_analyze.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run hooks and commit**

```bash
/usr/bin/env VIRTUAL_ENV=/tmp/isaaclab-kamino-dvi-benchmark/.venv-current ./isaaclab.sh -f
git add benchmarks/kamino_dvi/parsing.py benchmarks/kamino_dvi/tests/test_parsing.py
git commit -m "Reject non-finite benchmark series"
```

---

### Task 5: Reconnect the locked environments and run staged preflights

**Files:**
- No tracked files.
- Create ignored symlinks: `.venv`, `.venv-current`, `.venv-pr3570`, `_isaac_sim`, and `logs` in `/tmp/isaaclab-kamino-dvi-analysis`.
- Write artifacts under `/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist`.

**Interfaces:**
- Consumes: exact Newton revisions `c7ae7c7648cd0717df39e5c94b95d5a02c997320` and `7906676b2e5061273db96af179d7081fc6cbbba0`.
- Produces: completed five-iteration preflight manifests for Cartpole and ANYmal-D at the selected common environment counts.

- [ ] **Step 1: Create only missing ignored links**

From `/tmp/isaaclab-kamino-dvi-analysis`, resolve each source first, then run:

```bash
ln -s /tmp/isaaclab-kamino-dvi-benchmark/.venv-current .venv
ln -s /tmp/isaaclab-kamino-dvi-benchmark/.venv-current .venv-current
ln -s /tmp/isaaclab-kamino-dvi-benchmark/.venv-pr3570 .venv-pr3570
ln -s /home/antoiner/Documents/IsaacSim/_build/linux-x86_64/release _isaac_sim
ln -s /tmp/isaaclab-kamino-dvi-benchmark/logs logs
```

Skip an individual `ln` when that exact target already exists. Confirm `git status --short` remains clean.

- [ ] **Step 2: Verify locked provenance and DVI availability**

Run:

```bash
.venv-current/bin/python -c "import importlib.metadata as m; print(m.distribution('newton').read_text('direct_url.json'))"
.venv-pr3570/bin/python -c "import importlib.metadata as m; print(m.distribution('newton').read_text('direct_url.json')); from newton._src.solvers.kamino.config import DVISolverConfig; print(DVISolverConfig)"
```

Expected: current reports `c7ae7c...`, PR reports `7906676...`, and `DVISolverConfig` imports.

- [ ] **Step 3: Preflight tuned DVI first**

Run each command separately:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.run --preflight-only \
  --task Isaac-Cartpole-Direct --variant kamino_pr_dvi \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist --resume

./isaaclab.sh -p -m benchmarks.kamino_dvi.run --preflight-only \
  --task Isaac-Velocity-Flat-AnymalD --variant kamino_pr_dvi \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist --resume
```

Expected: both five-iteration DVI identities complete at 4096 environments, or ANYmal-D produces an explicit capacity classification that selects the next common count.

- [ ] **Step 4: Preflight all five variants per task**

Run:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.run --preflight-only \
  --task Isaac-Cartpole-Direct \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist --resume

./isaaclab.sh -p -m benchmarks.kamino_dvi.run --preflight-only \
  --task Isaac-Velocity-Flat-AnymalD \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist --resume
```

Expected: five completed preflight manifests per task at one common task count. Stop and diagnose any non-capacity failure rather than lowering the count.

- [ ] **Step 5: Audit preflight states**

Run:

```bash
jq -r '[.identity.task, .identity.variant, .identity.num_envs, .state, .failure_category] | @tsv' \
  /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist/preflight__*/manifest.json
```

Expected: Cartpole and ANYmal-D each have five completed variants at a single environment count.

---

### Task 6: Run fresh Cartpole full benchmarks

**Files:**
- No tracked files.
- Produce 15 raw Cartpole run directories in the shared ignored artifact root.

**Interfaces:**
- Consumes: the preflight-selected Cartpole count and all five physics variants.
- Produces: seeds 42–44, 300 iterations, reward/episode-length/success/runtime traces for each variant.

- [ ] **Step 1: Launch the resumable Cartpole matrix**

Run:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.run --full-only \
  --task Isaac-Cartpole-Direct \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist --resume
```

Expected: 15 completed identities. Existing stale Cartpole data is not reused because no completed post-tuning identities exist in this artifact root.

- [ ] **Step 2: Validate every Cartpole bundle**

Run:

```bash
./isaaclab.sh -p -c "
import json, math
from pathlib import Path

root = Path('/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist')
runs = []
for manifest_path in root.glob('full__*/manifest.json'):
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get('identity', {})
    if identity.get('task') == 'Isaac-Cartpole-Direct' and identity.get('seed') in (42, 43, 44):
        runs.append((manifest_path, manifest))
assert len(runs) == 15, len(runs)
for manifest_path, manifest in runs:
    assert manifest['state'] == 'completed', manifest_path
    bundles = tuple(manifest_path.parent.glob('benchmark_training_*.json'))
    assert len(bundles) == 1, (manifest_path, len(bundles))
    bundle = json.loads(bundles[0].read_text())
    assert bundle['run']['status'] == 'completed'
    assert bundle['runtime']['iterations_completed'] == 300
    for field in ('reward', 'ep_length', 'success_rate'):
        values = bundle['learning'][field]['series_per_iter']
        assert len(values) == 300, (manifest_path, field, len(values))
        assert all(math.isfinite(value) for value in values), (manifest_path, field)
print('validated', len(runs), 'Cartpole runs')
"
```

Expected: all assertions pass. If a required field is absent, stop and report it as a schema bug.

---

### Task 7: Run fresh ANYmal-D full benchmarks

**Files:**
- No tracked files.
- Produce 15 raw ANYmal-D run directories in the shared ignored artifact root.

**Interfaces:**
- Consumes: the preflight-selected ANYmal-D count and all five physics variants.
- Produces: seeds 42–44, 300 iterations, reward/episode-length/success/runtime traces for each variant.

- [ ] **Step 1: Launch the resumable ANYmal-D matrix**

Run:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.run --full-only \
  --task Isaac-Velocity-Flat-AnymalD \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist --resume
```

Expected: 15 completed identities at the preflight-selected common count. A completed run with weak reward or success remains valid and is documented as a learning-quality result.

- [ ] **Step 2: Validate every ANYmal-D bundle**

Run:

```bash
./isaaclab.sh -p -c "
import json, math
from pathlib import Path

root = Path('/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist')
runs = []
for manifest_path in root.glob('full__*/manifest.json'):
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get('identity', {})
    if identity.get('task') == 'Isaac-Velocity-Flat-AnymalD' and identity.get('seed') in (42, 43, 44):
        runs.append((manifest_path, manifest))
assert len(runs) == 15, len(runs)
for manifest_path, manifest in runs:
    assert manifest['state'] == 'completed', manifest_path
    bundles = tuple(manifest_path.parent.glob('benchmark_training_*.json'))
    assert len(bundles) == 1, (manifest_path, len(bundles))
    bundle = json.loads(bundles[0].read_text())
    assert bundle['run']['status'] == 'completed'
    assert bundle['runtime']['iterations_completed'] == 300
    for field in ('reward', 'ep_length', 'success_rate'):
        values = bundle['learning'][field]['series_per_iter']
        assert len(values) == 300, (manifest_path, field, len(values))
        assert all(math.isfinite(value) for value in values), (manifest_path, field)
print('validated', len(runs), 'ANYmal-D runs')
"
```

Expected: all 15 bundles pass. Stop on a missing field; document capacity, numerical, and learning failures separately.

---

### Task 8: Regenerate and commit the combined report

**Files:**
- Modify: `benchmarks/kamino_dvi/results/summary.json`
- Modify: `benchmarks/kamino_dvi/results/runtime.png`
- Modify: `benchmarks/kamino_dvi/results/learning.png`
- Modify: `benchmarks/kamino_dvi/results/kamino_dvi_benchmark.md`
- Modify: `benchmarks/kamino_dvi/results/kamino_dvi_benchmark.pdf`

**Interfaces:**
- Consumes: 45 complete runs forming 15 task/variant groups across Cartpole, Ant, and ANYmal-D, plus matching TensorBoard events under the shared `logs` target.
- Produces: combined JSON, Markdown, PDF, runtime figure, and learning figure with per-task three-seed 95% confidence intervals.

- [ ] **Step 1: Generate the combined artifacts**

Run:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/finalist \
  --logs-root /tmp/isaaclab-kamino-dvi-benchmark/logs \
  --output-dir benchmarks/kamino_dvi/results
```

Expected: `summary.json` contains 15 task/variant summaries; the Markdown and PDF include three tasks and both figures.

- [ ] **Step 2: Verify report structure and findings**

Run:

```bash
jq -e 'length == 15 and ([.[].task] | unique | length == 3) and all(.[]; .iteration_time_s.n == 3)' \
  benchmarks/kamino_dvi/results/summary.json
pdfinfo benchmarks/kamino_dvi/results/kamino_dvi_benchmark.pdf
rg -n "Isaac-Cartpole-Direct|Isaac-Ant-Direct|Isaac-Velocity-Flat-AnymalD|runtime.png|learning.png" \
  benchmarks/kamino_dvi/results/kamino_dvi_benchmark.md
```

Expected: jq returns true, the PDF has three pages, and all tasks/figures appear in Markdown.

- [ ] **Step 3: Run final focused verification**

Run:

```bash
./isaaclab.sh -p -m pytest \
  benchmarks/kamino_dvi/tests \
  source/isaaclab_newton/test/physics/test_kamino_manager_cfg.py \
  source/isaaclab_tasks/test/core/test_ant_physics_presets.py \
  source/isaaclab_tasks/test/core/test_cartpole_physics_presets.py \
  source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py -q
./isaaclab.sh -f
```

Expected: all focused tests pass; every available hook passes. If `git-lfs` remains unavailable, it is the only hook failure and is reported explicitly.

- [ ] **Step 4: Commit the combined report**

```bash
git add benchmarks/kamino_dvi/results/summary.json \
  benchmarks/kamino_dvi/results/runtime.png \
  benchmarks/kamino_dvi/results/learning.png \
  benchmarks/kamino_dvi/results/kamino_dvi_benchmark.md \
  benchmarks/kamino_dvi/results/kamino_dvi_benchmark.pdf
git commit -m "Report Cartpole and ANYmal DVI benchmarks"
```
