# ANYmal-D DVI Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible task-specific search that selects the fastest stable Kamino DVI configuration for `Isaac-Velocity-Flat-AnymalD` without degrading three-seed reward or success beyond the current DVI baseline's 95% confidence bounds.

**Architecture:** Add an isolated tuning layer beside the existing `benchmarks.kamino_dvi` comparison harness. A declarative matrix defines the baseline and 18 single-field candidates; pure functions resolve the six adaptive combined candidates and all promotion decisions; a dedicated runner reuses the existing process, environment, parsing, hashing, and failure primitives without weakening the strict five-variant report grammar. GPU stages execute only from committed clean source states, and the canonical ANYmal-D preset changes only after a finalist passes the override-based validation.

**Tech Stack:** Python 3.12, dataclasses, PyYAML, RSL-RL, IsaacLab Hydra overrides, TensorBoard event parsing, NumPy/matplotlib/reportlab already present in the benchmark environment, pytest, pre-commit.

## Global Constraints

- Use RSL-RL with `Isaac-Velocity-Flat-AnymalD`, 4096 environments, one physics substep, and the existing task timestep.
- Stage 0 and Stage 3 use seeds 42–44 and 300 iterations; Stage 1 uses seed 42 and 40 iterations; Stage 2 uses seeds 42–43 and 100 iterations.
- Runtime means exclude iterations 1–10; learning means use the final 20 iterations.
- Never lower the environment count during tuning. A capacity failure at 4096 disqualifies the candidate.
- Never substitute schema success for TensorBoard `Metrics/success_rate`.
- Every measured run must record exact source/command/config hashes and the matched TensorBoard event path and SHA-256.
- Do not add a required or optional dependency.
- Keep the canonical ANYmal-D `newton_kamino_dvi` preset unchanged until a finalist passes Stage 3.
- Run `./isaaclab.sh -f` before every commit; the known local `git-lfs` executable absence may be documented only if every other hook passes.
- Use 2026 in every new SPDX copyright header.

---

### Task 1: Declarative Tuning Matrix and Candidate Hashes

**Files:**
- Create: `benchmarks/kamino_dvi/tuning.py`
- Create: `benchmarks/kamino_dvi/tuning_matrix.yaml`
- Create: `benchmarks/kamino_dvi/tests/test_tuning.py`

**Interfaces:**
- Consumes: `benchmarks.kamino_dvi.matrix.DEFAULT_MATRIX_PATH` and `load_matrix()` for locked experiment revisions.
- Produces: `TuningCandidate`, `TuningMatrix`, `load_tuning_matrix(path)`, `resolve_config(matrix, candidate)`, `config_hash(config)`, and `hydra_overrides(matrix, candidate)`.

- [ ] **Step 1: Write the failing matrix and hash tests**

```python
def test_tuning_matrix_declares_exact_anymal_protocol_and_wave1():
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    assert matrix.task == "Isaac-Velocity-Flat-AnymalD"
    assert matrix.num_envs == 4096
    assert matrix.seeds == (42, 43, 44)
    assert (matrix.preflight_iterations, matrix.screen_iterations) == (5, 40)
    assert (matrix.halve_iterations, matrix.final_iterations) == (100, 300)
    assert len(matrix.wave1) == 18
    assert all(len(candidate.overrides) == 1 for candidate in matrix.wave1)


def test_resolved_hash_is_order_independent_and_hydra_values_are_canonical():
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = matrix.candidate("dynamics_preconditioning_true")
    resolved = resolve_config(matrix, candidate)
    assert resolved["dynamics_preconditioning"] is True
    assert config_hash(dict(reversed(tuple(resolved.items())))) == config_hash(resolved)
    assert hydra_overrides(matrix, candidate) == (
        "env.sim.physics.solver_cfg.dynamics_preconditioning=true",
    )
```

- [ ] **Step 2: Run the tests and confirm the RED state**

Run:

```bash
./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests/test_tuning.py -q
```

Expected: collection fails because `benchmarks.kamino_dvi.tuning` does not exist.

- [ ] **Step 3: Add the exact declarative matrix**

Create `benchmarks/kamino_dvi/tuning_matrix.yaml` with this complete content:

```yaml
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

task: Isaac-Velocity-Flat-AnymalD
variant: kamino_pr_dvi
preset: newton_kamino_dvi
num_envs: 4096
seeds: [42, 43, 44]
iterations:
  preflight: 5
  screen: 40
  halve: 100
  final: 300
warmup_iterations: 10
learning_window: 20
baseline:
  integrator: moreau
  dynamics_linear_solver_max_iterations: 9
  dvi_block_iterations: 16
  dvi_contact_iterations: 2
  dvi_bilateral_solve_period: 2
  dvi_omega: 0.3
  dvi_contact_jacobi_omega: 0.45
  dvi_contact_jacobi_relaxation: 0.9
  dynamics_preconditioning: false
  dvi_contact_block_preconditioner: false
  dvi_warmstart_mode: containers
wave1:
  - {name: integrator_euler, overrides: {integrator: euler}}
  - {name: cr_iterations_3, overrides: {dynamics_linear_solver_max_iterations: 3}}
  - {name: cr_iterations_5, overrides: {dynamics_linear_solver_max_iterations: 5}}
  - {name: cr_iterations_7, overrides: {dynamics_linear_solver_max_iterations: 7}}
  - {name: block_iterations_4, overrides: {dvi_block_iterations: 4}}
  - {name: block_iterations_8, overrides: {dvi_block_iterations: 8}}
  - {name: block_iterations_12, overrides: {dvi_block_iterations: 12}}
  - {name: contact_iterations_1, overrides: {dvi_contact_iterations: 1}}
  - {name: bilateral_period_4, overrides: {dvi_bilateral_solve_period: 4}}
  - {name: dvi_omega_0_5, overrides: {dvi_omega: 0.5}}
  - {name: jacobi_omega_0_3, overrides: {dvi_contact_jacobi_omega: 0.3}}
  - {name: jacobi_omega_0_6, overrides: {dvi_contact_jacobi_omega: 0.6}}
  - {name: jacobi_relaxation_0_7, overrides: {dvi_contact_jacobi_relaxation: 0.7}}
  - {name: jacobi_relaxation_1_0, overrides: {dvi_contact_jacobi_relaxation: 1.0}}
  - {name: dynamics_preconditioning_true, overrides: {dynamics_preconditioning: true}}
  - {name: contact_block_preconditioner_true, overrides: {dvi_contact_block_preconditioner: true}}
  - {name: warmstart_internal, overrides: {dvi_warmstart_mode: internal}}
  - {name: warmstart_none, overrides: {dvi_warmstart_mode: none}}
```

- [ ] **Step 4: Implement immutable models, validation, and canonical hashing**

Create `tuning.py` with frozen dataclasses and these public signatures:

```python
SolverValue = str | int | float | bool
DEFAULT_TUNING_MATRIX_PATH = Path(__file__).with_name("tuning_matrix.yaml")
HYDRA_PREFIX = "env.sim.physics.solver_cfg."


@dataclass(frozen=True)
class TuningCandidate:
    name: str
    overrides: dict[str, SolverValue]


@dataclass(frozen=True)
class TuningMatrix:
    task: str
    variant: Variant
    preset: str
    num_envs: int
    seeds: tuple[int, ...]
    preflight_iterations: int
    screen_iterations: int
    halve_iterations: int
    final_iterations: int
    warmup_iterations: int
    learning_window: int
    baseline: dict[str, SolverValue]
    wave1: tuple[TuningCandidate, ...]

    def candidate(self, name: str) -> TuningCandidate:
        for candidate in self.wave1:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


def config_hash(config: Mapping[str, SolverValue]) -> str:
    payload = json.dumps(dict(sorted(config.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_config(matrix: TuningMatrix, candidate: TuningCandidate) -> dict[str, SolverValue]:
    unknown = candidate.overrides.keys() - matrix.baseline.keys()
    if unknown:
        raise ValueError(f"candidate {candidate.name} has unknown fields: {sorted(unknown)}")
    return {**matrix.baseline, **candidate.overrides}


def _hydra_value(value: SolverValue) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def hydra_overrides(matrix: TuningMatrix, candidate: TuningCandidate) -> tuple[str, ...]:
    resolve_config(matrix, candidate)
    return tuple(
        f"{HYDRA_PREFIX}{name}={_hydra_value(value)}"
        for name, value in sorted(candidate.overrides.items())
    )
```

`load_tuning_matrix()` must reject duplicate names, non-ANYmal tasks, counts other than 4096, seeds other than `(42, 43, 44)`, a Wave 1 length other than 18, multi-field Wave 1 entries, missing baseline keys, and non-positive iteration/window values.

- [ ] **Step 5: Run focused tests and the existing matrix tests**

```bash
./isaaclab.sh -p -m pytest \
  benchmarks/kamino_dvi/tests/test_tuning.py \
  benchmarks/kamino_dvi/tests/test_matrix.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run hooks and commit**

```bash
./isaaclab.sh -f
git add benchmarks/kamino_dvi/tuning.py benchmarks/kamino_dvi/tuning_matrix.yaml \
  benchmarks/kamino_dvi/tests/test_tuning.py
git commit -m "Define ANYmal DVI tuning matrix"
```

### Task 2: Adaptive Wave 2 and Promotion Rules

**Files:**
- Modify: `benchmarks/kamino_dvi/tuning.py`
- Modify: `benchmarks/kamino_dvi/tests/test_tuning.py`

**Interfaces:**
- Consumes: `TuningMatrix`, `TuningCandidate`, `resolve_config()`, and `config_hash()` from Task 1.
- Produces: `TuningRunMetrics`, `PromotionDecision`, `resolve_wave2()`, `promote_stage2()`, `promote_finalists()`, `qualify_finalists()`, and `select_winner()`.

- [ ] **Step 1: Add failing Wave 2 tests**

```python
def make_screen_metric(candidate: str, index: int, *, failure: str | None = None) -> TuningRunMetrics:
    steady_time = 0.10 + index / 100
    return TuningRunMetrics(
        candidate=candidate,
        stage="wave1",
        seed=42,
        num_envs=4096,
        iteration_time_s=(1.0,) * 10 + (steady_time,) * 30,
        reward=(20.0,) * 40,
        success_rate=(1.0,) * 40,
        ep_length=(980.0,) * 40,
        failure=failure,
    )


def test_wave2_uses_cumulative_prefixes_of_fastest_compatible_changes(matrix):
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]
    wave2 = resolve_wave2(matrix, results)
    assert [candidate.name for candidate in wave2] == [
        "combined_top_02", "combined_top_03", "combined_top_04",
        "combined_top_05", "combined_top_06", "combined_top_07",
    ]
    assert len(wave2[0].overrides) == 2
    assert len(wave2[-1].overrides) == 7


def test_wave2_excludes_failed_candidates_but_requires_every_terminal_record(matrix):
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]
    results[-1] = dataclasses.replace(results[-1], failure="numerical")
    wave2 = resolve_wave2(matrix, results)
    assert all(candidate.overrides.get("dvi_warmstart_mode") != "none" for candidate in wave2)

    with pytest.raises(ValueError, match="18 terminal"):
        resolve_wave2(matrix, results[:-1])


def test_wave2_requires_seven_valid_changes(matrix):
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]
    results = [dataclasses.replace(result, failure="numerical") for result in results[:12]] + results[12:]
    with pytest.raises(ValueError, match="at least seven valid"):
        resolve_wave2(matrix, results)
```

- [ ] **Step 2: Run RED tests**

```bash
./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests/test_tuning.py -q
```

Expected: failures because the metrics and promotion interfaces do not exist.

- [ ] **Step 3: Add immutable metrics and decision records**

```python
@dataclass(frozen=True)
class TuningRunMetrics:
    candidate: str
    stage: str
    seed: int
    num_envs: int
    iteration_time_s: tuple[float, ...]
    reward: tuple[float, ...]
    success_rate: tuple[float, ...]
    ep_length: tuple[float, ...]
    failure: str | None = None

    def steady_time(self, warmup: int) -> float:
        if self.failure is not None or len(self.iteration_time_s) <= warmup:
            raise ValueError(f"{self.candidate} is not a valid runtime record")
        return statistics.mean(self.iteration_time_s[warmup:])

    def final_mean(self, values: tuple[float, ...], window: int) -> float:
        if len(values) < window:
            raise ValueError(f"{self.candidate} has fewer than {window} learning points")
        return statistics.mean(values[-window:])


@dataclass(frozen=True)
class PromotionDecision:
    source_stage: str
    selected: tuple[str, ...]
    rejected: dict[str, str]
    resolved_candidates: tuple[TuningCandidate, ...] = ()


@dataclass(frozen=True)
class FinalQualification:
    candidate: str
    qualified: bool
    reason: str | None
    runtime: Estimate
    reward: Estimate
    success_rate: Estimate
    ep_length: Estimate
```

- [ ] **Step 4: Implement deterministic Wave 2 resolution**

`resolve_wave2()` must require one terminal seed-42 result for each of the 18 Wave 1 candidates, exclude failed candidates, require at least seven valid one-field changes, and sort valid changes by `(steady_time, candidate_name)`. Because multiple Wave 1 candidates may change the same field, scan that ordering greedily and retain only the fastest candidate for each previously unused field. Require at least seven retained distinct fields, then create cumulative prefixes of retained changes two through seven. Persisted decision JSON contains each resolved override map and `config_hash(resolve_config(...))`.

```python
def resolve_wave2(matrix: TuningMatrix, results: Sequence[TuningRunMetrics]) -> tuple[TuningCandidate, ...]:
    terminal = {result.candidate: result for result in results if result.seed == 42}
    expected = {candidate.name for candidate in matrix.wave1}
    if len(terminal) != len(results) or set(terminal) != expected:
        raise ValueError("Wave 1 requires exactly 18 terminal candidate results")
    by_name = {name: result for name, result in terminal.items() if result.failure is None}
    if len(by_name) < 7:
        raise ValueError("Wave 2 requires at least seven valid one-field changes")
    ordered = sorted(
        (candidate for candidate in matrix.wave1 if candidate.name in by_name),
        key=lambda candidate: (by_name[candidate.name].steady_time(matrix.warmup_iterations), candidate.name),
    )
    compatible: list[TuningCandidate] = []
    selected_fields: set[str] = set()
    for candidate in ordered:
        field = next(iter(candidate.overrides))
        if field not in selected_fields:
            compatible.append(candidate)
            selected_fields.add(field)
    if len(compatible) < 7:
        raise ValueError("Wave 2 requires seven distinct valid one-field changes")
    resolved: list[TuningCandidate] = []
    for count in range(2, 8):
        overrides: dict[str, SolverValue] = {}
        for candidate in compatible[:count]:
            overrides.update(candidate.overrides)
        resolved.append(TuningCandidate(f"combined_top_{count:02d}", overrides))
    return tuple(resolved)
```

- [ ] **Step 5: Add failing Stage 2 and Stage 3 gate tests**

Cover these exact boundaries:

```python
def test_stage2_rejects_reward_below_eighty_percent(matrix, baseline_100):
    candidate = make_two_seed_metrics(reward_scale=0.799, success_delta=0.0, ep_length_scale=1.0)
    decision = promote_finalists(matrix, baseline_100, candidate)
    assert decision.selected == ()
    assert "reward" in decision.rejected[candidate[0].candidate]


def test_stage2_accepts_exact_learning_boundaries(matrix, baseline_100):
    candidate = make_two_seed_metrics(reward_scale=0.8, success_delta=-0.10, ep_length_scale=1.20)
    assert promote_finalists(matrix, baseline_100, candidate).selected == (candidate[0].candidate,)


def test_final_gate_uses_baseline_ci_lower_bounds(matrix, baseline_final, candidate_final):
    result = qualify_finalists(matrix, baseline_final, candidate_final)
    assert result[candidate_final[0].candidate].qualified is True
```

- [ ] **Step 6: Implement promotion and final selection exactly**

`promote_stage2()` selects the eight fastest successful Stage 1 candidates, or every valid candidate when fewer than eight remain. `promote_finalists()` requires seeds 42 and 43 and rejects a candidate if either seed violates reward `< 0.8 * baseline`, success `< baseline - 0.10`, or absolute episode-length ratio outside `[0.8, 1.2]`; it then selects the three fastest. Implement `qualify_finalists(matrix, baseline_results, candidate_results) -> dict[str, FinalQualification]`, requiring seeds 42–44 and comparing candidate means with `baseline_estimate.mean - baseline_estimate.half_width` from `mean_ci95()`. Implement `select_winner(matrix, qualifications, resolved_configs) -> str`; it orders qualified candidates by runtime, then applies the approved overlap tie-break from the spec.

- [ ] **Step 7: Run tests and commit**

```bash
./isaaclab.sh -p -m pytest \
  benchmarks/kamino_dvi/tests/test_tuning.py \
  benchmarks/kamino_dvi/tests/test_statistics.py -q
./isaaclab.sh -f
git add benchmarks/kamino_dvi/tuning.py benchmarks/kamino_dvi/tests/test_tuning.py
git commit -m "Select DVI tuning candidates"
```

### Task 3: Provenance-Safe Tuning Runner

**Files:**
- Create: `benchmarks/kamino_dvi/tune.py`
- Create: `benchmarks/kamino_dvi/tests/test_tune.py`
- Modify: `benchmarks/kamino_dvi/manifests.py`
- Modify: `benchmarks/kamino_dvi/tests/test_manifests.py`

**Interfaces:**
- Consumes: Task 1 candidates; `build_training_command()`, `execute_command()`, `inspect_bundle()`, `probe_environment()`, `validate_environment()`, `locate_rsl_rl_events()`, `sha256_file()`, `classify_failure()`, and the locked comparison matrix.
- Produces: `TuningIdentity`, `TuningManifest`, `build_tuning_command()`, `execute_tuning_identity()`, `select_tuning_identities()`, and CLI module `python -m benchmarks.kamino_dvi.tune`.

- [ ] **Step 1: Extract a tested generic atomic JSON writer**

Add this failing test to `test_manifests.py`:

```python
def test_write_json_atomic_replaces_complete_document(tmp_path):
    path = tmp_path / "decision.json"
    write_json_atomic(path, {"state": "old"})
    write_json_atomic(path, {"state": "new", "values": [1, 2]})
    assert json.loads(path.read_text()) == {"state": "new", "values": [1, 2]}
    assert not tuple(tmp_path.glob("*.tmp"))
```

Move the temporary-file/fsync/replace body from `write_manifest()` into `write_json_atomic(path: Path, data: Mapping[str, Any])`; make `write_manifest()` call it. Run the test before and after the refactor.

- [ ] **Step 2: Add failing command and identity tests**

```python
def test_tuning_command_appends_only_declared_candidate_overrides(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    candidate = tuning.candidate("cr_iterations_3")
    identity = TuningIdentity("wave1", candidate.name, 42, 4096, 40, 0)
    command = build_tuning_command(locked, tuning, candidate, identity, tmp_path, tmp_path / "run")
    assert command[-1] == "env.sim.physics.solver_cfg.dynamics_linear_solver_max_iterations=3"
    assert command.count("presets=newton_kamino_dvi") == 1
    assert not any("dynamics_solver=dvi" in value for value in command)


def test_resume_requires_exact_head_command_config_and_event_hash(tmp_path):
    manifest = completed_tuning_manifest(tmp_path)
    assert tuning_resume_matches(manifest, manifest.identity, manifest.command, manifest.config_hash, "f" * 40)
    assert not tuning_resume_matches(manifest, manifest.identity, manifest.command, "0" * 64, "f" * 40)
    assert not tuning_resume_matches(manifest, manifest.identity, manifest.command, manifest.config_hash, "0" * 40)


def test_canonical_command_uses_winner_identity_without_hydra_overrides(tmp_path):
    tuning = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    locked = load_matrix(DEFAULT_MATRIX_PATH)
    winner = write_winner_decision(tmp_path, tuning.candidate("cr_iterations_3"))
    identity, candidate, command = build_canonical_command(locked, tuning, winner, tmp_path)
    assert identity.candidate == "canonical_winner"
    assert candidate.config_hash == winner["config_hash"]
    assert not any(value.startswith("env.sim.physics.solver_cfg.") for value in command)
```

- [ ] **Step 3: Implement tuning identities and manifests**

Use these fields:

```python
@dataclass(frozen=True)
class TuningIdentity:
    stage: str
    candidate: str
    seed: int
    num_envs: int
    max_iterations: int
    attempt: int = 0


@dataclass(frozen=True)
class TuningManifest:
    run_id: str
    identity: TuningIdentity
    config_hash: str
    resolved_config: dict[str, SolverValue]
    command: tuple[str, ...]
    command_hash: str
    revisions: Revisions
    schema_version: str
    isaaclab_head: str
    artifact_root: str
    tensorboard_event_path: str | None
    tensorboard_event_hash: str | None
    artifact_hashes: dict[str, str]
    state: TerminalState
    failure_category: FailureCategory | None
    retry: RetryLineage
```

The stable ID is `{stage}__{candidate}__seed{seed}__env{num_envs}__iter{max_iterations}__attempt{attempt}`. On resume, skip only an exact completed manifest. If the latest exact attempt is not complete, create attempt `N+1` with `RetryLineage(attempt=N+1, parent_run_id=previous.run_id)` so raw evidence is never overwritten.

- [ ] **Step 4: Build commands from the existing PR3570 DVI variant**

Create the base `RunIdentity` with `TaskName.ANYMAL_D`, `Variant.KAMINO_PR_DVI`, the tuning seed/count/iterations, and the appropriate preflight/full phase. Call `build_training_command()`, then append exactly `hydra_overrides(tuning_matrix, candidate)`. The tuning command validator must reconstruct this full tuple and reject every extra, missing, reordered, or changed override.

- [ ] **Step 5: Implement execution and mandatory event integrity**

`execute_tuning_identity()` must:

1. write planned and running manifests atomically;
2. call `execute_command()` with the matrix timeout;
3. require `inspect_bundle(...).complete`;
4. locate the exact RSL-RL event under `repo_root / "logs"`;
5. require TensorBoard success through `parse_training_trace()`;
6. hash stdout, stderr, bundle, and event;
7. mark a missing event or metric as `FailureCategory.ARTIFACT`; and
8. write a completed or failed terminal manifest.

- [ ] **Step 6: Implement the stage CLI and exact schedule tests**

The CLI accepts:

```text
--stage baseline|wave1|wave2|halve|final|canonical
--candidate NAME
--seed INT
--artifact-root PATH
--decision-root PATH
--preflight-only
--measured-only
--resume
--dry-run
```

Schedules are:

- `baseline`: baseline candidate, seeds 42–44, 300 iterations;
- `wave1`: 18 candidates, seed 42, 40 iterations;
- `wave2`: six candidates loaded from `wave2.json`, seed 42, 40 iterations;
- `halve`: eight candidates loaded from `stage2.json`, seeds 42–43, 100 iterations;
- `final`: three candidates loaded from `finalists.json`, seeds 42–44, 300 iterations;
- `canonical`: baseline command with no tuning overrides, seeds 42–44, 300 iterations after the preset update.

The canonical stage loads `winner.json`, records the winner's literal resolved config and config hash under candidate name
`canonical_winner`, but constructs the command without Hydra tuning overrides so it tests the committed preset itself.
Every candidate gets one seed-42 five-iteration preflight before its first measured stage. `--measured-only` must refuse
to run when the exact preflight manifest is absent or failed for that candidate/config hash/source HEAD.

- [ ] **Step 7: Run runner tests and all existing harness tests**

```bash
./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests/test_tune.py -q
./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests -q
```

Expected: all tests pass without launching Isaac Sim.

- [ ] **Step 8: Run hooks and commit**

```bash
./isaaclab.sh -f
git add benchmarks/kamino_dvi/tune.py benchmarks/kamino_dvi/manifests.py \
  benchmarks/kamino_dvi/tests/test_tune.py benchmarks/kamino_dvi/tests/test_manifests.py
git commit -m "Run resumable DVI tuning stages"
```

### Task 4: Tuning Analysis, Decisions, and Addendum Report

**Files:**
- Create: `benchmarks/kamino_dvi/analyze_tuning.py`
- Create: `benchmarks/kamino_dvi/tuning_reporting.py`
- Create: `benchmarks/kamino_dvi/tests/test_analyze_tuning.py`
- Create: `benchmarks/kamino_dvi/tests/test_tuning_reporting.py`
- Modify: `benchmarks/kamino_dvi/README.md`

**Interfaces:**
- Consumes: tuning manifests/traces from Task 3 and selection functions from Task 2.
- Produces: `load_tuning_records()`, `write_decision()`, actions `resolve-wave2`, `promote-stage2`, `promote-finalists`, `select-winner`, and `report`.

- [ ] **Step 1: Write failing strict-loading tests**

```python
def test_loader_rejects_config_command_and_event_hash_mismatch(tmp_path):
    root, logs = write_completed_tuning_artifact(tmp_path)
    mutate_manifest(root, config_hash="0" * 64)
    with pytest.raises(ValueError, match="config hash"):
        load_tuning_records(root, logs, expected_stage="wave1")


def test_loader_rejects_incomplete_expected_candidate_seed_set(tmp_path):
    records = [make_record("integrator_euler", 42)]
    with pytest.raises(ValueError, match="missing tuning record"):
        validate_tuning_records(records, expected_wave1_identities())
```

- [ ] **Step 2: Implement strict trace loading**

For every completed manifest, validate typed identity fields, resolved config hash, locked revisions, exact source HEAD, command hash and full command reconstruction, bundle hashes, bundle task/seed/count/iterations, event path/hash, and finite aligned reward/success/episode/runtime series. Failed manifests are retained as rejection records. An undeclared directory or duplicate identity raises before aggregation.

- [ ] **Step 3: Add decision-action tests**

```python
def test_resolve_wave2_writes_canonical_decision(tmp_path, wave1_records):
    output = tmp_path / "wave2.json"
    main(["resolve-wave2", "--artifact-root", str(tmp_path), "--output", str(output)])
    data = json.loads(output.read_text())
    assert data["source_stage"] == "wave1"
    assert len(data["resolved_candidates"]) == 6
    assert all(len(item["config_hash"]) == 64 for item in data["resolved_candidates"])
```

Each decision JSON includes the source artifact root, source manifest hashes, selected/rejected candidates and reasons, resolved candidate configs/hashes, timestamp in UTC, and exact IsaacLab/Newton revisions. `write_json_atomic()` writes it.

- [ ] **Step 4: Implement report generation**

The Markdown/PDF addendum contains:

- a stage funnel table with attempted, valid, rejected, and promoted counts;
- a Wave 1/2 runtime ranking plot;
- Stage 2 reward/success/episode guardrail plots;
- final runtime and learning tables with 95% CIs;
- winner configuration and speedups against clean DVI, existing MJWarp, and existing PhysX;
- every failure/rejection reason; and
- provenance and environment-count disclosures.

Use existing matplotlib/reportlab dependencies and `mean_ci95()`. Write under `benchmarks/kamino_dvi/results/anymal_d_tuning/` only after Stage 3.

- [ ] **Step 5: Document exact commands**

Add an `ANYmal-D task-specific tuning` section to `README.md` listing the stage commands from Tasks 5–8 and explaining that decisions must exist before later stages.

- [ ] **Step 6: Run tests and commit**

```bash
./isaaclab.sh -p -m pytest \
  benchmarks/kamino_dvi/tests/test_analyze_tuning.py \
  benchmarks/kamino_dvi/tests/test_tuning_reporting.py -q
./isaaclab.sh -p -m pytest benchmarks/kamino_dvi/tests -q
./isaaclab.sh -f
git add benchmarks/kamino_dvi/analyze_tuning.py benchmarks/kamino_dvi/tuning_reporting.py \
  benchmarks/kamino_dvi/tests/test_analyze_tuning.py \
  benchmarks/kamino_dvi/tests/test_tuning_reporting.py benchmarks/kamino_dvi/README.md
git commit -m "Analyze staged DVI tuning"
```

### Task 5: Clean Baseline and Wave 1 Execution

**Files:**
- Runtime artifacts: `/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/`
- Decisions: `/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/`

**Interfaces:**
- Consumes: committed Tasks 1–4 and locked `.venv-pr3570`.
- Produces: three clean baseline traces and 18 Wave 1 traces at 4096 environments.

- [ ] **Step 1: Verify the clean immutable launch state**

```bash
git status --short
git rev-parse HEAD
.venv-pr3570/bin/python -c "import newton; print(newton.__version__)"
```

Expected: no tracked changes; HEAD is recorded in the campaign notes; Newton resolves to PR3570 commit `7906676b2e5061273db96af179d7081fc6cbbba0` through the environment probe.

- [ ] **Step 2: Dry-run baseline and Wave 1 schedules**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune --stage baseline --dry-run
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune --stage wave1 --dry-run
```

Expected: baseline prints 1 seed-42 preflight plus 3 measured commands; Wave 1 prints 18 preflights plus 18 measured commands, all at 4096 environments.

- [ ] **Step 3: Run baseline preflights and measured runs**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage baseline --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning --resume
```

Expected: four completed manifests; all three 300-iteration traces have finite reward, success, episode length, and TensorBoard hashes.

- [ ] **Step 4: Run Wave 1 sequentially**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage wave1 --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning --resume
```

Expected: every candidate has a terminal preflight and measured manifest. Failures are retained and not retried automatically.

- [ ] **Step 5: Audit baseline and Wave 1**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning validate \
  --stages baseline wave1 \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning
```

Expected: baseline is complete; Wave 1 reports exactly 18 terminal candidates and distinguishes valid from rejected.

### Task 6: Resolve Wave 2 and Run Successive Halving

**Files:**
- Create at runtime: `/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/wave2.json`
- Create at runtime: `/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/stage2.json`

**Interfaces:**
- Consumes: Stage 0/Wave 1 artifacts.
- Produces: six resolved combined candidates, their seed-42 screen traces, and eight promoted candidates with seed-42/43 100-iteration traces.

- [ ] **Step 1: Resolve and inspect Wave 2**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning resolve-wave2 \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --output /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/wave2.json
jq '.resolved_candidates[] | {name, overrides, config_hash}' \
  /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/wave2.json
```

Expected: six candidates named `combined_top_02` through `combined_top_07`, with two through seven overrides and unique hashes.

- [ ] **Step 2: Run Wave 2**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage wave2 \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions \
  --resume
```

Expected: six terminal preflights and six terminal 40-iteration runs.

- [ ] **Step 3: Select the eight fastest valid Stage 1 candidates**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning promote-stage2 \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --output /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/stage2.json
jq '{selected, rejected}' \
  /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/stage2.json
```

Expected: eight selected names unless fewer than eight candidates are valid; every omission has a recorded reason.

- [ ] **Step 4: Run Stage 2**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage halve \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions \
  --resume
```

Expected: each selected candidate has seed-42 and seed-43 100-iteration terminal traces after successful preflights.

### Task 7: Finalist Validation and Canonical Preset Update

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/flat_env_cfg.py`
- Modify: `source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py`
- Runtime decision: `/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/finalists.json`
- Runtime decision: `/tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/winner.json`

**Interfaces:**
- Consumes: baseline and Stage 2 traces.
- Produces: three finalist configurations, a statistically qualified winner, and literal task-preset values.

- [ ] **Step 1: Promote finalists with learning guardrails**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning promote-finalists \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --output /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/finalists.json
jq '{selected, rejected}' \
  /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/finalists.json
```

Expected: up to three fastest candidates satisfying both seeds' 80% reward, -0.10 success, and ±20% episode-length bounds.

- [ ] **Step 2: Run three-seed finalist validation**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage final \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions \
  --resume
```

Expected: each finalist has seeds 42–44, 300 iterations, 4096 environments, and complete finite metrics.

- [ ] **Step 3: Select and inspect the statistically qualified winner**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning select-winner \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --output /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/winner.json
jq '{winner, resolved_config, runtime, reward, success_rate, ep_length}' \
  /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/winner.json
```

Expected: one winner, or a clear no-qualifier failure that stops this task without modifying the preset.

- [ ] **Step 4: Write the failing preset regression from `winner.json`**

Read the literal `resolved_config` values and add assertions for every tunable field to `test_anymal_d_physics_presets.py`. Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py::test_anymal_d_flat_exposes_sparse_kamino_dvi_preset -q
```

Expected: FAIL on at least one field that differs from the existing preset. If no field differs, record the existing preset as the winner and skip the source edit.

- [ ] **Step 5: Copy the winner as literal preset values**

Set each field in `PhysicsCfg.newton_kamino_dvi.solver_cfg` to the literal value printed by:

```bash
jq -r '.resolved_config | to_entries[] | "\(.key)=\(.value)"' \
  /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions/winner.json
```

Do not load the tuning artifact at runtime. Preserve `max_contacts_per_world=64`, `num_substeps=1`, `debug_mode=False`, sparse Jacobian/dynamics, DVI selection, and every non-tuned task setting.

- [ ] **Step 6: Run focused tests, hooks, and commit the preset**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py \
  source/isaaclab_newton/test/physics/test_kamino_manager_cfg.py -q
./isaaclab.sh -p tools/changelog/cli.py check develop
./isaaclab.sh -f
git add source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/flat_env_cfg.py \
  source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py
git commit -m "Tune ANYmal DVI solver"
```

### Task 8: Canonical Validation, Addendum, and Final Verification

**Files:**
- Create: `benchmarks/kamino_dvi/results/anymal_d_tuning/summary.json`
- Create: `benchmarks/kamino_dvi/results/anymal_d_tuning/runtime.png`
- Create: `benchmarks/kamino_dvi/results/anymal_d_tuning/learning.png`
- Create: `benchmarks/kamino_dvi/results/anymal_d_tuning/anymal_d_dvi_tuning.md`
- Create: `benchmarks/kamino_dvi/results/anymal_d_tuning/anymal_d_dvi_tuning.pdf`

**Interfaces:**
- Consumes: committed winner preset and all tuning artifacts/decisions.
- Produces: clean canonical three-seed evidence and the final human-readable tuning addendum.

- [ ] **Step 1: Confirm the preset commit is clean**

```bash
git status --short
git rev-parse HEAD
```

Expected: no tracked changes. Record HEAD; the canonical manifests must contain it.

- [ ] **Step 2: Run canonical preset validation**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage canonical \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning/decisions \
  --resume
```

Expected: seeds 42–44 complete at 4096 environments and match the winner's Hydra-override results within their 95% runtime/reward/success intervals. If not, stop and diagnose before reporting.

- [ ] **Step 3: Generate the addendum**

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning report \
  --artifact-root /tmp/isaaclab-kamino-dvi-benchmark/benchmark_artifacts/kamino_dvi/anymal_tuning \
  --comparison-summary benchmarks/kamino_dvi/results/summary.json \
  --output-dir benchmarks/kamino_dvi/results/anymal_d_tuning
```

Expected: compact JSON, two readable PNG figures, Markdown, and a PDF with no clipped tables or missing metrics.

- [ ] **Step 4: Verify report structure and inspect the PDF**

```bash
jq 'length > 0' benchmarks/kamino_dvi/results/anymal_d_tuning/summary.json
pdfinfo benchmarks/kamino_dvi/results/anymal_d_tuning/anymal_d_dvi_tuning.pdf | rg '^(Pages|Page size)'
pdftotext benchmarks/kamino_dvi/results/anymal_d_tuning/anymal_d_dvi_tuning.pdf - | \
  rg '4096|Reward|Success rate|Episode length|MJWarp|PhysX'
```

Expected: JSON returns `true`; PDF metadata is valid; all required comparison labels appear in extracted text. Render each PDF page to PNG and visually check for overlap/clipping.

- [ ] **Step 5: Run final regression suites**

```bash
/usr/bin/env VIRTUAL_ENV=.venv-pr3570 \
  PYTHONPATH=/tmp/isaaclab-kamino-dvi-analysis/source/isaaclab_newton:/tmp/isaaclab-kamino-dvi-analysis/source/isaaclab_tasks \
  ./isaaclab.sh -p -m pytest \
  benchmarks/kamino_dvi/tests \
  source/isaaclab_newton/test/physics/test_kamino_manager_cfg.py \
  source/isaaclab_tasks/test/core/test_anymal_d_physics_presets.py -q
./isaaclab.sh -p tools/changelog/cli.py check develop
./isaaclab.sh -f
git diff --check
```

Expected: all focused tests pass; all hooks except any explicitly documented unavailable local `git-lfs` check pass; changelog and diff checks pass.

- [ ] **Step 6: Commit final artifacts**

```bash
git add benchmarks/kamino_dvi/results/anymal_d_tuning
git commit -m "Report ANYmal DVI tuning results"
```

- [ ] **Step 7: Request final code and report review**

Ask a fresh reviewer to inspect the full tuning diff, decision JSON, exact 4096/seed/iteration coverage, reward/success qualification, failure disclosure, figures, and PDF. Fix every Critical or Important finding in new commits, rerun the affected tests, and regenerate the report before handoff.
