# Training Success-Rate Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve per-iteration training success rates in Isaac Lab benchmark schema v1.1 while retaining the existing summary and play APIs.

**Architecture:** Extend the additive `Learning` schema with an optional `LearningCurve`, construct it through the existing learning builder, and pass each training adapter's already-collected `SuccessRateTracker.history` into that builder. Keep the top-level training tail mean and scalar play success rate unchanged for compatibility.

**Tech Stack:** Python 3.12 dataclasses, Isaac Lab benchmark schema/builders, RSL-RL/RL-Games/SB3/skrl adapters, pytest, pre-commit.

## Global Constraints

- Base all work on Isaac Lab `origin/develop` commit `79accca281128660a786abb599f40bd335963963` plus the committed design specification.
- Bump the benchmark schema from `"1.0"` to `"1.1"`; this is an additive minor revision.
- Do not remove, rename, or change the meaning of `TrainingBundle.success_rate` or `PlayBundle.success_rate`.
- Store no invented success values: absent or empty success history produces `Learning.success_rate is None`.
- `--no_series` removes the stored success history but retains its final raw and EMA values.
- Use the 2026 SPDX header for new source files and do not change years in existing headers; the changelog fragment follows the required header-first RST entry format.
- Use `./isaaclab.sh -p -m pytest` for tests and `./isaaclab.sh -f` before every commit and push.
- Add one `source/isaaclab/changelog.d/*.minor.rst` fragment; do not edit compiled changelog or version files.

---

## File Structure

- `source/isaaclab/isaaclab/test/benchmark/schema.py`: declare schema v1.1 and the optional training success curve.
- `source/isaaclab/isaaclab/test/benchmark/builders.py`: construct reward, episode-length, and optional success curves consistently.
- `scripts/benchmarks/{rsl_rl,rl_games,sb3,skrl}/benchmark_*_train.py`: feed each adapter's tracker history into the builder.
- `source/isaaclab/test/benchmark/test_schema.py`: verify v1.1 JSON serialization and compatibility behavior.
- `source/isaaclab/test/benchmark/test_builders.py`: verify populated, empty, absent, and omitted success histories.
- `scripts/benchmarks/test/test_benchmark_smoke.py`: verify end-to-end schema output and leave play semantics unchanged.
- `source/isaaclab/changelog.d/antoiner-benchmark-success-series.minor.rst`: user-facing schema addition.

### Task 1: Add the v1.1 Success Curve Schema and Builder

**Files:**

- Modify: `source/isaaclab/isaaclab/test/benchmark/schema.py`
- Modify: `source/isaaclab/isaaclab/test/benchmark/builders.py`
- Modify: `source/isaaclab/test/benchmark/test_schema.py`
- Modify: `source/isaaclab/test/benchmark/test_builders.py`

**Interfaces:**

- Produces: `Learning.success_rate: LearningCurve | None = None`.
- Produces: `build_learning(*, reward_series: Sequence[float], ep_length_series: Sequence[float], ema_alpha: float, success_rate_series: Sequence[float] | None = None, keep_series: bool = True) -> Learning`.
- Preserves: `TrainingBundle.success_rate: float | None` and `PlayBundle.success_rate: float | None`.

- [ ] **Step 1: Write failing schema and builder tests**

Update `_minimal_training_bundle()` in `test_schema.py` to include:

```python
success_rate=LearningCurve(final_raw=0.95, final_ema=0.91, series_per_iter=[0.1, 0.5, 0.95]),
```

inside `Learning(...)`, then add these assertions to `test_training_bundle_round_trip()`:

```python
assert SCHEMA_VERSION == "1.1"
assert data["learning"]["success_rate"]["final_raw"] == pytest.approx(0.95)
assert data["learning"]["success_rate"]["series_per_iter"] == pytest.approx([0.1, 0.5, 0.95])
assert data["success_rate"] == pytest.approx(0.91)
```

Extend `test_training_bundle_without_series()` so the replacement `Learning(...)` includes:

```python
success_rate=LearningCurve(final_raw=0.8, final_ema=0.7, series_per_iter=None),
```

and assert:

```python
assert data["learning"]["success_rate"]["series_per_iter"] is None
assert data["learning"]["success_rate"]["final_raw"] == pytest.approx(0.8)
```

Add to `test_builders.py`:

```python
def test_build_learning_includes_success_rate_curve():
    learning = builders.build_learning(
        reward_series=[1.0, 2.0],
        ep_length_series=[10.0, 20.0],
        success_rate_series=[0.1, 0.5, 0.9],
        ema_alpha=0.5,
    )

    assert learning.success_rate is not None
    assert learning.success_rate.final_raw == pytest.approx(0.9)
    assert learning.success_rate.final_ema == pytest.approx(0.6)
    assert learning.success_rate.series_per_iter == pytest.approx([0.1, 0.5, 0.9])


@pytest.mark.parametrize("success_rate_series", [None, []])
def test_build_learning_omits_absent_success_rate_curve(success_rate_series):
    learning = builders.build_learning(
        reward_series=[1.0],
        ep_length_series=[10.0],
        success_rate_series=success_rate_series,
        ema_alpha=0.1,
    )

    assert learning.success_rate is None
```

Extend `test_build_learning_keep_series_false()`:

```python
learning = builders.build_learning(
    reward_series=[1.0, 2.0],
    ep_length_series=[10.0],
    success_rate_series=[0.25, 0.75],
    ema_alpha=0.1,
    keep_series=False,
)
assert learning.success_rate is not None
assert learning.success_rate.final_raw == pytest.approx(0.75)
assert learning.success_rate.series_per_iter is None
```

- [ ] **Step 2: Run the new tests and verify regression coverage fails without the implementation**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_schema.py \
  source/isaaclab/test/benchmark/test_builders.py -q
```

Expected: FAIL because `Learning` and `build_learning()` do not accept a success-rate curve and `SCHEMA_VERSION` is still `"1.0"`.

- [ ] **Step 3: Implement the additive schema**

In `schema.py`, update the module documentation and constant to v1.1, describe `LearningCurve` as supporting reward, episode length, or success rate, add the field, and document it:

```python
SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class Learning:
    """Learning curves for a training run, plus their EMA smoothing factor.

    Args:
        ema_alpha: EMA smoothing factor in ``[0, 1]``.
        reward: Per-iteration mean-reward learning curve.
        ep_length: Per-iteration mean episode-length learning curve.
        success_rate: Per-iteration success-rate learning curve, or ``None``
            when the task does not report success.
    """

    ema_alpha: float
    reward: LearningCurve
    ep_length: LearningCurve
    success_rate: LearningCurve | None = None
```

Keep both top-level success-rate fields unchanged.

- [ ] **Step 4: Implement consistent curve construction**

In `builders.py`, add a private helper:

```python
def _build_learning_curve(values: Sequence[float], ema_alpha: float, keep_series: bool) -> LearningCurve:
    samples = list(values)
    return LearningCurve(
        final_raw=float(samples[-1]) if samples else 0.0,
        final_ema=ema(samples, ema_alpha),
        series_per_iter=samples if keep_series else None,
    )
```

Change `build_learning()` to this interface and behavior:

```python
def build_learning(
    *,
    reward_series: Sequence[float],
    ep_length_series: Sequence[float],
    ema_alpha: float,
    success_rate_series: Sequence[float] | None = None,
    keep_series: bool = True,
) -> Learning:
    rewards = list(reward_series)
    ep_lengths = list(ep_length_series)
    success_rates = list(success_rate_series) if success_rate_series else []

    return Learning(
        ema_alpha=ema_alpha,
        reward=_build_learning_curve(rewards, ema_alpha, keep_series),
        ep_length=_build_learning_curve(ep_lengths, ema_alpha, keep_series),
        success_rate=(
            _build_learning_curve(success_rates, ema_alpha, keep_series) if success_rates else None
        ),
    )
```

Update the Google-style docstring with `success_rate_series: Per-iteration success-rate values, or ``None`` when unavailable.` and state in `Returns:` that absent or empty success history produces `None`.

- [ ] **Step 5: Run schema and builder tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_schema.py \
  source/isaaclab/test/benchmark/test_builders.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all pre-commit hooks, review changes, stage, and rerun hooks**

Run:

```bash
PATH=/tmp/isaaclab-git-lfs-tool/extracted/usr/bin:$PATH ./isaaclab.sh -f
git diff --check
git add \
  source/isaaclab/isaaclab/test/benchmark/schema.py \
  source/isaaclab/isaaclab/test/benchmark/builders.py \
  source/isaaclab/test/benchmark/test_schema.py \
  source/isaaclab/test/benchmark/test_builders.py
PATH=/tmp/isaaclab-git-lfs-tool/extracted/usr/bin:$PATH ./isaaclab.sh -f
```

Expected: every hook passes twice and only the four listed files are staged.

- [ ] **Step 7: Commit the schema and builder change**

```bash
git commit -m "Add training success-rate curves"
```

### Task 2: Wire Training Adapters and Deliver the Schema PR

**Files:**

- Modify: `scripts/benchmarks/rsl_rl/benchmark_rsl_rl_train.py`
- Modify: `scripts/benchmarks/rl_games/benchmark_rl_games_train.py`
- Modify: `scripts/benchmarks/sb3/benchmark_sb3_train.py`
- Modify: `scripts/benchmarks/skrl/benchmark_skrl_train.py`
- Modify: `scripts/benchmarks/test/test_benchmark_smoke.py`
- Create: `source/isaaclab/changelog.d/antoiner-benchmark-success-series.minor.rst`

**Interfaces:**

- Consumes: `build_learning(..., success_rate_series: Sequence[float] | None, ...) -> Learning` from Task 1.
- Consumes: `SuccessRateTracker.history: list[float]` from `scripts/benchmarks/early_stop.py`.
- Produces: schema v1.1 training JSON with `learning.success_rate`; play JSON remains unchanged.

- [ ] **Step 1: Strengthen the end-to-end smoke assertions**

In `test_benchmark_smoke.py`, change:

```python
assert training_data["schema_version"] == "1.0"
```

to:

```python
assert training_data["schema_version"] == "1.1"
```

Then extend the existing `expect_success_rate` branch:

```python
if expect_success_rate:
    assert training_data["success_rate"] is not None
    success_curve = training_data["learning"]["success_rate"]
    assert success_curve is not None
    assert success_curve["series_per_iter"]
    assert success_curve["final_raw"] == pytest.approx(success_curve["series_per_iter"][-1])
else:
    assert training_data["learning"]["success_rate"] is None
```

Do not add any `learning` assertions to `play_data`; its scalar success API is intentionally unchanged.

- [ ] **Step 2: Verify the smoke regression fails before adapter wiring**

Run the skrl case, which reports success in the existing parametrization:

```bash
./isaaclab.sh -p -m pytest \
  scripts/benchmarks/test/test_benchmark_smoke.py -q -k skrl
```

Expected: FAIL because the adapter emits `learning.success_rate is None` even though the top-level success summary is populated.

- [ ] **Step 3: Wire RSL-RL and RL-Games histories**

In each adapter, move `get_success_tracker()` before `build_learning()` and pass its history:

```python
tracker = get_success_tracker(args_cli, early.tracker, log_data)  # observer.tracker for RL-Games
learning = builders.build_learning(
    reward_series=log_data.get(desc.reward_tag, []),
    ep_length_series=log_data.get(desc.ep_length_tag, []),
    success_rate_series=tracker.history if tracker is not None else None,
    ema_alpha=args_cli.ema_alpha,
    keep_series=not args_cli.no_series,
)
success_rate = round(tracker.tail_mean, 4) if (tracker and tracker.history) else None
```

Retain the existing top-level summary and `success_measurements(tracker)` calls.

- [ ] **Step 4: Wire SB3 and skrl histories**

In both adapters, move descriptor lookup, TensorBoard parsing, and `get_success_tracker()` above `build_learning()`. Pass:

```python
success_rate_series=success_tracker.history if success_tracker is not None else None,
```

Keep reward/episode-length sources, top-level tail mean, checkpoint handling, and play adapters unchanged.

- [ ] **Step 5: Add the changelog fragment**

Create `source/isaaclab/changelog.d/antoiner-benchmark-success-series.minor.rst` in the required changelog-fragment format:

```rst
Added
^^^^^

* Added per-iteration success-rate curves to schema v1.1 training benchmark
  bundles while preserving the existing summary success rate.
```

- [ ] **Step 6: Run focused unit and end-to-end tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_schema.py \
  source/isaaclab/test/benchmark/test_builders.py \
  scripts/benchmarks/test/test_training_adapters.py \
  scripts/benchmarks/test/test_early_stop.py \
  scripts/benchmarks/test/test_benchmark_smoke.py -q
```

Expected: PASS, including every supported training adapter and unchanged play bundle behavior.

- [ ] **Step 7: Run mandatory full pre-commit twice around staging**

Run:

```bash
PATH=/tmp/isaaclab-git-lfs-tool/extracted/usr/bin:$PATH ./isaaclab.sh -f
git diff --check
git add \
  scripts/benchmarks/rsl_rl/benchmark_rsl_rl_train.py \
  scripts/benchmarks/rl_games/benchmark_rl_games_train.py \
  scripts/benchmarks/sb3/benchmark_sb3_train.py \
  scripts/benchmarks/skrl/benchmark_skrl_train.py \
  scripts/benchmarks/test/test_benchmark_smoke.py \
  source/isaaclab/changelog.d/antoiner-benchmark-success-series.minor.rst
PATH=/tmp/isaaclab-git-lfs-tool/extracted/usr/bin:$PATH ./isaaclab.sh -f
```

Expected: every hook passes twice; review and restage any hook-modified file before the second pass.

- [ ] **Step 8: Commit the adapter integration**

```bash
git commit -m "Record training success-rate series"
```

- [ ] **Step 9: Verify the branch before push**

Run fresh verification:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_schema.py \
  source/isaaclab/test/benchmark/test_builders.py \
  scripts/benchmarks/test/test_training_adapters.py \
  scripts/benchmarks/test/test_early_stop.py \
  scripts/benchmarks/test/test_benchmark_smoke.py -q
PATH=/tmp/isaaclab-git-lfs-tool/extracted/usr/bin:$PATH ./isaaclab.sh -f
git status --short --branch
```

Expected: all tests and hooks pass; the branch is clean and ahead of `origin/develop` only by the design, plan, schema, and adapter commits.

- [ ] **Step 10: Push to the user fork and create the separate PR**

Push only to the `antoine` fork, never `origin`:

```bash
git push -u antoine antoiner/benchmark-success-series
```

Create a PR targeting `isaac-sim/IsaacLab:develop` with title:

```text
Add training success-rate series to benchmark schema
```

The PR body must state that schema v1.1 adds `Learning.success_rate`, retains both existing scalar success APIs, lists the focused verification results, and identifies the schema fix as a prerequisite for the separate Kamino DVI benchmark report.
