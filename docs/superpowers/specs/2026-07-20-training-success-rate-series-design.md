# Training Success-Rate Series Design

## Purpose

Isaac Lab's benchmark training adapters collect a per-iteration success-rate history, but schema v1.0 discards that history when it writes the canonical `TrainingBundle`. The bundle preserves full reward and episode-length series while exposing only a final success-rate tail mean.

This change makes success-rate training data available to benchmark consumers without requiring them to parse framework-specific TensorBoard logs.

## Scope

The change is limited to the benchmark schema, its builders and training adapters, and their tests. It does not change task success definitions, convergence thresholds, early-stopping behavior, play-benchmark semantics, or existing top-level summary fields.

## Schema

Increase `SCHEMA_VERSION` from `"1.0"` to `"1.1"`. Version 1.1 is an additive schema revision.

Add an optional success-rate curve to `Learning`:

```python
@dataclass(frozen=True)
class Learning:
    """Learning curves for a training run, plus their EMA smoothing factor."""

    ema_alpha: float
    reward: LearningCurve
    ep_length: LearningCurve
    success_rate: LearningCurve | None = None
```

The curve has the same representation as reward and episode length:

- `final_raw` is the last recorded per-iteration success rate.
- `final_ema` uses the benchmark's configured EMA factor.
- `series_per_iter` contains the complete tracker history unless `--no_series` is active.

When a task does not emit a success metric, `Learning.success_rate` is `None`.

Keep `TrainingBundle.success_rate` unchanged. It remains the final tail-mean summary used by existing consumers and therefore avoids a breaking API change. No public symbol is removed, renamed, or deprecated.

`PlayBundle` remains unchanged. Play benchmarks summarize completed evaluation episodes rather than training iterations, so their scalar success rate is not a learning curve.

## Data Flow

Each training adapter already obtains a `SuccessRateTracker` through `get_success_tracker()`. The adapter passes `tracker.history` to `build_learning()` together with reward and episode-length series. If the tracker is absent, it passes no success series.

`build_learning()` accepts an optional success-rate sequence and constructs `Learning.success_rate` with the same helper logic and `keep_series` policy used for the other curves. The adapter continues deriving `TrainingBundle.success_rate` from `tracker.tail_mean`.

This preserves two intentionally different statistics:

- `learning.success_rate.final_raw`: the last iteration's success rate.
- top-level `success_rate`: the established tail-window mean.

All supported training adapters must populate the field consistently when their success tracker has history. The RSL-RL path is the primary end-to-end regression target because the solver benchmark uses RSL-RL.

## Compatibility and Error Handling

Schema v1.0 readers can identify the newer artifact through `schema_version`. Existing Python callers constructing `Learning` remain source-compatible because the new field is optional and defaults to `None`.

An absent success metric is valid and serializes as `null`. An empty history is treated as absent rather than as a zero-valued curve. The implementation does not invent success values or infer success from reward.

The `--no_series` option sets `series_per_iter` to `None` while preserving `final_raw` and `final_ema`, matching reward and episode-length behavior.

## Testing

Tests will cover:

- schema version `1.1` in unit and smoke expectations;
- JSON round-trip of a populated success-rate `LearningCurve`;
- `Learning.success_rate is None` for tasks without a success metric;
- `--no_series` omission of success history while retaining final statistics;
- builder behavior for populated, empty, and absent success histories;
- RSL-RL benchmark smoke output containing the success curve when the task reports success;
- unchanged `PlayBundle.success_rate` behavior.

Before implementation, the focused baseline is:

```text
14 passed in 78.96s
```

from:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/benchmark/test_schema.py \
  scripts/benchmarks/test/test_benchmark_smoke.py -q
```

The final implementation must also pass `./isaaclab.sh -f` before commit and again before push.

## Delivery

Implement this schema change in the isolated `antoiner/benchmark-success-series` branch based on Isaac Lab `origin/develop` commit `79accca281128660a786abb599f40bd335963963`.

Add one minor changelog fragment under `source/isaaclab/changelog.d/` describing the new training success-rate curve. The schema fix is delivered as a separate pull request from the Kamino DVI benchmark and report work.
