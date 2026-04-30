# Odin Live Retry Ingestion

**Status:** Approved for implementation planning (operator: antoiner)
**Author:** Codex
**Date:** 2026-04-30
**Branch context:** `antoiner/feat/odin-retry-sqlite`

---

## Goal

When a dispatch is still running, clicking the dashboard retry button for a failed job should add that job back to the live runner's work queue for the same dispatch. The operator should not need to copy an `odin-dispatch --resume ... --retry-failed=...` command while the original runner is still alive.

The scope is deliberately narrow:

- The live runner only consumes retry rows for its own `dispatch_id`.
- The dashboard button still writes to `odin_runs/.retry.sqlite` through `DataLayer.toggle_retry_queue`.
- The runner polls the retry DB, converts eligible failed jobs back to `pending`, and re-enqueues them.
- The runner records the retry outcome with `RetryDB.mark_consumed(...)` when the retry attempt reaches a terminal state.

The existing manual resume path remains the fallback for finished dispatches.

---

## Non-goals

- **No cross-dispatch live retry.** A runner for `20260430-120000` must not consume retry rows for `20260430-110509`.
- **No dashboard-to-runner RPC.** The retry DB is the coordination surface; no sockets, HTTP server, file watches, or process signaling.
- **No restarting currently running jobs.** Live ingestion only requeues jobs whose current dispatch state is `failed`.
- **No skipped-row materialization.** `skipped[]` entries are audit records, not `JobEntry` rows. Turning a `SkippedEntry` into a runnable job requires a separate design.
- **No artificial long idle mode after dispatch completion.** If all work has drained and the runner has written `ended_at`, the dispatch is no longer live; use `odin-retry export-resume-cmd` / `odin-dispatch --resume`.
- **No automatic retry policy.** The operator still opts in by clicking the dashboard retry button or using `odin-retry queue`.

---

## Current state

The SQLite retry queue work adds:

- `tools/odin/valhalla/dashboard/retry_db.py`
  - `RetryDB.read_pending(dispatch_id) -> set[str]`
  - `RetryDB.toggle(dispatch_id, run_id, note=None) -> set[str]`
  - `RetryDB.mark_consumed(dispatch_id, run_id, retry_dispatch_id, outcome, failure_kind=None) -> None`
- `DataLayer.read_retry_queue` / `toggle_retry_queue` keep the dashboard-facing API and now write to `odin_runs/.retry.sqlite`.
- `odin-retry export-resume-cmd <dispatch_id>` emits the manual resume command.

The runner today does not observe this DB while running. `run_dispatch()` builds `job_q` once from jobs already marked `pending`, starts workers, immediately pushes `None` sentinels, then drains `StateEvent`s until the initial `remaining` count reaches zero.

That sentinel behavior is the main implementation trap: if workers see the upfront `None` sentinel and exit after the initial queue drains, a later dashboard click has no worker left to consume the newly requeued job.

---

## Proposed architecture

### 1. Retry DB remains the coordination surface

The dashboard does not need a new callback path. Clicking the retry button continues to call:

```python
data.toggle_retry_queue(dispatch_id, run_id)
```

For a live dispatch, the runner notices that pending retry row on its next poll. For a finished dispatch, the row stays pending for the existing manual resume flow.

### 2. Runner polls live retries for its own dispatch only

Add a small helper in `tools/odin/asgard/runner.py`:

```python
def _consume_live_retries(
    *,
    retry_db: RetryDB,
    dispatch_id: str,
    jobs_by_id: dict[str, JobEntry],
    job_q: queue.Queue,
) -> int:
    ...
```

Behavior:

1. Read `retry_db.read_pending(dispatch_id)`.
2. For each pending `run_id`:
   - If `run_id` is unknown to `jobs_by_id`, ignore it and leave it pending.
   - If the job status is not `failed`, ignore it and leave it pending.
   - If the job status is `failed`, reset it for a live retry:
     - `status = "pending"`
     - `failure = None`
     - `assigned_to = None`
     - `started_at = None`
     - `ended_at = None`
     - Preserve `attempts` so the dashboard still reflects total attempts.
   - Put the `JobEntry` back on `job_q`.
3. Return the number of jobs requeued so the runner can increment `remaining`.

This helper does not call `mark_consumed` yet. The DB row remains pending until the retry attempt completes or fails. This avoids losing a queued retry if the runner crashes after enqueueing but before a worker actually runs it.

### 3. Runner tracks live retry attempts

`run_dispatch()` keeps an in-memory set:

```python
live_retry_run_ids: set[str] = set()
```

When `_consume_live_retries(...)` requeues a job, add its `run_id` to this set.

When a `StateEvent` with transition `completed` or `failed` arrives:

1. Apply the state event as today.
2. If `ev.run_id in live_retry_run_ids`, call `RetryDB.mark_consumed(...)`.
3. Remove `ev.run_id` from `live_retry_run_ids`.

Mapping:

```python
outcome = "completed" if ev.transition == "completed" else "failed"
failure_kind = ev.failure.kind if ev.failure is not None else None
retry_db.mark_consumed(
    dispatch_id,
    ev.run_id,
    retry_dispatch_id=dispatch_id,
    outcome=outcome,
    failure_kind=failure_kind,
)
```

`retry_dispatch_id` is the same dispatch because this is live ingestion, not a resume into a new dispatch directory.

### 4. Workers stay alive until runner shutdown

Remove the upfront sentinel insertion:

```python
for _ in workers:
    job_q.put(None)
```

Workers already call `job_queue.get(timeout=0.5)` in a loop and check `shutdown_event`. That loop can support live enqueue if the runner does not send sentinels early.

At the end of `run_dispatch()`:

1. Set `shutdown_event`.
2. Optionally push one `None` per worker to wake blocked workers immediately.
3. Join workers.

This preserves the existing worker API and avoids creating a separate live-runner thread.

### 5. Runner loop polls retries while work remains

Add `DispatchOptions.live_retry_poll_s: float = 5.0`.

Expose this as `--live_retry_poll_s` in `tools/odin/asgard/cli.py`. Accept `--live-retry-poll-s` as a compatibility alias because the current Odin CLI mostly uses hyphenated flags, but document the snake_case spelling as canonical for new code.

Runner loop shape:

```python
last_retry_poll = time.monotonic()

while remaining > 0 and any(w.is_alive() for w in workers):
    try:
        ev = state_chan.get(timeout=1.0)
    except queue.Empty:
        now = time.monotonic()
        if now - last_retry_poll >= options.live_retry_poll_s:
            added = _consume_live_retries(...)
            if added:
                remaining += added
                write_dispatch_state(dispatch_dir, state)
            last_retry_poll = now
        ...
        continue

    remaining -= _apply_state_event(...)
    if ev.transition in {"completed", "failed"} and ev.run_id in live_retry_run_ids:
        mark_consumed(...)
    write_dispatch_state(dispatch_dir, state)
```

Also call `_consume_live_retries(...)` after processing each terminal event. This lets an operator click a retry immediately after a failure while other jobs are still active, without waiting for a quiet `queue.Empty` timeout.

### 6. No duplicate live enqueue

The status gate prevents duplicate queue insertion:

- First poll sees DB pending row + job status `failed`, resets to `pending`, enqueues once.
- Later polls see the same DB row, but job status is `pending` or `running`, so they do not enqueue again.
- When the retry attempt ends, `mark_consumed` makes the row non-pending.

If the retry fails again, the row is consumed as a failed retry. The operator can click again to create a fresh pending retry row.

---

## Dashboard behavior

No callback contract changes are required.

Recommended copy update in `jobs_table.py`:

- Current title: `Tag for retry on next --resume --retry-failed`
- New title: `Tag for live retry if the runner is active; otherwise retry on next resume`

The banner may continue to show the manual resume command because it is still valid for finished dispatches. A later UI polish can hide or soften the banner when the dispatch is live, but that is not required for correctness.

---

## Failure and race semantics

| Case | Behavior |
|---|---|
| Operator queues failed job while other jobs are running | Runner requeues it on the next live-retry poll. |
| Operator queues failed job after dispatch has ended | Runner is gone; row remains pending for manual resume. |
| Operator queues a running job manually via CLI | Runner ignores it while status is `running`; row remains pending. |
| Operator queues an unknown run_id | Runner ignores it; row remains pending for operator cleanup. |
| Retry attempt completes | Runner marks DB row consumed with `outcome="completed"`. |
| Retry attempt fails | Runner marks DB row consumed with `outcome="failed"` and `failure_kind=<kind>`. |
| Runner crashes after requeue but before terminal event | DB row remains pending; dispatch state determines resume behavior. |
| Runner crashes after terminal event but before `mark_consumed` | DB row may remain pending; operator can remove it or a future repair command can reconcile it. |

---

## Testing strategy

### Unit tests

Add tests under `tools/odin/tests/test_asgard_runner.py`:

- `_consume_live_retries` requeues a failed job and returns `1`.
- `_consume_live_retries` ignores completed, pending, running, and unknown run_ids.
- `_consume_live_retries` preserves `attempts` while clearing failure/assignment/timestamps.
- Terminal completed event for a live retry calls `mark_consumed(..., outcome="completed")`.
- Terminal failed event for a live retry calls `mark_consumed(..., outcome="failed", failure_kind=<kind>)`.
- Runner shutdown sends workers home after the queue drains without relying on upfront sentinels.

### Threaded integration test

Add one focused threaded test in `tools/odin/tests/test_asgard_runner.py` or `tools/odin/tests/test_asgard_integration.py` using fake SSH/rsync:

1. Dispatch has two jobs.
2. Job A fails quickly.
3. Job B blocks long enough to keep the runner live.
4. Test waits until `dispatch.json` shows Job A `failed`.
5. Test calls `RetryDB(tmp_path / "odin_runs").toggle(dispatch_id, job_a_run_id)`.
6. Release Job B / allow the queue to continue.
7. Assert final dispatch state shows Job A was retried and terminal.
8. Assert `RetryDB.list_for_dispatch(dispatch_id, pending_only=True)` is empty.
9. Assert `RetryDB.list_for_dispatch(dispatch_id)` records Job A with `retried_at` and `retry_dispatch_id == dispatch_id`.

### Existing suites

Run:

```bash
./isaaclab.sh -p -m pytest --noconftest -p no:cacheprovider \
    tools/odin/tests/test_asgard_runner.py \
    tools/odin/valhalla/dashboard/tests \
    -q
./isaaclab.sh -f
```

The dashboard suite should keep passing because the button still writes through the same `DataLayer` API.

---

## File-by-file impact

| File | Change | Estimated LOC |
|---|---|---:|
| `tools/odin/asgard/runner.py` | Add live retry poll helper, keep workers alive until shutdown, mark consumed on retry terminal events. | ~90 |
| `tools/odin/asgard/cli.py` | Optional `--live_retry_poll_s` parser support. | ~8 |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` | Update retry button tooltip copy. | ~2 |
| `tools/odin/tests/test_asgard_runner.py` | Unit + threaded fake-runner tests for live ingestion. | ~160 |
| `docs/odin/architecture.md` | Record live retry ingestion behavior. | ~1 |
| `tools/odin/README.md` | Add note that queued retries are consumed live while runner is active. | ~8 |

Estimated total: ~270 LOC.

---

## Open implementation notes

- Use `RetryDB(dispatch_dir.parent)` because `dispatch_dir` is `odin_runs/<dispatch_id>` and the DB lives at `odin_runs/.retry.sqlite`.
- Do not consume pending retry rows for completed jobs. The operator may have queued a typo or stale row; leaving it pending is more visible than silently deleting it.
- Preserve existing targeted resume behavior. `_apply_retry_options` still handles `--retry-failed` on startup; live ingestion is only for rows added after the dispatch loop is active.
- Keep `mark_consumed` best-effort but visible. If it raises, the runner should print a warning and continue dispatch completion rather than turning a successful training run into an infrastructure failure.
