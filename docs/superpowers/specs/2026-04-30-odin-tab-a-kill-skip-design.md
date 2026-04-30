# Odin Tab A — Per-row Kill / Skip Button

**Status:** Approved (operator: antoiner)
**Author:** Claude (handed off for implementation)
**Date:** 2026-04-30
**Branch context:** `antoiner/feat/odin`, atop the detached submit-and-poll merge + SQLite retry queue + Hugin streaming + orphan-sweep work (HEAD around `4a3b34b2101`).

---

## Goal

Give the operator a per-row Tab A button to **kill** a running job or **skip** a pending one. Today the dashboard is read-only past the retry-toggle: if a training run is wedged, the only options are wait out the per-job timeout (default 12 h) or `Ctrl-C` the whole dispatcher. After this change, the operator clicks a row's button, confirms within 5 s, and the dispatcher acts on it within one runner tick (~5 s on the existing `DispatchOptions.live_retry_poll_s` cadence).

The single most common motivating scenario: an operator notices an in-flight training is producing nonsense (reward stuck at chance, training.json mtime frozen) and wants the host freed for the next job *now*, plus the partial logs pulled back for post-mortem. Today they can do nothing.

---

## Non-goals

- **Bulk operations** ("kill all running on host X", "skip all pending matching task Y"). Per-row only in v1; bulk is a multiplier feature once kill/skip itself is proven.
- **Pause / resume of a single training run.** Trainers don't support checkpoint-and-resume mid-iteration today; pause would need framework-level work.
- **Cross-dispatch cancellations** (operator sees a stale dispatch in the dashboard and wants to clean it up). Cancel button is hidden once `dispatch.ended_at` is set.
- **Operator audit-log surfacing in the UI.** The `cancellations` SQLite table is the audit trail; a dashboard view of "who killed what when" is a follow-up.
- **Replacing the existing retry-toggle.** The retry-toggle stays exactly as-is; cancel is a new button slot in the row.

---

## Current state (what exists today)

- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` renders the Tab A jobs table. Each failed-status row carries an inline retry-toggle (`tab-a-retry-toggle`) and an expand-toggle (`tab-a-expand-toggle`). Pending and running rows have no per-row controls today.
- `tools/odin/valhalla/dashboard/retry_db.py` (`RetryDB`) is a SQLite-backed retry queue at `<runs_root>/.retry.sqlite`. WAL mode, migrations via `PRAGMA user_version`, `INSERT OR REPLACE` semantics, dashboard writes via `data.toggle_retry_queue`, runner consumes via `_consume_live_retries` on a configured `live_retry_poll_s` cadence.
- `tools/odin/asgard/runner.py: _consume_live_retries(retry_db, dispatch_id, jobs_by_id, job_q, live_retry_run_ids)` is the precedent: read pending rows from SQLite each tick, mutate matching `JobEntry` records, push into `job_q`, mark consumed on terminal events.
- `tools/odin/asgard/worker.py` runs detached submit-and-poll (post the recent merge). The worker has `_inflight: dict[run_id, JobInflight]` plus `_cleanup_remote_process(job)` (best-effort `pkill -9 -f '<run_id>'` over SSH with `pty=False`). Today it's only invoked on SSH timeout; we'll reuse it for operator kills.
- `FailureInfo.kind` (in `tools/odin/asgard/jobs.py`) currently enumerates `infrastructure`, `hugin_crash`, `hugin_malformed_bundle`, `timeout`, `preset_unsupported`, `gpu_lost`. Two new strings (`killed`, `skipped`) extend this enum.

---

## Proposed architecture

### 1. Control channel: new `cancellations` table in `.retry.sqlite`

Reuse the existing SQLite file. Adding a second SQLite at `<runs_root>/` would duplicate WAL setup, migration plumbing, and connection-locking for no benefit.

Migration #2 (sibling of the existing `retries` migration in `retry_db.py: _MIGRATIONS`):

```sql
CREATE TABLE cancellations (
  dispatch_id  TEXT NOT NULL,
  run_id       TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  kind         TEXT NOT NULL CHECK (kind IN ('kill','skip')),
  consumed_at  TEXT,
  outcome      TEXT,                                       -- "killed" | "skipped" | "noop"
  PRIMARY KEY (dispatch_id, run_id)
);
CREATE INDEX idx_cancellations_pending
  ON cancellations(dispatch_id) WHERE consumed_at IS NULL;
```

`outcome` is filled by the runner on consumption: `"killed"` (kill landed), `"skipped"` (skip landed), or `"noop"` (job was already in a terminal state — not an error, just bookkeeping).

### 2. Dashboard side

A new `tools/odin/valhalla/dashboard/cancel_db.py` mirrors `RetryDB`'s shape. Public surface:

```python
class CancelDB:
    def __init__(self, runs_root: Path) -> None: ...
    def request(self, dispatch_id: str, run_id: str, kind: str) -> None: ...
    def read_pending(self, dispatch_id: str) -> dict[str, str]: ...   # run_id → kind
    def mark_consumed(
        self,
        dispatch_id: str,
        run_id: str,
        *,
        outcome: str,
    ) -> None: ...
    def list_for_dispatch(
        self,
        dispatch_id: str,
        *,
        pending_only: bool = False,
    ) -> list[CancelRow]: ...
```

`request()` uses `INSERT OR REPLACE` so a Skip-then-Kill click sequence on the same row simply overwrites kind. `read_pending()` returns `{run_id: kind}` (not `set[str]`) because the runner needs the kind to decide between flow A and flow B below.

`tools/odin/valhalla/dashboard/data.py` grows two thin wrappers (`request_cancel`, `read_cancel_queue`) parallel to the existing `toggle_retry_queue` / `read_retry_queue`. The dashboard tabs never touch `CancelDB` directly; they go through `data.py` like the retry queue does.

### 3. Per-row button (Tab A)

`tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py: _data_row` renders a context-aware button per row. Mapping:

| Job status | Button label | Confirm label | Disabled when |
|---|---|---|---|
| `pending` | `Skip` | `Confirm skip` (red) | already cancellation-pending in DB |
| `running` | `Kill` | `Confirm kill` (red) | already cancellation-pending in DB |
| `completed` / `failed` | (existing retry-toggle stays) | — | — |

Two-click flow:

1. First click — pattern-matched callback `tab-a-cancel-toggle` flips a per-row `dcc.Store` entry (`{"run_id": ..., "expires_at_ms": now+5000}`) and re-renders the row with a red "Confirm…" button label.
2. A `dcc.Interval(id="tab-a-cancel-revert", interval=500ms)` callback checks every 500 ms whether any per-row store entries have expired; expired entries flip back to default.
3. Second click within the window — same callback observes the existing pending entry, calls `data.request_cancel(dispatch_id, run_id, kind)`, removes the per-row store entry, re-renders the row with a small "kill pending" / "skip pending" badge in the Status cell.

The pattern-matched id shape mirrors the existing retry-toggle: `{"type": "tab-a-cancel-toggle", "run_id": run_id}`. The badge is rendered when `run_id in cancel_queue` from `data.read_cancel_queue(dispatch_id)`.

When `dispatch.ended_at` is non-None (dispatch finished), the cancel button is omitted from the row entirely. There's no point in writing a cancellation row that no runner will consume.

### 4. Runner side

`tools/odin/asgard/runner.py` grows a sibling helper to `_consume_live_retries`:

```python
def _consume_cancellations(
    *,
    cancel_db: CancelDB,
    dispatch_id: str,
    jobs_by_id: dict[str, JobEntry],
    job_q: queue.Queue,
    workers: dict[str, ValkyrieWorker],
    state_chan: queue.Queue,
) -> int:
    """Consume pending cancellations; return how many landed terminal."""
    landed = 0
    for run_id, kind in cancel_db.read_pending(dispatch_id).items():
        job = jobs_by_id.get(run_id)
        if job is None or job.status in {"completed", "failed"}:
            cancel_db.mark_consumed(dispatch_id, run_id, outcome="noop")
            continue
        if kind == "skip" and job.status == "pending":
            # Flip status in-place. If a worker already pulled this job off
            # the queue, the worker re-reads job.status before submit and
            # bails out (see _submit_or_handle).
            job.status = "failed"
            job.failure = FailureInfo(
                kind="skipped",
                message="operator skipped before dispatch",
                details={"requested_at": _utc_now_iso()},
            )
            job.ended_at = _utc_now_iso()
            cancel_db.mark_consumed(dispatch_id, run_id, outcome="skipped")
            landed += 1
        elif kind == "skip" and job.status == "running":
            # Skip-on-already-running → promote to kill.
            cancel_db.upgrade_to_kill(dispatch_id, run_id)
            kind = "kill"
        if kind == "kill" and job.status == "running":
            worker = workers.get(job.assigned_to)
            if worker is None:
                continue
            worker.request_cancel(run_id)        # thread-safe setter on the worker
            # Outcome marked when the worker emits a failed/killed event;
            # runner's _apply_state_event handler does the mark_consumed.
    return landed
```

The runner's main loop calls `_consume_cancellations(...)` adjacent to the existing `_consume_live_retries(...)` call. The same `live_retry_poll_s` cadence (default 5 s) governs both — kill latency is ~5 s, well below the 12 h default per-job timeout we're optimising against.

`_apply_state_event` is extended to call `cancel_db.mark_consumed(..., outcome="killed")` on a `failed` event whose `failure.kind == "killed"`.

### 5. Worker side

`tools/odin/asgard/worker.py: ValkyrieWorker` grows:

- `self._cancel_request: dict[str, bool] = {}` — keyed by `run_id`, set by the runner via `request_cancel()`.
- `request_cancel(self, run_id: str) -> None` — thread-safe setter (a single dict assignment is atomic in CPython, but we'll add a `threading.Lock` to keep the contract explicit).
- In `_run_detached`'s tick, **before** `_poll_inflight_once`:

  ```python
  for run_id in list(self._cancel_request):
      inflight = self._inflight.get(run_id)
      if inflight is None:
          self._cancel_request.pop(run_id, None)
          continue
      if not inflight.kill_dispatched:
          self._cleanup_remote_process(inflight.job)   # existing pkill helper
          inflight.kill_dispatched = True
      self._cancel_request.pop(run_id, None)
  ```
- `JobInflight` grows a `kill_dispatched: bool = False` field, mirroring the existing `timeout_kill_dispatched`.
- `_finalize_terminal` precedence (highest first): `timeout_kill_dispatched` → `FailureInfo(kind="timeout")`; `kill_dispatched` → `FailureInfo(kind="killed", message="operator kill")`; otherwise → `_classify_remote(job)`. The timeout check stays first so an operator clicking Kill on a job that already tripped its budget gets the more accurate `kind="timeout"`.
- The rsync-pull path is unchanged: `_finalize_terminal` always pulls the bundle on terminal poll states, so the partial logs come back automatically (per Q4 answer).

The skip-race short-circuit lives in `_submit_or_handle`:

```python
def _submit_or_handle(self, job: JobEntry) -> None:
    # Skip race: the runner may have flipped this job's status to 'failed'
    # (kind=skipped) between the moment we pulled it off the queue and
    # now. Re-check before paying for an SSH submit.
    if job.status != "pending":
        return
    ...
```

`JobEntry` instances are shared by reference between the runner and workers (the queue ships the same Python object). Python's GIL makes a single attribute read atomic, so the worker sees the runner's status flip on the next read. No additional locking is required for this check.

### 6. Reconciliation on `--resume`

`tools/odin/asgard/reconcile.py` extends `reconcile_orphans` to:

1. After the existing `running` reconciliation, call `cancel_db.read_pending(dispatch_id)` once.
2. For each pending cancellation whose job is still in pending state in the loaded `dispatch.json`: apply the same skip flip (`status="failed"`, `kind="skipped"`) and `mark_consumed(..., outcome="skipped")` immediately, before any worker starts. This means a Skip request that arrived while the dispatcher was crashed lands on resume.
3. For each pending cancellation whose job is `running` after detached re-attach (status set to `running` by reconcile's `reattached_inflight` outcome): seed `worker._cancel_request[run_id] = True` before `worker.start()`, mirroring the in-flight reattach path.

### 7. Failure modes

| Scenario | Behaviour |
|---|---|
| Dashboard up, runner not running (dispatch ended) | Cancel button hidden when `dispatch.ended_at is not None`. No DB write possible. |
| Operator double-confirms (extra clicks on the red Confirm) | Second click is a no-op: callback checks `read_pending()` and skips the duplicate INSERT. |
| Click Skip then Kill on the same row mid-confirm (status changed under us) | `request()` is `INSERT OR REPLACE`; second request overwrites kind. Runner sees the latest. |
| Skip-on-already-running | Runner detects the status mismatch and calls `cancel_db.upgrade_to_kill(...)`; flow proceeds as Case B. Dashboard's next render shows "killing". |
| Worker dies mid-kill (SSH blip during `pkill`) | Same as today's timeout flow — the next poll tick reports `exited-no-manifest` (process is dead) or `alive` and the worker re-issues `pkill` on the next tick. `kill_dispatched` is sticky, so the eventual finalize uses `kind="killed"`. |
| Resume after dispatcher crash with pending cancellations | Reconcile applies skip rows immediately on pending jobs; sets `_cancel_request` on workers that re-attached running jobs. |
| Two operators click Kill simultaneously | Single row in `cancellations` (PK is `(dispatch_id, run_id)`); one INSERT wins, the other is a no-op via `INSERT OR REPLACE`. Both dashboards see the same pending state on next tick. |
| Cancellation row outlives a dispatch | Rows stay forever for audit. They're cheap (~50 bytes each). The same applies to today's `retries` rows. |
| Worker has a stale `_cancel_request` for a job that finished `completed` before the kill landed | The pre-poll cancel loop checks `self._inflight.get(run_id)` first; missing entry → drop the request silently. The runner's `_apply_state_event` will call `mark_consumed(..., outcome="noop")` if the event is `completed`. |

---

## Failure-mode mapping table (kind → action)

| `failure.kind` | Set by | When |
|---|---|---|
| `killed` | worker `_finalize_terminal` (or runner direct emit on noop-eq race) | running job pkilled by operator |
| `skipped` | runner `_consume_cancellations` (or reconcile on resume) | pending job skipped before dispatch |

Both kinds count toward the dispatch's `failed` total. The `failed_by_kind` summary line prints them with the rest. Tab A renders them as "failed" with the existing kind-pill mechanism (`tab-a-kind-pill-killed` / `tab-a-kind-pill-skipped` CSS classes need their colours added; mirror the muted-grey `preset_unsupported` palette rather than the alarming red of `gpu_lost` since these are operator-initiated, not failures).

---

## State machine (per job)

```
                   request_cancel('skip')
   pending ──────────────────────────────► failed (kind=skipped)
                       │
                       │ [worker hadn't picked it up: runner does the flip]
                       │ [worker had picked it up: skip_run_ids drains in submit_or_handle]
                       │
                       │ submit
                       ▼
                   running
                       │ request_cancel('kill')
                       ▼
              kill_dispatched=True
                       │ pkill -9 -f <run_id>
                       ▼
              poll: exited-no-manifest
                       │ rsync pull (best-effort)
                       ▼
              failed (kind=killed)
```

Today's other transitions (`running → completed`, `running → failed[other kinds]`, etc.) are unaffected.

---

## Backward compatibility

- The dispatch.json schema does **not** change. JobEntry stays as-is; two new strings join `failure.kind`'s enum, which the existing dashboard renders as "failed (kind)" without code changes.
- Pre-existing `.retry.sqlite` files migrate to schema version 2 transparently via `_migrate(con)` — the existing migration framework handles it. No action required from operators.
- Tab A renders no cancel button on rows whose status the new mapping table doesn't cover (e.g., `assigned`, an internal transient state). The retry-toggle behaviour for failed/completed rows is unchanged.

---

## Testing strategy

### Unit tests (~25 new tests, all pure-Python `python3 -m pytest --noconftest -p no:cacheprovider`)

- **`test_cancel_db.py`** (new, ~120 LOC):
  - `test_request_inserts_pending_row`
  - `test_request_kill_then_skip_replaces`
  - `test_read_pending_returns_dict_kind_keyed`
  - `test_mark_consumed_sets_outcome_and_consumed_at`
  - `test_concurrent_request_one_winner`
  - `test_migration_idempotent`
  - `test_request_rejects_invalid_kind`
  - `test_upgrade_to_kill_replaces_kind_on_pending_row`

- **`test_asgard_runner_cancellations.py`** (new, ~150 LOC):
  - `test_consume_cancellations_skips_pending_job`
  - `test_consume_cancellations_promotes_skip_to_kill_on_running_job`
  - `test_consume_cancellations_signals_worker_for_kill`
  - `test_consume_cancellations_noop_on_finished_job`
  - `test_consume_cancellations_marks_consumed_with_outcome`
  - `test_consume_cancellations_marks_consumed_killed_on_failed_event`

- **`test_asgard_worker_cancel.py`** (new, ~120 LOC):
  - `test_worker_dispatches_pkill_when_cancel_requested`
  - `test_worker_finalize_uses_killed_kind_when_kill_dispatched`
  - `test_worker_skips_submit_when_run_id_in_skip_set`
  - `test_worker_pulls_partial_bundle_on_kill`
  - `test_worker_drops_stale_cancel_request_for_finished_job`

- **`test_tab_a_cancel_button.py`** (new, ~80 LOC, dashboard render-level):
  - `test_pending_row_renders_skip_button`
  - `test_running_row_renders_kill_button`
  - `test_completed_row_does_not_render_cancel_button`
  - `test_pending_cancellation_renders_pending_badge`
  - `test_finished_dispatch_hides_cancel_button`

- **`test_tab_a_cancel_callback.py`** (new, ~100 LOC):
  - `test_first_click_flips_to_confirm_state`
  - `test_second_click_within_window_writes_db_row`
  - `test_confirm_state_reverts_after_timeout`
  - `test_callback_handles_kind_already_pending`

### Integration test

Extend `test_asgard_integration.py` with `test_loopback_detached_dispatch_skip_and_kill_via_db`:

1. Start a loopback detached dispatch with two jobs (Cartpole-fast, with a stub trainer that sleeps).
2. Wait until poll observes `alive` for at least one job.
3. Write a `kill` cancellation for the running one and a `skip` cancellation for the still-pending one directly via `CancelDB` (simulating what the dashboard would do).
4. Assert: both end in `failed`, with `kind="killed"` and `kind="skipped"` respectively. The killed one's bundle dir contains `logs/` (partial). Both `cancellations` rows have `consumed_at` set.

### Manual real-fleet validation

- Start a real dispatch, click Skip on a pending row → assert it lands `failed/skipped` within ~5 s.
- Start a real dispatch, click Kill on a running row → assert the host is freed (a queued pending job starts within one poll tick after the kill), the bundle's stderr is on the dispatcher, and the row shows `failed/killed`.

---

## Implementation order preview (suggested)

1. **`cancel_db.py`** — new schema migration, dataclass + `request` / `read_pending` / `mark_consumed` / `upgrade_to_kill` helpers + tests.
2. **`runner.py: _consume_cancellations`** — pure runner logic (no worker side yet) + skip-only path tests.
3. **`worker.py: request_cancel + kill_dispatched`** — wire kill path through detached run loop + tests.
4. **`runner.py` main-loop integration** — call `_consume_cancellations` next to `_consume_live_retries`, hand worker references through; mark consumed on `failed/killed` events.
5. **`reconcile.py`** — apply pending cancellations on `--resume`.
6. **`data.py`** — `request_cancel` / `read_cancel_queue` thin wrappers + tests.
7. **`jobs_table.py`** — render the cancel button + pending badge.
8. **`callbacks.py`** — two-click confirm flow + 5 s revert + tests.
9. Loopback integration test.
10. Real-fleet validation pass.

---

## Files touched (estimate)

| File | Change |
|---|---|
| `tools/odin/valhalla/dashboard/cancel_db.py` | New, ~150 LOC |
| `tools/odin/valhalla/dashboard/data.py` | +20 LOC (wrappers) |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` | +60 LOC (cancel-button render + pending badge) |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py` | +50 LOC (confirm-flow callback + interval revert) |
| `tools/odin/asgard/runner.py` | +60 LOC (`_consume_cancellations` + main-loop wiring + `_apply_state_event` tweak) |
| `tools/odin/asgard/worker.py` | +30 LOC (`request_cancel`, `_cancel_request`, kill_dispatched, skip-set short-circuit) |
| `tools/odin/asgard/reconcile.py` | +20 LOC (re-apply pending cancellations after reattach) |
| `tools/odin/asgard/jobs.py` | docstring only |
| `tools/odin/valhalla/dashboard/tests/test_cancel_db.py` | New, ~120 LOC |
| `tools/odin/tests/test_asgard_runner_cancellations.py` | New, ~150 LOC |
| `tools/odin/tests/test_asgard_worker_cancel.py` | New, ~120 LOC |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_button.py` | New, ~80 LOC |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_cancel_callback.py` | New, ~100 LOC |
| `tools/odin/tests/test_asgard_integration.py` | +120 LOC (skip + kill scenario) |
| **Total** | **~1,080 LOC including tests** |

---

## Open questions / decisions deferred

- **Live-cancel poll interval.** Reuses `live_retry_poll_s` (default 3 s) today. Kill latency of ~3 s is fine for the long-training case; if we ever want sub-second kills, the table is already structured to support that.
- **Hard-delete of consumed rows.** Out of scope; `cancellations` stays append-only forever, mirroring the `retries` table's behaviour.
- **Operator audit UI.** Showing "operator X killed run Y at time T" in the dashboard. Worth doing but not v1 — the SQLite table is the source of truth and post-mortem-readable.
- **Bulk operations.** Once kill/skip is proven, "kill all on host X" or "skip all matching filter F" is a multiplier feature. Out of scope here.

---

## What this does NOT solve

- A wedged trainer whose process *looks* alive but is making no progress. Kill fixes this from the operator side, but the auto-detection of "this job is stuck" is a dead-man-switch problem (model_*.pt mtime watcher) that's separate.
- A network blip during the kill request itself: if the SSH `pkill` call fails, the worker retries on the next poll tick because `kill_dispatched` is sticky. Worst case the trainer outlives the kill request by one poll interval.
- A misclick on the wrong row. The two-click confirmation is the safety net; we trust the operator after the second click.
