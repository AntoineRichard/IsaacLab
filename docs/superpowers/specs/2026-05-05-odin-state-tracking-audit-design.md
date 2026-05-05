# Odin Dispatcher State-Tracking Audit — Design

**Status:** approved
**Date:** 2026-05-05
**Task covered:** Centralize all dispatcher state transitions through a
single helper + an explicit allowed-transition graph, and fix four
state-tracking bugs observed during the 20260430-110509 (Blackwell)
and 20260505-095154 (Spark) dispatches.

## 1. Motivation

`JobEntry.status` is mutated from 25+ call sites across `worker.py`,
`reconcile.py`, and `runner.py`. Each site touches its own subset of
the related fields — `(status, ended_at, started_at, assigned_to,
failure, attempts, preferred_not)` — and there is no central contract
saying which fields a given target state requires. This caused four
distinct misclassifications during one debugging session:

**Bug 1 — Reconcile mis-classifies post-crash manifests as completed.**
On dispatcher restart, `reconcile.py:_manifest_indicates_clean_completion`
reads the remote manifest and accepts any non-empty `phases` dict where
every present phase is `completed exit=0`. Hugin writes the manifest
progressively: it stamps `phases.startup = {status: "completed", ...}`
as a placeholder *before* startup actually runs (`hugin/run.py:138`),
then either updates it on completion or leaves it stale on crash. If
reconcile reads while only the placeholder exists, every phase looks
healthy — the run gets adopted as `completed` and dispatch.json is
stamped accordingly. Hugin later overwrites the on-disk manifest with
both phases marked `failed exit=1`, but dispatch.json retains the
wrong verdict.

**Concrete victims:** in 20260505-095154,
`rsl-rl_physx_Isaac-Ant-Direct-v0_..._seed44` and
`rsl-rl_physx_Isaac-Quadcopter-Direct-v0_..._seed44`. Both have
on-disk `manifest.json` with `phases.startup.status="failed",
exit_code=1, run_duration_s≈0.5` but `dispatch.json` says
`status="completed"` with `ended_at=None`.

**Bug 2 — Worker holds `running` during slow rsync.** When a job
finishes on the host, `worker._finalize_terminal` rsync-pulls the
bundle before flipping status to terminal. With a slow controller↔host
link (~100 KB/s effective on the Blackwell tier this session), pulling
a 360 MB RGB-Camera bundle took ~1 hour per seed. During that hour the
dashboard correctly read `status="running"` from dispatch.json — but
the trainers had already finished, and the operator was looking at a
"running 2 h ago" row. UX-correctness gap: the dispatcher's worldview
lags reality by however long the rsync takes, with no signal that
finalization is in progress.

**Concrete victims:** all three RGB-Camera seeds in 20260430-110509.
Trainers exited around 10:23–10:28 UTC; rsyncs completed and dispatch
flipped to `completed` only at 11:25–11:34 UTC.

**Bug 3 — `gpu_lost` recovery silently drops state.** When a job's
pre-submit nvidia-smi probe fails, the worker classifies the failure
as `gpu_lost`. `worker._handle_synchronous_failure` calls
`recover_valkyrie_gpu`; on success it emits a `recovered` state event
and re-queues the JobEntry. But it never sets `job.status = "pending"`
or clears `started_at` / `assigned_to`. The re-queued JobEntry sits in
the queue with `status="running"` and a stale `started_at`. If another
job grabs the host slot before the queue cycles back, the original job
is effectively orphaned — neither running on any host nor visible as
pending.

**Concrete victim:** in 20260430-110509,
`rsl-rl_physx_Isaac-Navigation-Flat-Anymal-C-v0_..._seed43` on host
10.57.232.96. After the GPU-probe failure, the worker quickly
recovered the host and pulled seed44 from the queue; seed43 never
re-emerged because its in-memory state still said "running". The job
sat in dispatch.json as `running started_at=11:34:16Z` indefinitely.

**Bug 4 — `completed`-with-`ended_at=None` recurring pattern.** Every
reconcile-side flip writes `status` only:

- `reconcile.py:222` (`adopted_completed`)
- `reconcile.py:226` (`adopted_failed`)
- `reconcile.py:242` (post-detached-poll `adopted_completed`)
- `reconcile.py:252` (post-detached-poll `adopted_failed`)
- `runner.py:226` (legacy completion flip)

None of these set `ended_at`. The dashboard surfaces this as
"completed but no end time" rows. It is the same code-smell as Bug 3:
distributed state writes that touch a non-uniform subset of fields.

The four bugs share a single root cause: **state transitions are not
centralized**, so each call site can (and does) forget some field
update.

## 2. Goals

- All `JobEntry` state transitions go through a single helper that
  enforces field-update consistency per target state.
- An explicit allowed-transition graph rejects illegal transitions
  with a clear runtime error; legal transitions are well-defined and
  documented.
- The four observed bugs are fixed as a direct consequence of the
  refactor (Bugs 3 and 4) or via small targeted patches that
  complement it (Bugs 1 and 2).
- The dashboard can distinguish "training" from "pulling bundle" —
  the operator no longer sees a job stuck at `running` for an hour
  while finalization is in flight.
- Refactor is a no-behavior-change rewrite for legal transitions: the
  same dispatch outcomes the worker produces today, just with the
  field-update bugs removed.

## 3. Non-goals

- No changes to the dispatch protocol (no new state events, no
  schema_version bumps on `dispatch.json` or `manifest.json`).
- No new policy: the helper enforces what current correct behavior
  *should* be — it does not introduce new retry semantics, new
  failure kinds, new quarantine rules.
- No async / lock changes. The existing single-writer model on the
  runner's main thread (workers post StateEvents, runner mutates
  JobEntry on dequeue) is preserved.
- Not a fix for the slow-rsync transport itself
  (`project_odin_rsync_no_timeout.md` covers that). Bug 2's fix here
  is purely the UX surface — making the dashboard show "pulling
  bundle" while the rsync runs.
- No reconciliation strategy overhaul. The
  `_manifest_indicates_clean_completion` patch is a single-function
  defensive check, not a redesign of how reconcile reads the remote.

## 4. Design

### 4.1 The allowed-transition graph

The eight legal directed edges:

```
pending  → running    (worker assigns + submits)
pending  → failed     (skip-via-cancel, native_backend_mismatch,
                       newton_floor, preflight-rejects-all-hosts, …)
running  → completed  (worker happy path)
running  → failed     (training crash, hugin malformed bundle,
                       infrastructure error after retry budget,
                       killed, timeout, gpu_lost with recovery_failed)
running  → pending    (gpu_lost recovery succeeded → re-queue,
                       host_down re-queue with preferred_not,
                       reset_in_flight on resume,
                       reconcile says "process not actually running")
failed   → pending    (retry-failed CLI, retry-all-failed CLI,
                       live RetryDB consumption)
completed → pending   (live RetryDB consumption — operator can
                       request a re-run of an already-completed seed)
```

Self-loops are no-ops: calling `transition_to(current_state)` returns
without mutating fields and without emitting an event. This makes the
helper safe to call defensively from places that already-or-might-have
already transitioned.

Every other edge raises `ValueError` with both states named. The graph
is encoded as a class-level `_ALLOWED_TRANSITIONS: dict[str, set[str]]`
on `JobEntry`.

### 4.2 The `JobEntry.transition_to` API

```python
def transition_to(
    self,
    target: str,
    *,
    failure: FailureInfo | None = None,
    assigned_to: str | None = None,
    now: str | None = None,
    reset_attempts: bool = False,
    add_preferred_not: str | None = None,
) -> bool:
    """Transition this job to ``target``.

    Validates the (current, target) edge against
    :data:`_ALLOWED_TRANSITIONS`; raises ``ValueError`` for illegal
    edges, returns ``False`` for self-loops (no-op), returns ``True``
    after applying the field updates for legal cross-state edges.

    Per-target field contract:

    - ``pending``:  clears ``started_at``, ``ended_at``, ``assigned_to``,
      ``failure``, ``running_substate``. Optionally resets ``attempts``
      and adds ``add_preferred_not`` to the preferred-not set.
    - ``running``:  requires ``assigned_to``. Sets ``started_at = now``,
      clears ``ended_at`` and ``failure``, sets ``running_substate =
      "training"``. Does **not** touch ``attempts`` — the worker
      increments the counter at submit time (see §4.6) before the
      runner-side ``transition_to`` call observes the "started" event.
    - ``completed``: sets ``ended_at = now``. Clears ``failure`` and
      ``running_substate``. ``failure`` argument MUST be ``None``.
    - ``failed``:   sets ``ended_at = now``. Requires ``failure`` to
      be a ``FailureInfo``. Clears ``running_substate``.

    ``now`` defaults to ``_utc_now_iso()`` when ``None``."""
```

Invariants the helper enforces (asserted before applying updates):

1. `target == "running"` requires `assigned_to is not None`.
2. `target == "failed"` requires `failure is not None`.
3. `target == "completed"` rejects a non-None `failure` (callers never
   need to pass it; legacy code that does is a bug).
4. `add_preferred_not` is only honored when `target == "pending"`.

The helper does **not** post `StateEvent`s. State events are still the
runner-visible side effect; the helper only mutates the JobEntry. The
worker's existing `self._state_chan.put(StateEvent(...))` calls move
to immediately after the transition_to call. This keeps two concerns
separate: in-memory model (transition_to) vs. main-thread observation
(StateEvent on the runner's queue).

### 4.3 Field invariants on serialization

`tools/odin/asgard/state.py:write_dispatch_state` already serializes
`JobEntry` to `dispatch.json`. Add an invariant check before writing:

- `status in {"completed", "failed"}` ⇒ `ended_at is not None`
- `status == "failed"` ⇒ `failure is not None`
- `status == "running"` ⇒ `started_at is not None and assigned_to is not None`
- `status == "pending"` ⇒ `started_at is None and ended_at is None and assigned_to is None and failure is None`

Violations raise `AssertionError` with the offending JobEntry's
`run_id` and field values. This is a tripwire — if a code path slips
past `transition_to` and leaves a JobEntry inconsistent, the next
state-write fails loudly instead of writing a corrupt
dispatch.json.

In production, raising on serialization would risk losing dispatch
state. The check is therefore guarded by an
`ODIN_DISPATCH_STRICT_INVARIANTS` env flag (default `True` in tests,
recommended `True` in dev, can be set `False` for production
fallback). When the flag is `False`, violations are auto-repaired
(missing `ended_at` filled with `_utc_now_iso()`, missing `failure`
replaced with a stub `FailureInfo(kind="unknown",
message="state-write invariant violation; see logs")`) and a WARN is
logged.

### 4.4 Bug 1 supplemental fix — tighten `_manifest_indicates_clean_completion`

The reconcile classifier currently accepts any manifest whose
`phases` dict is non-empty and where each present phase is
`completed exit=0`. Replace with:

```python
def _manifest_indicates_clean_completion(manifest: dict) -> bool:
    phases = manifest.get("phases", {})
    # Require both phases to be present AND completed with exit 0.
    # The Hugin manifest writer stamps a placeholder
    # ``phases.startup = completed`` before startup runs, so a
    # manifest with only the startup key cannot be trusted as a
    # clean completion: the run may have crashed before training
    # was even invoked.
    required = {"startup", "training"}
    if not required.issubset(phases.keys()):
        return False
    return all(
        phases[k].get("status") == "completed" and phases[k].get("exit_code") == 0
        for k in required
    )
```

This is independent of the FSM refactor and can land first; the
`_validate_bundle` path in worker.py (which already requires both
phases via the training-phase content check it added for SIGKILL'd
orphans) already gets this right. The change brings reconcile in
line with worker.

### 4.5 Bug 2 supplemental fix — `running_substate`

Add a new field on `JobEntry`:

```python
running_substate: str | None = None  # "training" | "pulling_bundle" | None
```

Default `None`. Set to `"training"` when transitioning to `running`
(via `transition_to`'s field contract above). The worker flips it
to `"pulling_bundle"` immediately before invoking
`rsync.pull(...)` in `_finalize_terminal`, then transitions to the
terminal state (which clears the substate as part of its field
contract).

The flip is a separate state-event:

```python
self._state_chan.put(
    StateEvent(
        run_id=job.run_id,
        host=self.host.host,
        transition="finalizing",   # new transition kind
        running_substate="pulling_bundle",
    )
)
```

State-event consumption updates the JobEntry's `running_substate` on
the runner's main thread before the next `dispatch.json` write — same
pattern as today's `transition="started"` events.

The dashboard's render layer reads `running_substate` and shows
"pulling bundle" next to the status pill while it's set. No change
to the underlying `status` field. No change to the truth model: the
job is still authoritatively `running` until rsync + validate
succeed.

### 4.6 Reset-in-place vs. fresh-allocate of `attempts`

`attempts` is incremented by the worker at submit time
(`worker.py:655` / `:746`), *before* the runner observes the
"started" StateEvent and applies the corresponding `running`
transition. The helper therefore does **not** touch `attempts` on
the `running` edge — modifying it there would double-count.

The `retry-all-failed` CLI flag historically resets `attempts = 0`
(`runner.py:342`). To preserve that behavior, `transition_to(target=
"pending", reset_attempts=True)` zeroes the counter; the live-retry
path uses the same flag. Default is to preserve the counter, so the
gpu_lost recovery re-queue path (which used to leak attempts) keeps
its accumulated count.

## 5. Migration plan

The 25+ existing call sites get rewritten in three batches. Each
batch is one focused PR/commit; tests pass after each.

### 5.1 Batch 1 — Add the helper + wire `_emit_failed`

- Add `_ALLOWED_TRANSITIONS` and `transition_to` to `JobEntry` in
  `tools/odin/asgard/jobs.py`.
- Add unit tests for the helper covering each legal edge, each
  illegal edge, the self-loop no-op, and each per-target field
  contract.
- Replace `worker._emit_failed`'s body with a single
  `transition_to(target="failed", failure=failure)` call, leaving
  the StateEvent post in place. This proves the helper integrates
  with the existing worker before touching every other site.

### 5.2 Batch 2 — Sweep all 25 call sites

Catalog and replacement:

| File | Line | Current state-write | Replacement |
|---|---|---|---|
| reconcile.py | 222 | `j.status = "completed"` | `j.transition_to("completed")` |
| reconcile.py | 226 | `j.status = "failed"; j.failure = …` | `j.transition_to("failed", failure=…)` |
| reconcile.py | 242 | `j.status = "completed"` | `j.transition_to("completed")` |
| reconcile.py | 252 | `j.status = "failed"; j.failure = _classify_pulled_bundle(…)` | `j.transition_to("failed", failure=…)` |
| reconcile.py | 259 | `j.status = "pending"` | `j.transition_to("pending")` |
| reconcile.py | 269 | `j.status = "pending"` | `j.transition_to("pending")` |
| reconcile.py | 274 | `j.status = "pending"` | `j.transition_to("pending")` |
| reconcile.py | 287 | `job.status = "failed"; job.failure = …` | `transition_to("failed", failure=…)` |
| worker.py | 807 | `job.status = "failed"; job.ended_at = …; job.failure = …` | `transition_to("failed", failure=…)` |
| worker.py | 833 | same pattern | same replacement |
| worker.py | 846 | same pattern | same replacement |
| worker.py | 854 | `job.status = "completed"; job.ended_at = …` | `transition_to("completed")` |
| worker.py | 1137 | `job.status = "completed"; job.ended_at = …` | `transition_to("completed")` |
| worker.py | 1174 | (inside `_emit_failed`, see Batch 1) | already replaced |
| worker.py | 684–702 | gpu_lost recovery: re-queue without status flip | `transition_to("pending", add_preferred_not=…)` then re-queue. **This is the Bug 3 fix.** |
| runner.py | 216 | `j.status = "running"` | `transition_to("running", assigned_to=host)` |
| runner.py | 226 | `j.status = "completed"` | `transition_to("completed")`. **This is the Bug 4 fix.** |
| runner.py | 237 | `j.status = "failed"; j.failure = …` | `transition_to("failed", failure=…)` |
| runner.py | 270 | `j.status = "pending"` | `transition_to("pending")` |
| runner.py | 306 | `j.status = "failed"` | `transition_to("failed", failure=…)` |
| runner.py | 334 / 340 | retry-failed/retry-all-failed flip | `transition_to("pending", reset_attempts=…)` |
| runner.py | 364 | `job.status = "pending"` | `transition_to("pending")` |
| runner.py | 438 | `job.status = "failed"` | `transition_to("failed", failure=…)` |
| runner.py | 727 | `j.status = "failed"` | `transition_to("failed", failure=…)` |

The non-`JobEntry` `f.status = ...` lines in `runner.py` (216, 221,
230, 242, 260) are `FleetEntry` mutations, not `JobEntry` — they
stay as-is. The audit only covers `JobEntry`.

### 5.3 Batch 3 — Targeted Bug 1 + Bug 2 fixes

- Replace `_manifest_indicates_clean_completion` per §4.4.
- Add `running_substate` field per §4.5; flip it in
  `worker._finalize_terminal` immediately before rsync.pull; teach
  the runner state-event consumer to apply it; teach the dashboard
  jobs-table renderer to show "pulling bundle" badge.

### 5.4 Batch 4 — Strict-invariants tripwire

- Add the serialization-time invariant check per §4.3.
- Add an env-flag opt-out for production safety.
- Add unit test that crafts each invariant violation and confirms it
  raises in strict mode and auto-repairs (with WARN) in lenient
  mode.

## 6. Testing

### 6.1 Unit tests for `transition_to` (Batch 1)

Coverage matrix:

- One test per legal edge (8 edges × 1 test each).
- One test per illegal edge family: `completed → running`,
  `completed → failed`, `failed → running`, `failed → completed`,
  `running → running` (self-loops are not illegal — they no-op).
- Per-target field contract: each target verified to clear /
  set the documented fields, and reject illegal arguments
  (`"completed"` with non-None failure, `"running"` without
  `assigned_to`, etc.).
- Self-loop: every state's `transition_to(self.status)` returns
  `False` and leaves all fields untouched.

### 6.2 Regression tests for the four bugs

- Bug 1: Construct a manifest with only `phases.startup = completed`
  and confirm `_manifest_indicates_clean_completion` returns
  `False`. Confirm the legacy three-phase healthy manifest still
  returns `True`.
- Bug 3: Drive a worker through the gpu_lost recovery path with a
  fake SSH that simulates probe failure → recovery success.
  Assert that the JobEntry comes out of recovery with
  `status="pending"`, `started_at=None`, `assigned_to=None`,
  `attempts` unchanged, `preferred_not` unchanged. (The gpu_lost-
  recovery-failed branch already adds the host to preferred_not;
  that's preserved.)
- Bug 4: Drive `reconcile_orphans` through the
  `manifest-indicates-clean-completion` path and assert the
  resulting JobEntry has `status="completed"` AND
  `ended_at != None`.
- Bug 2: Render `jobs_table.py` with a JobEntry where
  `running_substate="pulling_bundle"` and assert the "pulling
  bundle" badge appears.

### 6.3 End-to-end loopback

The existing `tests/test_asgard_integration.py` loopback dispatch
covers the happy path. Add one new loopback test: a job hits
gpu_lost on first submit, recovers, completes on second attempt.
Assert dispatch.json's final state has the job at `completed`,
`attempts=2`, no orphaned `running` rows. This is the integration
analog of the Bug 3 unit test — guards against regressions in the
re-queue plumbing.

### 6.4 Strict-invariants tripwire (Batch 4)

Existing tests that write `dispatch.json` should run under strict
mode by default and pass. Add three negative tests that craft an
invariant violation (terminal status with no `ended_at`, etc.),
attempt a state write, and confirm the strict mode raises. Set
`ODIN_DISPATCH_STRICT_INVARIANTS=False` and re-run; confirm the
auto-repair path runs and a WARN is logged.

## 7. Risks and open questions

- **Risk: a transition we don't currently recognize.** The 25-site
  catalog is grep-derived; if any state mutation hides behind
  attribute access via a string (e.g.,
  `setattr(job, "status", ...)`) it will be missed. Mitigation: a
  pre-merge audit (`ripgrep "JobEntry.*status"` and
  `ripgrep "\.status\s*="` across the asgard tree).
- **Risk: production strictness blowing up live dispatches.** The
  `ODIN_DISPATCH_STRICT_INVARIANTS=False` env flag exists for this.
  Default to strict in tests and dev; document the prod toggle.
- **Open question: should `running → running` be an error or a
  no-op?** Currently spec says no-op (idempotent helper). Two
  legitimate uses: re-attaching to an already-running orphan after
  reconcile, and the worker's hypothetical "extend started_at"
  retry path. If the no-op is too permissive, we can tighten to
  "raise unless `assigned_to` matches" — but that's stricter than
  any current call site needs.
- **Open question: do we want `completed → pending` as a legal
  edge?** Live-retry (RetryDB) lets the operator request a re-run
  of any job, including one that's already completed. The current
  code (`_consume_live_retries`) only flips `failed` to `pending`,
  not `completed` to `pending`. If we ever add "re-run a completed
  seed" we need this edge. For now: include in the graph but
  unused by any caller; cheap insurance.

## 8. Out of scope (filed separately)

- `project_odin_periodic_preflight.md` — fleet watcher loop +
  auto-recovery on health flips.
- `project_odin_arm_nccl_shadow.md` — bake the NCCL symlink into
  bootstrap.
- `project_odin_arm_nvrtc_builtin.md` — sibling NVRTC builtin
  symlink in bootstrap.
- `project_odin_rsync_no_timeout.md` — add rsync stalled-data-
  timeout + drop `-z` on already-binary bundles.

This spec depends on none of those; all four are independent.
