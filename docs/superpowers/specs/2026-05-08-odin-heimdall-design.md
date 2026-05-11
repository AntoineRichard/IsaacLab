# Heimdall — Odin fleet watcher

**Date:** 2026-05-08
**Status:** Design
**Branch:** `worktree-heimdall-design` (worktree off `antoiner/feat/odin`)
**Spec author:** Antoine Richard

## Summary

Add **Heimdall**, an Odin component that periodically re-probes the Asgard fleet and watches for stale jobs while a dispatch is in flight. Closes two existing gaps:

1. The dispatcher's preflight runs once at startup; hosts that wedge mid-dispatch (NVML loss, SSH death, GPU drop) go undetected until the next dispatcher restart. Real incident: dispatch `20260430-110509` ran 22h with two hosts wedged and no work assigned to them.
2. Jobs that wedge in transport (rsync hang) or have a wedged worker thread sit in `running` indefinitely, misleading the dashboard and tying up host slots.

Named after Heimdall, watchman of Asgard with supernatural senses (the role of fleet watcher fits the canonical Norse mythology more tightly than Mimir, who is held in reserve for a wisdom/oracle component later).

## Motivation

Two TODOs converge here:

- *Periodic re-preflight + fleet-watcher* — auto-recovery on the initial probe landed in commit `40c964696d9` (2026-05-05), but only fires at startup. Real fleets need a periodic re-probe so wedges are detected and self-healed without operator intervention.
- *Stale-job detection* — jobs holding `status="running"` after their trainers have ended (rsync wedge, worker stuck in SSH call) make the dashboard misleading and block the in-flight slot. Observed 2026-05-05: three RGB-Camera bundles took ~1h each to rsync back, holding `running` for an extra hour.

Both are "make the dispatcher tolerate slow / partially-wedged hosts." Solving them together is cheaper than separately because they share the watcher loop, the persistence file, and the dashboard surface.

## Goals

- Detect mid-dispatch host wedges within ~10 min worst case (K=2 consecutive probe failures at 300s cadence) and either auto-recover or quarantine + re-queue assigned jobs.
- Detect stale jobs (worker thread alive but not making progress) within ~3 min of the last heartbeat.
- Branch the response on host health: stale + healthy host = trainer wedge (no retry); stale + unhealthy host = host wedge (infrastructure retry).
- Surface watcher state to the Valhalla dashboard as a panel beside the fleet table.
- Persist host health and recent watcher activity to a new `fleet.json` so the dashboard and a future restart can read it without polling SSH.

## Non-goals

- Not a separate sidecar process. Heimdall lives inside the dispatcher process; if the dispatcher exits, Heimdall exits with it. (A sidecar variant is plausible later but adds two-writer file-locking complexity for limited benefit today.)
- Not a periodic full-preflight chain. Probes are cheap (`nvidia-smi -L`, ssh-alive). The full preflight stays at startup.
- Not auto-restart of a quarantined host. Once Heimdall has tried `recover_valkyrie_gpu` and failed, the host stays quarantined for the rest of the dispatch. Operator un-quarantines manually.
- Not a transport-layer hardening (rsync `--timeout`, dropping `-z`). That work is tracked separately in `project_odin_rsync_no_timeout` and remains useful even with Heimdall, since Heimdall is detection-of-last-resort, not the primary fix.
- Not a replacement for `reconcile_orphans`. Heimdall fills the *during-dispatch* hole; reconcile still runs at startup. The two are complementary; reconcile gains a small change to attempt host recovery before flipping `running` jobs to `pending`.

## Architecture

A single in-process **HeimdallWatcher** thread owns probe cadence; the dispatcher's main loop is the sole writer of `DispatchState`.

```
┌──────────────────── dispatcher process ────────────────────┐
│                                                            │
│  ┌─ HeimdallWatcher (daemon thread) ─────────────────┐     │
│  │  every ~5 min:                                    │     │
│  │   • parallel-probe all hosts (thread-per-host),   │     │
│  │     reusing preflight's executor pattern          │     │
│  │   • compute stale jobs from latest DispatchState  │     │
│  │     view: now - JobEntry.last_heartbeat_at        │     │
│  │     > stale_threshold_s                           │     │
│  │   • publish HeimdallSnapshot under a lock         │     │
│  │   • atomic-write fleet.json                       │     │
│  └───────────────────────┬───────────────────────────┘     │
│                          │ snapshot                        │
│  ┌─ run_dispatch main loop ──────────────────────────┐     │
│  │  per tick (alongside _consume_live_retries /      │     │
│  │  _consume_cancellations):                         │     │
│  │    _consume_heimdall_snapshot(snap, state, …)     │     │
│  │      • flips → recovery → quarantine + re-queue   │     │
│  │      • stale jobs → kill + classify + emit event  │     │
│  │      • mark snap.generated_at as consumed         │     │
│  └───────────────────────────────────────────────────┘     │
│                                                            │
│  ┌─ worker thread (per Valkyrie) ────────────────┐         │
│  │  while job in flight:                         │         │
│  │   spawn heartbeat daemon → emit Event.        │         │
│  │   heartbeat(run_id, ts) every ~30s            │         │
│  │  on terminal-state entry: stop heartbeat      │         │
│  └───────────────────────────────────────────────┘         │
│                                                            │
│  ┌─ Valhalla dashboard (separate process, reads only) ─┐   │
│  │  dispatch_fleet/heimdall_card.py renders            │   │
│  │  fleet.json + dispatch event log.                   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Why the worker thread is the heartbeat source

The worker thread that owns a job is already the authoritative "this job is making progress" sentinel. If it wedges in rsync or any SSH call, heartbeats stop — *for free*, no remote-side cooperation needed. This means:

- No SSH probe per running job per cycle. With N hosts and M jobs/host, that scales as O(N·M); heartbeats scale as O(1) — we read in-memory state.
- No new file on the remote filesystem. No staleness-of-mtime ambiguity.
- A genuine trainer crash that the worker thread *handles cleanly* still reaches a terminal state via the existing path; the heartbeat case catches the worker-stuck variant specifically.

The heartbeat is delivered as a `state event`, identical in shape to the events the worker already emits for `started` / `progress` / terminal — bumping a single new field on `JobEntry`.

## Components

### New: `tools/odin/asgard/heimdall.py`

```python
@dataclass(frozen=True)
class HostHealth:
    name: str
    healthy: bool
    last_probe_at: str               # ISO-8601 UTC
    consecutive_failures: int        # for K-failure flip gate
    failure_reason: str | None       # e.g. "ssh_timeout", "nvml_missing"
    recovery_attempts: int
    recovery_history: list[str]      # last ≤5 attempts as ISO timestamps
    quarantined: bool

@dataclass(frozen=True)
class StaleJob:
    run_id: str
    host: str
    last_heartbeat_at: str
    age_seconds: float
    host_was_healthy: bool           # decides timeout vs infra classification

@dataclass(frozen=True)
class HeimdallSnapshot:
    generated_at: str
    hosts: dict[str, HostHealth]
    stale_jobs: list[StaleJob]
    recent_events: list[dict]        # ring buffer, last ≤20 actions

class HeimdallWatcher:
    """Periodically probes the fleet and publishes snapshots.

    The watcher is the sole writer of fleet.json. Consumers (dispatcher
    main loop, Valhalla dashboard) read snapshots; they never mutate
    HeimdallWatcher state.
    """

    def __init__(
        self,
        fleet: Fleet,
        dispatch_dir: Path,
        ssh: SSHRunner,
        state_view: Callable[[], DispatchState],   # returns a copy
        *,
        probe_interval_s: int = 300,
        stale_threshold_s: int = 180,
        flip_after_k_failures: int = 2,
        probe_timeout_s: int = 15,
    ) -> None: ...

    def start(self) -> None: ...                  # spawns daemon thread
    def stop(self, timeout_s: float = 10.0) -> None: ...
    def latest(self) -> HeimdallSnapshot | None: ...   # thread-safe read
    def is_alive(self) -> bool: ...               # for crash detection
```

`fleet.json` schema (single owner = HeimdallWatcher):

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-05-08T14:32:18Z",
  "hosts": { "<name>": { ...HostHealth... } },
  "recent_events": [
    {"ts": "...", "kind": "host_flipped",   "host": "175", "reason": "ssh_timeout"},
    {"ts": "...", "kind": "host_recovered", "host": "175"},
    {"ts": "...", "kind": "host_quarantined","host": "183","reason": "recovery_failed"},
    {"ts": "...", "kind": "stale_job_killed","run_id": "...","classification": "infrastructure"}
  ]
}
```

### Touched: `tools/odin/asgard/runner.py`

- Start the watcher inside `run_dispatch` after `_snapshot_fleet_yaml` and before the main dispatch loop.
- Stop it in the `finally` block alongside other cleanup.
- Add `_consume_heimdall_snapshot(snap, state, fleet, ssh, rsync, *, on_event, last_consumed_at)` — analogous to the existing `_consume_live_retries`. Idempotent via `last_consumed_at == snap.generated_at` short-circuit.
- Watcher liveness check: if `watcher.is_alive()` returns False, log a single warning and continue without Heimdall-driven recovery. No auto-restart.

### Touched: `tools/odin/asgard/worker.py`

- Spawn a heartbeat daemon thread when transitioning a job to `running`.
- The heartbeat thread emits `Event.heartbeat(run_id, ts=utc_now_iso())` every 30s using the same event-queue mechanism the worker already uses.
- The heartbeat thread stops when the worker enters its terminal-state finalization path (signal via `threading.Event`, set just before the rsync pull begins). Stop is idempotent and safe to call from any terminal path.

### Touched: `tools/odin/asgard/state.py`

- Add `last_heartbeat_at: str | None = None` to `JobEntry`.
- Add `heartbeat` to the event vocabulary handled by `_apply_state_event` — handler bumps `last_heartbeat_at`, no other state changes.
- Bump dispatch-state schema minor (e.g., `1.4.0 → 1.5.0`). `_schema_version_compatible` already permits same-major; resume from older dispatches treats `last_heartbeat_at` as `None`, which the consumer interprets as "use job's `started_at` as the floor for the staleness clock."

### Touched: `tools/odin/asgard/reconcile.py`

- In `reconcile_orphans`, before flipping a `running` job to `pending` because its host is unreachable, attempt one `recover_valkyrie_gpu` call. On success, leave the job alone (it'll be picked up by the next `_consume_heimdall_snapshot`). On failure, fall through to the existing flip-to-pending behavior.
- This addresses the canonical "today's mishap" case from `project_odin_periodic_preflight.md` where 3 Shadow-Vision rows were flipped to pending unnecessarily.

### New: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/heimdall_card.py`

- Renders a panel beside `fleet_table.py`:
  - Header: `Heimdall — last look-up: <ts> (<age> ago)`.
  - Per-host status row with icon + reason: ✓ healthy / ✗ unhealthy (`reason`) / ⚠ recovering / ⛔ quarantined.
  - Recent activity (last 5 entries from `fleet.json:recent_events`).
  - Stale-job count badge if `len(stale_jobs) > 0`.
- `layout.py`: add `html.Div(id="tab-a-heimdall-card")` next to `tab-a-fleet-table`.
- `callbacks.py`: register a callback at the same polling cadence as the fleet table; reads `fleet.json` from the dispatch dir; gracefully renders an empty state if the file is missing (older dispatches pre-Heimdall).

## Data flow

### Probe cycle (every `probe_interval_s`, default 300)

```
HeimdallWatcher tick:
  ├── for each host in fleet (parallel, thread-per-host):
  │     try ssh `nvidia-smi -L` (timeout = probe_timeout_s)
  │     on success: HostHealth(healthy=True, consecutive_failures=0)
  │     on failure: increment consecutive_failures
  │                 healthy = consecutive_failures < flip_after_k_failures
  │                 (i.e. flip only after K consecutive failures)
  ├── snapshot DispatchState via state_view()
  │     for each JobEntry where status == "running":
  │       baseline = last_heartbeat_at or started_at
  │       age = now - baseline
  │       if age > stale_threshold_s:
  │         host_was_healthy = hosts[job.host].healthy
  │         stale_jobs.append(StaleJob(...))
  ├── snapshot = HeimdallSnapshot(now, hosts, stale_jobs, recent_events)
  ├── publish snapshot under self._lock
  └── atomic-write fleet.json (write to .tmp, fsync, rename)
```

### Main-loop consumer (next dispatch tick)

```
snap = watcher.latest()
if snap is None or snap.generated_at == last_consumed_at:
    return

for host_name, h in snap.hosts.items():
    prev = state.fleet_health.get(host_name)
    if prev and prev.healthy and not h.healthy:                # flip
        rec = recover_valkyrie_gpu(host, ssh)
        if rec.success:
            on_event(Event.host_recovered(host_name))
        else:
            on_event(Event.host_quarantined(host_name, reason=rec.reason))
            for job in state.jobs:
                if job.host == host_name and job.status == "running":
                    on_event(Event.job_requeued(
                        job.run_id, classification="infrastructure"))

for sj in snap.stale_jobs:
    if state.jobs[sj.run_id].status != "running":
        continue                                               # already terminal
    best_effort_kill_remote(sj.host, sj.run_id, ssh, timeout_s=10)
    if sj.host_was_healthy:
        on_event(Event.job_failed(sj.run_id, classification="timeout"))
    else:
        on_event(Event.job_requeued(sj.run_id, classification="infrastructure"))

last_consumed_at = snap.generated_at
```

### Worker heartbeat (per running job)

```
worker thread: transition job → running
  spawn heartbeat daemon (stop_event = threading.Event())
  heartbeat loop (every 30s, until stop_event):
    on_event(Event.heartbeat(run_id, ts=utc_now_iso()))

worker thread: enter terminal-state finalization (before rsync pull)
  stop_event.set()
  heartbeat thread observes and exits
  proceed with rsync pull, terminal state event
```

`Event.heartbeat` flows through the same single-writer state-event queue the worker already uses; `_apply_state_event` handles it by bumping `JobEntry.last_heartbeat_at` and writing through to disk on the existing flush cadence.

## Error handling

**Probe failures are not host failures.** A single SSH timeout doesn't flip a host to unhealthy. K consecutive failures (default K=2) are required. Avoids flapping on transient network blips.

**Recovery is at-most-once per flip.** When a host flips healthy→unhealthy, `recover_valkyrie_gpu` runs once. On failure, host quarantines for the rest of the dispatch. Operator can un-quarantine via existing `recovery_cli` flag if they fix the host. We do not retry recovery in a loop — that path was already considered and rejected as a tight-loop hazard during the initial-probe auto-recovery design.

**Best-effort kill is genuinely best-effort.** When killing a stale job's remote process, the SSH command itself can hang (wedged host). Wrap with a 10s timeout; on timeout log it, proceed with the state-event flip anyway. Remote-process leakage is acceptable — `reconcile_orphans` catches stragglers at the next dispatcher restart.

**Watcher thread crash must not silently degrade the dispatcher.** Main loop checks `watcher.is_alive()` before `watcher.latest()`. If dead, log one warning and continue without Heimdall (no regression vs. today's behavior). We do *not* auto-restart the watcher; loud failure beats a silent retry loop that masks bugs.

**`fleet.json` write failures are non-fatal.** Atomic write-then-rename; on failure log and retry next cycle. The in-memory snapshot is the source of truth for action; `fleet.json` is for the dashboard.

**Heartbeat without a running job.** If `Event.heartbeat` arrives for a `run_id` that's already in a terminal state, `_apply_state_event` ignores it. Race-tolerant by design.

**State view race.** `state_view()` returns a copy. The watcher computes stale-job candidates against a snapshot that may be 0–N seconds old by the time the main loop acts. The consumer re-checks `state.jobs[run_id].status == "running"` before any kill/event. Skipping silently when the job has already terminated is correct.

## Testing strategy

Following the existing asgard test pattern (`tools/odin/tests/asgard/test_*.py`, ~120 unit tests, one slow-marked integration test).

### Unit tests

- `test_heimdall_watcher.py` — fake `SSHRunner` returning scripted probe outputs; verify K-consecutive-failure flip gate, recovery-attempt accounting in `HostHealth`, snapshot publication ordering, fleet.json round-trip (write + read), atomic write semantics under failure injection.
- `test_heimdall_consumer.py` — `_consume_heimdall_snapshot` against synthetic `HeimdallSnapshot` + `DispatchState`. Cover:
  - healthy→unhealthy flip with successful recovery: emits `host_recovered`, no requeue.
  - healthy→unhealthy flip with failed recovery: emits `host_quarantined` + N `job_requeued` (one per running job on host).
  - stale job + host healthy: emits `job_failed(timeout)`, no requeue.
  - stale job + host unhealthy: emits `job_requeued(infrastructure)`.
  - already-terminal job in snapshot: skipped silently.
  - idempotent re-consumption when `generated_at` matches `last_consumed_at`.
- `test_worker_heartbeat.py` — heartbeat thread emits at expected cadence (use a fake clock); stops promptly on terminal-state signal; doesn't outlive worker thread; double-stop is a no-op.
- `test_state_heartbeat_event.py` — `_apply_state_event` correctly bumps `last_heartbeat_at`; minor schema bump round-trips through `read_dispatch_state` / `write_dispatch_state`; resume from a pre-Heimdall dispatch state gracefully treats missing `last_heartbeat_at` as `None`.

### Regression tests (must fail without the fix — verify by temporary revert)

- *22h-wedge incident* — simulate a host that probes healthy at startup, then flips unhealthy mid-dispatch, with two jobs assigned. Verify Heimdall detects flip + quarantines + re-queues. Without the watcher, the test must hang or time out, confirming the new path is actually exercised. Implementation: a `FakeFleet` whose probe returns healthy for the first call and unhealthy thereafter.
- *Reconcile-after-restart 3-Shadow-Vision case* — pre-populate state with a `running` job whose host is reachable. Trigger `reconcile_orphans`. Verify host recovery is attempted *before* the job is flipped to `pending`, and that on recovery success the job is left in `running`.

### Integration test (slow-marked, `ssh localhost` required)

One full dispatch with two loopback hosts. The first host has a sentinel file the probe checks; flipping the file mid-dispatch makes the probe fail. Verify end-to-end: probe flips, recovery attempted, quarantine, jobs re-queue to the second host, second host succeeds. Skip when `ssh localhost` is unavailable, matching existing convention.

### Dashboard tests

Snapshot tests of `heimdall_card.py` against three `fleet.json` fixtures: all-healthy, mixed (one quarantined + one unhealthy + one healthy), all-quarantined. Verify the empty-state path when `fleet.json` is missing.

## Migration / rollout

- Schema bump on `dispatch.json` is minor; existing in-flight dispatches resume cleanly. Resumed jobs without `last_heartbeat_at` use `started_at` as the staleness baseline until the next heartbeat.
- Watcher is opt-out via a `--no-heimdall` CLI flag for the first release, in case probe overhead or false positives surface in the wild. Default on.
- Dashboard renders an empty Heimdall card when `fleet.json` is absent — older dispatches in the archive don't break.

## Open questions

- **Probe interval default.** 300s is the value sketched in the original TODO. Worth measuring in practice on a real fleet — too long, we miss short wedges; too short, we burn fleet time on probes. 300s is a reasonable starting point.
- **Heartbeat cadence vs. event flood.** 30s × N concurrent jobs adds N writes to the dispatch event queue every 30s. On a 10-host fleet with 10 jobs each that's 200 events/min. The existing event queue has comfortably handled live-retry / cancellation traffic; should be fine, but worth confirming during implementation that `_apply_state_event` for `heartbeat` is cheap (no full state-write per event — flush-on-cadence only).
- **Should `fleet.json` accumulate across resumes?** The schema includes `recovery_history` per host. If a host wedges, recovers, wedges again on the same dispatch, history captures both. On resume, history is reloaded. Cap at 5 entries per host to bound size.

## File touch summary

New files:
- `tools/odin/asgard/heimdall.py`
- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/heimdall_card.py`
- `tools/odin/tests/asgard/test_heimdall_watcher.py`
- `tools/odin/tests/asgard/test_heimdall_consumer.py`
- `tools/odin/tests/asgard/test_worker_heartbeat.py`
- `tools/odin/tests/asgard/test_state_heartbeat_event.py`
- `tools/odin/tests/asgard/test_reconcile_recovery_first.py` (regression)
- `tools/odin/tests/asgard/test_heimdall_e2e.py` (slow-marked integration)
- `tools/odin/tests/valhalla/test_heimdall_card.py`

Modified files:
- `tools/odin/asgard/runner.py` (start/stop watcher, add `_consume_heimdall_snapshot`)
- `tools/odin/asgard/worker.py` (heartbeat thread)
- `tools/odin/asgard/state.py` (add `last_heartbeat_at`, `heartbeat` event, schema bump)
- `tools/odin/asgard/reconcile.py` (try recovery before flipping `running` to `pending`)
- `tools/odin/asgard/cli.py` (`--no-heimdall` flag)
- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py` (add card div)
- `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py` (register card callback)

Memory follow-ups (after merge):
- Update `project_odin_periodic_preflight.md` to reference Heimdall as the landed solution.
- Update `project_odin.md` component list under "Component naming reserved": move Heimdall from reserved-but-unused to landed; note that Mimir remains reserved.
