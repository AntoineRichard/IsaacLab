# Odin T3.1 — Headless Dispatch Across Asgard Design

**Project:** Odin (multi-backend IsaacLab evaluation harness)
**Task:** T3.1 — headless end-to-end distributed dispatch (Asgard + Valkyries + Bifrost)
**Date:** 2026-04-22
**Branch:** `antoiner/feat/odin`
**Status:** Draft — pending user review

## Context

T1 delivered the per-run benchmark pipeline (one bundle per
`(framework, backend, task, seed)` tuple). T2.1 delivered the curated
env lists (`physx_envs.yaml`, `newton_envs.yaml` with `keep: true` on
the rows we want benchmarked). T2.2 closed the remaining reliability
caveats in the Layer-1 outputs.

T3 is the Layer-3 dispatcher: given a fleet of SSH-accessible
Valkyrie machines, a dispatcher reads the curated env lists, spawns one
Hugin/Munin per `(task, seed)` across the fleet, monitors them,
collects the bundles back, and handles retries. `eval_plan.md`
describes T3 as a single task; during T3 brainstorming (2026-04-22) we
split it into two sub-specs:

- **T3.1 (this spec) — headless end-to-end dispatch.** Library +
  thin CLI. Bring-up, queue, dispatch, monitor, collect, retry.
  No UI.
- **T3.2 (deferred) — local web UI.** Read-only view on the T3.1
  on-disk state. May fold into T4 / Valhalla instead of shipping as a
  standalone deliverable.

T3.1 is entirely Odin-side (`tools/odin/asgard/`). No
`source/isaaclab*` touches; the `./docker/container.py` interface is
used as-is.

## Goals & non-goals

**In scope:**

- A Python library **`tools/odin/asgard/`** + a thin CLI that:
  - Reads `fleet.yaml` (list of Valkyrie hosts with per-host
    overrides) and the T2.1 env YAMLs.
  - Expands each `keep: true` / non-stale row across a CLI-supplied
    seed list into one `JobEntry` per `(row, seed)`.
  - **Provisions** each Valkyrie via rsync of the controller's
    working tree and `./docker/container.py start` (smart-sync by
    default; `--fresh` wipes and re-clones).
  - **Preflight** checks every host (SSH reach, docker running,
    container up, IsaacLab present) before any job dispatches;
    fails fast unless `--skip-preflight`.
  - **Dispatches** jobs concurrently (one thread per Valkyrie)
    over SSH-into-docker (`docker exec`), tees remote stdout to a
    per-run log on the controller, waits for the remote process
    to exit, rsyncs the bundle directory back.
  - **Retries** infrastructure failures (SSH / docker exec before
    Hugin starts) up to `--max-infrastructure-retries` (default 2).
    Does NOT retry Hugin crashes, malformed bundles, or timeouts;
    those mark the job `failed` with a classified `failure.kind`.
  - **Writes** all bundles into
    `odin_runs/<dispatch_id>/<run_id>/`; writes
    `odin_runs/<dispatch_id>/dispatch.json` with the full state
    (atomically rewritten after every transition).
  - **Resumes** when re-invoked against an existing dispatch
    directory: in-flight jobs flip back to `pending`; completed
    and failed jobs are left alone.
- Unit tests for pure dispatch logic (queue expansion, state
  transitions, retry classification, preflight aggregation) with
  injected fake SSH/rsync runners.
- One slow-marked integration test against `ssh localhost` + a
  local docker container.
- Update `docs/odin/architecture.md` (§6 task map, §9 change-log)
  at closeout.

**Out of scope:**

- **Web UI** — T3.2 or folded into T4's Valhalla dashboard.
- **Dynamic fleet changes during a dispatch.** Fleet is fixed at
  CLI start; a host going down mid-dispatch flips to `status: down`
  in `dispatch.json` but no new host is recruited.
- **Multi-job-per-Valkyrie.** One job per node at a time;
  training uses the whole GPU.
- **Smart job ordering.** Strict FIFO out of the queue. Fleet is
  homogeneous per `eval_plan.md`.
- **Multiple concurrent dispatches against the same fleet.** One
  dispatch at a time. No locking; users coordinate out-of-band.
- **Per-task timeout tuning from the YAML.** Single
  `--per-job-timeout` CLI flag (default 4h). Can be pushed into
  the YAML later if we hit recurring issues.
- **Upstream IsaacLab changes.** T3.1 is entirely Odin-side.
  `./docker/container.py` is used as-is.
- **Cross-commit retry of failed runs.** If a run fails, it
  stays failed; the user re-runs the CLI with
  `--retry-failed run_id1,...` or prunes the failed row.
  (Cross-commit tracking is T4's concern.)

**Success criteria:**

1. `odin-dispatch --fleet fleet.yaml --physx-yaml tools/odin/config/physx_envs.yaml --seeds 42`
   produces one bundle per `keep: true` PhysX row under
   `odin_runs/<dispatch_id>/` with matching `dispatch.json`
   entries marked `status: completed`.
2. Killing the controller mid-dispatch and re-invoking with
   `--resume LATEST` resumes: `running` / `assigned` jobs flip to
   `pending` and re-dispatch; `completed` / `failed` are preserved.
3. A Valkyrie without a running docker container triggers a
   preflight-stage failure before any job dispatches. Passing
   `--skip-preflight` marks it `down` and continues on the
   healthy hosts.
4. A Hugin crash on one Valkyrie produces
   `status: failed, failure.kind: "hugin_crash"` with exit code
   and log tail; the remaining queue is unaffected.
5. Unit tests cover the queue state machine and retry
   classification with no real SSH.
6. Loopback integration test (slow-marked) passes on a host
   where `ssh localhost` and the local docker container work.

## Architecture — library + threading model

### Module layout

New directory `tools/odin/asgard/`:

```
tools/odin/
└── asgard/
    ├── __init__.py               # public-API re-exports
    ├── fleet.py                  # Fleet, ValkyrieConfig, load_fleet()
    ├── queue.py                  # JobEntry, build_queue_from_env_lists()
    ├── state.py                  # DispatchState, atomic read/write
    ├── transport.py              # SSHRunner, RsyncRunner protocols + shell impls
    ├── provisioner.py            # provision_valkyrie()
    ├── preflight.py              # preflight_valkyrie()
    ├── worker.py                 # ValkyrieWorker thread
    ├── runner.py                 # run_dispatch(), DispatchOptions
    └── cli.py                    # thin CLI
```

Each file has one responsibility; soft cap ~250 lines per file.
Tests live under `tools/odin/tests/test_asgard_*.py`.

### Public API

`tools/odin/asgard/__init__.py` re-exports:

```python
from .fleet import Fleet, ValkyrieConfig, load_fleet
from .queue import JobEntry, build_queue_from_env_lists
from .state import DispatchState, read_dispatch_state, write_dispatch_state
from .runner import run_dispatch, DispatchOptions
```

`run_dispatch(...)` is the one entry point the CLI calls and the
one a future web UI (T3.2) or programmatic caller would invoke.

### Threading model

```
┌───────────────────────────────────────────────────────────────┐
│ Main thread (runs run_dispatch)                               │
│ 1. Load fleet + build queue + ensure dispatch_dir exists       │
│ 2. Run preflight: one quick SSH per Valkyrie, fail fast        │
│ 3. Spawn one ValkyrieWorker thread per Valkyrie                │
│ 4. Loop: drain state-event queue, print status lines,          │
│    periodically rewrite dispatch.json to disk                  │
│ 5. When all workers report "no more jobs" → join + exit        │
└──────────────▲──────────────────────────────▲─────────────────┘
               │ state events                 │ keep-alive heartbeat
               │ (thread-safe queue)          │ (main rewrites .json on timer)
┌──────────────┴──────────────┐  ┌────────────┴─────────────┐
│ ValkyrieWorker "valk-01"    │  │ ValkyrieWorker "valk-02"  │
│  ensure_provisioned()       │  │  ...                       │
│  loop until queue empty:    │  │                            │
│    job = queue.get()        │  │                            │
│    ssh + docker exec hugin  │  │                            │
│    tee stdout to log file   │  │                            │
│    wait for exit (timeout)  │  │                            │
│    rsync bundle back        │  │                            │
│    validate bundle          │  │                            │
│    emit state event         │  │                            │
└─────────────────────────────┘  └────────────────────────────┘
```

- **Work distribution** — single `queue.Queue[JobEntry]`, FIFO,
  all workers pull from it.
- **State events** — workers post `StateEvent(run_id, transition, ...)`
  to a thread-safe channel; the main thread drains. Workers
  never write `dispatch.json` directly.
- **Write cadence** — after every state event and on a 5s
  heartbeat, so file mtime reflects liveness even in quiet
  periods (useful for a future web UI tailing the file).
- **Graceful shutdown** — SIGINT / SIGTERM sets a shared
  `_shutting_down` flag. Workers finish their current job
  (don't kill the remote process mid-run), then exit. Main
  thread writes a final `dispatch.json` with `ended_at` set
  and any unfinished jobs flipped to `status: pending` so
  resume picks them up.
- **Retry-on-different-node hint** — an infrastructure-failed
  job re-queued with `preferred_not: {host}` hint. Workers that
  pull a job first check the hint; if their host is excluded
  and any other worker is idle, they put the job back and yield.
  Degenerate case (all other workers busy) falls back to
  "any worker can pull any job."

### Injectability for tests

```python
class SSHRunner(Protocol):
    def run(self, host: ValkyrieConfig, cmd: str,
            *, timeout_s: float | None = None,
            stdout_tee: Path | None = None) -> SSHResult: ...

class RsyncRunner(Protocol):
    def pull(self, host: ValkyrieConfig, remote_path: str, local_path: Path) -> RsyncResult: ...
    def push(self, host: ValkyrieConfig, local_path: Path, remote_path: str) -> RsyncResult: ...
```

Default impls (`ShellSSHRunner`, `ShellRsyncRunner`) shell out
to `ssh` / `rsync` via `subprocess`. Tests inject fakes.
`SSHResult` / `RsyncResult` dataclasses expose `exit_code`,
`stdout`, `stderr`, `duration_s`, `timed_out`.

SSH options baked in: `StrictHostKeyChecking=accept-new`,
`ServerAliveInterval=30`, `ConnectTimeout=10`. Explicit
identity via `-i <ssh_key>` when set.

## Data shapes

### `fleet.yaml`

```yaml
fleet_name: h100-sweep-2026-04
default_ssh_user: odinrunner
default_ssh_key: ~/.ssh/odin_id_ed25519
hosts:
  - host: valkyrie-01.internal
  - host: valkyrie-02.internal
    ssh_user: svc-odin
    isaaclab_path: /mnt/scratch/IsaacLab
  - host: 10.0.0.42
    ssh_key: ~/.ssh/alt_key
```

Resolved into per-host `ValkyrieConfig`:

```python
@dataclass
class ValkyrieConfig:
    host: str
    ssh_user: str
    ssh_key: Path | None
    isaaclab_path: str = "~/IsaacLab"
    container_name: str = "isaac-lab"
    labels: list[str] = field(default_factory=list)

@dataclass
class Fleet:
    fleet_name: str
    hosts: list[ValkyrieConfig]
```

### `JobEntry`

```python
@dataclass
class JobEntry:
    run_id: str
    task_id: str
    framework: str                   # "rsl_rl" | "skrl"
    backend: str                     # "physx" | "newton"
    num_envs: int
    max_iterations: int
    seed: int
    bundle_dir_name: str
    status: str = "pending"          # pending | assigned | running | completed | failed
    assigned_to: str | None = None
    attempts: int = 0
    failure: FailureInfo | None = None
    preferred_not: set[str] = field(default_factory=set)
    started_at: str | None = None
    ended_at: str | None = None
```

### `FailureInfo`

```python
@dataclass
class FailureInfo:
    kind: str                        # "infrastructure" | "hugin_crash" | "hugin_malformed_bundle" | "timeout"
    message: str                     # one-line human description
    details: dict[str, object]       # kind-specific: exit_code / duration_s / attempts / log_tail_path
```

### `DispatchState`

```python
@dataclass
class FleetSnapshot:
    host: str
    status: str                      # "idle" | "busy" | "down"
    current_run_id: str | None
    last_error: str | None

@dataclass
class DispatchState:
    schema_version: str              # "1.0"
    dispatch_id: str
    started_at: str
    ended_at: str | None
    seeds: list[int]
    commit_sha: str                  # from controller working tree at dispatch start; -dirty suffix if uncommitted changes
    fleet: list[FleetSnapshot]
    jobs: list[JobEntry]
```

### `DispatchOptions`

```python
@dataclass
class DispatchOptions:
    seeds: list[int]
    max_infrastructure_retries: int = 2
    per_job_timeout_s: int = 14400          # 4 h
    fresh: bool = False
    skip_preflight: bool = False
    include_filter: list[str] | None = None # fnmatch on task_id
    verbose: bool = False                   # interleave remote stdout to controller stdout
    retry_failed: list[str] | None = None   # specific run_ids to re-attempt despite failure
```

## On-disk layout

```
odin_runs/
└── 20260422-220000/                         # dispatch_id (UTC at first CLI invocation)
    ├── dispatch.json                         # DispatchState, atomically rewritten
    ├── fleet.yaml.snapshot                   # copy of fleet.yaml at dispatch start
    ├── preflight.json                        # per-host PreflightResult from opening check
    ├── rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42/
    │   ├── manifest.json                     # from Hugin, rsync'd back
    │   ├── training.json
    │   ├── startup.json
    │   ├── tb/
    │   └── logs/
    │       ├── ssh-tail.log                  # controller-side tee of remote stdout (NEW in T3)
    │       ├── startup.stderr.log            # on failure only (T1 Hugin behaviour)
    │       └── startup.stdout.log
    └── ...
```

**Additions vs T1 bundle layout:**

- `dispatch.json` — new, at dispatch-dir level.
- `fleet.yaml.snapshot` — new. "Which hosts ran this work"
  answerable months later without the live YAML.
- `preflight.json` — new. Opening-moment health of each host.
- `<run_id>/logs/ssh-tail.log` — new, appended to T1's log dir.
  Complements Hugin's own stderr/stdout logs.

### `dispatch.json` schema (v1.0)

Illustrative:

```json
{
  "schema_version": "1.0",
  "dispatch_id": "20260422-220000",
  "started_at": "2026-04-22T22:00:00Z",
  "ended_at": null,
  "seeds": [42, 43],
  "commit_sha": "0c09e96be67",
  "fleet": [
    {"host": "valkyrie-01.internal", "status": "busy",
     "current_run_id": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42",
     "last_error": null},
    {"host": "valkyrie-02.internal", "status": "idle",
     "current_run_id": null, "last_error": null},
    {"host": "valkyrie-03.internal", "status": "down",
     "current_run_id": null,
     "last_error": "preflight: docker ps timed out"}
  ],
  "jobs": [
    {
      "run_id": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42",
      "task_id": "Isaac-Ant-Direct-v0",
      "framework": "rsl_rl",
      "backend": "physx",
      "num_envs": 4096,
      "max_iterations": 300,
      "seed": 42,
      "status": "running",
      "assigned_to": "valkyrie-01.internal",
      "attempts": 1,
      "started_at": "2026-04-22T22:04:15Z",
      "ended_at": null,
      "failure": null,
      "bundle_dir_name": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42"
    },
    {
      "run_id": "skrl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42",
      "status": "failed",
      "attempts": 1,
      "failure": {
        "kind": "hugin_crash",
        "message": "exit code 1; stderr tail: 'CUDA out of memory'",
        "details": {
          "exit_code": 1,
          "log_tail_path": "skrl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42/logs/ssh-tail.log"
        }
      }
    }
  ]
}
```

**Invariants:**

- `commit_sha` resolved once at dispatch start; every Valkyrie
  is rsync'd to match. If working tree is dirty, SHA is
  suffixed `-dirty`.
- `fleet[].status` values: `idle`, `busy`, `down`. `down` = a
  host that preflight-failed OR exhausted its infrastructure
  retry budget. The entry stays in the list; length equals
  the original fleet size.
- `jobs[].status` values: `pending`, `assigned`, `running`,
  `completed`, `failed`.

### `preflight.json` schema

```json
{
  "schema_version": "1.0",
  "dispatch_id": "20260422-220000",
  "checked_at": "2026-04-22T21:59:48Z",
  "hosts": [
    {"host": "valkyrie-01.internal", "ok": true,
     "checks": {"ssh_reach": true, "docker_running": true,
                "container_up": true, "isaaclab_present": true},
     "message": ""},
    {"host": "valkyrie-03.internal", "ok": false,
     "checks": {"ssh_reach": true, "docker_running": false,
                "container_up": false, "isaaclab_present": true},
     "message": "docker daemon not responding"}
  ]
}
```

Preflight failures fatal by default. `--skip-preflight` marks
failed hosts `down` in `dispatch.json` and continues with the
healthy ones. If ALL hosts fail preflight, dispatch exits 1
unconditionally.

## Resume semantics

When the CLI is invoked against an existing `dispatch_dir`:

1. Load existing `dispatch.json`.
2. Validate — same `dispatch_id`, same `seeds`, same job list
   as would be regenerated from YAMLs + seeds. Mismatch aborts
   with a clear error (resume must match; starting fresh means
   a new `dispatch_id`).
3. Per-job state preserved as-is unless:
   - `status == running`: flip to `pending` (previous controller
     crashed; can't trust in-flight remote state). Worker
     re-dispatches.
   - `status == assigned`: flip to `pending` for the same reason.
   - `status in (completed, failed)`: left alone.
4. Fleet snapshot rebuilt fresh — preflight + re-provision as
   needed. Fleet can have changed between CLI invocations.

**Failed jobs are NOT automatically retried on resume.** To
re-attempt a specific failure, pass `--retry-failed <run_id>`
(or edit `dispatch.json` by hand to flip the status, if you
really want to). This is deliberate — a Hugin crash is usually
a real bug; retrying burns compute without diagnosis.

Dispatch-dir resolution:

```
odin-dispatch ...              → creates NEW odin_runs/<now_utc>/
odin-dispatch ... --resume 20260422-220000   → resumes that dir
odin-dispatch ... --resume LATEST            → resumes most-recent dir
```

## Provisioner flow

`provision_valkyrie(host, working_tree, *, fresh, ssh, rsync)`:

1. **Working-tree sync:**
   - `fresh=True`: `ssh host "rm -rf {isaaclab_path}"` + rsync push full tree.
   - `fresh=False`: rsync push (additive; `--delete` to prune
     removed files is an implementation detail — rsync has options).
2. **Container state:**
   - Query: `ssh host "./docker/container.py status"` (if supported)
     OR `ssh host "docker inspect {container_name}"`.
   - If container not running and `fresh=False`: start it
     (`./docker/container.py start`).
   - If `fresh=True`: stop (`./docker/container.py stop`) and
     start again.
3. **Record the commit SHA** actually synced (from the controller's
   working tree), surfaced in `dispatch.json` at top level.

Concrete `container.py` invocation form is an open question
(plan will inspect `./docker/container.py` and choose).

## Retry policy

| Failure class | Behaviour |
|---|---|
| **Infrastructure** — SSH unreachable, docker exec errored, container died before Hugin started | Retry up to `max_infrastructure_retries` (default 2). First retry on same host; second on a different host (via `preferred_not` hint). On final failure, `status=failed`, `failure.kind="infrastructure"`. |
| **Hugin crash** — remote process exited non-zero | NO retry. `status=failed`, `failure.kind="hugin_crash"`, `exit_code` + stdout/stderr tail preserved. |
| **Malformed bundle** — Hugin exited 0 but manifest.json missing / `status != "completed"` / required files missing | NO retry. `status=failed`, `failure.kind="hugin_malformed_bundle"`, message identifies the missing surface. |
| **Timeout** — elapsed wall-clock > `per_job_timeout_s` | Controller kills remote process. `status=failed`, `failure.kind="timeout"`, `duration_s` recorded. NO retry. |

The retry-failed escape hatch (`--retry-failed run_id1,run_id2,...`)
lets a user explicitly re-attempt specific run_ids after a dispatch
exits, without editing `dispatch.json` by hand.

## CLI surface

```
./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    --newton-yaml tools/odin/config/newton_envs.yaml \
    --seeds 42 \
    [--include 'Isaac-Ant-*' 'Isaac-Humanoid-*']  \
    [--resume <dispatch_id|LATEST>] \
    [--fresh] \
    [--skip-preflight] \
    [--per-job-timeout 14400] \
    [--max-infrastructure-retries 2] \
    [--retry-failed run_id1,run_id2,...] \
    [--verbose]
```

Shorthand wrappers (e.g. `tools/odin/scripts/odin-dispatch.sh`)
may be added later if ergonomics demand.

### Log presentation

Per Section 5 of brainstorming (option D):

```
[22:04:15] DISPATCH rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42 → valkyrie-03
[22:04:18] DISPATCH rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed43 → valkyrie-04
[22:04:22] PREFLIGHT valkyrie-05 FAIL: docker ps: container 'isaaclab' not running
[22:37:02] COMPLETE rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42 on valkyrie-03
                    (32m47s, reward_final_ema=1823.4)
[22:41:19] FAIL     rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed43 on valkyrie-04
                    (kind=hugin_crash, exit=1, tail: "CUDA OOM …")
...
[23:12:58] ALL DONE 24 completed, 3 failed (2 hugin_crash, 1 timeout), 0 pending
```

`--verbose` turns on interleaved `[host] ...`-prefixed remote
stdout for 1-2-node debugging runs.

## Testing approach

**Tier 1 — unit tests (fast, CI-safe, no SSH, no GPU):**

- `test_asgard_fleet.py` — `load_fleet()` round-trips, default
  resolution, missing-field errors.
- `test_asgard_queue.py` — `build_queue_from_env_lists()` expands
  seeds correctly, respects `keep: true` / `status != "stale"`,
  applies `include_filter`, unique `run_id`s.
- `test_asgard_state.py` — `DispatchState` round-trip, atomic
  write (temp-file + rename verified), resume rewrite flips
  `running`/`assigned` → `pending` without touching
  `completed`/`failed`.
- `test_asgard_worker.py` — `ValkyrieWorker` with fake
  `SSHRunner` / `RsyncRunner`: happy path, infrastructure-retry,
  hugin_crash, timeout, malformed bundle.
- `test_asgard_retry.py` — retry classification matrix +
  `preferred_not` hint.
- `test_asgard_preflight.py` — preflight pass/fail matrix,
  aggregation, `preflight.json` shape.

**Tier 2 — loopback integration test (slow, opt-in):**

`tools/odin/tests/test_asgard_integration.py`, marked
`@pytest.mark.slow`. Skips if `ssh localhost "echo ok"` fails or
the configured container isn't running. Dispatches one trivial
job via a 1-host fleet pointing at localhost, asserts
`dispatch.json` status is `completed`, a bundle arrived,
`ssh-tail.log` is non-empty.

**Tier 3 — manual acceptance (one-shot at T3.1 delivery):**

Run dispatch against a real ≥2-host fleet with the curated
`physx_envs.yaml`'s `keep: true` rows and `--seeds 42`. Hand-induce
one failure (kill a container) and confirm classification.

## What lands where

All Odin-side. No upstream IsaacLab changes.

| Path | Change |
|---|---|
| `tools/odin/asgard/__init__.py` | **New.** Public-API re-exports. |
| `tools/odin/asgard/fleet.py` | **New.** `Fleet`, `ValkyrieConfig`, `load_fleet()`. |
| `tools/odin/asgard/queue.py` | **New.** `JobEntry`, `build_queue_from_env_lists()`. |
| `tools/odin/asgard/state.py` | **New.** `DispatchState`, atomic read/write. |
| `tools/odin/asgard/transport.py` | **New.** `SSHRunner` / `RsyncRunner` protocols + shell impls. |
| `tools/odin/asgard/provisioner.py` | **New.** `provision_valkyrie()`. |
| `tools/odin/asgard/preflight.py` | **New.** `preflight_valkyrie()`. |
| `tools/odin/asgard/worker.py` | **New.** `ValkyrieWorker` thread. |
| `tools/odin/asgard/runner.py` | **New.** `run_dispatch()`, `DispatchOptions`. |
| `tools/odin/asgard/cli.py` | **New.** Thin CLI. |
| `tools/odin/tests/test_asgard_*.py` | **New.** 6 unit-test files + 1 slow integration. |
| `tools/odin/README.md` | Update — new section on `odin-dispatch` invocation. |
| `docs/odin/architecture.md` | §6 task map: T3 → 🟡 in-progress on T3.1 start, → ✅ on T3.1 close (with T3.2 status noted). §9 change-log entry. §3 layer-3 diagram annotation (Asgard is now code). |

### Architecture-doc handling

Per the doc's self-rule: architecture.md update lands in the
**same commit as the T3.1 closeout**, not in intermediate
per-task commits. Implementation plan has this as its final
closeout task, mirroring T2.1 / T2.2's pattern.

## Verification gates

- `./isaaclab.sh -f` clean on every touched file before each
  commit.
- `./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_*.py -v --confcutdir=tools/odin -m "not slow"`
  — all unit tests green.
- Loopback integration optional:
  `./isaaclab.sh -p -m pytest tools/odin/tests/ -v --confcutdir=tools/odin -m slow`
  — passes on a machine where `ssh localhost` + local docker
  work.
- Tier-3 acceptance per section above.

## Open questions (to resolve during implementation)

- **Exact `container.py` invocation form.** Whether to use
  `./docker/container.py python <script>` (if supported) or
  `docker exec <name> bash -lc '...'`. Decided at implementation
  time after reading the current `docker/container.py`.
- **rsync flag set.** `-avz --delete` for controller → Valkyrie
  push; `-avz` without `--delete` for Valkyrie → controller pull
  (don't want to prune prior bundles). Confirm exact flags
  during implementation.
- **Valkyrie-side `odin_runs/` retention.** After pull-back,
  the Valkyrie's host-side bundle dir is either left as a debug
  artefact or cleaned. Default: leave; add `--cleanup-remote`
  later if required.
- **Controller → Valkyrie rsync excludes.** Exclude
  `__pycache__/`, `.git/`, `odin_runs/`, `tools/tests/` etc. so
  the Valkyrie working tree stays tidy. Exact exclude list
  decided at implementation time.
- **CLI `--include` fnmatch vs regex.** Specified as fnmatch;
  confirm no task_id contains a character that breaks fnmatch
  semantics before locking in.

These are execution-time questions; the architecture and
deliverables don't change based on how they resolve.
