# Odin Asgard — Detached Submit-and-Poll Worker

**Status:** Approved (operator: antoiner)
**Author:** Claude (handed off for implementation)
**Date:** 2026-04-30
**Branch context:** `antoiner/feat/odin`, atop the hardening sprint (HEAD around `f17e1e4eba0`).

---

## Goal

Replace today's **PTY-tied per-job SSH** worker with a **submit-and-poll** model so transient network blips between the dispatcher and a Valkyrie no longer terminate in-flight training. Concretely: a 90-second VPN flutter today kills `ssh -tt`, propagates SIGHUP to the remote `bash`, kills the running training, and the job lands as `hugin_crash` with `Connection to <host> closed.`. After this change, the same flutter delays the dispatcher's view of state by ≤ one poll interval and the training keeps running.

A 12-minute-into-dispatch event on `20260430-110509` killed 5 jobs simultaneously across 5 different hosts, all with `Timeout, server <host> not responding.` — the network event was on the dispatcher side, not the hosts. With detached submission, those 5 jobs would have continued and been picked up cleanly when SSH reconnected.

---

## Non-goals

- **Replacing recovery** (`gpu_lost` → container restart). The recovery path is unchanged; only how we *observe* the remote process changes.
- **Replacing aggregation, dispatch.json schema, dashboard.** This is a worker-internal refactor.
- **Removing `_classify`.** The classifier still maps remote stderr → `FailureInfo`. The change is just *when* and *how* we read that stderr.
- **Replacing the existing PTY-mode for backward-compat.** New mode is opt-in initially (CLI flag), then becomes the default, then the old path is removed in a separate change.

---

## Current state (what exists today)

- `tools/odin/asgard/worker.py` :: `ValkyrieWorker.run` — pulls jobs from a queue and for each runs:

  ```python
  ssh_result = self._ssh.run(host, _build_docker_exec_cmd(host, job), timeout_s=options.per_job_timeout_s)
  ```

  with `ssh -tt -o ServerAliveInterval=30 -o ConnectTimeout=10 -o BatchMode=yes` (transport.py `_DEFAULT_SSH_OPTS`). The SSH session stays open for the whole training run (10 minutes to 10 hours). Default `ServerAliveCountMax` is 3, so 90 seconds of unresponsive network kills the session and SIGHUPs the remote process group (intentional — added by the hardening branch to prevent zombie-on-disconnect).

- `_build_docker_exec_cmd` runs (inside a `bash -lc '…'` over SSH):

  ```bash
  cd /workspace/isaaclab \
    && mkdir -p odin_runs/<bundle>/logs \
    && (nvidia-smi -L >/dev/null 2>logs/nvidia-probe.log || { echo 'odin: gpu_unavailable: …' >&2; exit 1; }) \
    && PYTHONPATH=. _isaac_sim/python.sh tools/odin/hugin/run.py …
  ```

  with stdout/stderr redirected to `<bundle>/logs/hugin-stdout.log` and `…stderr.log`.

- `_classify(ssh_result, …)` examines `r.timed_out` / `r.exit_code` / `r.stderr` to map to `FailureInfo(kind=…)`. SSH-side stderr is what the classifier sees today; remote bundle stderr lives at `<bundle>/logs/training.stderr.log` and is **not** read by classify (only present in the bundle after rsync-pull, on success).

- After a non-failure SSH exit, the worker rsync-pulls the bundle, validates `manifest.json`, emits `completed` or a malformed-bundle failure.

- `tools/odin/asgard/reconcile.py` exists (added by the hardening branch) and runs on `--resume`. It currently reconciles `running` rows in `dispatch.json` whose process the previous dispatcher abandoned, by checking docker for orphaned exec sessions. This already gives us a foothold for orphan recovery in the new model.

---

## Proposed architecture

### 1. Submit phase

Replace the single PTY-tied `ssh -tt … 'docker exec … bash -lc "…"'` with a **submit** that fires the inner training script detached on the remote and returns immediately:

```bash
# inner script (still called over a single SSH connection, but the SSH exits as soon as the bash backgrounds the training)
cd /workspace/isaaclab
mkdir -p odin_runs/<bundle>/logs
nvidia-smi -L >/dev/null 2>odin_runs/<bundle>/logs/nvidia-probe.log || \
  { echo "odin: gpu_unavailable: $(tr -d '\n' < odin_runs/<bundle>/logs/nvidia-probe.log)" > odin_runs/<bundle>/logs/odin-submit-error.log; exit 1; }

# Detach the training. nohup + setsid + redirect everything → no PTY tie.
nohup setsid bash -c '
  echo $$ > odin_runs/<bundle>/.run.pid
  exec PYTHONPATH=. _isaac_sim/python.sh tools/odin/hugin/run.py … \
    > odin_runs/<bundle>/logs/hugin-stdout.log \
    2> odin_runs/<bundle>/logs/hugin-stderr.log
' > /dev/null 2>&1 < /dev/null &

# Submit-side success: print a sentinel the worker greps for.
echo "odin-submit: ok run_id=<run_id> bundle=<bundle>"
```

The outer SSH executes this script via `docker exec isaac-lab-base bash -lc '<inner>'`. Total session time is sub-second once the docker exec dispatches the background bash; SSH client closes cleanly without affecting the detached training (because of `setsid` + `nohup` + closed stdio).

**Pre-job probe (`nvidia-smi -L`) stays in the submit phase.** If the probe fails, the submit script writes the marker to `odin-submit-error.log` and exits non-zero; the worker reads exit code + that file and tags `gpu_lost` as before.

### 2. Tracker file

Right before `exec`-ing the trainer, the inner script writes a tracker JSON to the remote bundle dir:

```jsonc
// <bundle>/.tracker.json
{
  "schema_version": "1.0",
  "run_id": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260430-110509_seed42",
  "container_name": "isaac-lab-base",
  "host": "10.59.114.176",
  "submitted_at": "2026-04-30T11:05:34Z",
  "pid": 12345,                        // host-side PID (the setsid bash)
  "container_pid": null,               // optional; populated by docker if cheap
  "per_job_timeout_s": 43200
}
```

Worker reads this on first poll to confirm submission landed; reconciler (on `--resume`) reads it to recover orphans.

### 3. Poll phase

The worker keeps an in-memory map `run_id → tracker` and runs a poll loop. Per poll tick (default **30 s**):

```python
for run_id, tracker in self._inflight.items():
    state = self._poll_one(tracker)
    if state.terminal:
        self._finalize(tracker, state)  # rsync pull, classify, emit StateEvent
```

`_poll_one` issues **one short SSH call per host** that batches all that host's in-flight jobs:

```bash
# poll script — short, idempotent, no PTY
for bundle in <bundle1> <bundle2> …; do
  if [ -f /workspace/isaaclab/odin_runs/$bundle/manifest.json ]; then
    echo "$bundle done"
  elif [ -f /workspace/isaaclab/odin_runs/$bundle/.run.pid ]; then
    pid=$(cat /workspace/isaaclab/odin_runs/$bundle/.run.pid)
    if kill -0 "$pid" 2>/dev/null; then
      echo "$bundle alive"
    else
      echo "$bundle exited-no-manifest"
    fi
  else
    echo "$bundle no-pidfile"   # sub-second window between submit and pidfile write
  fi
done
```

Three terminal poll states map to existing classifier outputs:

| Poll output | Action | Next step |
|---|---|---|
| `done` | manifest.json present | rsync-pull, validate, emit `completed` |
| `exited-no-manifest` | training exited without writing manifest | rsync-pull (best effort), read remote `training.stderr.log`, `_classify_remote(stderr_text, exit_marker)` → `gpu_lost` / `hugin_crash` / `infrastructure` / `preset_unsupported`, emit `failed` |
| `alive` for > `per_job_timeout_s` | training still running past budget | submit a `kill` over fresh SSH (best effort `pkill -9 -f <run_id>`), wait for next poll, expect `exited-no-manifest` next time, classify as `timeout` |

`alive` and `no-pidfile` are non-terminal — keep polling.

### 4. Remote stderr is now first-class

A huge side benefit: `_classify_remote` reads `<bundle>/logs/training.stderr.log` directly via SSH `cat` (or rsync) on terminal-fail. That fixes the long-standing problem where `_classify` could only see `Connection to <host> closed.` from the SSH session and misclassified GPU wedges as `hugin_crash`. No more pre-job `nvidia-smi -L` probe-based marker hack (we keep the probe for pre-flight, but the *post-fail* classification gets robust automatically).

### 5. Reconciliation on `--resume`

`reconcile.py` extends to:

1. Walk `odin_runs/<dispatch_id>/*/`. For each bundle dir with `.tracker.json` and no `manifest.json`:
   - SSH the tracker's host, run the same poll snippet for that one bundle.
   - `done` → re-attach the worker, finalize (pull + emit `completed`).
   - `alive` → re-add to the in-memory inflight map; the runner will keep polling.
   - `exited-no-manifest` → finalize (pull + classify remote stderr + emit `failed`).
2. Without a tracker (legacy bundle from PTY-mode dispatcher): existing behavior — reset to `pending`.

This makes dispatcher restarts cheap. Today, a `Ctrl-C` of the runner = lost in-flight work; new model = restart, `--resume`, training continues.

### 6. Network resilience

Concrete blip-tolerance numbers under the new model:

| Event | Old behavior | New behavior |
|---|---|---|
| 30 s blackout | Old: tolerated (under 90 s ServerAliveCountMax). New: tolerated trivially. | One poll tick may miss; next poll catches up. |
| 90 s blackout | Old: SSH dies, training dies, job → `hugin_crash`. | One poll tick may miss; training continues. |
| 5 min blackout | Old: same as above. | At most one poll missed; training continues. |
| 30 min blackout | Old: same. | Same — training continues. Dispatcher catches up on reconnect. |
| Dispatcher local crash | Old: training dies (SIGHUP). | Training continues; `--resume` re-attaches. |

Current SSH `_DEFAULT_SSH_OPTS` keep `ServerAliveInterval=30` + a tighter `ConnectTimeout=10` for the short submit/poll calls — these are good for fast-fail when actually unreachable. The PTY (`-tt`) flag is **removed** for the new mode (we never want SIGHUP propagation now).

---

## Components and responsibilities

### `tools/odin/asgard/worker.py`

- **New:** `_build_submit_script(host, job) -> str` — emits the detached-submit inner script (returns the bash snippet that goes inside `docker exec bash -lc '…'`). Replaces `_build_docker_exec_cmd` for the new path. Old function stays until the legacy mode is removed.
- **New:** `_build_poll_script(bundles: list[str]) -> str` — emits the per-host batched poll snippet.
- **New:** `_submit_job(self, job) -> SubmitResult` — runs the submit SSH call, parses stdout for the `odin-submit: ok` sentinel; on failure, returns the appropriate `FailureInfo` directly (no poll needed).
- **New:** `_poll_host(self, host, bundle_ids) -> dict[bundle_id, PollState]` — one SSH per host per tick.
- **New:** `_finalize_terminal(self, job, poll_state) -> None` — pulls bundle, calls `_classify_remote`, emits StateEvent, removes from inflight map.
- **Refactor:** the run loop becomes:
  ```python
  while not self._down_event.is_set():
      self._submit_pending_until_capacity()   # pulls from queue, calls _submit_job
      self._poll_inflight()                   # batched per host
      time.sleep(self._options.poll_interval_s)
      self._sweep_timeouts()                  # any inflight job past per_job_timeout_s
  ```
- **`_classify_remote(self, host, bundle_dir) -> FailureInfo | None`** — SSH reads `<bundle>/logs/training.stderr.log` and `odin-submit-error.log`, applies the same signature regex set as today's `_classify` (gpu_lost, preset_unsupported, infrastructure docker exit codes, fall-through `hugin_crash`).

### `tools/odin/asgard/runner.py`

- Add `DispatchOptions.poll_interval_s: int = 30`.
- Add `DispatchOptions.detached_mode: bool = True` (with a CLI flag to opt out for the legacy path during initial rollout).
- Re-attach orphans during `--resume` via the extended `reconcile.py`.

### `tools/odin/asgard/reconcile.py`

- Extend to read `.tracker.json` files and call into worker's poll-one path.

### `tools/odin/asgard/cli.py`

- New flag: `--legacy-pty-mode` (or equivalent). Default is the new mode; flag opts back into the PTY model. Once two real-fleet runs prove stable, the flag and the legacy code path are removed.

### `tools/odin/asgard/transport.py`

- The existing `_DEFAULT_SSH_OPTS` keep `ServerAliveInterval=30`, `ConnectTimeout=10`, `BatchMode=yes`, `StrictHostKeyChecking=accept-new`. Drop `-tt` for the new-mode invocations (a separate ssh-options profile, or a `pty: bool = False` kwarg on `SSHRunner.run`). Keep the PTY profile for the legacy path during the rollout.

### Tracker schema file

- `<bundle>/.tracker.json` — schema documented above. Validator: `_validate_tracker(payload) -> None` raises on schema mismatch. Living in `tools/odin/asgard/tracker.py` (new file).

---

## State machine (per job)

```
                     submit_job
   pending ─────────────────────► submitted
                                      │
                                      ▼
                                  inflight ──poll: alive──► inflight (stay)
                                      │ poll: done            │
                                      │                       │ poll: alive AND elapsed > timeout
                                      ▼                       ▼
                                  pulling ──validate──►   timing-out
                                      │ ok                    │ best-effort kill
                                      ▼                       ▼
                                  completed              inflight ──poll: exited-no-manifest──►
                                                               │
                                                               ▼
                                                            failed (kind=timeout)

                            poll: exited-no-manifest
   inflight ─────────────────────────────────────────►   failed (kind from _classify_remote)
```

States are **in-memory in the worker** (`self._inflight: dict[run_id, JobInflight]`). The dispatch.json row is updated on every state transition by the existing `_apply_state_event` runner code.

A `JobInflight` carries: `job: JobEntry`, `tracker: dict`, `submitted_at: float`, `last_poll_state: PollState | None`.

---

## Failure modes and classification

| Source | Detection | Classifier output |
|---|---|---|
| GPU wedged at job start | nvidia-smi probe writes `gpu_unavailable` to `<bundle>/logs/odin-submit-error.log`; submit returns exit 1 with the marker on stdout | `gpu_lost` (auto-recovery fires) |
| Container down at submit | `docker exec` exits 125; submit captures it | `infrastructure` (existing path) |
| Training crashes mid-run | poll → `exited-no-manifest`. classify reads remote `training.stderr.log`, looks for: `Failed to initialize NVML`, `CUDA error: no CUDA-capable device`, `Vulkan ERROR_INCOMPATIBLE_DRIVER`, `RuntimeError: No CUDA GPUs are available`, `preset_unsupported:`, otherwise `hugin_crash` | `gpu_lost` / `preset_unsupported` / `hugin_crash` |
| Training exceeds wall-clock | `_sweep_timeouts` sees `submitted_at + per_job_timeout_s < now`; sends best-effort `pkill`; next poll classifies | `timeout` |
| Manifest written but malformed | After pull, existing `_validate_bundle` flags it | `hugin_malformed_bundle` |
| Network blip | Poll SSH errors out, worker retries on next tick (no state change) | None — non-terminal |

Note: a network blip during the **submit** phase still terminal-fails the job today (because submit is one shot). Mitigation: wrap `_submit_job` with a small retry loop (3 attempts, 2 s backoff, configurable) so a one-second SSH glitch at submit doesn't kill a job before it starts.

---

## Backward compatibility

- The dispatch.json schema does **not** change. JobEntry stays as-is; the tracker is a remote-side file invisible to anything reading dispatch.json.
- Legacy bundles (no tracker) on `--resume` go through the existing reset-to-pending path. Mixed dispatch_ids (legacy + new) work without any operator coordination.
- The `--legacy-pty-mode` flag exists for one release. After two real-fleet validations, both the flag and the old code paths in `worker.py` / `transport.py` are deleted in a separate small change.

---

## Testing strategy

### Unit tests (new, all pure-Python, plain `python3 -m pytest --noconftest -p no:cacheprovider`)

- `test_asgard_worker_submit.py`
  - `test_build_submit_script_includes_setsid_and_pidfile_write`
  - `test_build_submit_script_runs_nvidia_probe_before_train`
  - `test_submit_parses_ok_sentinel`
  - `test_submit_returns_gpu_lost_when_probe_fails` (via fake SSH returning exit 1 + the marker)
  - `test_submit_returns_infrastructure_on_docker_exit_125`
  - `test_submit_retries_on_transient_ssh_error_then_succeeds`

- `test_asgard_worker_poll.py`
  - `test_build_poll_script_batches_multiple_bundles`
  - `test_parse_poll_output_recognises_done_alive_exited_no_manifest`
  - `test_finalize_done_emits_completed_after_pull_and_validate`
  - `test_finalize_exited_no_manifest_classifies_via_remote_stderr`
  - `test_classify_remote_recognises_gpu_lost_signatures` (parameterised over the existing `_GPU_LOST_SIGNATURES`)
  - `test_classify_remote_falls_back_to_hugin_crash_with_no_signature`
  - `test_sweep_timeouts_kills_remote_and_marks_failed_with_kind_timeout`

- `test_asgard_tracker.py`
  - `test_tracker_round_trip` (write → read)
  - `test_validate_tracker_rejects_missing_required_fields`

- `test_asgard_reconcile.py` (extend)
  - `test_reconcile_reattaches_inflight_with_tracker_done` → finalizes as completed
  - `test_reconcile_reattaches_inflight_with_tracker_alive` → re-adds to inflight
  - `test_reconcile_finalizes_inflight_with_tracker_exited_no_manifest`

### Integration test (loopback)

- `test_asgard_integration.py::test_loopback_detached_dispatch_survives_dispatcher_restart`:
  1. Start a dispatch with one Cartpole job (fast).
  2. Wait until poll observes `alive` (the inner sleep stub — replace the trainer with a `sleep 30` for the integration test).
  3. SIGTERM the runner.
  4. Restart with `--resume`.
  5. Assert the job ends `completed` and the bundle has the same `.tracker.json` UUID.

### Manual real-fleet validation

- Repeat the 153-job physx batch on the 5-host Blackwell fleet. Acceptance:
  - 0 SSH-timeout-induced `hugin_crash` failures (i.e. the `Connection to <host> closed.` family disappears).
  - Dispatcher kill-and-resume in the middle of a run completes all in-flight jobs without re-running them.
  - Per-host total submit + poll SSH time per job < 5 % of training wall-clock.

---

## Implementation order preview (suggested)

1. **`tracker.py`** — schema + tiny read/write helpers + tests.
2. **`worker.py: _build_submit_script` + `_build_poll_script`** — pure string builders + their tests.
3. **`worker.py: _classify_remote`** — pulls remote stderr, applies signature set + tests.
4. **`worker.py` run-loop refactor** behind `DispatchOptions.detached_mode`. Existing PTY path stays callable. Tests for both.
5. **`reconcile.py`** extension for tracker-driven re-attach.
6. **`cli.py: --legacy-pty-mode`** and default-to-detached.
7. Loopback integration test.
8. Real-fleet validation pass on the 5-host Blackwell fleet, on the same physx_envs × 3-seed matrix.
9. After validation: remove `--legacy-pty-mode` + the legacy code paths (separate commit).

---

## Open questions / decisions deferred

- **Poll interval default** — 30 s feels right; finalize on the first real-fleet pass.
- **Tracker GC** — `.tracker.json` lives forever on the remote unless we sweep. Leave for now; the bundle dir is anyway preserved as the artifact.
- **Multi-job-per-host concurrency** — out of scope; one job per host stays the rule (the submit-and-poll model would technically allow multiple in-flight jobs per host, but GPU contention makes this useless). If revisited later, the poll script already batches multiple bundles per host so the SSH-call accounting is already right.
- **SSH multiplexing (ControlMaster/ControlPersist)** — orthogonal optimisation that could land before *or* after this; reduces TCP setup cost on the poll path. Suggest landing after the new model proves out, so we don't conflate two changes.

---

## Files touched (estimate)

| File | Change |
|---|---|
| `tools/odin/asgard/worker.py` | Refactor run loop, add submit / poll / finalize / classify_remote helpers (~250 LOC delta) |
| `tools/odin/asgard/tracker.py` | New, ~80 LOC |
| `tools/odin/asgard/runner.py` | `DispatchOptions` field additions (~10 LOC) |
| `tools/odin/asgard/reconcile.py` | Tracker-driven re-attach (~60 LOC) |
| `tools/odin/asgard/transport.py` | Optional `pty: bool = True` kwarg on `SSHRunner.run` (~10 LOC) |
| `tools/odin/asgard/cli.py` | `--legacy-pty-mode` + DispatchOptions wiring (~10 LOC) |
| `tools/odin/tests/test_asgard_worker_submit.py` | New (~200 LOC) |
| `tools/odin/tests/test_asgard_worker_poll.py` | New (~250 LOC) |
| `tools/odin/tests/test_asgard_tracker.py` | New (~60 LOC) |
| `tools/odin/tests/test_asgard_reconcile.py` | Extend (~80 LOC) |
| `tools/odin/tests/test_asgard_integration.py` | Add detached-resume test (~120 LOC) |
| Total | ~1,100 LOC including tests |

---

## What this does NOT solve

- The dispatcher's local clock and the Valkyries' clocks may drift. `submitted_at` is set on the host (UTC). For timeout enforcement we use `time.monotonic()` on the dispatcher, not the tracker timestamp. (Tracker timestamp is for audit only.)
- The first jobs in a dispatch still all submit at roughly the same time. If the burst itself trips a network throttle, we could see N parallel submit failures. The submit-retry loop (above) covers this; if 3 attempts fail, the job genuinely belongs as failed.
- Recovery for a wedged container that *looks* alive (PID is up but training hung). Today's heartbeat is "PID exists"; a hung process is indistinguishable from progress. A future enhancement could check `model_*.pt` mtimes from the tracker (dead-man-switch on bundle file activity), but that's out of scope here.
