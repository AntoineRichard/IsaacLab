# Odin GPU-Loss Detection + Recovery — Design Spec

**Status:** approved (pending user review of this written form)
**Branch:** `antoiner/feat/odin`
**Lineage:** follows `2026-04-27-odin-native-backend-design.md`. Schema bump 1.2 → 1.3.

## Background

The first real-fleet T4.1+ validation of the native-backend routing fix (dispatch `20260427-141302`, 2026-04-27) reproduced the previously-crashing Anymal-C Flat case successfully (3/3, reward 11.40 ± 0.19), but exposed a separate infrastructure bug: after several consecutive Isaac Sim runs in the same container, NVML inside the container failed with `Failed to initialize NVML: Unknown Error`. Host-level `nvidia-smi` worked fine. 8/15 jobs in that dispatch ghost-completed — subprocess exited 0 in ~13 s producing no output, but Hugin's manifest was written claiming `status="completed", exit_code=0`.

Two latent bugs surfaced together:

1. **Hugin's `_run_phase` only checks returncode**, not whether the expected output JSON exists. Silent-exit-0 crashes are recorded as successes.
2. **The worker's stderr classifier doesn't recognize GPU-loss signatures** (`Failed to initialize NVML`, `CUDA error: no CUDA-capable device is detected`, `Vulkan ERROR_INCOMPATIBLE_DRIVER`). They fall into the generic `hugin_crash` bucket with no operational hint.

Compounding both: there is no automated recovery for in-dispatch GPU dropout. The only mitigation today is to manually `docker restart` the container between dispatches.

## Goal

Detect GPU loss before, during, and after a job runs; auto-recover the host's container when possible; fail the job out cleanly to another host when not. Three layers ship together as one feature on `antoiner/feat/odin`.

## Non-goals

- Driver-level reset (`nvidia-smi --gpu-reset`, `systemctl restart nvidia-persistenced`). The observed signature is container-level; the host driver was healthy. Driver reset would require NOPASSWD sudo on Valkyries and has nasty failure modes (stuck reset, conflicts with concurrent jobs). Excluded as YAGNI; revisit if a future incident demands it.
- Per-job pre-flight GPU probe in the worker. Adds ~200 ms per job for negligible additional coverage versus the post-job stderr classifier.
- Operator-only recovery CLI as the primary path. The recovery tool is a Python module the worker calls directly. A thin `odin-recover` CLI wraps it for ad-hoc use, not for runtime escalation.
- Per-(task, machine) timeout estimates. Tracked separately; not in this feature's scope.
- Detection of mid-job GPU loss that manifests as a hang (no stderr, no exit). The existing timeout path catches it via `kind="timeout"`. Recovery does not run on timeouts.

## Architecture

Four cooperating layers. Each is independently safe to land; together they form the detect-and-recover loop.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Dispatch start                                                       │
│   Layer 3 — Preflight: gpu_present check                             │
│     docker exec <ctr> nvidia-smi -L → marks host down if absent      │
│                                                                      │
│ Per-job loop (worker, one thread per host):                          │
│   docker exec ... → Hugin/Munin                                      │
│     Layer 1 — Hugin _run_phase: output-existence check               │
│       returncode==0 AND output_json exists → completed               │
│       else → failed; main() exits non-zero                           │
│                                                                      │
│   Layer 2 — Worker._classify: scans stderr after job returns         │
│     matches GPU-loss signatures → FailureInfo(kind="gpu_lost")       │
│                                                                      │
│   Layer 4 — Worker recovery integration:                             │
│     gpu_lost ⇒ run RecoveryTool (docker restart + probe)             │
│       recovered → retry on same host (counts vs infra retry cap)     │
│       not recovered → mark host down + preferred_not + emit final    │
└──────────────────────────────────────────────────────────────────────┘
```

### Key invariants

1. `gpu_lost` shares the existing `infrastructure`-retry contract: counts against `WorkerOptions.max_infrastructure_retries` (default 2 ⇒ up to 3 attempts per job), distinct kind in `dispatch.json` for telemetry.
2. Recovery is best-effort and time-bounded (~60 s for `docker restart` + `nvidia-smi -L` probe). On failure, host transitions to `down` + the job's `preferred_not` is updated; existing bounded fallback handles routing.
3. `dispatch.json` schema bumps minor 1.2 → 1.3: `FailureInfo.kind` enum gains `"gpu_lost"`; `FleetSnapshot.last_error` may now contain `"gpu_lost: <detail>"`. Additive only. Validators do major-match, so 1.2 files load on resume unchanged.
4. The recovery module is reusable: `odin-recover` console-script wraps it for ad-hoc operator use, but the worker calls the module directly without going through the CLI.

## Components

### New: `tools/odin/asgard/recovery.py`

```python
@dataclass
class RecoveryResult:
    host: str
    container_name: str
    attempted: bool          # False if precondition failed (e.g., SSH unreachable)
    recovered: bool          # True iff post-recovery probe sees GPU
    duration_s: float
    message: str             # short human-readable summary
    details: dict[str, Any]  # phase-by-phase: docker_restart, container_up, gpu_probe

def recover_valkyrie_gpu(host: ValkyrieConfig, *, ssh: SSHRunner) -> RecoveryResult:
    """Restart container, wait for running, probe nvidia-smi -L."""
```

Phases:

1. `docker restart <container>` (SSH timeout 60 s).
2. Poll `docker inspect -f '{{.State.Status}}'` every 2 s for up to 30 s; require `running`.
3. `docker exec <container> nvidia-smi -L`; require exit 0 with at least one non-empty line.

Failure at any phase short-circuits subsequent phases. `details` records the exit codes / outputs / durations of attempted phases.

### New: `tools/odin/asgard/recovery_cli.py`

Thin CLI exposing `recover_valkyrie_gpu` for ad-hoc operator use. Accepts `--fleet fleet.yaml` + `--host <ip>`; prints `RecoveryResult.message`; exits 0 iff `recovered=True`. Registered as `odin-recover` console-script in `pyproject.toml` (matches existing `odin-dispatch`).

### Modified: `tools/odin/hugin/run.py`

`_run_phase` adds an output-existence check:

```python
if completed.returncode != 0 or not os.path.exists(output_json):
    status = "failed"
    if completed.returncode == 0:
        # Silent-exit-0: derive a non-zero exit_code so Hugin's main()
        # propagates the failure upstream.
        derived_exit = 1
    else:
        derived_exit = completed.returncode
    # write stderr/stdout tails
else:
    status = "completed"
    derived_exit = 0
return ManifestPhase(file=..., status=status, duration_s=..., exit_code=derived_exit)
```

`main()` continues to exit non-zero when any phase's `exit_code != 0`, so the worker's existing `_classify` path picks up silent-exit-0 cases as `hugin_crash`.

### Modified: `tools/odin/munin/run.py`

Mirror the Hugin change in Munin's `_run_phase` (Munin has its own copy).

### Modified: `tools/odin/asgard/preflight.py`

Add a 5th check `gpu_present`:

```python
checks = {
    "ssh_reach": False,
    "docker_running": False,
    "container_up": False,
    "isaaclab_present": False,
    "gpu_present": False,
}
# ... existing checks 1–4 ...
# 5. gpu_present — at least one GPU visible inside container.
r = ssh.run(host, f"docker exec {host.container_name} nvidia-smi -L", timeout_s=15.0)
if r.exit_code != 0 or not r.stdout.strip():
    return PreflightResult(
        host=host.host,
        ok=False,
        checks=checks,
        message=f"GPU absent in container: {r.stderr.strip() or 'empty stdout'}",
    )
checks["gpu_present"] = True
```

If `container_up` fails, `gpu_present` is reported as `False` without running the probe (existing short-circuit pattern).

### Modified: `tools/odin/asgard/jobs.py`

Add `"gpu_lost"` to the `FailureInfo` docstring kind enum:

> ``gpu_lost``: training process exited non-zero with a GPU-loss
> signature in stderr (NVML init failure, CUDA "no device", Vulkan
> driver mismatch). Worker attempts `docker restart`-based recovery
> before retrying on the same host. Counts against
> ``max_infrastructure_retries``.

No code change to the dataclass.

### Modified: `tools/odin/asgard/worker.py`

**(a) New module-level constant:**
```python
_GPU_LOST_SIGNATURES = (
    "Failed to initialize NVML",
    "CUDA error: no CUDA-capable device is detected",
    "Vulkan ERROR_INCOMPATIBLE_DRIVER",
)
```

**(b) `_classify` extension** — new branch *before* `preset_unsupported`, *after* docker-infrastructure exit codes:

```python
if r.exit_code != 0:
    stderr_text = r.stderr or ""
    if any(sig in stderr_text for sig in _GPU_LOST_SIGNATURES):
        return FailureInfo(
            kind="gpu_lost",
            message="GPU-loss signature in stderr",
            details={
                "exit_code": r.exit_code,
                "log_tail_path": str(ssh_tail.relative_to(self._dispatch_dir)),
            },
        )
    if "preset_unsupported:" in stderr_text:
        ...
```

Order: `timed_out` → `_INFRASTRUCTURE_DOCKER_EXIT_CODES` → `gpu_lost` signatures → `preset_unsupported` → `hugin_crash`. Timeout always wins (we wouldn't want to keep restarting docker for a job that is wedged on something else).

**(c) `_execute` recovery integration.** The retry loop currently re-loops on `kind="infrastructure"`. Extend to also re-loop on `kind="gpu_lost"`, but only after a successful recovery:

```python
if failure is not None and failure.kind == "infrastructure":
    if job.attempts <= self._options.max_infrastructure_retries:
        continue
elif failure is not None and failure.kind == "gpu_lost":
    rec = recover_valkyrie_gpu(self.host, ssh=self._ssh)
    self._state_chan.put(StateEvent(
        run_id=job.run_id,
        host=self.host.host,
        transition="recovered" if rec.recovered else "host_down",
        failure=failure if not rec.recovered else None,
    ))
    if not rec.recovered:
        job.preferred_not = set(job.preferred_not) | {self.host.host}
        # Fall through to terminal failure emission.
    elif job.attempts <= self._options.max_infrastructure_retries:
        continue
    # else: retries exhausted → terminal failure.

break
```

Recovery output is written to `<dispatch_dir>/<bundle_dir_name>/logs/recovery.log`.

**(d) New `StateEvent` transitions:** `"recovered"` and `"host_down"`. Existing `"completed" | "failed" | "running" | "shutdown_idle"` are unchanged.

### Modified: `tools/odin/asgard/runner.py`

Consume the two new state transitions:

- `"recovered"` → update `FleetSnapshot[host].last_error = "gpu_lost: recovered"`. Status stays `idle`/`busy`.
- `"host_down"` → `FleetSnapshot[host].status = "down"`, `last_error = f"gpu_lost: recovery_failed ({rec.message})"`. Remove host from worker pool (signal worker to drain its queue claim and exit).

### Modified: `tools/odin/asgard/state.py`

`SCHEMA_VERSION = "1.3"`. No struct changes. Validators continue to do major-match.

## Data flow

**Flow 1 — Healthy job (unchanged):**
```
worker pops job → docker exec → Hugin returncode==0 + output exists
  → status=completed → rsync pull → bundle validates → emit completed
```

**Flow 2 — Hugin silent-exit (Layer 1 fix):**
```
docker exec → returncode==0 BUT output_json missing
  → Hugin _run_phase marks phase failed, exit_code derived as 1
  → Hugin main() exits non-zero
  → worker._classify sees exit≠0, no preset_unsupported, no GPU signatures
  → FailureInfo(kind="hugin_crash")
```

**Flow 3 — GPU loss detected mid-dispatch (Layer 2):**
```
docker exec → training fails inside container with NVML/CUDA/Vulkan error
  → worker._classify scans stderr, matches signature
  → FailureInfo(kind="gpu_lost") → recovery branch (3a or 3b)
```

**Flow 3a — Recovery succeeds:**
```
worker calls recover_valkyrie_gpu(host)
  → docker restart <ctr>           [~5–15 s]
  → wait for State.Status == "running"  [poll, max 30 s]
  → docker exec <ctr> nvidia-smi -L     [exit 0 with ≥ 1 line]
  → RecoveryResult(recovered=True)
worker:
  - emit StateEvent(transition="recovered", host=X)
    → runner: fleet[X].last_error="gpu_lost: recovered"
  - if job.attempts <= max_infrastructure_retries: continue retry loop on same host
  - else: emit terminal failed (gpu_lost)
```

**Flow 3b — Recovery fails:**
```
recover_valkyrie_gpu returns recovered=False
worker:
  - emit StateEvent(transition="host_down", host=X, failure=gpu_lost)
  - runner: FleetSnapshot[X].status="down",
            last_error="gpu_lost: recovery_failed"
  - runner: remove X from healthy worker pool
  - worker (this job): add X to job.preferred_not
  - if other hosts available: return job to queue, fallback routing
  - if X was the last host: emit terminal failed (gpu_lost)
```

**Flow 4 — Preflight catches GPU absent at dispatch start (Layer 3):**
```
preflight_valkyrie(host) → 5th check: docker exec <ctr> nvidia-smi -L → fails
  → PreflightResult(ok=False, checks={..., "gpu_present": False},
                    message="GPU absent in container: <stderr>")
runner:
  - down_hosts.add(host)
  - FleetSnapshot[host].status="down", last_error="gpu_lost: preflight"
  - existing skip_preflight + healthy-only logic handles the rest
```

## dispatch.json schema 1.3

Additive only:

- `failure.kind` enum: `infrastructure | hugin_crash | hugin_malformed_bundle | timeout | preset_unsupported | gpu_lost`
- `fleet[].last_error`: free-form string. New conventional prefixes: `"gpu_lost: preflight"`, `"gpu_lost: recovered"`, `"gpu_lost: recovery_failed"`.
- `skipped[].reason` enum unchanged.

Validator: existing `_assert_schema_compatible` does major-match. Resume from a 1.2 file works unchanged.

## Error handling and edge cases

### Recovery tool internal failures

| Phase | Failure mode | Behavior |
|---|---|---|
| SSH unreachable | Cannot run `docker restart` | `RecoveryResult(attempted=False, recovered=False, message="ssh_unreachable")`. Worker treats as recovery failure → host-down + preferred_not. |
| `docker restart` returns non-zero | Container failed to restart | Wait skipped; probe skipped; `recovered=False`, message starts `"docker_restart_failed"`. |
| `docker restart` hangs > 60 s | Daemon wedged | Hard timeout via `ssh.run(timeout_s=60)`. `recovered=False`, message `"docker_restart_timeout"`. |
| Container takes > 30 s to reach `running` | `docker inspect` polled every 2 s for up to 30 s; if state never reaches `running`, `recovered=False`, message `"container_not_running_after_restart"`. |
| `nvidia-smi -L` returns empty / non-zero | GPU still gone | `recovered=False`, message starts `"gpu_probe_failed"`. |
| `nvidia-smi -L` returns ≥ 1 GPU UUID | Acceptance test passes | `recovered=True`, message `"recovered_via_container_restart"`. |

### Worker-side edge cases

1. **Stderr signature in non-failing run.** A run that completes successfully but logs a transient NVML warning that matches a signature. Mitigation: signature check is gated by `r.exit_code != 0`. Won't false-positive on green runs.
2. **Stderr signature alongside other failures.** Order of checks: `timed_out` first → `_INFRASTRUCTURE_DOCKER_EXIT_CODES` → `gpu_lost` signatures → `preset_unsupported` → `hugin_crash`. Timeout always wins.
3. **Recovery succeeds but next attempt also drops GPU immediately.** `max_infrastructure_retries=2` (default) means worker tries up to 3 attempts total per job. Three consecutive `gpu_lost` events on the same host for the same job → terminal `failed` with `kind="gpu_lost"`, host marked down, no further jobs route there.
4. **Recovery succeeds for *this* job but the host drops mid-next-job for a different job.** Each job has its own attempt counter. Per-job, not per-host counting matches existing infrastructure-retry semantics.
5. **All hosts go down.** If recovery fails on the last healthy host, runner has no workers left. Existing behavior: `state_chan` drains, queue contains pending jobs, runner emits final `dispatch.json` with `fleet[*].status="down"`, returns. Pending jobs land in `dispatch.json` with `status="pending"` and can be resumed later via `--resume <dispatch_id>` after fleet repair.

### Preflight `gpu_present` failure semantics

Preflight is short-circuit: if `container_up` fails, `gpu_present` is reported as `False` without running the probe (matches existing pattern). Standalone preflight failure on `gpu_present` does *not* trigger the recovery tool — that's the worker's job mid-dispatch. Operator can run `odin-recover --host X` manually.

### Schema migration

None required. 1.2 → 1.3 is additive only. Validator does major-match; resume-from-1.2 works.

### Concurrency

`recover_valkyrie_gpu` is host-scoped and SSH-serial. Workers are one-thread-per-host; only one can call recovery on its host at a time. No locking needed.

### Logging

Recovery phase outputs go to `<dispatch_dir>/<bundle_dir_name>/logs/recovery.log` (alongside `ssh-tail.log`). Captures `docker restart` stderr, `docker inspect` polls, and `nvidia-smi -L` output for post-mortem.

## Testing strategy

Six test layers — each maps to a concrete file. All target the existing pytest suite (`./isaaclab.sh -p -m pytest tools/odin/tests/...`).

### 1. `tools/odin/tests/test_recovery.py` (new)

Reuses scripted-`FakeSSHRunner` pattern from existing transport tests.

| Test | Scripted SSH responses | Assertion |
|---|---|---|
| `test_recovery_happy_path` | `docker restart` exit 0 → `inspect` returns `"running"` → `nvidia-smi -L` exit 0 with `"GPU 0: ..."` | `recovered=True`, `details.docker_restart="ok"`, `details.gpu_probe="ok"` |
| `test_recovery_docker_restart_fails` | `docker restart` exit 1 stderr `"Error response from daemon"` | `recovered=False`, message starts `"docker_restart_failed"` |
| `test_recovery_container_never_running` | `docker restart` exit 0 → `inspect` returns `"created"` 15 times | `recovered=False`, message `"container_not_running_after_restart"` |
| `test_recovery_gpu_probe_empty` | `docker restart` exit 0 → `inspect` returns `"running"` → `nvidia-smi -L` exit 0 with empty stdout | `recovered=False`, message starts `"gpu_probe_failed"` |
| `test_recovery_ssh_unreachable` | First SSH call exits non-zero | `attempted=False`, `recovered=False`, message `"ssh_unreachable"` |

### 2. `tools/odin/tests/test_asgard_preflight.py` (extend)

| Test | Behavior |
|---|---|
| `test_preflight_gpu_present_pass` | All four prior checks pass + `nvidia-smi -L` exit 0 with GPU listed → `ok=True`, `checks["gpu_present"]=True` |
| `test_preflight_gpu_absent` | First four pass, `nvidia-smi -L` exit non-zero → `ok=False`, message `"GPU absent: ..."` |
| `test_preflight_gpu_short_circuits_on_container_down` | `container_up` fails → `gpu_present` reported `False` without probe call (verify SSH call count) |

### 3. `tools/odin/tests/test_asgard_worker.py` (extend)

Inject a fake `recover_valkyrie_gpu` via module monkeypatch.

| Test | Setup | Assertion |
|---|---|---|
| `test_classify_gpu_lost_signature_nvml` | exit 1, stderr contains `"Failed to initialize NVML"` | `FailureInfo.kind == "gpu_lost"` |
| `test_classify_gpu_lost_signature_cuda` | exit 1, stderr contains `"CUDA error: no CUDA-capable device is detected"` | same |
| `test_classify_gpu_lost_signature_vulkan` | exit 1, stderr contains `"Vulkan ERROR_INCOMPATIBLE_DRIVER"` | same |
| `test_classify_no_false_positive_on_success` | exit 0, stderr coincidentally contains `"Failed to initialize NVML"` (warning) | `_classify` returns `None` |
| `test_classify_timeout_wins_over_gpu_signature` | timed_out=True AND stderr has CUDA error | `kind == "timeout"` |
| `test_worker_gpu_lost_recovery_succeeds_retries_same_host` | First attempt: exit 1 + NVML stderr → fake recover returns `recovered=True` → second attempt: exit 0 success | Job ends `completed`, `attempts == 2`, recovery called once |
| `test_worker_gpu_lost_recovery_fails_marks_host_down` | First attempt: gpu_lost → fake recover `recovered=False` | StateEvent `host_down` emitted, job's `preferred_not` includes host, terminal failed |
| `test_worker_gpu_lost_three_in_a_row` | Three consecutive gpu_lost + successful recovery | After third attempt, terminal failed (`max_infrastructure_retries=2` default) |

### 4. `tools/odin/tests/test_hugin_run.py` (extend) — and corresponding Munin file

| Test | Behavior |
|---|---|
| `test_run_phase_returncode_zero_output_exists` | Subprocess exits 0 + creates output_json → status `"completed"` |
| `test_run_phase_returncode_zero_output_missing` | Subprocess exits 0 but writes nothing → status `"failed"`, exit_code derived as 1, log tails written |
| `test_run_phase_returncode_nonzero_unchanged` | Existing behavior preserved (regression guard) |

### 5. `tools/odin/tests/test_asgard_state.py` (extend)

| Test | Behavior |
|---|---|
| `test_schema_version_is_1_3` | constant `SCHEMA_VERSION == "1.3"` |
| `test_resume_from_1_2_state_works` | Write a 1.2-shape `dispatch.json`, read back, assert no error (major-match) |
| `test_failure_kind_gpu_lost_round_trips` | JobEntry with `failure=FailureInfo(kind="gpu_lost", ...)` serializes + deserializes intact |

### 6. `tools/odin/tests/test_asgard_integration.py` (extend)

One slow-marked test (`@pytest.mark.slow`):

| Test | Behavior |
|---|---|
| `test_dispatch_handles_gpu_lost_with_recovery` | Two-host fleet, scripted SSH so host A returns NVML error on first job → fake recovery succeeds → second job on A succeeds → host B runs all of its jobs unaffected. Assert final `dispatch.json` has expected job statuses and `fleet[A].last_error` mentions `"gpu_lost"` once. |

**Coverage targets:** every new module ≥ 95 % line coverage. Existing modules: only changed lines need new test coverage.

**Performance:** None of the new tests run actual Isaac Sim. Recovery tests are pure SSH-fake (millisecond-scale).

## Implementation order (preview — full plan in writing-plans phase)

10–11 commits, similar shape to `2026-04-27-odin-native-backend-design.md`:

1. `recovery.py` module + `test_recovery.py` (TDD).
2. `recovery_cli.py` + console-script registration in `pyproject.toml`.
3. Hugin `_run_phase` output-existence check + `test_hugin_run.py`.
4. Munin mirror of step 3.
5. Preflight `gpu_present` check + `test_asgard_preflight.py` extensions.
6. `FailureInfo` docstring update + jobs.py.
7. Worker `_classify` GPU-loss matcher + classifier tests.
8. Worker `_execute` recovery integration + integration tests.
9. Runner `host_down` / `recovered` event handling.
10. State schema 1.3 bump + `test_asgard_state.py` extensions.
11. End-to-end integration test + arch-doc change-log entry.

Step ordering keeps every commit independently safe: detection without recovery (steps 3, 7) just adds finer-grained classification; recovery without detection (steps 1, 2) is a stand-alone tool. The integration flips on at step 8.

## Non-functional requirements

- **No new required dependencies.** Recovery tool uses the existing `SSHRunner` interface; no new pip packages.
- **Schema compatibility.** Validators do major-match; 1.2 dispatches resume on 1.3 code without modification.
- **Telemetry.** All new failure paths emit `StateEvent` with classified `FailureInfo`. Recovery results land in `recovery.log` for post-mortem; coarse outcome is in `dispatch.json` `last_error` strings.
- **Performance.** Recovery on success: ~5–15 s (`docker restart` dominates). On failure: bounded ≤ ~95 s (60 s restart timeout + 30 s wait for running + 5 s probe). Acceptable as occasional cost for an in-dispatch GPU dropout — this is well below the 4-hour per-job timeout.
