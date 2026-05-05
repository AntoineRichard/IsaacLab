# Odin — Bifrost (OSMO) backend design

- **Date:** 2026-05-05
- **Branch:** `antoiner/feat/odin`
- **Status:** Approved (brainstorm complete; awaiting plan-writing pass)

## 1. Summary

Add a second peer dispatch path to Odin that submits eval jobs to **OSMO** ([NVIDIA/OSMO](https://github.com/NVIDIA/OSMO)), the Kubernetes-native workflow orchestrator. The new module, `tools/odin/bifrost/`, sits next to today's `tools/odin/asgard/` direct-machine-access path. Each is a self-contained CLI with its own config file. They share the run-id format, manifest schema, bundle layout (`odin_runs/<dispatch_id>/<run_id>/`), Hugin/Munin runners, and Valhalla aggregator — i.e. everything **inside** and **after** the dispatcher. They share **none** of the host-lifecycle plumbing because OSMO subsumes all of it.

The design rests on one observation: OSMO already provides scheduling, image pull, retry on infra failures (`exitActions`), output upload (datasets), log streaming, group barriers, and a `--rsync` daemon for live development. Trying to fit OSMO behind a `Backend` interface shaped around asgard's `preflight → provision → ssh + docker exec` lifecycle would force a wrong abstraction. Bifrost is therefore a peer entry point, not a backend plugin.

## 2. Scope

**In v1:**

- Dispatch a row list of `(task, framework, backend, seed)` to OSMO as a single workflow with N parallel tasks.
- Per-task bundle pull-back via OSMO datasets named `{prefix}-{dispatch_id}-{run_id}`.
- Four-kind failure classification (`infrastructure` / `hugin_crash` / `hugin_malformed_bundle` / `timeout`) preserved across both backends.
- `--resume`, `--retry-failed <run_ids>`, `--dry-run`, optional `--rsync` for active development.
- Hugin/Munin code reaches the container via a `tools/odin/` tarball uploaded with `files: [{localpath: ...}]` (default), or via `--rsync` (dev), or assumed image-baked (out of scope for v1).
- Atomic `dispatch.json` with new optional fields: `dispatcher`, `osmo_workflow_id`, `parent_dispatch_id`, plus per-job `osmo_task_name`. (Field name `dispatcher` chosen because the existing per-job `backend` field already means physics backend — `physx` / `newton`.)

**Out of v1 (deferred, noted as follow-ups):**

- Mixed-backend dispatches (OSMO + asgard in one invocation).
- Shared cross-backend retry SQLite queue (`.retry.sqlite` is asgard-only for now).
- Multi-node OSMO `groups` (data-parallel training across nodes).
- Custom Odin docker image build / publish.
- Dashboard live-link rendering to OSMO Web UI for `osmo_workflow_id`.
- Per-task tuning of `exitActions` ranges (defaults applied uniformly).

**Open (decisions deferred, no blockers):**

- Whether to rename `asgard → horde` in code/CLI for symmetry with the SSH user. Cosmetic; defer to follow-up PR.
- Whether `osmo workflow status --output json` exists in OSMO 6.2 (verify on first integration test); fallback table parser is still required either way.

## 3. Context

### 3.1 Today's direct-machine path (asgard)

Per `tools/odin/README.md`:

- `fleet.yaml` — list of SSH-reachable Valkyrie hosts.
- `odin-bootstrap` — wipes `{isaaclab_path}`, rsyncs the working tree, runs `docker/container.py start`, polls `docker inspect`.
- `odin-cuda` — SSH-driven CUDA upgrade with reboot orchestration.
- `odin-dispatch` — preflight (SSH + `docker ps` + `docker inspect`), provisioner (rsync + container start), runner (one thread per Valkyrie, SSH + `docker exec`, tee stdout, rsync bundle back).
- State: `odin_runs/<dispatch_id>/dispatch.json` (atomic), `.retry.sqlite` (cross-dispatch retry queue), per-bundle `manifest.json` + `training.json` + `startup.json` + `training_data/` + `logs/`.
- Failure kinds: `infrastructure` (retried up to N), `hugin_crash` (not retried), `hugin_malformed_bundle` (not retried), `timeout` (not retried).

The host-coupled modules (`bootstrap*`, `cuda_install*`, `preflight`, `provisioner`, `transport`, host-lifecycle parts of `recovery`/`worker`) **all exist because OSMO doesn't.** Their entire job is to reproduce on bare hosts what OSMO ships out of the box.

### 3.2 OSMO 6.2, in one paragraph

OSMO ([NVIDIA/OSMO](https://github.com/NVIDIA/OSMO), `/home/antoiner/Documents/OSMO`) is a workflow orchestrator built on Kubernetes plus NVIDIA Run:AI for scheduling. Users submit a YAML workflow via the `osmo` CLI or web UI; the API server validates, the workflow engine builds a DAG, the scheduler places tasks on pool/platform-tagged nodes, and tasks run as pods with images pulled from a registry. Inputs come from datasets, S3 URLs, or upstream task outputs. Outputs go to datasets or S3 URLs. State streams to the API server. Logs are tailed via `osmo workflow logs`. Failure handling is declarative: `exitActions: {COMPLETE: 0-10, RESTART: 11-20, RESCHEDULE: 21-255}`. For active development, `osmo workflow submit ... --rsync ./code:/workspace` keeps a local directory in sync with the running container.

The Isaac Lab cookbook ([`cookbook/reinforcement_learning/single_gpu/train_policy.yaml`](file:///home/antoiner/Documents/OSMO/cookbook/reinforcement_learning/single_gpu/train_policy.yaml)) shows the canonical pattern: one `task` running `bash /tmp/entry.sh` from inline `files:`, image `nvcr.io/nvidia/isaac-lab:2.2.0`, log dir moved to `{{output}}/`, output uploaded to a named dataset.

### 3.3 What this means for Odin

For each `(task, seed)` row that asgard dispatches as a thread + SSH + `docker exec`, bifrost dispatches as one parallel task in a single OSMO workflow. The per-row work (Hugin/Munin running inside an `nvcr.io/nvidia/isaac-lab` container with the same CLI args) is identical. What's different is the wrapper around the row list, and bifrost's wrapper is dramatically smaller than asgard's.

## 4. Architecture

### 4.1 Module layout

```
tools/odin/
├── bifrost/                          # NEW — peer to asgard/
│   ├── __init__.py
│   ├── cli.py                        # odin-bifrost-dispatch entry point
│   ├── config.py                     # bifrost-osmo.yaml load + validate
│   ├── workflow.py                   # render dispatch.yaml.j2 from rows + config
│   ├── client.py                     # thin osmo-CLI subprocess wrappers
│   ├── poller.py                     # status poll loop, OSMO-state → failure kind
│   ├── bundle.py                     # osmo dataset download + place in odin_runs/
│   └── templates/
│       └── dispatch.yaml.j2          # one workflow, N parallel tasks
├── asgard/                           # UNCHANGED for this work
├── common/                           # UNCHANGED — run_id, manifest, log_tail
├── hugin/                            # UNCHANGED — runs inside the container
├── munin/                            # UNCHANGED — runs inside the container
├── valhalla/                         # UNCHANGED — bundles look identical
└── tests/
    ├── test_bifrost_config.py        # NEW
    ├── test_bifrost_workflow.py      # NEW
    ├── test_bifrost_client.py        # NEW
    ├── test_bifrost_poller.py        # NEW
    ├── test_bifrost_cli.py           # NEW
    └── test_bifrost_integration.py   # NEW (slow-marked, gated by env flag)
```

### 4.2 Why Bifrost is a peer, not a plugin

A unifying `Backend` ABC over `preflight / submit / poll / fetch_logs / fetch_bundle / cancel` was considered and rejected:

- `preflight` does not exist for OSMO. The user's OSMO admin has already configured pools, platforms, image pull credentials, and pod templates; client-side preflight is a `osmo profile list` sanity check and nothing more.
- `submit` per-target does not exist for OSMO. OSMO submits one workflow with N tasks; the per-task placement is the scheduler's problem.
- `fetch_logs` is `osmo workflow logs` — no per-target SSH.
- `fetch_bundle` is `osmo dataset download` — no rsync.

The shared abstraction would be a 1-method `dispatch(rows) -> bundles` interface, which is so thin it's not worth the indirection. Two peer entry points with shared post-dispatch primitives (run-id, manifest, valhalla) is the honest shape.

If a `DispatchPlanner` (rows-from-yaml + seed-expansion + include-filter) emerges as obviously-shared during implementation, it can be lifted into `tools/odin/common/` post-hoc. We don't pre-design it.

### 4.3 What asgard does NOT change

For this work, asgard is frozen. No flag renames, no module moves, no behavior changes. Every existing `odin-dispatch`, `odin-bootstrap`, `odin-cuda`, `odin-retry` invocation continues to work exactly as it does today.

### 4.4 What downstream tooling DOES change

Two minimal additions:

1. `dispatch.json` schema: two **optional** new fields (see §6.2). Existing aggregator code handles missing-field cases (defaults to `backend: "asgard"` for parity with historical dispatches).
2. Valhalla aggregator: no changes for v1. It reads bundles from disk and doesn't care which backend produced them.
3. Dashboard: no changes for v1. A follow-up may add a "View on OSMO" link when `backend == "osmo"`.

## 5. Externals

### 5.1 `bifrost-osmo.yaml`

Sibling to `fleet-blackwell.yaml` etc., but a different shape (no host list):

```yaml
# Copyright (c) 2022-2026, ...

osmo_profile: prod                      # which `osmo profile` is selected (passed via OSMO_PROFILE env)
pool: rtx-pro-6000-eval                 # default pool; --pool overrides
priority: NORMAL                        # HIGH | NORMAL | LOW

image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
  pull_credential: ngc-readonly         # OSMO credential name (optional)

defaults:
  resources:                            # OSMO workflow.resources.default
    cpu: 16
    gpu: 1
    memory: 64Gi
    storage: 64Gi
    platform: rtx-pro-6000              # platform tag, pool-specific
  exec_timeout: 14400                   # seconds, per task
  queue_timeout: 7200                   # seconds, per task

retry:                                  # rendered into per-task exitActions
  reschedule_codes: "3001-3006"         # FAILED_BACKEND_ERROR..FAILED_PREEMPTED OSMO range
  restart_codes: ""                     # empty: we don't trust in-container RESTART for hugin/munin

bundle_dataset_prefix: odin             # final dataset name = "{prefix}-{dispatch_id}-{run_id}"

code_delivery:
  mode: files_upload                    # files_upload | rsync | image_baked
  source_root: tools/odin               # what to upload (relative to repo root)
```

Validation rules (in `bifrost/config.py`):

- All required fields present; no silent defaults for `pool`, `image.reference`.
- `priority ∈ {HIGH, NORMAL, LOW}`.
- `defaults.resources` has all five sub-keys (cpu/gpu/memory/storage/platform).
- `code_delivery.mode ∈ {files_upload, rsync, image_baked}`.
- `code_delivery.source_root` exists relative to the controller's CWD.
- `retry.reschedule_codes` and `retry.restart_codes` are valid OSMO exit-code-range strings (`""` allowed) — disjoint ranges; no overlap with the user code range `0-255`.

Config errors raise typed `BifrostConfigError` with the offending key path; modeled after `asgard.fleet`'s validator.

### 5.2 `odin-bifrost-dispatch` CLI

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/bifrost/cli.py \
    --osmo-config bifrost-osmo.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    [--newton-yaml tools/odin/config/newton_envs.yaml] \
    --seeds 42,43,44 \
    [--include 'Isaac-Ant-*'] \
    [--pool rtx-pro-6000-eval] \
    [--priority HIGH|NORMAL|LOW] \
    [--rsync] \
    [--dry-run] \
    [--resume <dispatch_id|LATEST>] \
    [--retry-failed <run_id1,run_id2,...>] \
    [--poll-interval 15] \
    [--verbose]
```

Flags absent compared to asgard (intentional — OSMO subsumes them):

- `--build-timeout`, `--skip-preflight`, `--fresh` — no host bootstrap.
- `--per-job-timeout` — moved into `bifrost-osmo.yaml: defaults.exec_timeout`.
- `--live_retry_poll_s` — `--poll-interval` covers this.

### 5.3 `dispatch.json` schema delta

Today's schema (`tools/odin/asgard/state.py` is at `SCHEMA_VERSION = "1.4"`) gains additive fields and bumps to **v1.5**. The bump is back-compatible by construction: `_schema_version_compatible` already accepts any same-major version, and every new field is optional with a sensible implicit default (`dispatcher → "asgard"`). Old `dispatch.json` files from prior asgard runs continue to load unmodified. New optional top-level keys:

```json
{
  "schema_version": "1.5",
  "dispatch_id": "20260505-150000",
  "started_at": "2026-05-05T15:00:00Z",
  "dispatcher": "osmo",                                    // NEW top-level: "osmo" | "asgard"; absent → "asgard"
  "osmo_workflow_id": "odin-disp-20260505-150000-1",       // NEW: present iff dispatcher == "osmo"
  "parent_dispatch_id": null,                              // NEW: set when this is a --retry-failed child
  "seeds": [42, 43, 44],
  "commit_sha": "...",
  "fleet": [],                                             // empty list for osmo (no host fleet)
  "jobs": [
    {
      "run_id": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260505-150000_seed42",
      "task_id": "Isaac-Ant-Direct-v0",
      "framework": "rsl-rl",
      "backend": "physx",                                   // EXISTING field, physics backend
      "num_envs": 4096,
      "max_iterations": 500,
      "seed": 42,
      "bundle_dir_name": "rsl-rl_physx_Isaac-Ant-Direct-v0_20260505-150000_seed42",
      "status": "completed",
      "assigned_to": null,                                  // SSH host for asgard; null for osmo
      "osmo_task_name": "rsl-rl-physx-isaac-ant-seed42",    // NEW: present iff dispatcher == "osmo"; DNS-1123-safe
      "attempts": 1,
      "failure": null
    }
  ],
  "skipped": [],
  "quarantined_hosts": []
}
```

For `dispatcher == "asgard"` (or absent), all existing fields keep their meaning. Aggregator and retry CLI must treat the new fields as optional (default `dispatcher → "asgard"`, `osmo_workflow_id → null`, `parent_dispatch_id → null`, `osmo_task_name → null`).

## 6. Lifecycle

### 6.1 One bifrost dispatch, end-to-end

1. **Plan.** Read `--physx-yaml` + `--newton-yaml` + `--seeds` + `--include` → row list of `(run_id, task, framework, backend_phys, seed, num_envs, max_iterations)`. Identical logic to today's asgard planning (lifted from `asgard.queue` if helpful; if not, re-implemented in `bifrost/cli.py` — code is small).
2. **Initialize state.** Allocate `dispatch_id` (today's format: `YYYYMMDD-HHMMSS`). Create `odin_runs/<dispatch_id>/`. Write `dispatch.json` with `backend: "osmo"`, `state: "pending"` jobs.
3. **Render workflow.** Build the Jinja2 context (`dispatch_id`, rows, config), render `templates/dispatch.yaml.j2` to `odin_runs/<dispatch_id>/workflow.yaml`. If `--dry-run`, print and exit 0 here.
4. **Stage code.** If `code_delivery.mode == files_upload`, tar `code_delivery.source_root` to `odin_runs/<dispatch_id>/odin-source.tar.gz` and reference its absolute path in the rendered workflow (Jinja already wired). If `mode == rsync`, skip. If `mode == image_baked`, skip.
5. **Submit.** `osmo workflow submit odin_runs/<dispatch_id>/workflow.yaml` (plus `--rsync tools/odin:/workspace/odin-source` if `--rsync`). Parse the returned `Workflow ID` and persist into `dispatch.json`.
6. **Poll.** Every `--poll-interval` seconds, `osmo workflow status <wf-id>` (try `--output json` first; fall back to a table parser if unavailable). For each task whose terminal state changed since the last poll, update its `dispatch.json` entry, classify failure (§7), and on `COMPLETED` enqueue a bundle download.
7. **Pull bundles.** A small thread pool (default 4) drains the bundle download queue: `osmo dataset download {prefix}-{dispatch_id}-{run_id} odin_runs/<dispatch_id>/<run_id>/`. After each successful download, validate manifest; on mismatch, mark the row as `hugin_malformed_bundle`.
8. **Live tail (optional).** With `--verbose`, the poller picks the first `RUNNING` task and runs `osmo workflow logs --follow <wf-id> <task-name>` in a background thread, tee'd to `odin_runs/<dispatch_id>/<run_id>/logs/osmo-tail.log`. When that task completes, advance to the next still-running task. Single-tail by design (per-task tails are a follow-up).
9. **Terminal.** When all rows reach terminal, finalize `dispatch.json`, then unless `--skip-aggregate` invoke `valhalla.cli` against this dispatch (parity with asgard tail behavior).

### 6.2 `--resume`

`--resume LATEST` (or an explicit `<dispatch_id>`) reads the existing `dispatch.json`, picks up the persisted `osmo_workflow_id`, and re-enters step 6. Cheap because OSMO holds the truth — we're just reattaching the poller. Bundles already downloaded are not re-downloaded (path existence + manifest hash check).

### 6.3 `--retry-failed`

`--retry-failed <run_ids>` reads the parent `dispatch.json`, filters to those rows (must currently be in a non-success terminal state), allocates a fresh `dispatch_id`, sets `parent_dispatch_id: <orig>`, and runs steps 1–9 with the filtered row list. The new dispatch lands its own bundles on disk; aggregation across both dispatches is a Valhalla concern (today's aggregator already supports re-running across an extended bundle set).

OSMO workflows are immutable once submitted, so this is the OSMO-idiomatic retry shape — there is no "edit in place" alternative.

## 7. Failure mapping

`OSMO terminal state → Odin failure.kind`. Lives as `OSMO_STATE_TO_FAILURE_KIND` in `bifrost/poller.py`, single source of truth, table-tested.

| OSMO terminal state                                                                                    | Odin `failure.kind`           | Notes |
|---|---|---|
| `COMPLETED` (exit 0) + valid manifest                                                                  | — (success)                   | dataset downloaded, manifest validated |
| `COMPLETED` (exit 0) + bundle missing/malformed                                                        | `hugin_malformed_bundle`      | matches asgard semantics |
| `FAILED` (user exit 1–255)                                                                             | `hugin_crash`                 | not retried automatically |
| `FAILED_EXEC_TIMEOUT`                                                                                  | `timeout`                     | matches asgard |
| `FAILED_BACKEND_ERROR`, `FAILED_PREEMPTED`, `FAILED_EVICTED`, `FAILED_IMAGE_PULL`, `FAILED_START_ERROR`, `FAILED_START_TIMEOUT`, `FAILED_QUEUE_TIMEOUT`, `FAILED_SERVER_ERROR` | `infrastructure`              | OSMO already attempted reschedule per `exitActions`; surfaces only after OSMO gave up |
| `FAILED_CANCELED`                                                                                      | `infrastructure` (sub: `canceled`) | external cancel only |
| `FAILED_SUBMISSION` (workflow-level, not per task)                                                     | abort dispatch                | bad spec/creds; surface verbatim, exit non-zero, do not write per-row failures |
| `FAILED_UPSTREAM`                                                                                      | impossible                    | our N tasks have no inter-deps |
| `RESCHEDULED` (intermediate)                                                                           | not terminal                  | poller continues to poll the replacement task |

OSMO's exit-code ranges (User 0–255, OSMO init 256–257, Service 2002–4000) are recorded verbatim in `failure.exit_code` for forensics; the failure kind drives behavior.

## 8. Retry semantics

- **Infra-class auto-retry** is delegated to OSMO via `exitActions: RESCHEDULE: <range>` rendered from `bifrost-osmo.yaml: retry.reschedule_codes`. Default range covers the OSMO service codes for backend/preempted/evicted/image-pull/start/queue-timeout/server-error (3001–3006). The bifrost poller does **not** maintain a per-row infra-retry counter; OSMO does that.
- **User-class retry** (`hugin_crash`, `timeout`, `hugin_malformed_bundle`) is explicit operator action via `--retry-failed <run_ids>`. There is no auto-retry for these — same as asgard (real bugs repeat).
- **Cross-dispatch retry queue** (`.retry.sqlite`) is **asgard-only** for v1. Bifrost dispatches do not poll the SQLite queue. A future unification can lift retry into a backend-neutral facility, but only after we have lived experience with the OSMO failure shape.

## 9. Internals

### 9.1 `templates/dispatch.yaml.j2`

```yaml
workflow:
  name: odin-disp-{{ dispatch_id }}
  pool: {{ pool }}
  resources:
    default:
      cpu: {{ defaults.cpu }}
      gpu: {{ defaults.gpu }}
      memory: {{ defaults.memory }}
      storage: {{ defaults.storage }}
      platform: {{ defaults.platform }}
  timeouts:
    exec: {{ defaults.exec_timeout }}
    queue: {{ defaults.queue_timeout }}
  tasks:
  {%- for row in rows %}
  - name: {{ row.osmo_task_name }}
    image: {{ image.reference }}
    {%- if image.pull_credential %}
    credentials:
      registry: {{ image.pull_credential }}
    {%- endif %}
    environment:
      ACCEPT_EULA: "Y"
      NO_NUCLEUS: "Y"
      OMNI_KIT_ALLOW_ROOT: "1"
      ODIN_DISPATCH_ID: "{{ dispatch_id }}"
      ODIN_RUN_ID: "{{ row.run_id }}"
    command: ["bash"]
    args: ["/tmp/entry.sh"]
    files:
    - path: /tmp/entry.sh
      contents: |
        set -euxo pipefail
        {%- if code_delivery.mode == "files_upload" %}
        # Overlay our tools/odin/ onto the image-baked IsaacLab tree
        tar -xzf /workspace/odin-source.tar.gz -C /workspace/IsaacLab
        {%- elif code_delivery.mode == "rsync" %}
        # --rsync daemon syncs into /workspace/IsaacLab/tools/odin live
        :
        {%- endif %}
        cd /workspace/IsaacLab
        # --runs_root pins Hugin/Munin's bundle root at the OSMO per-task output dir.
        # Hugin/Munin then write `{{output}}/<run_id>/{manifest.json, training.json, ...}`
        # which becomes the dataset content; bundle.py downloads to odin_runs/<dispatch_id>/.
        PYTHONPATH=. ./isaaclab.sh -p tools/odin/{{ row.framework_runner }}/run.py \
          --task {{ row.task_id }} --backend {{ row.backend }} --seed {{ row.seed }} \
          --num_envs {{ row.num_envs }} --max_iterations {{ row.max_iterations }} \
          --runs_root '{{ '{{output}}' }}'
    {%- if code_delivery.mode == "files_upload" %}
    # tarball_path is the CONTROLLER-LOCAL path; OSMO uploads it at submit time
    - localpath: {{ tarball_path }}
      path: /workspace/odin-source.tar.gz
    {%- endif %}
    outputs:
    - dataset:
        name: {{ bundle_dataset_prefix }}-{{ dispatch_id }}-{{ row.run_id }}
    {%- if retry.reschedule_codes or retry.restart_codes %}
    exitActions:
      {%- if retry.reschedule_codes %}
      RESCHEDULE: {{ retry.reschedule_codes }}
      {%- endif %}
      {%- if retry.restart_codes %}
      RESTART: {{ retry.restart_codes }}
      {%- endif %}
    {%- endif %}
  {%- endfor %}
```

Notes on the template:

- `'{{output}}'` is OSMO's special token for the per-task output directory; we double-brace-escape inside Jinja so the literal `{{output}}` survives the render and OSMO substitutes at runtime.
- `osmo_task_name` is derived from `run_id` and must be DNS-1123-compliant (≤63 chars, lowercase alphanumeric + `-`, no leading/trailing `-`). Transform: lowercase, replace `[_.\s]+` with `-`, strip non-alphanumeric-or-dash, truncate to 63 chars, then suffix with a 6-hex hash of the full run_id if truncation occurred. Implementation in `bifrost.workflow.osmo_safe_task_name`.
- `framework_runner` is `hugin` for `framework == rsl-rl` and `munin` for `framework == skrl`.
- **No new flag is needed on Hugin/Munin.** Both already accept `--runs_root` (defaults to `./odin_runs`). Setting `--runs_root '{{output}}'` makes Hugin write `{{output}}/<run_id>/{manifest.json, ...}`. After OSMO uploads `{{output}}` as the dataset content, downloading the dataset into `odin_runs/<dispatch_id>/` gives `odin_runs/<dispatch_id>/<run_id>/manifest.json` — the canonical layout, with no glue code.
- `code_delivery.mode == "files_upload"` overlays the tarball onto the image-baked `/workspace/IsaacLab` tree. v1 ships only `tools/odin/`; if a future row needs newer `source/isaaclab/*` or `scripts/benchmarks/*`, expand `code_delivery.source_root` (it's already a config knob).

### 9.2 `client.py`

Five thin subprocess wrappers:

```python
def submit(yaml_path: pathlib.Path, *, rsync_pairs: list[tuple[str, str]] = ()) -> str:
    """Returns the workflow_id."""

def status(workflow_id: str) -> WorkflowStatus:
    """Returns a structured snapshot. Tries --output json; falls back to table parser."""

def logs(workflow_id: str, task_name: str, *, follow: bool = False) -> Iterator[bytes]:
    """Tails or fetches task logs."""

def dataset_download(name: str, dest_dir: pathlib.Path) -> None: ...

def cancel(workflow_id: str) -> None: ...
```

Errors are typed:

- `OsmoCliError` — non-zero exit, non-empty stderr.
- `OsmoTransientError` — exit codes/stderr patterns indicating "retry the API call" (e.g. transient HTTP 5xx). Caller may retry.
- `OsmoAuthError` — auth failure; caller surfaces immediately, does not retry.

Retries in `client.py` are only for `OsmoTransientError` and capped at 3 with exponential backoff. Anything else propagates.

### 9.3 `poller.py`

Single-threaded poll loop with optional log-tail thread. Pseudocode:

```
while not all_terminal(state):
    snap = client.status(wf_id)
    for task in snap.tasks:
        prev = state.jobs[task.name]
        if task.state == prev.state:
            continue
        if task.state in TERMINAL:
            kind = OSMO_STATE_TO_FAILURE_KIND.get(task.state, "infrastructure")
            update_job(task, kind)
            if task.state == "COMPLETED":
                bundle_queue.put(task)
        else:
            update_job(task, kind=None)
    write_dispatch_json_atomic(state)
    sleep(poll_interval)
```

`write_dispatch_json_atomic` reuses `tools/odin/asgard/state.py::_atomic_write` if it's importable, else replicates it (write to `dispatch.json.tmp`, `os.replace`).

### 9.4 `bundle.py`

```
for task in bundle_queue:
    dataset_name = f"{prefix}-{dispatch_id}-{run_id}"
    dest = runs_root / dispatch_id / run_id
    client.dataset_download(dataset_name, dest)
    if not validate_manifest(dest):
        mark_malformed_bundle(task)
```

`validate_manifest` is the same validator used by asgard today (lifted into `tools/odin/common/manifest.py` already per project memory).

## 10. Testing

### 10.1 Unit tests

All tests run with `PYTHONPATH=. python3 -m pytest tools/odin/tests/test_bifrost_*.py -v --confcutdir=tools/odin` — no Kit runtime needed.

| Test file | Coverage |
|---|---|
| `test_bifrost_config.py`     | yaml validation: required fields, enum constraints, error message quality |
| `test_bifrost_workflow.py`   | template rendering: 1 row, 5 rows, mixed physx+newton, retry block on/off, special-token escape, DNS-safe task names |
| `test_bifrost_client.py`     | submit/status/logs/download wrappers; JSON path + table-parser fallback; transient-vs-fatal error classification |
| `test_bifrost_poller.py`     | full failure-kind table; pending → running → terminal transitions; bundle-queue trigger on COMPLETED; atomic dispatch.json |
| `test_bifrost_cli.py`        | argparse, --dry-run, --resume LATEST, --retry-failed parent linkage, --include filter |
| `test_bifrost_bundle.py`     | dataset download → bundle path placement; manifest validation marks `hugin_malformed_bundle`; idempotent re-download skipped when path exists with matching manifest hash |

`subprocess.run` is patched with a fixture that returns canned stdout/stderr/exit-code triples per `osmo` subcommand. No real `osmo` CLI required for unit tests.

### 10.2 Integration test

`test_bifrost_integration.py` — slow-marked, gated by `ODIN_OSMO_INTEGRATION=1`. Requires a working local OSMO deployment (per OSMO's `run/start_backend_kind.py`). Optimistic-path only:

1. Render a 1-row workflow against a tiny local image (alpine + a stub Hugin script that writes a fake manifest).
2. Submit, poll until COMPLETED, download dataset.
3. Assert the bundle layout matches the canonical schema.

Failure-path integration testing is out of scope for v1 — too brittle to maintain.

### 10.3 Pre-commit

Standard: `./isaaclab.sh -f` before commit.

## 11. Naming & inheritance from existing project

- **Bifrost** (Norse mythology: the burning rainbow bridge connecting Asgard to other realms) was reserved for inter-node communication / SSH transport in T3 per `project_odin.md`. The reservation is repurposed here: bifrost is the bridge between the Odin controller and the OSMO realm. The original "SSH transport" usage was a placeholder that never landed; reusing the name is a net improvement.
- **Asgard** stays as-is for the direct-machine path. Possible future rename to "Horde" (matching the SSH user `horde`) is noted but out of scope.
- **Hugin / Munin** runners are unchanged; they execute identically in both backends.

## 12. References

- OSMO source repo (full local clone): `/home/antoiner/Documents/OSMO/`
- OSMO RL example (canonical Isaac Lab pattern): `/home/antoiner/Documents/OSMO/cookbook/reinforcement_learning/single_gpu/train_policy.yaml`
- OSMO workflow lifecycle states: `/home/antoiner/Documents/OSMO/docs/user_guide/workflows/lifecycle/index.rst`
- OSMO exit codes: `/home/antoiner/Documents/OSMO/docs/user_guide/workflows/exit_codes.rst`
- OSMO templating: `/home/antoiner/Documents/OSMO/docs/user_guide/workflows/specification/templates_and_tokens.rst`
- OSMO submission & CLI features: `/home/antoiner/Documents/OSMO/docs/user_guide/workflows/submission.rst`
- Odin project pointer & memory: `~/.claude/projects/-home-antoiner-Documents-IsaacLab/memory/project_odin.md`
- Asgard internals (reference for structure to mirror in bifrost): `tools/odin/README.md` and `tools/odin/asgard/`
