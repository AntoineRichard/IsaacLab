# Odin Live Job Tail — streaming Hugin output + Tab A "View tail" button

**Status:** Approved (operator: antoiner)
**Author:** Claude (handed off for implementation)
**Date:** 2026-04-30
**Branch context:** `antoiner/feat/odin`, atop the retry-toggle UI commit `f17e1e4eba0`.

---

## Goal

Let the operator see what a *running* training job is producing right now — last 50 lines of the training subprocess's stdout, on demand from Tab A. Today, Hugin captures the training subprocess's stdout/stderr **in memory** and only writes them to disk on failure, so for an in-flight job there is no tail-able text on disk. The TB event file has per-iter metrics but it's binary protobuf and tooling-heavy to parse for a quick read.

Two coupled changes deliver this:

1. **Stream Hugin's child stdout/stderr directly to disk** (drop `subprocess.run(capture_output=True)` for `stdout=fh, stderr=fh`). The bundle's `<phase>.stdout.log` / `<phase>.stderr.log` files are populated *during* the run, not just on failure.
2. **Tab A "View tail" button on running rows** — same pattern as the existing `ssh-tail.log` button on failed rows. Click → `docker exec tail -50` the right file → render inline in an expand row.

---

## Non-goals

- **Live streaming / SSE** — no realtime push. Click reads the current state, no auto-refresh; operator clicks again to see more. (Polling is a follow-up if useful.)
- **Replacing failure-side ssh-tail** — that path stays for `failed` rows. The new button is `running` rows only. The two are visually distinct so the operator knows which they're seeing.
- **Parsing TB event files** — the protobuf parse is a separate (richer) feature — out of scope here.
- **Backward compat on already-running jobs** — they're using the old Hugin (buffered output, nothing on disk). Once the Hugin patch lands and a future container rebuild propagates it, *new* dispatches benefit. Old in-flight jobs are stuck with the old behaviour for the rest of their runs.

---

## Background

`tools/odin/hugin/run.py:_run_phase`:

```python
def _run_phase(cmd, bundle_dir, phase_name, output_json):
    logs_dir = os.path.join(bundle_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    completed = _subprocess_run(cmd, capture_output=True)        # ← buffers in RAM
    if completed.returncode != 0 or not os.path.exists(output_json):
        # ... only on FAILURE: write captured tail to <phase>.stderr.log / .stdout.log
        with open(.../f"{phase_name}.stderr.log", "wb") as f:
            f.write(tail_bytes(completed.stderr))
        with open(.../f"{phase_name}.stdout.log", "wb") as f:
            f.write(tail_bytes(completed.stdout))
```

For a running phase, both files don't exist. We confirmed this on the live `20260430-110509` Anymal-C Rough seed 42 job (27 model checkpoints saved, training clearly progressing, but `hugin-stdout.log`, `hugin-stderr.log`, `nvidia-probe.log` were all 0 bytes; no `training.stdout.log` on disk).

The `tail_bytes(...)` truncation is presumably to keep failure logs small. Streaming would write the *whole* log; for `Isaac-Repose-Cube-Allegro-Direct-v0` (10000 iters, ~hourly runs) the stdout text is ~5-15 MB. Bundle dir is already 200 MB+ from model checkpoints — single-digit MB more is acceptable.

The dashboard's existing failed-row UI uses `ssh-tail.log` (per-job dispatcher-side SSH tee'd log). That file is unrelated to what we're adding here — the new button reads bundle-side `training.stdout.log`.

---

## Part A — Hugin streaming patch

### Change

```python
def _run_phase(cmd, bundle_dir, phase_name, output_json):
    logs_dir = os.path.join(bundle_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stdout_path = os.path.join(logs_dir, f"{phase_name}.stdout.log")
    stderr_path = os.path.join(logs_dir, f"{phase_name}.stderr.log")
    start = datetime.now(timezone.utc)
    with open(stdout_path, "wb") as so, open(stderr_path, "wb") as se:
        completed = _subprocess_run(cmd, stdout=so, stderr=se)
    end = datetime.now(timezone.utc)
    duration_s = (end - start).total_seconds()
    output_exists = os.path.exists(output_json)
    if completed.returncode != 0 or not output_exists:
        status = "failed"
        exit_code = completed.returncode or 1
        # No more tail-on-failure write — the files are already populated by the subprocess.
    else:
        status = "completed"
        exit_code = 0
    return ManifestPhase(...)
```

The subprocess inherits the file descriptors, the OS handles buffering. Output appears on disk as the child's libc flushes (typically 4 KB block-buffered for redirected stdout).

### Buffering caveat

For per-iter visibility we want lines to land within a second or two, not on 4 KB block boundaries. Two options:

- **`-u` on the Python child** (cheapest): if `cmd[0]` ends in `python` / `python.sh`, prepend `-u` to force unbuffered. Inspect `cmd` and only inject when safe.
- **`stdbuf -oL -eL` wrapper**: not always available in minimal containers; skip.

Pick option 1: if `_run_phase` detects a Python invocation, insert `-u` after the interpreter. Document the assumption in the docstring.

### Tests

`tools/odin/tests/test_hugin.py` (extend, ~80 LOC):

- `test_run_phase_streams_stdout_to_log_during_run` — fake subprocess writes lines with delays; assert log file is non-empty before subprocess returns.
- `test_run_phase_writes_full_stdout_on_completed` — file contents on success match what subprocess emitted.
- `test_run_phase_no_tail_truncation_on_failure` — failure path has the same content as success path (no separate tail-on-failure code).
- `test_run_phase_injects_dash_u_for_python_child` — `cmd = ["python.sh", "script.py", ...]` becomes `["python.sh", "-u", "script.py", ...]`.
- `test_run_phase_does_not_inject_dash_u_for_non_python` — bare `cmd = ["nvidia-smi"]` is passed through unchanged.

### Failure-classification side note

The existing `worker._classify_remote` (proposed in the submit-and-poll spec) reads `<bundle>/logs/training.stderr.log` to extract GPU sigs. With this Hugin patch, that file is **always** populated regardless of failure status, so the classifier sees richer context (the actual training stack trace, not just a tail). No change needed in the classifier itself.

### Rollout

Since Hugin lives **inside the container's image**, this change requires a container rebuild on each Valkyrie before it takes effect. Same propagation issue we discussed for the skrl trainer fix earlier today. Bind-mounting the source tree (separate change) would make this kind of patch live without a rebuild — flag as a follow-up.

---

## Part B — DataLayer

### New method

```python
class DataLayer:
    def read_running_job_tail(
        self,
        dispatch_id: str,
        run_id: str,
        *,
        host: str,
        ssh_user: str = "horde",
        ssh_key: Path | None = None,
        container_name: str = "isaac-lab-base",
        n: int = 50,
    ) -> list[str]:
        """SSH-and-docker-exec tails the last ``n`` lines of the running job's
        ``training.stdout.log``, falling back to ``startup.stdout.log`` when
        training hasn't started yet.

        Returns the lines (without trailing newlines). Empty list when both
        log files are missing or empty (e.g. the job hasn't reached the
        Hugin _run_phase stage yet).

        Tries `training.stdout.log` first; if it doesn't exist or is empty,
        falls back to `startup.stdout.log`. This matches the natural Hugin
        phase order — startup runs first, training picks up the GPU once
        the sim is loaded.
        """
```

Implementation:

```python
import shlex
ssh_cmd = (
    f"docker exec {shlex.quote(container_name)} bash -c "
    + shlex.quote(
        f'ts=/workspace/isaaclab/odin_runs/{run_id}/logs/training.stdout.log; '
        f'ss=/workspace/isaaclab/odin_runs/{run_id}/logs/startup.stdout.log; '
        f'for f in $ts $ss; do if [ -s $f ]; then tail -n {n} $f; exit 0; fi; done'
    )
)
# subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", f"{ssh_user}@{host}", ssh_cmd], …)
```

The SSH command uses the same baked options as the existing `_compute_ssh_tail_store` helper — short timeout, batch mode (no password prompt), accept-new host key.

### Tests

`tools/odin/valhalla/dashboard/tests/test_data_running_tail.py` (new, ~120 LOC):

- `test_read_running_tail_returns_lines_from_training_log_when_present` — fake host (monkeypatched subprocess) returning 50 lines; assert list length + content.
- `test_read_running_tail_falls_back_to_startup_when_training_empty` — training file exists but 0 bytes → reads startup instead.
- `test_read_running_tail_returns_empty_list_when_no_logs_yet` — both files missing.
- `test_read_running_tail_caps_at_n_lines` — when log has 200 lines and n=50, returns 50.
- `test_read_running_tail_ssh_failure_returns_empty_with_warning` — SSH timeout → empty list + stderr warning (don't crash the dashboard).
- `test_read_running_tail_handles_binary_garbage_gracefully` — log contains non-UTF-8; we lossy-decode.

---

## Part C — Tab A UI

### Layout

Existing per-row pattern on `failed`/`skipped` rows:

```
[task]  [fw × be]  [seed]  [Failed pill]  [kind pill] ▸ ↻  [host]  [time]
```

`▸` toggles the failure-detail expand row. `↻` toggles the retry queue (just shipped).

Add a new button on **`running`** rows:

```
[task]  [fw × be]  [seed]  [Running pill]  [—]   👁    [host]  [time]
```

`👁` (or `📺` / `Tail` text — bikeshed in implementation) is a `dcc.Button` with id `{type: "tab-a-running-tail-toggle", run_id: <id>}`.

Click toggles a per-run-id presence in `dcc.Store(id="tab-a-running-tail-shown", data=[])`. When a run_id is in that store, an expand row renders below the data row showing the most recent tail (loaded once at expand time + a "Refresh" button to re-fetch).

### State machine

- `tab-a-running-tail-shown` — list of run_ids currently expanded (mirrors `tab-a-expanded-run-ids` for failed rows).
- `tab-a-running-tail-store` — dict mapping `run_id → {"lines": [...], "fetched_at": "ISO"}`. Populated by the click + Refresh callbacks.

### Two callbacks (mirrors the existing ssh-tail pattern)

1. **Toggle visibility:**

   ```python
   @app.callback(
       Output("tab-a-running-tail-shown", "data"),
       Input({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "n_clicks"),
       State({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "id"),
       State("tab-a-running-tail-shown", "data"),
   )
   def _on_running_tail_toggle(n_clicks_list, ids_list, current):
       return _toggle_run_id(current or [], ident["run_id"])  # pure helper
   ```

2. **Fetch lines on first show / refresh click:**

   ```python
   @app.callback(
       Output("tab-a-running-tail-store", "data"),
       Input({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "n_clicks"),
       Input({"type": "tab-a-running-tail-refresh", "run_id": ALL}, "n_clicks"),
       State({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "id"),
       State({"type": "tab-a-running-tail-refresh", "run_id": ALL}, "id"),
       State("tab-a-dispatch-id", "data"),
       State("tab-a-running-tail-store", "data"),
   )
   def _on_running_tail_fetch(...):
       # Use ctx.triggered to find the run_id, find the job's host from
       # data.load_dispatch(dispatch_id), call data.read_running_job_tail(...).
   ```

### Render

Extend `_data_row` to render the `👁` button on `status="running"` rows. Add a sibling `_expand_running_row(job, store)` that renders the tail in a `<pre>` block (analogous to `_expand_row` for failed jobs). The jobs-table loop already inserts an expand row when a run_id is in `expanded_run_ids` — extend it to also check `running_tail_shown`.

### CSS

Reuse `.tab-a-expand-toggle` styling for the button. The expand row reuses `.tab-a-expand-row`. Optional: prefix the rendered `<pre>` text with the source filename (`training.stdout.log` vs `startup.stdout.log`) so the operator knows which phase they're seeing.

### Tests

`tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py` (new, ~150 LOC):

- `test_running_row_has_tail_button` — `_data_row(job, ..., status="running")` includes a `tab-a-running-tail-toggle` button.
- `test_completed_row_does_not_have_tail_button` — `status="completed"` rows don't.
- `test_failed_row_has_retry_and_expand_but_not_tail_button` — orthogonal to retry / expand.
- `test_expand_running_row_renders_lines_in_pre_block`.
- `test_expand_running_row_shows_filename_marker` — `training.stdout.log` vs `startup.stdout.log` annotation visible.
- `test_running_tail_callback_round_trip` (in `test_tab_a_callbacks.py`) — toggle on, store populated; toggle off, store untouched.

---

## File-by-file change estimate

| File | Change | LOC |
|---|---|---|
| `tools/odin/hugin/run.py` | Streaming refactor of `_run_phase`; add `-u` injection | ~40 delta |
| `tools/odin/tests/test_hugin.py` | 5 new streaming tests | ~80 |
| `tools/odin/valhalla/dashboard/data.py` | `read_running_job_tail` | ~50 |
| `tools/odin/valhalla/dashboard/tests/test_data_running_tail.py` | New unit tests | ~120 |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py` | `👁` button on running rows; `_expand_running_row` helper | ~70 delta |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/layout.py` | 2 new dcc.Store entries | ~6 delta |
| `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/callbacks.py` | Toggle + fetch callbacks; pure helpers | ~80 delta |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_running_tail.py` | New UI tests | ~150 |
| `tools/odin/valhalla/dashboard/tests/test_tab_a_callbacks.py` | Add round-trip test | ~30 delta |
| `tools/odin/valhalla/dashboard/assets/style.css` | Reuse existing classes; minor | ~5 delta |
| **Total** | | **~630** including tests |

---

## Implementation order

1. **Hugin streaming** + tests — lands first since it's container-side. Operator rebuilds the image (or waits for the next bootstrap) before downstream work matters.
2. **`read_running_job_tail`** + tests — pure dispatcher-side, no image rebuild.
3. **Tab A UI** + tests — wires (1) + (2) into the dashboard.
4. **Manual smoke** — run a dispatch, click `👁` on a running Anymal-C Rough job, see the per-iter reward / fps lines streaming.

Each step is independently mergeable and testable.

---

## Open questions / decisions

- **Auto-poll the running tail?** No. Adding a 30 s auto-refresh would be simple, but it adds SSH chatter for every operator viewing the dashboard. Ship click-to-fetch first; auto-poll is a follow-up if requested.
- **Tail more than 50 lines?** Make `n` configurable via a `?lines=200` URL param or a small selector. Out of scope for v1.
- **Truncate huge log files?** Hugin's previous `tail_bytes` truncation is gone in this design — full logs always written. Bundle dir grows; the existing rsync-pull already handles arbitrary-sized log files. If this ever becomes a disk problem, add a Hugin-side rotation, not a dashboard truncation.
- **Streaming buffering** — `-u` for Python children handles 95 % of cases. If a non-Python child needs line-buffering and we don't see live output, layer on a `stdbuf -oL -eL` wrapper later.
- **Coexistence with submit-and-poll spec** — when the submit-and-poll spec lands, `_classify_remote` will read these same `training.stderr.log` files. Both specs are mutually compatible and reinforce each other (richer remote stderr → better classifier signal).

---

## What this does NOT solve

- **Active monitoring** — the operator has to click. A future "always-on" tail (Server-Sent Events or a 5 s poll) is a separate feature.
- **Live TB scalars** — per-iter reward/fps remain in the binary tfevents file. A protobuf-parsing UI element is its own feature.
- **Multi-phase view** — startup + training in one pane. Today we show one phase at a time (training, falling back to startup). Could be expanded to two `<pre>` blocks side-by-side later.
- **Already-running jobs** — they used the old Hugin and have no streamed logs. Only matters until they finish.
