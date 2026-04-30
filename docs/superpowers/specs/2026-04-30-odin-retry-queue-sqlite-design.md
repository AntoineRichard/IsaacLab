# Odin Retry Queue — SQLite-backed

**Status:** Approved (operator: antoiner)
**Author:** Claude (handed off for implementation)
**Date:** 2026-04-30
**Branch context:** `antoiner/feat/odin`, atop the retry-toggle UI commit `f17e1e4eba0`.

---

## Goal

Replace the per-dispatch flat-file retry queue (`odin_runs/<dispatch_id>/retry_queue.txt`) with a single SQLite database at `odin_runs/.retry.sqlite`, plus an `odin-retry` CLI. The flat-file model works for the basic toggle UI we just shipped, but it can't answer cross-dispatch questions, doesn't track whether a queued retry actually ran, and the read-modify-write toggle has a small last-writer-wins race window.

Concretely, after this lands the operator can:
- See pending retries across **all** dispatches in one view.
- Track whether a retry was consumed and whether it ultimately completed or re-failed (and with what failure kind).
- Drive queue/remove from a shell (`odin-retry queue 20260430-110509 rsl-rl_physx_X_seed42`) without poking the dashboard.
- Run multiple dashboard tabs / a CLI / a script concurrently without races (WAL mode handles concurrent readers + one writer).

The dashboard's existing UI (the per-row ↻ button + banner) keeps working unchanged — `DataLayer.read_retry_queue` / `toggle_retry_queue` keep the same signatures; only the implementation switches from file to DB.

---

## Non-goals

- **Replacing the dispatcher's `--retry-all-failed` / `--retry-failed=<csv>` flags.** Those still drive the actual retry; the queue just *records what the operator wants retried*. Outcome wiring (was the retry consumed by a particular dispatch run?) lives in v2; v1 only persists the queue.
- **Net-new auth / multi-user concerns.** The DB is owned by the operator who runs the dashboard; same trust model as `runs_root` today.
- **Migrating any other Odin state to SQLite.** Dispatch.json / aggregate.json / hardware.json stay flat-file; they are read-mostly historical records and don't benefit from a DB.

---

## Background — what exists today

`f17e1e4eba0` added:

- `DataLayer.read_retry_queue(dispatch_id) -> set[str]` — reads `odin_runs/<dispatch_id>/retry_queue.txt`, one run_id per line, returns dedup'd set, empty set if file missing.
- `DataLayer.toggle_retry_queue(dispatch_id, run_id) -> set[str]` — read current set, add or remove `run_id`, atomic write via `tempfile.mkstemp` + `os.replace`.
- Tab A jobs table renders a `↻` per failed/skipped row (pattern-matching id `{type: tab-a-retry-toggle, run_id}`); on click, callback invokes `toggle_retry_queue` and bumps a `tab-a-retry-bump` store so the live-poll re-renders. Banner appears when the per-dispatch queue is non-empty, with the exact `odin-dispatch --resume <id> --retry-failed=<csv>` command.

Tests today: 5 backend + 4 UI/callback. 196 total dashboard tests pass.

The two pain points motivating this change:

1. **Cross-dispatch visibility is impossible.** "Did I queue retries on yesterday's dispatch and forget to run them?" requires `cat odin_runs/*/retry_queue.txt`. A `SELECT * FROM retries WHERE retried_at IS NULL` is the proper shape.
2. **No outcome tracking.** When the operator runs the resume command, we lose track — did the retry succeed? Re-fail? Today the only signal is opening the resumed dispatch's `dispatch.json` and matching run_ids by hand.

---

## Schema

Single table `retries` in `odin_runs/.retry.sqlite`:

```sql
PRAGMA user_version = 1;
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE retries (
    dispatch_id        TEXT    NOT NULL,
    run_id             TEXT    NOT NULL,
    queued_at          TEXT    NOT NULL,   -- ISO 8601 UTC, when toggle_retry_queue added it
    note               TEXT,               -- free-form operator note ('flaky network', etc.)
    retried_at         TEXT,               -- ISO 8601 UTC when a --resume consumed it; NULL = pending
    retry_dispatch_id  TEXT,               -- dispatch_id that re-ran it (= dispatch_id when --resume)
    retry_outcome      TEXT CHECK (retry_outcome IN ('completed', 'failed') OR retry_outcome IS NULL),
    retry_failure_kind TEXT,               -- if retry_outcome = 'failed', the FailureInfo.kind
    PRIMARY KEY (dispatch_id, run_id)
);

CREATE INDEX idx_retries_pending ON retries(dispatch_id) WHERE retried_at IS NULL;
CREATE INDEX idx_retries_global_pending ON retries(retried_at) WHERE retried_at IS NULL;
```

`PRAGMA user_version = 1` is the schema version — bump on future migrations.
`WAL` mode lets multiple readers (dashboard tabs, CLI, scripts) coexist with a single writer without lock contention.

---

## Components and responsibilities

### `tools/odin/valhalla/dashboard/retry_db.py` (new, ~120 LOC)

Pure functions and a `RetryDB` class that wraps short-lived connections:

```python
class RetryDB:
    def __init__(self, runs_root: Path) -> None: ...
    def _connect(self) -> sqlite3.Connection: ...      # opens, sets WAL, applies pending migrations
    def read_pending(self, dispatch_id: str) -> set[str]: ...
    def toggle(self, dispatch_id: str, run_id: str, *, note: str | None = None) -> set[str]: ...
    def list_all(self, *, pending_only: bool = False) -> list[RetryRow]: ...
    def list_for_dispatch(self, dispatch_id: str) -> list[RetryRow]: ...
    def mark_consumed(self, dispatch_id: str, run_id: str, *, retry_dispatch_id: str,
                      outcome: str, failure_kind: str | None = None) -> None: ...
    def remove(self, dispatch_id: str, run_id: str) -> None: ...   # hard delete (rare; toggle is the usual path)
```

`RetryRow` is a frozen dataclass mirroring the table columns.

Schema migration shape:

```python
_MIGRATIONS: dict[int, str] = {
    1: """CREATE TABLE retries (...); CREATE INDEX ...""",
}
def _migrate(con: sqlite3.Connection) -> None:
    cur = con.execute("PRAGMA user_version")
    current = cur.fetchone()[0]
    for v, sql in sorted(_MIGRATIONS.items()):
        if v > current:
            con.executescript(sql)
            con.execute(f"PRAGMA user_version = {v}")
    con.commit()
```

The first connection on a fresh `runs_root` runs migration 1 (creates the table). Subsequent connections see `user_version=1` and skip.

### `DataLayer` shim

`DataLayer.read_retry_queue` / `toggle_retry_queue` keep their existing public signatures. Their bodies become trivial passthroughs to `self._retry_db`:

```python
def read_retry_queue(self, dispatch_id: str) -> set[str]:
    return self._retry_db.read_pending(dispatch_id)

def toggle_retry_queue(self, dispatch_id: str, run_id: str) -> set[str]:
    return self._retry_db.toggle(dispatch_id, run_id)
```

`self._retry_db = RetryDB(self._runs_root)` is constructed lazily on first access (so a DataLayer pointed at a missing `runs_root` doesn't fail constructing the DB).

### `tools/odin/valhalla/dashboard/retry_cli.py` (new, ~150 LOC)

Argparse-based CLI installed as the `odin-retry` console_script:

```
odin-retry list                                # all pending retries across all dispatches
odin-retry list --dispatch 20260430-110509     # pending retries for one dispatch
odin-retry list --all                          # include consumed retries (history)
odin-retry queue <dispatch_id> <run_id> [--note "..."]
odin-retry remove <dispatch_id> <run_id>
odin-retry status                              # summary: N pending across M dispatches
odin-retry export-resume-cmd <dispatch_id>     # emit the odin-dispatch --resume … --retry-failed=… string
```

`list` output is tab-separated for grep / cut composability:

```
dispatch_id          run_id                                       queued_at             note
20260430-110509      rsl-rl_physx_Isaac-Ant-Direct-v0_…seed42     2026-04-30T11:48:12Z  network-blip
20260430-110509      rsl-rl_physx_Isaac-Ant-Direct-v0_…seed43     2026-04-30T11:48:13Z
20260430-110509      rsl-rl_physx_Isaac-Ant-Direct-v0_…seed44     2026-04-30T11:48:14Z
```

Console_script wiring goes in `source/isaaclab/setup.py` next to the existing `odin-dispatch`, `odin-bootstrap`, `odin-recover`, `odin-cuda`, `odin-aggregate`, `odin-dashboard`.

### Migration from existing `retry_queue.txt`

On first `_connect` for a given `runs_root`, after the schema migration, scan the runs root once for legacy text files:

```python
def _maybe_import_legacy(con, runs_root: Path) -> None:
    # Only import if the retries table is empty AND legacy txts exist.
    count = con.execute("SELECT COUNT(*) FROM retries").fetchone()[0]
    if count > 0:
        return
    rows = []
    for txt in runs_root.glob("*/retry_queue.txt"):
        dispatch_id = txt.parent.name
        for line in txt.read_text().splitlines():
            run_id = line.strip()
            if not run_id:
                continue
            rows.append((dispatch_id, run_id, _file_mtime_iso(txt)))
    if rows:
        con.executemany(
            "INSERT OR IGNORE INTO retries(dispatch_id, run_id, queued_at) VALUES (?, ?, ?)",
            rows,
        )
        con.commit()
```

The legacy `.txt` files are **left in place** after import — they're an audit artifact. Subsequent toggles only touch the DB. A clear-out helper can be added later if it ever becomes confusing.

### Optional dispatcher hook (out of scope for v1)

The runner already supports `--retry-failed=<csv>` and `--retry-all-failed`. A v2 enhancement would have the runner call `RetryDB.mark_consumed(...)` for every retried run_id at the end of the dispatch with the resulting outcome, so the dashboard / CLI can show "X of these have been re-run and Y succeeded." The DB schema supports this today (the `retried_at` / `retry_outcome` / `retry_failure_kind` columns); writing the wiring is its own change.

For v1, those columns stay NULL except in tests that exercise `mark_consumed` directly.

---

## Concurrency model

- `journal_mode=WAL` on every connection.
- Each `RetryDB` method opens a fresh connection, runs in a single transaction, commits, closes. No long-lived shared connection.
- `toggle` uses a single transaction:

  ```sql
  BEGIN IMMEDIATE;
  -- check if (dispatch_id, run_id) is in the table with retried_at IS NULL
  -- if yes: DELETE
  -- if no: INSERT OR REPLACE (re-queueing a previously-consumed row clears the consume metadata)
  COMMIT;
  ```

- `BEGIN IMMEDIATE` acquires a RESERVED lock at txn start so two simultaneous toggles serialise rather than racing read-modify-write. Last writer doesn't clobber — second toggle sees the post-commit state and acts on it.

- Readers don't block writers in WAL.

---

## Backward compatibility

- DataLayer's two existing methods (`read_retry_queue`, `toggle_retry_queue`) keep their public signatures and behavior. The dashboard UI (8 modules touched in `f17e1e4eba0`) is unchanged.
- Existing `retry_queue.txt` files are imported on first DB write. Operators who never touched the retry UI before this change end up with an empty DB and a normal first-run experience.
- The 5 backend tests in `test_data.py` need an update to assert against the DB rather than the txt file (see *Testing*).
- The existing UI tests (4 jobs-table tests, 1 callback test) keep passing as-is — they exercise behavior at the DataLayer interface, which is preserved.

---

## Testing strategy

### Unit tests, plain `python3 -m pytest --noconftest -p no:cacheprovider`

`tools/odin/valhalla/dashboard/tests/test_retry_db.py` (new, ~250 LOC):

- `test_fresh_db_creates_schema_and_pragmas` — connect on empty runs_root, assert table exists, `user_version=1`, `journal_mode=wal`.
- `test_toggle_adds_then_removes` — round-trip identical to today's flat-file behavior.
- `test_toggle_repeat_after_consume_re_queues` — once `mark_consumed` has set `retried_at`, a fresh `toggle` re-queues (clears the consume metadata).
- `test_read_pending_excludes_consumed` — adds two; consumes one; `read_pending` returns the other.
- `test_list_all_pending_only_default` — only pending rows by default; `--all` returns history too.
- `test_list_for_dispatch` — scoped to one dispatch.
- `test_mark_consumed_records_outcome_and_kind`.
- `test_concurrent_toggle_serialises` — two threads toggling the same `(dispatch_id, run_id)` 1000 times each end with deterministic state (count of completed toggles is even ⇒ row absent; odd ⇒ row present).
- `test_legacy_txt_imported_on_first_write` — pre-populate `<runs_root>/<id>/retry_queue.txt`, open DB, assert rows imported with `queued_at = file mtime`.
- `test_legacy_txt_import_skipped_when_db_nonempty` — second open is a no-op.

`tools/odin/valhalla/dashboard/tests/test_data.py` (extend, ~5 LOC delta):

- The 5 existing retry-queue tests remain. Their assertions on disk-side `retry_queue.txt` get loosened to "the persisted state matches what `read_retry_queue` returns" so they don't pin the storage backend. Or replace them with the equivalent assertions through `read_retry_queue` / `toggle_retry_queue` only.

`tools/odin/valhalla/dashboard/tests/test_retry_cli.py` (new, ~150 LOC):

- `test_list_pending_outputs_tsv` — populates DB, `odin-retry list` writes the TSV with the right columns.
- `test_list_dispatch_filters` — `--dispatch <id>` scopes correctly.
- `test_queue_and_remove_round_trip` — CLI surface mirrors `RetryDB.toggle`.
- `test_export_resume_cmd_emits_csv_in_alphabetical_order` — output is one line, parseable by an operator's shell.
- `test_status_summarises` — counts per-dispatch.

### Manual smoke

1. Start a dispatch.
2. Click `↻` on a few failed rows in the dashboard.
3. `odin-retry list` shows them.
4. `odin-retry export-resume-cmd <dispatch_id>` emits the resume command.
5. Run that command (the dispatcher); jobs re-attempt.
6. (v2 only) `odin-retry list --all` shows the consumed rows with `retry_outcome` set.

---

## Implementation order preview

1. **`retry_db.py`** with schema + migrations + `RetryDB` class + tests. (~120 src + 250 test LOC)
2. **`DataLayer` shim** — replace the txt-backed methods with passthroughs. Update existing `test_data.py` retry tests to be backend-agnostic. (~10 src + 10 test LOC delta)
3. **Migration step** — `_maybe_import_legacy` runs once on first connect; tested. (~30 src + 30 test LOC)
4. **`retry_cli.py`** + `odin-retry` console_script registration in `source/isaaclab/setup.py`. (~150 src + 150 test LOC)
5. **Doc note** in `docs/odin/architecture.md` change-log + a one-line README pointer. (~5 LOC)

Total ~500 LOC including tests.

---

## Open questions / decisions

- **DB path**: `odin_runs/.retry.sqlite` vs `~/.odin/retry.sqlite`. I picked the former so the queue lives next to the bundles it references and a fresh runs-root starts clean. If multiple `runs_root`s coexist, each gets its own DB — fine.
- **Legacy txt cleanup**: leave for now (post-import audit). Add a `odin-retry vacuum-legacy` command later if the txts ever cause confusion.
- **Schema versioning**: PRAGMA-based, monotone integer. Future migrations append entries to `_MIGRATIONS`.
- **Outcome wiring**: deferred to v2 (the dispatcher learns to call `mark_consumed` after `--retry-failed=…`). Not blocking the queue work.
- **Dashboard "All retries" tab**: out of scope; operator can `odin-retry list` for now.

---

## Files touched

| File | Change | LOC |
|---|---|---|
| `tools/odin/valhalla/dashboard/retry_db.py` | New: `RetryDB` + schema + migrations + legacy-import. | ~120 |
| `tools/odin/valhalla/dashboard/retry_cli.py` | New: argparse `odin-retry` CLI. | ~150 |
| `tools/odin/valhalla/dashboard/data.py` | Replace 2 method bodies; add `_retry_db` lazy attribute. | ~20 delta |
| `source/isaaclab/setup.py` | Register `odin-retry` console_script. | ~3 |
| `tools/odin/valhalla/dashboard/tests/test_retry_db.py` | New unit tests. | ~250 |
| `tools/odin/valhalla/dashboard/tests/test_retry_cli.py` | New CLI tests. | ~150 |
| `tools/odin/valhalla/dashboard/tests/test_data.py` | Loosen / update 5 retry tests. | ~10 delta |
| `docs/odin/architecture.md` | Change-log entry. | ~5 |
| **Total** | | **~700** including tests |

---

## What this does NOT solve

- Retries that the operator typed by hand into `--retry-failed=…` (without using the dashboard / CLI) won't be tracked. v2 wiring closes that gap by hooking the runner.
- The DB is single-user. If two operators ever share a runs_root, the queue is shared too — that's intentional (collaborative ops) but worth documenting.
- Backups: if the operator deletes `.retry.sqlite`, all queue history is gone. The legacy `.txt` files (left in place after import) are a minor backstop for the *first* import; subsequent state has no shadow. If durability beyond that ever matters, daily backup of the file is trivial — out of scope here.
