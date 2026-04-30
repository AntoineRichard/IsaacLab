# Odin Live Retry Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a live `odin-dispatch` runner consume retry rows for its own dispatch from `.retry.sqlite` and requeue failed jobs while workers are still active.

**Architecture:** Keep the retry DB as the coordination surface. `run_dispatch()` polls `RetryDB(dispatch_dir.parent)` while work remains, resets eligible failed `JobEntry` rows to `pending`, and records retry outcomes through `mark_consumed`.

**Tech Stack:** Python 3.12, `queue.Queue`, existing Odin Asgard runner/worker state events, SQLite `RetryDB`.

---

### Task 1: Live Retry Helper

**Files:**
- Modify: `tools/odin/asgard/runner.py`
- Modify: `tools/odin/tests/test_asgard_runner.py`

- [x] Write tests for `_consume_live_retries` that queue only failed jobs from the current dispatch and preserve `attempts`.
- [x] Run those tests and verify they fail because `_consume_live_retries` does not exist.
- [x] Add `_consume_live_retries` in `runner.py`.
- [x] Run the tests and verify they pass.

### Task 2: Mark Consumed

**Files:**
- Modify: `tools/odin/asgard/runner.py`
- Modify: `tools/odin/tests/test_asgard_runner.py`

- [x] Write tests for marking completed and failed live retry events consumed.
- [x] Run those tests and verify they fail because the helper does not exist.
- [x] Add `_mark_live_retry_consumed` in `runner.py`.
- [x] Run the tests and verify they pass.

### Task 3: Runner Loop Integration

**Files:**
- Modify: `tools/odin/asgard/runner.py`
- Modify: `tools/odin/tests/test_asgard_runner.py`

- [x] Write a threaded fake-runner regression test: one job fails, another keeps the runner live, a DB click requeues the failed job.
- [x] Run it and verify it fails because workers exit after upfront sentinels / runner does not poll DB.
- [x] Remove upfront sentinels, poll `RetryDB` while `remaining > 0`, and mark live retry terminal events consumed.
- [x] Run the threaded test and targeted runner tests.

### Task 4: CLI and Docs

**Files:**
- Modify: `tools/odin/asgard/cli.py`
- Modify: `tools/odin/tests/test_asgard_cli.py`
- Modify: `tools/odin/valhalla/dashboard/tabs/dispatch_fleet/jobs_table.py`
- Modify: `docs/odin/architecture.md`
- Modify: `tools/odin/README.md`

- [x] Add `DispatchOptions.live_retry_poll_s` and CLI `--live_retry_poll_s` with hyphen alias.
- [x] Update retry button tooltip copy.
- [x] Document current-dispatch-only live ingestion.
- [x] Run dashboard + runner tests and `./isaaclab.sh -f`.
