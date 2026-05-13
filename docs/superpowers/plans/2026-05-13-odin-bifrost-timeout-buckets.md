# Bifrost per-task timeouts via timeout-class buckets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split a Bifrost dispatch into N OSMO workflows keyed on `timeout_class`, each chunk capped at `chunk_size` (default 25) tasks, so per-task budgets can vary (Cartpole 30m, Shadow-Vision 8h).

**Architecture:** New planner step buckets rows by `timeout_class` (read from curated env YAML), chunks each bucket, renders one OSMO workflow per chunk with the class's `exec_timeout`. State adds `osmo_workflow_ids: list[str]` (plus per-job `osmo_workflow_id`); poller walks all workflows each tick. Aggregate is unchanged.

**Tech Stack:** Python 3.12, pytest, pre-commit (`./isaaclab.sh -f`). Working directory `/home/antoiner/Documents/IsaacLab` on branch `antoiner/feat/odin`.

**Spec:** `docs/superpowers/specs/2026-05-13-odin-bifrost-timeout-buckets-design.md`.

---

## File Structure

**Modified (existing):**
- `tools/odin/bifrost/cli.py` — `_PlannedRow` gains `timeout_class`; new `_bucket_and_chunk` planner; `main()` orchestrates N submissions.
- `tools/odin/bifrost/config.py` — `BifrostConfig.timeout_classes: dict[str, str]`, `BifrostConfig.default_timeout_class: str`, `BifrostConfig.chunk_size: int`; deprecation path for `defaults.exec_timeout`.
- `tools/odin/bifrost/workflow.py` — `RenderRow` already carries enough; render function takes per-chunk `exec_timeout` instead of pulling from config.
- `tools/odin/bifrost/templates/dispatch.yaml.j2` — replace `{{ cfg.defaults.exec_timeout }}` with `{{ exec_timeout }}` (template-level variable, not from cfg).
- `tools/odin/bifrost/poller.py` — `poll_until_terminal` walks `state.osmo_workflow_ids` each tick instead of single `state.osmo_workflow_id`.
- `tools/odin/asgard/state.py` — `DispatchState.osmo_workflow_ids: list[str]`; serializer + deserializer; migration from old `osmo_workflow_id` field.
- `tools/odin/asgard/jobs.py` — `JobEntry.osmo_workflow_id` semantics doc updated (already exists, no schema change).
- `tools/odin/config/physx_envs.yaml`, `tools/odin/config/newton_envs.yaml` — add `timeout_class` per kept env (separate PR / commit, can land alongside parser).
- `tools/odin/config/bifrost-osmo.yaml.example` — add `timeout_classes`, `default_timeout_class`, `chunk_size`.
- `tools/odin/config/bifrost-osmo.yaml` — same fields, user's local copy.

**New tests:**
- `tools/odin/tests/test_bifrost_bucket_and_chunk.py` — pure-function tests on the bucketing logic.
- `tools/odin/tests/test_bifrost_config_timeout_classes.py` — config loader for new fields + deprecation.
- `tools/odin/tests/test_bifrost_envs_timeout_class.py` — curated YAML loader handles `timeout_class`, fallback, unknown → raise.
- `tools/odin/tests/test_bifrost_cli_multi_workflow.py` — `main()` orchestration with mock `OsmoClient` submits N workflows.
- `tools/odin/tests/test_bifrost_poller_multi_workflow.py` — poller walks multiple wf-ids per tick.
- `tools/odin/tests/test_asgard_state_workflow_ids_migration.py` — old dispatch.json (`osmo_workflow_id` only) loads into new schema.

**Modified tests (existing):**
- `tools/odin/tests/test_bifrost_cli.py` — update fixtures to set `timeout_classes` in config.
- `tools/odin/tests/test_bifrost_poller.py` — update fixtures to use the new state field.

**Out of scope:** Asgard-side per-task timeout plumbing (curated YAML field is shared but Asgard's read path is separate). The dashboard's heimdall card / wedge detection (separate).

---

## Conventions

- **Test command:** `PYTHONPATH=. python3 -m pytest --noconftest -p no:cacheprovider <path>::<name> -v`. Always include `--noconftest -p no:cacheprovider`.
- **Pre-commit:** `./isaaclab.sh -f` before every commit. Re-stage if it reformats.
- **Commit messages:** Imperative mood, ~50-char subject, blank line, body wrapping at 72 chars. No AI-coauthor lines.
- **TDD:** Each code-changing task starts with a failing test, demonstrates the failure, implements, demonstrates pass.

---

# Batch 1 — Config + curated-YAML schema

Goal: introduce `timeout_class` everywhere the rows are read, without
yet changing the submit path. Existing dispatches continue to work
through the legacy single-workflow path.

## Task 1.1 — Add `timeout_classes` + `default_timeout_class` + `chunk_size` to `BifrostConfig`

**Files:**
- Modify: `tools/odin/bifrost/config.py` (add fields, parse from YAML, deprecation warning on `defaults.exec_timeout`)
- Create: `tools/odin/tests/test_bifrost_config_timeout_classes.py`

**Test first:** Config with `timeout_classes: { short: 30m, medium: 2h }` loads into a `dict[str, str]`. Missing `timeout_classes` keeps old behavior. Both present → deprecation warning logged.

- [ ] Write failing tests in `test_bifrost_config_timeout_classes.py`
- [ ] Run tests, confirm failures
- [ ] Add fields to `BifrostConfig` dataclass
- [ ] Update `load_bifrost_config` to parse new fields
- [ ] Tests pass
- [ ] Run `./isaaclab.sh -f`, commit

## Task 1.2 — Curated-YAML env loader reads `timeout_class`

**Files:**
- Modify: `tools/odin/bifrost/cli.py:_load_envs_yaml` and/or `_build_rows`
- Create: `tools/odin/tests/test_bifrost_envs_timeout_class.py`

**Test first:** An env with `timeout_class: short` produces a `_PlannedRow` with that field. Missing field → fallback to `cfg.default_timeout_class`. Class not in `cfg.timeout_classes` → `BifrostConfigError` with a clear message.

- [ ] Add `timeout_class: str` to `_PlannedRow` dataclass
- [ ] Update `_build_rows` to read + validate
- [ ] Failing tests pass
- [ ] Pre-commit, commit

## Task 1.3 — Populate `timeout_class` in curated YAMLs

**Files:**
- Modify: `tools/odin/config/physx_envs.yaml`, `tools/odin/config/newton_envs.yaml`

Pure data change: pick a class per kept env. Use the existing
`per_job_timeout_s` values asgard's `dispatch_cli.py` references (if
present) as a guide. Default heuristic:

- `< 30m`: `short`
- `30m – 2h`: `medium`
- `2h – 8h`: `long`
- `> 8h`: `very_long`

- [ ] Run a quick analysis script to suggest a class per env from observed asgard runtimes (manifests in `odin_runs/*/`)
- [ ] Apply suggestions; spot-check
- [ ] Pre-commit, commit

---

# Batch 2 — Planner: `_bucket_and_chunk`

Goal: ship the pure-function bucketing/chunking step. Submit path
still unchanged until Batch 3.

## Task 2.1 — Implement `_bucket_and_chunk(rows, chunk_size) -> list[Bucket]`

**Files:**
- Modify: `tools/odin/bifrost/cli.py` (new helper)
- Create: `tools/odin/tests/test_bifrost_bucket_and_chunk.py`

**Test first:**

- Empty rows → empty list.
- 3 rows all same class → 1 bucket.
- 3 rows, 2 classes → 2 buckets, deterministic order by class name.
- 50 rows same class, chunk_size=25 → 2 chunks of 25.
- 51 rows same class, chunk_size=25 → 3 chunks of (25, 25, 1).
- Rows within a chunk are sorted by `(task_id, backend, seed)`.

- [ ] Write all 6 failing test cases
- [ ] Implement helper
- [ ] Pass
- [ ] Pre-commit, commit

---

# Batch 3 — State schema: `osmo_workflow_ids`

Goal: storage layer accepts a list of OSMO workflow ids, with
forward + backward compatibility.

## Task 3.1 — `DispatchState.osmo_workflow_ids: list[str]`

**Files:**
- Modify: `tools/odin/asgard/state.py`
- Create: `tools/odin/tests/test_asgard_state_workflow_ids_migration.py`

**Test first:**

- New dispatch: `state.osmo_workflow_ids` defaults to `[]`.
- Legacy `dispatch.json` with `osmo_workflow_id: "abc"` and no
  `osmo_workflow_ids` → loads as `osmo_workflow_ids=["abc"]`.
- Both fields present (post-migration write) → list wins; single
  field ignored on read but written for old-reader compat.
- Round-trip stability: write → read → equality.

- [ ] Failing tests
- [ ] Implement field, migration in `read_dispatch_state`, serialization in `write_dispatch_state`
- [ ] Pass
- [ ] Pre-commit, commit

## Task 3.2 — `JobEntry.osmo_workflow_id` per-job (already exists; doc update)

The field exists; semantics document needs updating to reflect
"the chunk-workflow this run lives in" instead of "the only
workflow." No code change. One-line docstring update.

- [ ] Docstring update
- [ ] Pre-commit, commit

---

# Batch 4 — Template + workflow.py

Goal: render one workflow per chunk with the class's exec_timeout.

## Task 4.1 — Template takes `exec_timeout` as a parameter

**Files:**
- Modify: `tools/odin/bifrost/templates/dispatch.yaml.j2` (replace `cfg.defaults.exec_timeout` with `exec_timeout`)
- Modify: `tools/odin/bifrost/workflow.py` (`render_workflow_yaml` gains `exec_timeout: str` param)
- Modify: `tools/odin/tests/test_bifrost_workflow.py` (existing render tests pass `exec_timeout` explicitly)

**Test first:** Two render calls with the same rows but different
`exec_timeout` produce YAML with the right `workflow.timeout.exec_timeout`.

- [ ] Failing tests
- [ ] Implement
- [ ] Pass
- [ ] Pre-commit, commit

---

# Batch 5 — CLI orchestration: submit N workflows

Goal: `main()` buckets, renders, submits, persists state per
workflow.

## Task 5.1 — `main()` submits one workflow per bucket

**Files:**
- Modify: `tools/odin/bifrost/cli.py` (`main()` for the forward path; resume path follows in Task 5.2)
- Create: `tools/odin/tests/test_bifrost_cli_multi_workflow.py`

**Test first (with mock OsmoClient):**

- 3 rows × 2 classes → 2 calls to `client.submit()`, each with the
  correct `exec_timeout` in the rendered YAML.
- After all submits, `state.osmo_workflow_ids` has 2 entries.
- Each `JobEntry.osmo_workflow_id` is set to the workflow its row was
  in.
- If submit #2 raises, submit #1's workflow_id is persisted in
  dispatch.json (resumable).

- [ ] Failing tests
- [ ] Implement (loop over buckets, append to `osmo_workflow_ids`, write state after each submit)
- [ ] Pass
- [ ] Pre-commit, commit

## Task 5.2 — Resume path uses `osmo_workflow_ids`

**Files:**
- Modify: `tools/odin/bifrost/cli.py` (`--resume` branch)
- Modify: `tools/odin/tests/test_bifrost_cli.py` (resume tests use list)

**Test first:** Resume with a dispatch.json containing 2 workflow ids
re-attaches the poller to both.

- [ ] Failing tests
- [ ] Implement
- [ ] Pass
- [ ] Pre-commit, commit

---

# Batch 6 — Poller: walk all workflows per tick

Goal: `poll_until_terminal` aggregates per-task state across N
workflows.

## Task 6.1 — Poller iterates `state.osmo_workflow_ids`

**Files:**
- Modify: `tools/odin/bifrost/poller.py`
- Modify: `tools/odin/tests/test_bifrost_poller.py` (update fixtures)
- Create: `tools/odin/tests/test_bifrost_poller_multi_workflow.py`

**Test first (with mock client):**

- 2 workflows, 3 tasks each. Tick 1 returns all RUNNING on both →
  state has 6 jobs RUNNING. Tick 2 returns all COMPLETED → state has
  6 COMPLETED. `_all_terminal` returns True.
- Workflow 1 RUNNING, workflow 2 COMPLETED → poll loop continues until
  workflow 1 also reaches terminal.
- Empty `osmo_workflow_ids` → loop exits immediately (defensive).

- [ ] Failing tests
- [ ] Implement (a thin wrapper that loops `for wf_id in
      state.osmo_workflow_ids: snap = client.status(wf_id); merge...`)
- [ ] Pass
- [ ] Pre-commit, commit

---

# Batch 7 — Integration

## Task 7.1 — End-to-end Bifrost dispatch with 2 classes (mock OSMO)

**Files:**
- Modify: `tools/odin/tests/test_bifrost_e2e.py` (or sibling)

**Test:** 6 tasks, 3 short + 3 medium, with a `FakeOsmoClient` that
records every submit/query call. Assert:

- 2 workflows submitted with the right timeouts.
- After all tasks reach COMPLETED, dispatch.json shows all jobs
  completed.
- `aggregate.json` is written.

- [ ] Tests
- [ ] Pre-commit, commit

## Task 7.2 — Slow-marked real-OSMO smoke

**Files:**
- New script under `tools/odin/bifrost/scripts/smoke_timeout_buckets.py` (or extend existing)

Manually run: 6 tasks (3 Cartpole short + 3 Ant medium) against the
real `isaac-dev-l40s-04` pool. Verify in OSMO web UI that the two
workflows have different `exec_timeout` values; verify bundles
download; verify `aggregate.json` is correct.

- [ ] Smoke runs clean
- [ ] Document in PR description

---

# Batch 8 — Polish

- [ ] Update `tools/odin/README.md` — add a "Bifrost: timeout classes" section.
- [ ] Update `docs/odin/2026-04-28-eval-run-punchlist.md` — close the per-task-timeouts item with a pointer to this spec/plan.
- [ ] Pre-commit, commit.
