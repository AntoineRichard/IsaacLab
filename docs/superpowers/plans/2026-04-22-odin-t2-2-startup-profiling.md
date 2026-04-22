# Odin T2.2 — Startup Profiling Survey & Reliability Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two T1-carried reliability caveats (`CProfileFunction.calls` always 0; `Resources.*.peak` copying mean), re-tune the startup whitelist to cover all five phases, and publish `docs/odin/startup_profiling_survey.md` grounded in a fresh profile run.

**Architecture:** Four narrow upstream-IsaacLab changes (parse-cprofile ncalls, benchmark-startup call sites, recorder peak tracking, training-bundle peak wiring) plus a survey doc and an updated whitelist. No schema version bump — both fixes populate existing v1.0 fields correctly. No Odin-side (`tools/odin/**`) code touched.

**Tech Stack:** Python 3.10+, pytest, cProfile/pstats, psutil, pynvml/nvidia-smi/torch.cuda, PyYAML, `@configclass` dataclasses.

**Spec:** `docs/superpowers/specs/2026-04-22-odin-t2-2-startup-profiling-design.md`.

**Branch:** `antoiner/feat/odin` (local commits only; do not push).

**Commit convention:** Imperative subject ~50 chars, body explains *why*, no AI co-authorship. `commit.gpgsign=true` is configured globally — do not pass `-c commit.gpgsign=false` or `--no-gpg-sign`. If GPG signing times out, prime the agent with `echo "test" | gpg --clearsign > /dev/null` and retry the commit.

**Project rules**: Python always via `./isaaclab.sh -p`; tests via `./isaaclab.sh -p -m pytest PATH -v`. Run `./isaaclab.sh -f` BEFORE `git commit`. Pre-existing codespell / ruff failures in unrelated files are noise — check the files *you* touched pass.

---

## Task 1: Write failing tests for `parse_cprofile_stats` ncalls contract

**Goal:** Pin the new 4-tuple contract `(label, tottime_ms, cumtime_ms, ncalls)` via a red test before touching production code.

**Files:**
- Create: `source/isaaclab/test/benchmark/test_parse_cprofile_stats.py`

- [ ] **Step 1: Create the new test file**

Create `source/isaaclab/test/benchmark/test_parse_cprofile_stats.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`scripts.benchmarks.utils.parse_cprofile_stats`.

The function is expected to return 4-tuples
``(label, tottime_ms, cumtime_ms, ncalls)`` after the T2.2 reliability fix.
Before the fix, the function returned 3-tuples and CProfileFunction.calls was
always 0 in the downstream startup bundle.
"""

from __future__ import annotations

import cProfile
import os
import sys

# scripts/benchmarks/utils.py is not an installable package; add the repo
# root to sys.path so the import works.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.benchmarks.utils import parse_cprofile_stats  # noqa: E402


def _profiled_call(n_outer: int, n_inner: int) -> cProfile.Profile:
    """Run a couple of nested helpers a known number of times under cProfile."""

    def inner():
        return sum(range(10))

    def outer():
        for _ in range(n_inner):
            inner()

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(n_outer):
        outer()
    prof.disable()
    return prof


def test_top_n_returns_ncalls():
    # The synthetic functions live in THIS test file, so _is_isaaclab will
    # not match them — they come through the "first-level external call from
    # an IsaacLab caller" path only if we pass this file's directory as an
    # isaaclab prefix. Do so to include them.
    test_dir = os.path.abspath(os.path.dirname(__file__))
    prof = _profiled_call(n_outer=3, n_inner=5)

    results = parse_cprofile_stats(prof, isaaclab_prefixes=[test_dir], top_n=30)

    # Each row must be a 4-tuple now.
    assert results, "parse_cprofile_stats should return at least one row"
    for row in results:
        assert len(row) == 4, f"expected (label, tot, cum, ncalls) 4-tuple, got {row!r}"
        label, tot, cum, ncalls = row
        assert isinstance(label, str)
        assert isinstance(tot, float)
        assert isinstance(cum, float)
        assert isinstance(ncalls, int)
        assert ncalls >= 0

    # Locate our two functions by suffix and check their call counts.
    outer_rows = [r for r in results if r[0].endswith(":outer")]
    inner_rows = [r for r in results if r[0].endswith(":inner")]
    assert outer_rows, f"outer() should be in results, got labels: {[r[0] for r in results]}"
    assert inner_rows, f"inner() should be in results, got labels: {[r[0] for r in results]}"
    assert outer_rows[0][3] == 3, f"outer ncalls should be 3, got {outer_rows[0][3]}"
    assert inner_rows[0][3] == 15, f"inner ncalls should be 3*5=15, got {inner_rows[0][3]}"


def test_whitelist_path_returns_ncalls():
    test_dir = os.path.abspath(os.path.dirname(__file__))
    prof = _profiled_call(n_outer=2, n_inner=4)

    results = parse_cprofile_stats(
        prof,
        isaaclab_prefixes=[test_dir],
        whitelist=["*:inner", "*:definitely_not_a_real_function"],
    )

    # Matched row carries the real ncalls; placeholder row carries 0.
    labels = {r[0]: r for r in results}
    inner_label = next((l for l in labels if l.endswith(":inner")), None)
    assert inner_label is not None, f"inner() should match wildcard whitelist, labels: {list(labels)}"
    assert labels[inner_label][3] == 8, f"inner ncalls should be 2*4=8, got {labels[inner_label][3]}"

    placeholder = labels.get("*:definitely_not_a_real_function")
    assert placeholder is not None, "placeholder row should be emitted for unmatched pattern"
    assert placeholder == ("*:definitely_not_a_real_function", 0.0, 0.0, 0)
```

- [ ] **Step 2: Run the tests and verify they fail**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_parse_cprofile_stats.py -v
```

Expected failure mode: `AssertionError: expected (label, tot, cum, ncalls) 4-tuple, got (...)` — the current implementation returns 3-tuples.

- [ ] **Step 3: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/test/benchmark/test_parse_cprofile_stats.py
git commit -m "Add failing ncalls contract test for parse_cprofile_stats

parse_cprofile_stats currently drops CPython's ncalls field from
pstats.Stats.stats, which is why CProfileFunction.calls is always 0
in the schema-v1 startup bundle. Pin the new 4-tuple contract
(label, tottime_ms, cumtime_ms, ncalls) with a red test before
changing the function signature in the next commit."
```

---

## Task 2: Implement ncalls in `parse_cprofile_stats` and update call sites

**Goal:** Return 4-tuples from `parse_cprofile_stats`; feed the real call count into `CProfileFunction.calls` at both call sites in `benchmark_startup.py`.

**Files:**
- Modify: `scripts/benchmarks/utils.py` (lines around the main `for func_key, ... in stats.stats.items()` loop and the whitelist return block)
- Modify: `scripts/benchmarks/benchmark_startup.py` (two call sites near lines 254 and 408)

- [ ] **Step 1: Update the main loop in `parse_cprofile_stats`**

Find in `scripts/benchmarks/utils.py` the block:

```python
    results = []
    for func_key, (_, _, tottime, cumtime, callers) in stats.stats.items():
        filename, _, funcname = func_key
        if _is_isaaclab(filename):
            label = _make_label(filename, funcname)
            results.append((label, tottime * 1000.0, cumtime * 1000.0))
        else:
            # Check if any direct caller is an IsaacLab function
            for caller_key in callers:
                caller_filename = caller_key[0]
                if _is_isaaclab(caller_filename):
                    label = _make_label(filename, funcname)
                    results.append((label, tottime * 1000.0, cumtime * 1000.0))
                    break
```

Replace with (unpack `ncalls` as the second value in the stats tuple — the inline comment in the file already documents the layout as `(pcalls, ncalls, tottime, cumtime, callers)`):

```python
    results = []
    for func_key, (_, ncalls, tottime, cumtime, callers) in stats.stats.items():
        filename, _, funcname = func_key
        if _is_isaaclab(filename):
            label = _make_label(filename, funcname)
            results.append((label, tottime * 1000.0, cumtime * 1000.0, ncalls))
        else:
            # Check if any direct caller is an IsaacLab function
            for caller_key in callers:
                caller_filename = caller_key[0]
                if _is_isaaclab(caller_filename):
                    label = _make_label(filename, funcname)
                    results.append((label, tottime * 1000.0, cumtime * 1000.0, ncalls))
                    break
```

- [ ] **Step 2: Update the whitelist block in `parse_cprofile_stats`**

Find the block:

```python
    # Whitelist mode: filter by fnmatch patterns, emit placeholders for unmatched patterns
    matched: dict[str, tuple[str, float, float]] = {}
    matched_patterns: set[str] = set()
    for label, tottime, cumtime in results:
        for pattern in whitelist:
            if fnmatch.fnmatch(label, pattern):
                if label not in matched:
                    matched[label] = (label, tottime, cumtime)
                matched_patterns.add(pattern)

    # Add 0.0 placeholders for patterns that matched nothing
    for pattern in whitelist:
        if pattern not in matched_patterns:
            print(
                f"[WARNING] Whitelist pattern '{pattern}' matched no profiled functions. "
                "Check for typos or verify the function ran during this phase."
            )
            matched[pattern] = (pattern, 0.0, 0.0)

    filtered = list(matched.values())
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered
```

Replace with:

```python
    # Whitelist mode: filter by fnmatch patterns, emit placeholders for unmatched patterns
    matched: dict[str, tuple[str, float, float, int]] = {}
    matched_patterns: set[str] = set()
    for label, tottime, cumtime, ncalls in results:
        for pattern in whitelist:
            if fnmatch.fnmatch(label, pattern):
                if label not in matched:
                    matched[label] = (label, tottime, cumtime, ncalls)
                matched_patterns.add(pattern)

    # Add 0.0 placeholders for patterns that matched nothing. Placeholder rows
    # keep the schema shape (still a 4-tuple) and carry ncalls=0 — semantically
    # "this pattern matched nothing, so no call count is meaningful."
    for pattern in whitelist:
        if pattern not in matched_patterns:
            print(
                f"[WARNING] Whitelist pattern '{pattern}' matched no profiled functions. "
                "Check for typos or verify the function ran during this phase."
            )
            matched[pattern] = (pattern, 0.0, 0.0, 0)

    filtered = list(matched.values())
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered
```

- [ ] **Step 3: Update the docstring return type**

Find the docstring in `parse_cprofile_stats` and replace the `Returns:` section:

```python
    Returns:
        List of (function_label, tottime_ms, cumtime_ms) tuples sorted by
        tottime descending.
```

with:

```python
    Returns:
        List of ``(function_label, tottime_ms, cumtime_ms, ncalls)`` tuples
        sorted by tottime descending. ``ncalls`` is the primitive (non-recursive)
        call count reported by ``pstats.Stats.stats``. Whitelist placeholder
        rows carry ``ncalls=0``.
```

Also update the signature's return annotation:

```python
def parse_cprofile_stats(
    profile: cProfile.Profile,
    isaaclab_prefixes: list[str],
    top_n: int = 30,
    whitelist: list[str] | None = None,
) -> list[tuple[str, float, float, int]]:
```

- [ ] **Step 4: Update the top_n call site in `benchmark_startup.py`**

Find the block around line 254 (look for `for label, tottime_ms, cumtime_ms in parse_cprofile_stats(`):

```python
        for label, tottime_ms, cumtime_ms in parse_cprofile_stats(
            imports_profile, _ISAACLAB_PREFIXES, top_n=args_cli.top_n
        ):
            benchmark.add_measurement(
                "python_imports",
                SingleMeasurement(name=f"tot_{label}", value=tottime_ms, unit="ms"),
            )
            benchmark.add_measurement(
                "python_imports",
                SingleMeasurement(name=f"cum_{label}", value=cumtime_ms, unit="ms"),
            )
```

(Exact surrounding code may differ slightly — locate by the `parse_cprofile_stats(` call.)

Wherever a call site unpacks 3-tuples, update to 4-tuples. If the call site only uses `label/tottime_ms/cumtime_ms` and discards `ncalls`, bind it to `_` so the unpacking doesn't error. If the call site needs `ncalls`, bind it properly.

The two call sites in `benchmark_startup.py` are:

**Call site A (inside the per-phase SingleMeasurement emission loop, around line 254):**

The unpacking pattern is `for label, tottime_ms, cumtime_ms in parse_cprofile_stats(...)` — it feeds `add_measurement` and does NOT need `ncalls` here. Add `_` to swallow it:

```python
        for label, tottime_ms, cumtime_ms, _ncalls in parse_cprofile_stats(
            imports_profile, _ISAACLAB_PREFIXES, top_n=args_cli.top_n
        ):
```

Repeat the `_ncalls` swallow for any other unpacking in this first loop block.

**Call site B (inside `_build_startup_bundle` where `CProfileFunction` is constructed, around line 408):**

Find the block that currently looks like:

```python
            functions = parse_cprofile_stats(
                profile,
                _ISAACLAB_PREFIXES,
                top_n=phase_top_n,
                whitelist=whitelist_for_phase,
            )
            top_functions = [
                CProfileFunction(
                    name=label,
                    own_time_s=tottime_ms / 1000.0,
                    cum_time_s=cumtime_ms / 1000.0,
                    # parse_cprofile_stats does not currently return call counts;
                    # pass 0 as a placeholder until the upstream fix lands.
                    calls=0,
                )
                for label, tottime_ms, cumtime_ms in functions
            ]
```

Replace with:

```python
            functions = parse_cprofile_stats(
                profile,
                _ISAACLAB_PREFIXES,
                top_n=phase_top_n,
                whitelist=whitelist_for_phase,
            )
            top_functions = [
                CProfileFunction(
                    name=label,
                    own_time_s=tottime_ms / 1000.0,
                    cum_time_s=cumtime_ms / 1000.0,
                    calls=ncalls,
                )
                for label, tottime_ms, cumtime_ms, ncalls in functions
            ]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_parse_cprofile_stats.py -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Sanity-check the full benchmark test suite**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/ -v 2>&1 | tail -20
```

Expected: all existing benchmark tests pass (no regression from the unpack change).

- [ ] **Step 7: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/utils.py scripts/benchmarks/benchmark_startup.py
git commit -m "Populate CProfileFunction.calls from parse_cprofile_stats ncalls

parse_cprofile_stats now returns 4-tuples (label, tottime_ms,
cumtime_ms, ncalls). CPython's pstats.Stats.stats dict already carries
ncalls as the second field of each entry — the previous 3-tuple
signature was discarding it, forcing CProfileFunction.calls to a
placeholder 0 in every startup bundle.

Both call sites in benchmark_startup.py are updated: the SingleMeasurement
emission loop swallows the new field with _ncalls, the schema-v1
CProfileFunction builder passes it through as calls=. Whitelist
placeholder rows carry ncalls=0 (semantic: no call count is meaningful
for a non-matching pattern).

Closes one of the two T1-carried reliability caveats in
docs/odin/architecture.md §9."
```

---

## Task 3: Write failing peak tests for `MemoryInfoRecorder`

**Goal:** Pin the new `_rss_peak`/`_vms_peak`/`_uss_peak` contract on `MemoryInfoRecorder` via red tests in the existing `TestMemoryInfoRecorder` class.

**Files:**
- Modify: `source/isaaclab/test/benchmark/test_recorders.py` (append methods to `TestMemoryInfoRecorder` around line 296)

- [ ] **Step 1: Read the existing `TestMemoryInfoRecorder` to match style**

```bash
sed -n '296,432p' source/isaaclab/test/benchmark/test_recorders.py
```

Note the mocking strategy (likely `unittest.mock.patch.object` or `monkeypatch` on `psutil.Process.memory_info`). Match it exactly.

- [ ] **Step 2: Append failing peak tests**

Add these methods INSIDE the existing `class TestMemoryInfoRecorder:` (just before the class ends, maintaining the indentation of existing methods):

```python
    def test_rss_peak_is_zero_before_any_record(self):
        from isaaclab.test.benchmark.recorders.record_memory_info import MemoryInfoRecorder

        rec = MemoryInfoRecorder()
        data = rec.get_data()
        peak_rows = [m for m in data.measurements if m.name == "System Memory RSS peak"]
        assert peak_rows, "expected a 'System Memory RSS peak' SingleMeasurement"
        assert peak_rows[0].value == 0.0

    def test_rss_peak_tracks_running_max(self, monkeypatch):
        import psutil

        from isaaclab.test.benchmark.recorders.record_memory_info import MemoryInfoRecorder

        # Scripted RSS sequence; peak must equal the max seen so far.
        scripted_values = [100 * 1024**3, 200 * 1024**3, 150 * 1024**3]  # bytes
        scripted_iter = iter(scripted_values)

        class _FakeMemInfo:
            def __init__(self, rss):
                self.rss = rss
                self.vms = rss  # mirror so VMS also moves
                # USS is read via memory_full_info, not memory_info; leave alone.

        def _fake_memory_info(self):  # noqa: ARG001 — bound method, self is the process
            return _FakeMemInfo(next(scripted_iter))

        monkeypatch.setattr(psutil.Process, "memory_info", _fake_memory_info)

        rec = MemoryInfoRecorder()
        for _ in scripted_values:
            rec.update()

        data = rec.get_data()
        rss_peak = next(m for m in data.measurements if m.name == "System Memory RSS peak")
        # The recorder emits GB; input was in bytes. 200 GiB -> 200.0 after rounding.
        assert rss_peak.value == 200.0, f"expected peak=200.0 GB, got {rss_peak.value}"

        vms_peak = next(m for m in data.measurements if m.name == "System Memory VMS peak")
        assert vms_peak.value == 200.0

    def test_rss_peak_does_not_decrease(self, monkeypatch):
        import psutil

        from isaaclab.test.benchmark.recorders.record_memory_info import MemoryInfoRecorder

        # Decreasing sequence — peak is set by the first sample and then stays.
        scripted_values = [300 * 1024**3, 50 * 1024**3, 25 * 1024**3]
        scripted_iter = iter(scripted_values)

        class _FakeMemInfo:
            def __init__(self, rss):
                self.rss = rss
                self.vms = rss

        def _fake_memory_info(self):  # noqa: ARG001
            return _FakeMemInfo(next(scripted_iter))

        monkeypatch.setattr(psutil.Process, "memory_info", _fake_memory_info)

        rec = MemoryInfoRecorder()
        for _ in scripted_values:
            rec.update()

        data = rec.get_data()
        rss_peak = next(m for m in data.measurements if m.name == "System Memory RSS peak")
        assert rss_peak.value == 300.0
```

**IMPORTANT matching detail:** the existing tests may use `mocker` (pytest-mock) instead of `monkeypatch`. If `grep -n 'mocker\|monkeypatch' source/isaaclab/test/benchmark/test_recorders.py` shows one style predominates, switch the three new tests to that style. Don't introduce a new mocking library.

- [ ] **Step 3: Run tests and verify they fail**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_recorders.py::TestMemoryInfoRecorder -v 2>&1 | tail -20
```

Expected failure: `StopIteration` or `AssertionError: expected a 'System Memory RSS peak' SingleMeasurement` — the recorder doesn't emit peak yet.

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/test/benchmark/test_recorders.py
git commit -m "Add failing peak-tracking tests for MemoryInfoRecorder

Three new tests on TestMemoryInfoRecorder pin the contract for
_rss_peak / _vms_peak / _uss_peak: peak is 0.0 before any record,
tracks the running max across samples, and does not decrease on a
descending sequence. Wire the implementation in the next commit."
```

---

## Task 4: Implement peak tracking in `MemoryInfoRecorder`

**Goal:** Add `_rss_peak`/`_vms_peak`/`_uss_peak` state and emit `System Memory RSS peak` / `System Memory VMS peak` / `System Memory USS peak` SingleMeasurements.

**Files:**
- Modify: `source/isaaclab/isaaclab/test/benchmark/recorders/record_memory_info.py`

- [ ] **Step 1: Add peak attributes to `__init__`**

In `MemoryInfoRecorder.__init__` (around line 17), after the existing Welford-state blocks for RSS/VMS/USS, add:

```python
        # Peak (running max) alongside the Welford mean/std. Initialised to
        # 0.0 so emit-before-record returns a meaningful zero.
        self._rss_peak = 0.0
        self._vms_peak = 0.0
        self._uss_peak = 0.0
```

- [ ] **Step 2: Update `_get_runtime_info` to track peak**

Find the RSS block:

```python
        # RSS (Resident Set Size) - physical memory used
        self._rss_mean, self._rss_m2, self._rss_n, rss_std = self._update_welford(
            mem_info.rss, self._rss_mean, self._rss_m2, self._rss_n
        )
        self._memory_runtime_info["rss_mean"] = self._rss_mean
        self._memory_runtime_info["rss_std"] = rss_std
        self._memory_runtime_info["rss_n"] = self._rss_n
```

Replace with:

```python
        # RSS (Resident Set Size) - physical memory used
        self._rss_mean, self._rss_m2, self._rss_n, rss_std = self._update_welford(
            mem_info.rss, self._rss_mean, self._rss_m2, self._rss_n
        )
        self._rss_peak = max(self._rss_peak, float(mem_info.rss))
        self._memory_runtime_info["rss_mean"] = self._rss_mean
        self._memory_runtime_info["rss_std"] = rss_std
        self._memory_runtime_info["rss_n"] = self._rss_n
        self._memory_runtime_info["rss_peak"] = self._rss_peak
```

Apply the identical pattern to the VMS block (using `mem_info.vms` and `self._vms_peak`) and the USS block (using `uss` and `self._uss_peak`). Keep the existing `try/except` around USS.

- [ ] **Step 3: Emit peak SingleMeasurements in `get_data`**

In `get_data()` (around line 107), find the RSS measurements block:

```python
            SingleMeasurement(
                name="System Memory RSS",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_mean", 0)),
                unit="GB",
            ),
            SingleMeasurement(
                name="System Memory RSS std",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_std", 0)),
                unit="GB",
            ),
            SingleMeasurement(name="System Memory RSS n", value=self._memory_runtime_info.get("rss_n", 0), unit=""),
```

Append a peak entry immediately after the RSS std SingleMeasurement:

```python
            SingleMeasurement(
                name="System Memory RSS",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_mean", 0)),
                unit="GB",
            ),
            SingleMeasurement(
                name="System Memory RSS std",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_std", 0)),
                unit="GB",
            ),
            SingleMeasurement(
                name="System Memory RSS peak",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_peak", 0)),
                unit="GB",
            ),
            SingleMeasurement(name="System Memory RSS n", value=self._memory_runtime_info.get("rss_n", 0), unit=""),
```

Do the same for VMS — insert a `System Memory VMS peak` row between the std and n rows.

For USS — find the conditional block (only runs when USS is available) and add a `System Memory USS peak` entry there, mirroring the RSS/VMS pattern.

- [ ] **Step 4: Run tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_recorders.py::TestMemoryInfoRecorder -v 2>&1 | tail -15
```

Expected: all existing `TestMemoryInfoRecorder` tests pass PLUS the three new peak tests pass.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/test/benchmark/recorders/record_memory_info.py
git commit -m "Track RSS / VMS / USS peak in MemoryInfoRecorder

Three new instance attributes (_rss_peak / _vms_peak / _uss_peak)
updated inline with the existing Welford mean/std state. New
SingleMeasurement entries 'System Memory {RSS,VMS,USS} peak' expose
the value to consumers (benchmark_rsl_rl.py and benchmark_skrl.py
wire them into Resources.*.peak in a later commit).

Peak is the running max of bytes observed across .update() calls;
0.0 when .update() has never been called. Welford state is untouched
— peak is additive, never read by Welford.

First half of the second T1-carried reliability caveat in
docs/odin/architecture.md §9."
```

---

## Task 5: Write failing peak tests for `GPUInfoRecorder`

**Goal:** Pin the new per-device `_mem_peak`/`_util_peak` contract on `GPUInfoRecorder`.

**Files:**
- Modify: `source/isaaclab/test/benchmark/test_recorders.py` (append methods to `TestGPUInfoRecorder` around line 117)

- [ ] **Step 1: Read the existing `TestGPUInfoRecorder` to match style**

```bash
sed -n '117,296p' source/isaaclab/test/benchmark/test_recorders.py
```

Note: the existing tests almost certainly mock `torch.cuda.is_available()`, `torch.cuda.device_count()`, `torch.cuda.get_device_properties()`, and the underlying NVML / nvidia-smi surfaces. Match the existing mock scaffold exactly — if it uses a `_setup_single_device_recorder()` helper, use it.

- [ ] **Step 2: Append failing peak tests**

Add these methods INSIDE the existing `class TestGPUInfoRecorder:` (before the class ends):

```python
    def test_mem_peak_is_zero_before_any_record(self, monkeypatch):
        import torch

        from isaaclab.test.benchmark.recorders.record_gpu_info import GPUInfoRecorder

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

        class _FakeProps:
            name = "FakeGPU"
            total_memory = 80 * 1024**3
            major = 9
            minor = 0
            multi_processor_count = 132

        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _FakeProps())

        rec = GPUInfoRecorder()
        data = rec.get_data()
        peaks = [m for m in data.measurements if "peak" in m.name.lower() and "GPU" in m.name]
        # At minimum there should be a GPU memory peak row for device 0.
        mem_peak_rows = [m for m in peaks if "Memory" in m.name]
        assert mem_peak_rows, f"expected a GPU memory peak row, got names: {[m.name for m in data.measurements]}"
        assert mem_peak_rows[0].value == 0.0

    def test_mem_peak_tracks_running_max(self, monkeypatch):
        """Feed the recorder a scripted memory sequence; peak must match the max."""
        import torch

        from isaaclab.test.benchmark.recorders.record_gpu_info import GPUInfoRecorder

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

        class _FakeProps:
            name = "FakeGPU"
            total_memory = 80 * 1024**3
            major = 9
            minor = 0
            multi_processor_count = 132

        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _FakeProps())

        rec = GPUInfoRecorder()

        # Bypass nvml / nvidia-smi entirely and drive memory_allocated.
        scripted_mem = iter([10 * 1024**3, 50 * 1024**3, 30 * 1024**3])  # 10 GB, 50 GB, 30 GB
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: next(scripted_mem))
        rec._nvml_available = False
        rec._nvidia_smi_available = False

        for _ in range(3):
            rec.update()

        data = rec.get_data()
        mem_peak_rows = [m for m in data.measurements if "Memory" in m.name and "peak" in m.name.lower()]
        assert mem_peak_rows, "expected a GPU memory peak row"
        # 50 GB is the max.
        assert mem_peak_rows[0].value == 50.0, f"expected 50.0 GB peak, got {mem_peak_rows[0].value}"
```

**IMPORTANT adaptation:** if the existing `TestGPUInfoRecorder` tests use a shared fixture (e.g. `_setup_single_device_recorder`) instead of inline monkey-patching, refactor these three tests to use that fixture. Do NOT introduce a parallel mock scaffold.

- [ ] **Step 3: Run tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_recorders.py::TestGPUInfoRecorder -v 2>&1 | tail -15
```

Expected failure: `AssertionError: expected a GPU memory peak row, got names: [...]` — the GPU recorder doesn't emit peak yet.

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/test/benchmark/test_recorders.py
git commit -m "Add failing peak-tracking tests for GPUInfoRecorder

Two new tests on TestGPUInfoRecorder pin the contract: GPU memory
peak is 0.0 before any record, and tracks the running max across
scripted torch.cuda.memory_allocated values when nvml and
nvidia-smi are disabled. Wire the implementation in the next commit."
```

---

## Task 6: Implement peak tracking in `GPUInfoRecorder`

**Goal:** Add per-device `_mem_peak[]`/`_util_peak[]` state and emit `GPU {i} Memory Used peak` / `GPU {i} Utilization peak` SingleMeasurements (exact name format to match the existing `GPU Memory Used` / `GPU Utilization` naming).

**Files:**
- Modify: `source/isaaclab/isaaclab/test/benchmark/recorders/record_gpu_info.py`

- [ ] **Step 1: Read the existing emission format for GPU memory**

```bash
sed -n '214,303p' source/isaaclab/isaaclab/test/benchmark/recorders/record_gpu_info.py
```

Note the exact SingleMeasurement name convention used for GPU memory mean and std (likely includes a device index, e.g. `"GPU 0 Memory Used"`). The new peak rows MUST follow the same convention for `_capture_resources` in the training-bundle emitters to find them.

- [ ] **Step 2: Add peak state to `__init__` and `_get_hardware_info`**

In `GPUInfoRecorder.__init__` (around line 21-38), add peak lists alongside the existing Welford lists:

```python
        # Per-device peak (running max) for memory (bytes) and utilization (%).
        self._mem_peak = []
        self._util_peak = []
```

Then in `_get_hardware_info` (around line 74-82), in the per-device initialization loop, append `0.0` to each new list:

```python
            # Initialize Welford stats for this device
            self._mem_mean.append(0)
            self._mem_std.append(0)
            self._mem_n.append(0)
            self._mem_m2.append(0)
            self._util_mean.append(0)
            self._util_std.append(0)
            self._util_n.append(0)
            self._util_m2.append(0)
            # Peak state (running max)
            self._mem_peak.append(0.0)
            self._util_peak.append(0.0)
```

- [ ] **Step 3: Update `_get_runtime_info` to track peak**

Find the memory block at the end of `_get_runtime_info` that looks like:

```python
            self._mem_n[i] += 1
            delta = memory_bytes - self._mem_mean[i]
            self._mem_mean[i] += delta / self._mem_n[i]
            delta2 = memory_bytes - self._mem_mean[i]
            self._mem_m2[i] += delta * delta2
            if self._mem_n[i] > 1:
                self._mem_std[i] = math.sqrt(self._mem_m2[i] / (self._mem_n[i] - 1))

            self._gpu_runtime_info["devices"][i]["memory_used_mean_bytes"] = self._mem_mean[i]
            self._gpu_runtime_info["devices"][i]["memory_used_std_bytes"] = self._mem_std[i]
            self._gpu_runtime_info["devices"][i]["memory_n"] = self._mem_n[i]
```

Insert peak tracking:

```python
            self._mem_n[i] += 1
            delta = memory_bytes - self._mem_mean[i]
            self._mem_mean[i] += delta / self._mem_n[i]
            delta2 = memory_bytes - self._mem_mean[i]
            self._mem_m2[i] += delta * delta2
            if self._mem_n[i] > 1:
                self._mem_std[i] = math.sqrt(self._mem_m2[i] / (self._mem_n[i] - 1))
            self._mem_peak[i] = max(self._mem_peak[i], float(memory_bytes))

            self._gpu_runtime_info["devices"][i]["memory_used_mean_bytes"] = self._mem_mean[i]
            self._gpu_runtime_info["devices"][i]["memory_used_std_bytes"] = self._mem_std[i]
            self._gpu_runtime_info["devices"][i]["memory_used_peak_bytes"] = self._mem_peak[i]
            self._gpu_runtime_info["devices"][i]["memory_n"] = self._mem_n[i]
```

Apply the parallel pattern for the utilization block: `self._util_peak[i] = max(self._util_peak[i], float(gpu_util))` and emit `utilization_peak_percent`.

- [ ] **Step 4: Emit peak SingleMeasurements in `get_data`**

Find the per-device SingleMeasurement block in `get_data()` (around lines 258-303). The existing shape is something like:

```python
                        SingleMeasurement(
                            name=f"GPU {i} Memory Used",
                            value=self._bytes_to_gb(device_rt.get("memory_used_mean_bytes", 0)),
                            unit="GB",
                        ),
                        SingleMeasurement(
                            name=f"GPU {i} Memory Used std",
                            value=self._bytes_to_gb(device_rt.get("memory_used_std_bytes", 0)),
                            unit="GB",
                        ),
```

Append a peak row after each mean / std pair:

```python
                        SingleMeasurement(
                            name=f"GPU {i} Memory Used",
                            value=self._bytes_to_gb(device_rt.get("memory_used_mean_bytes", 0)),
                            unit="GB",
                        ),
                        SingleMeasurement(
                            name=f"GPU {i} Memory Used std",
                            value=self._bytes_to_gb(device_rt.get("memory_used_std_bytes", 0)),
                            unit="GB",
                        ),
                        SingleMeasurement(
                            name=f"GPU {i} Memory Used peak",
                            value=self._bytes_to_gb(device_rt.get("memory_used_peak_bytes", 0)),
                            unit="GB",
                        ),
```

Do the same for utilization — new row `f"GPU {i} Utilization peak"` with unit `"%"`.

**Also add the aggregate (non-per-device) rows if the recorder emits them.** If `get_data` also emits a `"GPU Memory Used"` (no device index) aggregate — check lines 258-303 carefully — emit a matching `"GPU Memory Used peak"` aggregate that's the max across devices. If no aggregate is emitted today, don't invent one.

- [ ] **Step 5: Run tests and verify they pass**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/test_recorders.py::TestGPUInfoRecorder -v 2>&1 | tail -15
```

Expected: all existing + both new peak tests pass.

- [ ] **Step 6: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/test/benchmark/recorders/record_gpu_info.py
git commit -m "Track GPU memory and utilization peak in GPUInfoRecorder

Per-device _mem_peak[] / _util_peak[] lists updated alongside the
existing Welford mean/std state. New SingleMeasurement entries
'GPU {i} {Memory Used,Utilization} peak' expose the value to
consumers (benchmark_rsl_rl.py and benchmark_skrl.py wire them into
Resources.*.peak in a later commit).

Second half of the second T1-carried reliability caveat in
docs/odin/architecture.md §9."
```

---

## Task 7: Wire real peak into `benchmark_rsl_rl.py` Resources

**Goal:** Replace the `peak=mean` placeholder in `_capture_resources` with a real read from the new recorder SingleMeasurements.

**Files:**
- Modify: `scripts/benchmarks/benchmark_rsl_rl.py` (around lines 200-230)

- [ ] **Step 1: Identify the exact peak SingleMeasurement names**

Look at what the recorders now emit (from Tasks 4 and 6):
- RAM peak comes from `MemoryInfoRecorder` as `"System Memory RSS peak"`.
- GPU memory peak comes from `GPUInfoRecorder`. If the recorder emits an aggregate (non-per-device) row (check Task 6 step 4), use that. Otherwise use `"GPU 0 Memory Used peak"` (device 0 — most Odin runs are single-GPU).

The existing `_capture_resources` already reads the mean from `"GPU Memory Used"` (aggregate, no index). Mirror that: if the mean key has no device index, the peak key also has no device index. Confirm by running:

```bash
grep -E 'SingleMeasurement\(\s*name=' source/isaaclab/isaaclab/test/benchmark/recorders/record_gpu_info.py | head -20
```

and note whether `"GPU Memory Used"` (aggregate) exists. Use the matching aggregate peak name; fall back to `f"GPU 0 Memory Used peak"` only if no aggregate exists.

- [ ] **Step 2: Replace the placeholder in `_capture_resources`**

Find the block:

```python
def _capture_resources(bm: BaseIsaacLabBenchmark):
    """Build a schema-v1 :class:`Resources` dataclass from GPU/CPU/Memory recorders.

    The underlying recorders track Welford online mean/std but not peak, so
    ``peak`` fields fall back to the mean when no peak is tracked. This is a
    known v1 limitation (see docs/odin/architecture.md §9).
    """
    from isaaclab.test.benchmark.standard_schema import MeanStd, MeanStdPeak, Resources

    gpu_m = bm._manual_recorders["GPUInfo"].get_data().measurements
    cpu_m = bm._manual_recorders["CPUInfo"].get_data().measurements
    mem_m = bm._manual_recorders["MemoryInfo"].get_data().measurements

    gpu_util_mean = _find_measurement(gpu_m, "GPU Utilization") or 0.0
    gpu_util_std = _find_measurement(gpu_m, "GPU Utilization std") or 0.0
    gpu_mem_mean = _find_measurement(gpu_m, "GPU Memory Used") or 0.0
    gpu_mem_std = _find_measurement(gpu_m, "GPU Memory Used std") or 0.0
    cpu_util_mean = _find_measurement(cpu_m, "CPU Utilization") or 0.0
    cpu_util_std = _find_measurement(cpu_m, "CPU Utilization std") or 0.0
    ram_mean = _find_measurement(mem_m, "System Memory RSS") or 0.0
    ram_std = _find_measurement(mem_m, "System Memory RSS std") or 0.0

    return Resources(
        gpu_util_pct=MeanStd(mean=gpu_util_mean, std=gpu_util_std),
        gpu_mem_gb=MeanStdPeak(mean=gpu_mem_mean, std=gpu_mem_std, peak=gpu_mem_mean),
        cpu_util_pct=MeanStd(mean=cpu_util_mean, std=cpu_util_std),
        ram_gb=MeanStdPeak(mean=ram_mean, std=ram_std, peak=ram_mean),
    )
```

Replace with (use the peak-name convention you confirmed in Step 1):

```python
def _capture_resources(bm: BaseIsaacLabBenchmark):
    """Build a schema-v1 :class:`Resources` dataclass from GPU/CPU/Memory recorders."""
    from isaaclab.test.benchmark.standard_schema import MeanStd, MeanStdPeak, Resources

    gpu_m = bm._manual_recorders["GPUInfo"].get_data().measurements
    cpu_m = bm._manual_recorders["CPUInfo"].get_data().measurements
    mem_m = bm._manual_recorders["MemoryInfo"].get_data().measurements

    gpu_util_mean = _find_measurement(gpu_m, "GPU Utilization") or 0.0
    gpu_util_std = _find_measurement(gpu_m, "GPU Utilization std") or 0.0
    gpu_mem_mean = _find_measurement(gpu_m, "GPU Memory Used") or 0.0
    gpu_mem_std = _find_measurement(gpu_m, "GPU Memory Used std") or 0.0
    gpu_mem_peak = _find_measurement(gpu_m, "GPU Memory Used peak") or 0.0
    cpu_util_mean = _find_measurement(cpu_m, "CPU Utilization") or 0.0
    cpu_util_std = _find_measurement(cpu_m, "CPU Utilization std") or 0.0
    ram_mean = _find_measurement(mem_m, "System Memory RSS") or 0.0
    ram_std = _find_measurement(mem_m, "System Memory RSS std") or 0.0
    ram_peak = _find_measurement(mem_m, "System Memory RSS peak") or 0.0

    return Resources(
        gpu_util_pct=MeanStd(mean=gpu_util_mean, std=gpu_util_std),
        gpu_mem_gb=MeanStdPeak(mean=gpu_mem_mean, std=gpu_mem_std, peak=gpu_mem_peak),
        cpu_util_pct=MeanStd(mean=cpu_util_mean, std=cpu_util_std),
        ram_gb=MeanStdPeak(mean=ram_mean, std=ram_std, peak=ram_peak),
    )
```

(If the GPU recorder emits only per-device rows — no aggregate — replace `"GPU Memory Used peak"` with `"GPU 0 Memory Used peak"` and similarly for `gpu_mem_mean` / `gpu_mem_std` to stay consistent with device 0.)

- [ ] **Step 3: Sanity-check existing tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/ -v 2>&1 | tail -10
```

Expected: all tests still pass. No regression from the Resources wiring.

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/benchmark_rsl_rl.py
git commit -m "Populate Resources.*.peak from the recorder peak in RSL-RL bundle

_capture_resources now reads the 'GPU Memory Used peak' and 'System
Memory RSS peak' SingleMeasurements (added to GPUInfoRecorder and
MemoryInfoRecorder in the previous commits) instead of copying
mean into peak. The stale 'known v1 limitation' note in the
docstring is dropped; the caveat is closed."
```

---

## Task 8: Wire real peak into `benchmark_skrl.py` Resources

**Goal:** Mirror Task 7 in the SKRL benchmark script.

**Files:**
- Modify: `scripts/benchmarks/benchmark_skrl.py` (around lines 200-220)

- [ ] **Step 1: Apply the same edit as Task 7**

Find the `_capture_resources` function in `scripts/benchmarks/benchmark_skrl.py`. It has the same shape as the RSL-RL version — mean/std fields read from SingleMeasurements, `peak=mean` placeholder on `MeanStdPeak`. Apply the same diff: add `gpu_mem_peak = _find_measurement(gpu_m, "GPU Memory Used peak") or 0.0` and `ram_peak = _find_measurement(mem_m, "System Memory RSS peak") or 0.0`, then use those in the `MeanStdPeak(peak=...)` calls instead of the means.

(Match the exact peak-name convention used in Task 7 — whatever you settled on.)

- [ ] **Step 2: Sanity-check tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/ -v 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 3: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/benchmark_skrl.py
git commit -m "Populate Resources.*.peak from the recorder peak in SKRL bundle

Mirror the RSL-RL fix in the SKRL benchmark script: read the new
'GPU Memory Used peak' and 'System Memory RSS peak' SingleMeasurements
and feed them into Resources.gpu_mem_gb.peak and Resources.ram_gb.peak
instead of copying mean."
```

---

## Task 9: Bump IsaacLab CHANGELOG and extension version

**Goal:** Document both fixes in `source/isaaclab/docs/CHANGELOG.rst` and bump the extension version from `4.6.9` to `4.6.10` (patch — both are bug fixes, no new features).

**Files:**
- Modify: `source/isaaclab/docs/CHANGELOG.rst`
- Modify: `source/isaaclab/config/extension.toml`

- [ ] **Step 1: Insert a new version entry at the top of `CHANGELOG.rst`**

Open `source/isaaclab/docs/CHANGELOG.rst`. After the line `Changelog\n---------\n`, insert:

```rst
4.6.10 (2026-04-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :func:`~scripts.benchmarks.utils.parse_cprofile_stats` discarding
  the per-function call count. The function now returns 4-tuples
  ``(label, tottime_ms, cumtime_ms, ncalls)``; the schema-v1
  ``CProfileFunction.calls`` field in ``startup.json`` is now the real
  call count instead of the placeholder ``0``.
* Fixed :class:`~isaaclab.test.benchmark.recorders.MemoryInfoRecorder`
  and :class:`~isaaclab.test.benchmark.recorders.GPUInfoRecorder` not
  tracking a running max alongside the Welford mean/std. New
  SingleMeasurement entries expose the peak
  (``System Memory {RSS,VMS,USS} peak`` and
  ``GPU {i} {Memory Used,Utilization} peak``).
  :func:`scripts.benchmarks.benchmark_rsl_rl._capture_resources` and the
  SKRL equivalent now populate ``Resources.*.peak`` with the real peak
  instead of copying ``mean``.

```

(Keep the blank line after the block so the next existing version heading is cleanly separated.)

- [ ] **Step 2: Bump the extension version**

In `source/isaaclab/config/extension.toml`, change:

```toml
version = "4.6.9"
```

to:

```toml
version = "4.6.10"
```

- [ ] **Step 3: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/docs/CHANGELOG.rst source/isaaclab/config/extension.toml
git commit -m "Bump isaaclab to 4.6.10 for T2.2 reliability fixes

Changelog entry covers both T2.2 closures: parse_cprofile_stats
now returns ncalls (was hardcoded 0), MemoryInfoRecorder and
GPUInfoRecorder now track peak (was copying mean). Schema v1.0
unchanged — existing fields are now populated correctly."
```

---

## Task 10: Run fresh `benchmark_startup.py` and collect top-function data

**Goal:** Produce a fresh `startup.json` with the Task 1–8 fixes in place, which the survey doc and whitelist tuning will cite.

**Files:**
- Write: `/tmp/t2_2_startup.json` (intermediate; not committed)

- [ ] **Step 1: Run the profiler, top_n mode, no whitelist**

```bash
PYTHONPATH=. ./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
    --task Isaac-Ant-Direct-v0 \
    --top_n 30 \
    --schema_v1_output /tmp/t2_2_startup.json
```

Expected runtime: ~30-90 seconds (Isaac Sim startup + env creation + one step).
Expected output: `/tmp/t2_2_startup.json` exists, valid JSON, five phases.

- [ ] **Step 2: Eyeball the output — confirm ncalls is real**

```bash
./isaaclab.sh -p -c "
import json
d = json.load(open('/tmp/t2_2_startup.json'))
for phase_name, phase in d['phases'].items():
    tops = phase['top_functions'][:5]
    nonzero = [t for t in tops if t['calls'] > 0]
    print(f'{phase_name}: {len(tops)} rows, {len(nonzero)} with calls>0')
    for t in tops[:3]:
        print(f'  {t[\"calls\"]:>10} calls  {t[\"own_time_s\"]*1000:.2f} ms  {t[\"name\"]}')
"
```

Expected: every phase reports at least one row with `calls > 0`. If a phase has zero non-zero rows, that's a Task 1/2 bug — revisit before proceeding.

- [ ] **Step 3: Dump top-30 for each phase into a scratch file (survey input)**

```bash
./isaaclab.sh -p -c "
import json
d = json.load(open('/tmp/t2_2_startup.json'))
for phase in ['app_launch', 'python_imports', 'task_config', 'env_creation', 'first_step']:
    print(f'=== {phase} (total_time_s: {d[\"phases\"][phase][\"total_time_s\"]:.2f}) ===')
    for t in d['phases'][phase]['top_functions']:
        print(f'  {t[\"calls\"]:>10} {t[\"own_time_s\"]*1000:>10.2f} ms  {t[\"cum_time_s\"]*1000:>10.2f} ms  {t[\"name\"]}')
    print()
" > /tmp/t2_2_top_functions.txt
head -40 /tmp/t2_2_top_functions.txt
```

Nothing to commit in this task — the intermediate files are consumed by Tasks 11 and 12.

No commit.

---

## Task 11: Re-tune `scripts/benchmarks/startup_whitelist.yaml`

**Goal:** Update the whitelist to cover all five phases, based on the top functions observed in Task 10. Each phase either has an explicit pattern list (for stable top functions) or an explicit `# fall through to top_n` comment (for phases where the dominant functions are too variable to pin).

**Files:**
- Modify: `scripts/benchmarks/startup_whitelist.yaml`

- [ ] **Step 1: Inspect `/tmp/t2_2_top_functions.txt`**

Open the file. For each phase, scan the top-30 for stable, meaningful functions — ones you'd want a downstream dashboard to track across commits. Ignore:
- Functions with `own_time_s < 10 ms` (too small to matter).
- Functions whose label contains obvious run-specific noise (temporary-module paths, generated code).
- Duplicate entries for different phases of the same pipeline step (prefer the most specific).

Target 3-5 patterns per phase. Phases where the top functions are too variable or all below the 10 ms threshold stay on `top_n` fallback with a comment.

- [ ] **Step 2: Write the updated YAML**

Replace the contents of `scripts/benchmarks/startup_whitelist.yaml` with (substitute your actual observed patterns for each phase):

```yaml
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Per-phase function whitelist for benchmark_startup.py. Patterns are
# fnmatch-style; patterns matching no function emit a placeholder row
# (tottime=0, cumtime=0, ncalls=0) so downstream dashboards always receive
# consistent keys.
#
# A phase MAY be absent from this file — in that case benchmark_startup
# falls back to top_n selection (default: 30). Phases documented below
# that intentionally fall through say so explicitly in a comment rather
# than listing patterns.

app_launch:
  - "isaaclab.utils.configclass:_wrap_resolvable_strings"
  - "isaaclab.utils.configclass:_custom_post_init"
  - "isaaclab.utils.configclass:_field_module_dir"
  # TODO (from T2.2 re-tuning): add/replace based on observed top functions.

python_imports:
  # REPLACE with 3-5 stable patterns observed in Task 10 output.
  # Likely candidates: major import chains (torch, gymnasium, isaaclab).
  # If no stable pattern emerges, DELETE this phase entry and add a
  # top-of-file comment: "python_imports: intentional top_n fallback —
  # dominant functions are import-chain noise."
  - "REPLACE_ME"

task_config:
  # REPLACE or delete with top_n-fallback comment. Typical candidates:
  # isaaclab.utils.configclass:* and isaaclab_tasks.utils.parse_cfg:*.
  - "REPLACE_ME"

env_creation:
  - "isaaclab.cloner.*:usd_replicate"
  - "isaaclab.cloner.*:filter_collisions"
  - "isaaclab_physx.cloner.*:attach_end_fn"
  - "isaaclab.scene.*:_init_scene"
  - "isaaclab.envs.mdp.observations:*"
  - "isaaclab.utils.assets:_find_usd_dependencies"
  # TODO (from T2.2 re-tuning): add/replace based on observed top functions.

first_step:
  - "isaaclab.envs.mdp.rewards:*"
  - "isaaclab.envs.mdp.terminations:*"
  - "isaaclab.envs.mdp.observations:*"
  - "isaaclab.actuators.*:compute"
  - "warp.*:launch"
  - "warp.*:to_torch"
  # TODO (from T2.2 re-tuning): add/replace based on observed top functions.
```

**Substitute the REPLACE_ME entries with your actual observed patterns.** If a phase has no stable patterns (after scanning the top-30 in Task 10 step 3), delete the phase block and add a one-line comment at the top of the file: `# <phase>: intentional top_n fallback — <reason>`.

- [ ] **Step 3: Verify every pattern matches at least one function**

Re-run the profiler with the new whitelist:

```bash
PYTHONPATH=. ./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
    --task Isaac-Ant-Direct-v0 \
    --whitelist_config scripts/benchmarks/startup_whitelist.yaml \
    --schema_v1_output /tmp/t2_2_startup_whitelisted.json 2>&1 | tee /tmp/t2_2_whitelist_run.log
```

Expected: no `[WARNING] Whitelist pattern '...' matched no profiled functions.` lines in the log.

If any pattern emits that warning, either fix the pattern or remove it (and switch that phase to the fall-through comment).

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/startup_whitelist.yaml
git commit -m "Re-tune startup_whitelist.yaml to cover all five phases

Added explicit patterns for python_imports and task_config (previously
falling through to top_n silently); refreshed app_launch / env_creation
/ first_step patterns against the current top functions observed on
antoiner/feat/odin.

Patterns are validated against a fresh benchmark_startup.py run —
every pattern matches at least one profiled function (no warning
from parse_cprofile_stats's unmatched-pattern check).

<If any phases intentionally stay on top_n fallback, document them
 in the body of this commit message.>"
```

---

## Task 12: Author `docs/odin/startup_profiling_survey.md`

**Goal:** Publish the survey doc with per-phase commentary grounded in the Task 10 profile + whitelist rationale from Task 11.

**Files:**
- Create: `docs/odin/startup_profiling_survey.md`

- [ ] **Step 1: Write the doc skeleton with grounded content**

Create `docs/odin/startup_profiling_survey.md`:

```markdown
# Startup profiling survey

**Scope.** What `scripts/benchmarks/benchmark_startup.py` captures today,
what the numbers mean, and what Valhalla / comparison tooling should look
at. Content grounded in a fresh local run on `antoiner/feat/odin` at
`<INSERT current HEAD sha>`.

**Audience.** Anyone reading a `startup.json` from an Odin bundle (T4
dashboard, debugging a slow startup, investigating a regression).

## 1. Pipeline overview

`benchmark_startup.py` wraps each of five phases in its own
`cProfile.Profile` session and records wall-clock time plus the top
functions by own-time:

- **`app_launch`** — `AppLauncher(...)` starts Kit and returns a running
  `SimulationApp`.
- **`python_imports`** — `import gymnasium / numpy / torch / isaaclab.envs`
  plus `isaaclab_tasks.utils.launch_simulation / resolve_task_config`.
- **`task_config`** — `resolve_task_config(task, None)` loads and
  instantiates the env config dataclass tree.
- **`env_creation`** — `launch_simulation(env_cfg)` creates the
  environment, clones it across `num_envs` instances, instantiates
  sensors / actuators.
- **`first_step`** — one `env.step(action)` call to force kernel
  compilation and first-iteration setup.

Each phase emits a schema-v1 `StartupPhase`:

```json
{
  "total_time_s": 18.4,
  "top_functions": [
    {"name": "...", "own_time_s": 1.82, "cum_time_s": 2.41, "calls": 4312}
  ]
}
```

Selection is either `top_n` (default 30, or 5 with `--whitelist_config`)
or explicit fnmatch patterns from `scripts/benchmarks/startup_whitelist.yaml`.

`startup.json` lives at `<run_id>/startup.json` in an Odin bundle.
`docs/superpowers/specs/2026-04-22-odin-t1-evaluation-runner-design.md`
documents the full schema.

## 2. Phase reference

Numbers below are from the `Isaac-Ant-Direct-v0` baseline run at the
grounding commit. Absolute timings vary between hardware; relative
breakdowns are the stable signal.

### 2.1 `app_launch`

**What it is.** `AppLauncher(headless=True)` — brings up the Kit app
runtime, loads the base Omniverse extensions, acquires GPU resources.

**Typical wall-time.** `<INSERT observed total_time_s from /tmp/t2_2_startup.json>`.

**Top functions (fresh run).**

```
<INSERT top 3-5 rows from /tmp/t2_2_top_functions.txt for app_launch,
 as own_time_ms | function_label>
```

**Commentary.** <Short paragraph: what these functions do, why they're
expected to dominate, what a big change in any of them would signal.>

**Caveats.** Kit-subsystem load time is not visible to cProfile at
Python level — if `app_launch` total jumps but no Python function owns
the jump, the cost is inside Kit (native code). See §5.

### 2.2 `python_imports`

**What it is.** The profiled `import` block between `AppLauncher.app`
and `resolve_task_config` — torch, gymnasium, numpy, the isaaclab env
modules, and `isaaclab_tasks`.

**Typical wall-time.** `<INSERT observed total_time_s>`.

**Top functions (fresh run).**

```
<INSERT top 3-5 rows>
```

**Commentary.** <short paragraph>.

**Caveats.** First-time imports after a package update may show cache
population (`.pyc` compilation) inflating times.

### 2.3 `task_config`

**What it is.** `resolve_task_config(task_id, None)` — the gym registry
lookup + env cfg instantiation, including all `@configclass`
`__post_init__` chains.

**Typical wall-time.** `<INSERT>`.

**Top functions (fresh run).**

```
<INSERT top 3-5 rows>
```

**Commentary.** <paragraph>.

**Caveats.** Deeply nested `PresetCfg` resolution (e.g. on tasks with a
`newton` preset) can skew this phase's total independently of the top
Python functions.

### 2.4 `env_creation`

**What it is.** `launch_simulation(env_cfg)` — simulation context
initialisation, scene creation, env cloning, sensor / actuator / event
wiring.

**Typical wall-time.** `<INSERT>`.

**Top functions (fresh run).**

```
<INSERT top 3-5 rows>
```

**Commentary.** <paragraph covering cloner, sensor init, USD load>.

**Caveats.** Cloner and USD-load timings scale non-linearly with
`num_envs`; comparing env_creation across different `num_envs` is not
meaningful.

### 2.5 `first_step`

**What it is.** One `env.step(sample_action)` call. Forces warp kernel
compilation, first observation compute, first reward / termination
evaluation.

**Typical wall-time.** `<INSERT>`.

**Top functions (fresh run).**

```
<INSERT top 3-5 rows>
```

**Commentary.** <paragraph covering warp JIT, torch autograd setup>.

**Caveats.** Warp kernel compile is not visible to cProfile at the
per-function level — the cost shows up as large `warp.*:launch` own-time
on first call only.

## 3. Reading the data (cross-cutting)

**cProfile semantics.** `own_time_s` (tottime) is time spent in a
function excluding its callees; `cum_time_s` (cumtime) is the total
including callees. For hot-spot hunting use `own_time_s`; for
"which dispatch path is expensive" use `cum_time_s`. The filter in
`parse_cprofile_stats` keeps functions that are (a) inside an IsaacLab
source directory, or (b) directly called by an IsaacLab function —
enough context for diagnosis, not so much that the output is dominated
by torch internals.

**`ncalls` is now populated.** Before 2026-04-22 (isaaclab 4.6.9 and
earlier), `CProfileFunction.calls` was hardcoded to `0` due to an
upstream bug in `parse_cprofile_stats`. Bundles dated after 4.6.10 carry
the real primitive call count. When comparing across commits that
straddle the fix, ignore `calls` on older bundles.

**Whitelist vs `top_n`.** Use whitelist mode when the downstream
consumer needs a stable schema across commits (dashboards with
hard-coded series names). Use `top_n` when you want to catch newly
dominant functions — e.g. after a refactor that introduces a hot path.
A whitelist pattern that matches nothing emits a placeholder row
`(pattern, 0.0, 0.0, 0)` so the dashboard key is stable even when the
function disappears.

**Comparing across commits.** Total per-phase time is the stable
headline metric. Per-function own-time is noisy within-phase; apply a
median or mean-of-N smoothing over repeated runs before alerting on a
regression. `ncalls` is deterministic for most phases and makes a good
"did the call graph change" signal.

**Comparing across backends (PhysX vs Newton).** Whole-phase totals are
the best first-order signal. Inside-phase top functions diverge too
much for direct row-level comparison (different physics code paths);
compare at the phase-total level, then drill into per-function only if
a total diverges.

**Resource caveats.** After 2026-04-22, `Resources.*.peak` in
`training.json` is the real running-max of RSS and GPU memory during
training. Bundles dated 4.6.9 or earlier carry `peak == mean` — do not
use `peak` as a "max seen" signal on those.

## 4. Whitelist recommendations

The committed `scripts/benchmarks/startup_whitelist.yaml` provides
explicit patterns for <N> of the five phases and lets <M> fall through
to `top_n` (reasons inline in the YAML).

**When to add a pattern.** If a phase's regression or cross-commit
comparison is blocked by the `top_n` cut dropping a function you care
about, add a targeted fnmatch pattern. The placeholder-row behaviour
(zero-value row for an unmatched pattern) keeps the dashboard key
stable across runs.

**When to remove a pattern.** If a pattern consistently emits the
"matched no profiled functions" warning across runs, the function has
moved or been renamed; remove the stale pattern and pick a new one
from the fresh `top_n` output.

## 5. Open questions (seeds for T4 / later work)

Things noticed during T2.2 that are not solved here:

- **Kit-subsystem cost.** `app_launch` total can change by several
  seconds without any Python function owning the change; the cost is
  inside Kit native code. Surfacing it probably needs per-extension
  timing at the Kit level, not cProfile.
- **Warp kernel compile time.** `warp.*:launch` shows up as a huge
  own-time on first step only; a subsequent step would show near-zero.
  A dedicated "first-call kernel compile" metric separate from the
  steady-state step time might be useful.
- **USD asset-load timing per asset.** `env_creation` rolls USD loading
  into a few opaque cProfile entries; a per-asset timing breakdown
  would help regressions that come from a heavier asset being added
  to a scene.
- **GPU memory delta per phase.** Today `Resources.*.peak` is
  training-wide. A per-phase peak delta (how much memory a phase
  added) would make startup-phase memory regressions observable.

These are not promises — they're candidates for whoever builds T4 or
a future extension to T2.2.
```

- [ ] **Step 2: Fill in the `<INSERT ...>` placeholders from `/tmp/t2_2_top_functions.txt`**

For each `<INSERT>` marker, copy the corresponding data from the Task 10 scratch file. Top rows per phase are the 3-5 highest-`own_time_s` entries; per-phase wall-times come from the `=== phase (total_time_s: X.XX) ===` headers.

For each `<INSERT current HEAD sha>`, run `git rev-parse --short HEAD` and paste the output.

- [ ] **Step 3: Write the phase commentary paragraphs**

Replace each `<short paragraph>` / `<paragraph>` with 1-3 sentences on what the top functions do and why they dominate. Draw on:
- Code-level inspection of the top-ranked functions (e.g. `git grep -n "_custom_post_init" source/isaaclab/isaaclab/utils/configclass.py` to understand what `_custom_post_init` does).
- The existing `benchmark_startup.py` docstring commentary.

Keep it tight — 1-3 sentences per phase. The survey is reference material, not an essay.

- [ ] **Step 4: Replace `<N>` and `<M>` in §4**

Count how many phases got explicit patterns in Task 11's YAML vs how many stayed on `top_n` fallback. Substitute.

- [ ] **Step 5: Run pre-commit**

```bash
./isaaclab.sh -f
```

If codespell complains about a technical term (`fnmatch`, `tottime`, `pstats`), add it to `pyproject.toml`'s codespell ignore list. Otherwise fix the typo.

- [ ] **Step 6: Commit**

```bash
git add docs/odin/startup_profiling_survey.md
git commit -m "Author startup profiling survey for Odin T2.2

Pipeline overview plus per-phase reference sections for app_launch,
python_imports, task_config, env_creation, first_step — each with
typical wall-time, current top functions, and caveats. Grounded in a
fresh benchmark_startup.py run on antoiner/feat/odin.

Cross-cutting section covers cProfile semantics, the ncalls fix
cutover (4.6.9 vs 4.6.10), whitelist vs top_n, comparing across
commits and backends, and the Resources.*.peak fix cutover.

Whitelist recommendations document §4 explains which phases use
explicit patterns vs top_n fallback and when to add / remove a
pattern. Section 5 lists open observations (Kit-subsystem cost,
warp compile, USD timing, per-phase memory delta) as seeds for T4
and future work."
```

---

## Task 13: Smoke pass — verify fixes land end-to-end

**Goal:** One short training run that confirms `calls > 0` and `peak != mean` on the emitted bundle. Not a committed artefact — a sign-off check.

**Files:**
- None committed. Outputs to `/tmp/`.

- [ ] **Step 1: Run a ~10-iteration RSL-RL bundle**

```bash
PYTHONPATH=. ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
    --task Isaac-Ant-Direct-v0 \
    --max_iterations 10 \
    --num_envs 1024 \
    --headless \
    --schema_v1_output /tmp/t2_2_training.json
```

Expected runtime: ~2-5 minutes.

- [ ] **Step 2: Inspect the peak fields**

```bash
./isaaclab.sh -p -c "
import json
d = json.load(open('/tmp/t2_2_training.json'))
r = d.get('resources', {})
print('GPU Memory Used  mean / std / peak:', r.get('gpu_mem_gb'))
print('RAM              mean / std / peak:', r.get('ram_gb'))
gm = r.get('gpu_mem_gb', {})
ram = r.get('ram_gb', {})
print('GPU peak > mean:', gm.get('peak', 0) > gm.get('mean', 0))
print('RAM peak >= mean:', ram.get('peak', 0) >= ram.get('mean', 0))
"
```

Expected: `peak > mean` on at least one of GPU memory or RAM. `peak >= mean` should be True always (peak can equal mean if only one sample was recorded).

- [ ] **Step 3: Verify a fresh startup.json also has real calls**

```bash
./isaaclab.sh -p -c "
import json
d = json.load(open('/tmp/t2_2_startup.json'))
for phase_name, phase in d['phases'].items():
    tops = [t for t in phase['top_functions'] if t['calls'] > 0]
    print(f'{phase_name}: {len(tops)}/{len(phase[\"top_functions\"])} rows with calls>0')
"
```

Expected: every phase reports at least one row with `calls > 0`.

If either check fails, investigate and fix the underlying task before proceeding. No commit in this task — it's a sign-off.

---

## Task 14: Update `docs/odin/architecture.md` change log

**Goal:** Record T2.2 in §9 of the architecture reference. No task-map change (T2.2 was added to the map by T2.1's closeout).

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 1: Update "Last updated" line**

Change the current line near the top of the file:

```markdown
**Last updated:** 2026-04-22 (end of T2.1)
```

to:

```markdown
**Last updated:** 2026-04-22 (end of T2.2)
```

- [ ] **Step 2: Flip T2.2 status in the task map (§6)**

Find the T2.2 row in the §6 task map:

```markdown
| T2.2 | Dense startup profiling survey | — | ⚪ |
```

Replace with:

```markdown
| T2.2 | Dense startup profiling survey | `docs/superpowers/specs/2026-04-22-odin-t2-2-startup-profiling-design.md` | ✅ |
```

- [ ] **Step 3: Add a change-log row in §9**

Append to the change-log table at the bottom:

```markdown
| 2026-04-22 | T2.2 delivered. Closed both T1-carried reliability caveats: `parse_cprofile_stats` returns `ncalls` so `CProfileFunction.calls` is real (was always `0`); `MemoryInfoRecorder` and `GPUInfoRecorder` track peak so `Resources.*.peak` is the real running max (was copying `mean`). No schema bump — v1.0 fields are now populated correctly. `scripts/benchmarks/startup_whitelist.yaml` re-tuned to cover all five phases (or explicit `top_n` fallback comments). Survey published at `docs/odin/startup_profiling_survey.md` grounded in a fresh `Isaac-Ant-Direct-v0` profile. isaaclab version bumped 4.6.9 → 4.6.10. Open observations carried to T4 / future work: Kit-subsystem cost (not visible to cProfile), warp kernel compile time, per-asset USD load timing, per-phase GPU memory delta. | Odin T2.2 |
```

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark Odin T2.2 complete in architecture reference

Task-map status: T2.2 flips from ⚪ to ✅ with spec link.

Change log: one row summarising the T2.2 deliverables (two
reliability fixes, whitelist re-tune, survey doc), version bump
4.6.9 → 4.6.10, and the open observations carried to T4.

Last-updated line moves from 'end of T2.1' to 'end of T2.2'."
```

---

## Self-review notes (for the implementer)

Before calling T2.2 done, verify:

1. **All tests pass.**
   - `./isaaclab.sh -p -m pytest source/isaaclab/test/benchmark/ -v` — all green.
   - No new warnings from pre-commit on any touched file.

2. **Pre-commit clean** on `HEAD` for the full set of touched files.

3. **Deliverables exist and are coherent.**
   - `docs/odin/startup_profiling_survey.md` has no `<INSERT>` placeholders left.
   - `scripts/benchmarks/startup_whitelist.yaml` has no `REPLACE_ME` patterns.
   - Re-running `benchmark_startup.py --whitelist_config` emits zero `[WARNING] Whitelist pattern '...' matched no profiled functions.` lines.

4. **Smoke pass (Task 13) passed:** `calls > 0` and `peak > mean` on a real run.

5. **CHANGELOG + extension version** at `4.6.10`, architecture doc at "end of T2.2".

6. **Gap doc from T2.1 still referenced correctly.** T2.2 doesn't consume `newton_api_gaps.md`; ensure nothing accidentally broke the link from the architecture change log.
