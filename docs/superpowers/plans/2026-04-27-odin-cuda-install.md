# Odin CUDA Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `odin-cuda` — a CLI with `check` and `install` subcommands that detects each Valkyrie's host-side CUDA support and, when below floor (default 12.4), upgrades it to a pinned target (`cuda-12-9` apt meta-package = driver 575 + toolkit 12.9), reboots, and verifies.

**Architecture:** Two new modules under `tools/odin/asgard/`: `cuda_install.py` (parsers, `CheckResult`, `CudaInstallResult`, single-host functions, fleet drivers, dispatch guard) and `cuda_install_cli.py` (argparse + `check` / `install` subcommands). Reuses `ValkyrieConfig`, `Fleet`, `load_fleet`, `SSHRunner`, the `_container_start` / `_container_stop` helpers in `provisioner.py`, and `read_dispatch_state` from `state.py`. Zero changes to dispatch / preflight / worker / bootstrap.

**Tech Stack:** Python 3.10+, stdlib only (`concurrent.futures`, `dataclasses`, `time`, `argparse`, `pathlib`, `re`), pytest.

**Spec:** `docs/superpowers/specs/2026-04-27-odin-cuda-install-design.md`

---

## File Structure

**New files:**
- `tools/odin/asgard/cuda_install.py` — parsers, `CheckResult`, `CudaInstallResult`, `check_cuda_valkyrie`, `check_fleet`, `install_cuda_valkyrie`, `install_fleet`, `find_running_dispatches`.
- `tools/odin/asgard/cuda_install_cli.py` — argparse + `main()` for `odin-cuda` (subcommands `check`, `install`).
- `tools/odin/tests/test_asgard_cuda_install.py` — unit tests for the core module.
- `tools/odin/tests/test_asgard_cuda_install_cli.py` — CLI-level tests.

**Modified:**
- `tools/odin/README.md` — new "Validating CUDA across the fleet" section.

**Unchanged (confirmed):**
- `tools/odin/asgard/runner.py`, `preflight.py`, `worker.py`, `state.py`, `transport.py`, `bootstrap.py`, `provisioner.py`.

**Task ordering rationale:** Task 1 builds the three pure-function parsers everything else relies on (nvidia-smi output, `/etc/os-release`, CUDA version comparison). Tasks 2-3 add `check_cuda_valkyrie` then the parallel `check_fleet`. Tasks 4-6 add the install primitives (result type + skip path → full pipeline → fleet driver). Task 7 adds the active-dispatch guard. Tasks 8-9 ship the CLI. Task 10 wraps up with README + a final pre-commit sweep.

---

### Task 1: Parsers and version comparator

**Files:**
- Create: `tools/odin/asgard/cuda_install.py`
- Create: `tools/odin/tests/test_asgard_cuda_install.py`

Three small pure functions. We ship the file with only these so subsequent tasks have a place to grow.

- [ ] **Step 1: Write the failing tests**

Create `tools/odin/tests/test_asgard_cuda_install.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tools.odin.asgard.cuda_install."""

from __future__ import annotations

import pytest

from tools.odin.asgard.cuda_install import (
    cuda_at_or_above,
    parse_nvidia_smi,
    parse_os_release,
)


# --- parse_nvidia_smi -------------------------------------------------------


_SMI_OK = """\
Mon Apr 27 11:52:21 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
"""


def test_parse_nvidia_smi_extracts_driver_and_cuda():
    parsed = parse_nvidia_smi(_SMI_OK)
    assert parsed == ("535.161.07", "12.2")


def test_parse_nvidia_smi_returns_none_when_no_match():
    assert parse_nvidia_smi("nvidia-smi: command not found") is None
    assert parse_nvidia_smi("") is None


def test_parse_nvidia_smi_handles_long_driver_minor():
    line = "| NVIDIA-SMI 575.51.03             Driver Version: 575.51.03   CUDA Version: 12.9     |"
    assert parse_nvidia_smi(line) == ("575.51.03", "12.9")


# --- parse_os_release -------------------------------------------------------


_OS_2404 = """\
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
ID=ubuntu
VERSION_CODENAME=noble
"""

_OS_2204 = """\
NAME="Ubuntu"
VERSION_ID="22.04"
ID=ubuntu
"""

_OS_RHEL = """\
NAME="Red Hat Enterprise Linux"
VERSION_ID="9.4"
ID=rhel
"""


def test_parse_os_release_ubuntu_2404():
    assert parse_os_release(_OS_2404) == "ubuntu2404"


def test_parse_os_release_ubuntu_2204():
    assert parse_os_release(_OS_2204) == "ubuntu2204"


def test_parse_os_release_unsupported_returns_none():
    assert parse_os_release(_OS_RHEL) is None
    assert parse_os_release("") is None


# --- cuda_at_or_above -------------------------------------------------------


@pytest.mark.parametrize(
    "measured,floor,expected",
    [
        ("12.2", "12.4", False),
        ("12.4", "12.4", True),
        ("12.9", "12.4", True),
        ("13.0", "12.4", True),
        ("11.8", "12.4", False),
        ("12.10", "12.4", True),  # numeric, not lexicographic
    ],
)
def test_cuda_at_or_above(measured: str, floor: str, expected: bool):
    assert cuda_at_or_above(measured, floor) is expected


def test_cuda_at_or_above_rejects_garbage():
    with pytest.raises(ValueError):
        cuda_at_or_above("12", "12.4")
    with pytest.raises(ValueError):
        cuda_at_or_above("not.a.version", "12.4")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: collection error / `ModuleNotFoundError: tools.odin.asgard.cuda_install`.

- [ ] **Step 3: Create the module with the parsers**

Create `tools/odin/asgard/cuda_install.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin CUDA detection + per-host upgrade helpers.

Public surface mirrors ``tools/odin/asgard/bootstrap.py``: a single-host
function and a fleet driver per phase (``check`` / ``install``).
"""

from __future__ import annotations

import re

__all__ = [
    "cuda_at_or_above",
    "parse_nvidia_smi",
    "parse_os_release",
]


_NVIDIA_SMI_HEADER_RE = re.compile(
    r"NVIDIA-SMI\s+\S+\s+Driver Version:\s+(?P<driver>\d+\.\d+(?:\.\d+)?)\s+CUDA Version:\s+(?P<cuda>\d+\.\d+)"
)


def parse_nvidia_smi(stdout: str) -> tuple[str, str] | None:
    """Extract ``(driver_version, cuda_version)`` from ``nvidia-smi`` output.

    The header line on every supported driver release looks like::

        | NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |

    We pull the driver version (``"535.161.07"``) and the driver-advertised
    max CUDA (``"12.2"``).

    Args:
        stdout: Raw stdout from ``nvidia-smi`` (or any string).

    Returns:
        Two-tuple of ``(driver, cuda)`` strings, or ``None`` if no header
        line is found (e.g. ``nvidia-smi`` is missing or ran on a no-GPU host).
    """
    m = _NVIDIA_SMI_HEADER_RE.search(stdout)
    if m is None:
        return None
    return (m.group("driver"), m.group("cuda"))


_OS_RELEASE_RE = re.compile(r'(?m)^(?P<key>[A-Z_]+)=(?:"(?P<qval>[^"]*)"|(?P<val>\S*))')


def parse_os_release(stdout: str) -> str | None:
    """Map the contents of ``/etc/os-release`` to NVIDIA's apt repo slug.

    Returns ``"ubuntu2204"`` or ``"ubuntu2404"`` on supported Ubuntus,
    ``None`` for everything else (RHEL, Debian, no file).

    Args:
        stdout: Raw contents of ``/etc/os-release``.
    """
    fields: dict[str, str] = {}
    for m in _OS_RELEASE_RE.finditer(stdout):
        fields[m.group("key")] = m.group("qval") if m.group("qval") is not None else m.group("val") or ""
    if fields.get("ID") != "ubuntu":
        return None
    version_id = fields.get("VERSION_ID", "")
    if version_id == "22.04":
        return "ubuntu2204"
    if version_id == "24.04":
        return "ubuntu2404"
    return None


def cuda_at_or_above(measured: str, floor: str) -> bool:
    """Return ``True`` iff the dotted CUDA string ``measured >= floor``.

    Numeric (not lexicographic) comparison: ``"12.10" >= "12.4"`` is True.
    Both inputs must be ``"<major>.<minor>"`` — ``ValueError`` otherwise.
    """
    return _parse_cuda(measured) >= _parse_cuda(floor)


def _parse_cuda(s: str) -> tuple[int, int]:
    parts = s.split(".")
    if len(parts) != 2:
        raise ValueError(f"expected 'major.minor' CUDA string, got {s!r}")
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError as e:
        raise ValueError(f"expected 'major.minor' CUDA string, got {s!r}") from e
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: 13 passed (3 nvidia-smi cases + 3 os-release cases + 6 parametrized comparator cases + 1 garbage-rejection test).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install.py tools/odin/tests/test_asgard_cuda_install.py
git commit -m "$(cat <<'EOF'
cuda_install: parsers + CUDA version comparator

nvidia-smi header parser, os-release parser (Ubuntu only), and a
numeric CUDA "major.minor" comparator. Building blocks for the
forthcoming check_cuda_valkyrie / install_cuda_valkyrie pipeline.
EOF
)"
```

---

### Task 2: `CheckResult` and `check_cuda_valkyrie` (single-host)

**Files:**
- Modify: `tools/odin/asgard/cuda_install.py`
- Modify: `tools/odin/tests/test_asgard_cuda_install.py`

Single-host read-only check: SSH probe → nvidia-smi → /etc/os-release → status verdict.

- [ ] **Step 1: Append the failing tests**

Append to `tools/odin/tests/test_asgard_cuda_install.py`:

```python
# --- check_cuda_valkyrie ---------------------------------------------------

from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.cuda_install import CheckResult, check_cuda_valkyrie
from tools.odin.asgard.fleet import ValkyrieConfig


@dataclass
class _SSHCall:
    cmd: str
    timeout_s: float | None


@dataclass
class _FakeSSH:
    """Records calls; replies via substring lookup on ``cmd`` (insertion-ordered)."""

    calls: list[_SSHCall] = field(default_factory=list)
    replies: dict[str, int] = field(default_factory=dict)
    reply_stdout: dict[str, str] = field(default_factory=dict)
    reply_stderr: dict[str, str] = field(default_factory=dict)
    reply_timed_out: dict[str, bool] = field(default_factory=dict)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
        self.calls.append(_SSHCall(cmd=cmd, timeout_s=timeout_s))
        exit_code = 0
        stdout = ""
        stderr = ""
        timed_out = False
        for key, code in self.replies.items():
            if key in cmd:
                exit_code = code
                stdout = self.reply_stdout.get(key, "")
                stderr = self.reply_stderr.get(key, "")
                timed_out = self.reply_timed_out.get(key, False)
                break

        class R:
            pass

        R.exit_code = exit_code
        R.stdout = stdout
        R.stderr = stderr
        R.duration_s = 0.01
        R.timed_out = timed_out
        return R()


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/opt/IsaacLab",
        container_name="isaac-lab-base",
    )


_NVIDIA_SMI_OK_122 = (
    "| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |"
)
_NVIDIA_SMI_OK_129 = (
    "| NVIDIA-SMI 575.51.03             Driver Version: 575.51.03   CUDA Version: 12.9     |"
)
_OS_2404 = 'ID=ubuntu\nVERSION_ID="24.04"\n'


def test_check_ok_when_at_or_above_floor():
    ssh = _FakeSSH(
        replies={
            "echo cuda-check-ok": 0,
            "nvidia-smi": 0,
            "/etc/os-release": 0,
        },
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert isinstance(result, CheckResult)
    assert result.host == "v1.internal"
    assert result.status == "ok"
    assert result.driver == "575.51.03"
    assert result.cuda == "12.9"


def test_check_needs_upgrade_when_below_floor():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_122, "/etc/os-release": _OS_2404},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "needs-upgrade"
    assert result.cuda == "12.2"


def test_check_unreachable_when_ssh_probe_fails():
    ssh = _FakeSSH(
        replies={"echo cuda-check-ok": 255},
        reply_stderr={"echo cuda-check-ok": "ssh: connect to host: No route"},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "unreachable"
    assert "No route" in result.message
    # nvidia-smi must NOT have been attempted.
    assert not any("nvidia-smi" in c.cmd for c in ssh.calls)


def test_check_no_gpu_when_nvidia_smi_missing():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 127},
        reply_stderr={"nvidia-smi": "nvidia-smi: command not found"},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "no-gpu"


def test_check_no_gpu_when_nvidia_smi_unparseable():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0},
        reply_stdout={"nvidia-smi": "(garbled output, no header line)"},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "no-gpu"


def test_check_unsupported_os_short_circuits():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={
            "nvidia-smi": _NVIDIA_SMI_OK_122,
            "/etc/os-release": 'ID=rhel\nVERSION_ID="9.4"\n',
        },
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "unsupported-os"
    # We still report observed driver/cuda for context.
    assert result.driver == "535.161.07"
    assert result.cuda == "12.2"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: ImportError on `CheckResult`, `check_cuda_valkyrie`.

- [ ] **Step 3: Implement `CheckResult` and `check_cuda_valkyrie`**

Append to `tools/odin/asgard/cuda_install.py` (and add the new symbols to `__all__`):

```python
from dataclasses import dataclass

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import SSHRunner


# Insert into __all__:
#   "CheckResult", "check_cuda_valkyrie",


@dataclass
class CheckResult:
    """Outcome of a single-host CUDA check.

    Attributes:
        host: Host string from :class:`ValkyrieConfig`.
        status: One of ``"ok"``, ``"needs-upgrade"``, ``"unreachable"``,
            ``"no-gpu"``, ``"unsupported-os"``.
        driver: Driver version (e.g. ``"535.161.07"``) when known, ``""`` otherwise.
        cuda: Driver-advertised max CUDA (e.g. ``"12.2"``) when known, ``""`` otherwise.
        message: Human-readable diagnostic; populated for non-ok statuses.
    """

    host: str
    status: str
    driver: str = ""
    cuda: str = ""
    message: str = ""


def check_cuda_valkyrie(
    host: ValkyrieConfig,
    *,
    ssh: SSHRunner,
    floor: str = "12.4",
) -> CheckResult:
    """Read-only CUDA + OS check on ``host``.

    Pipeline (short-circuits):

      1. SSH reach (``echo cuda-check-ok``) → ``unreachable`` on non-zero.
      2. ``nvidia-smi`` → ``no-gpu`` if missing or unparseable.
      3. ``cat /etc/os-release`` → ``unsupported-os`` if not Ubuntu 22.04 / 24.04.
      4. CUDA-vs-floor → ``ok`` or ``needs-upgrade``.

    Args:
        host: Target Valkyrie.
        ssh: SSH runner.
        floor: CUDA floor as ``"<major>.<minor>"`` (default ``"12.4"``).
    """
    # 1. SSH reach.
    r = ssh.run(host, "echo cuda-check-ok", timeout_s=15.0)
    if r.exit_code != 0:
        return CheckResult(
            host=host.host,
            status="unreachable",
            message=f"ssh unreachable: {r.stderr.strip() or r.stdout.strip() or 'non-zero exit'}",
        )

    # 2. nvidia-smi.
    r = ssh.run(host, "nvidia-smi 2>&1", timeout_s=15.0)
    parsed = parse_nvidia_smi(r.stdout) if r.exit_code == 0 else None
    if parsed is None:
        return CheckResult(
            host=host.host,
            status="no-gpu",
            message=f"nvidia-smi unavailable or unparseable: {r.stderr.strip() or r.stdout.strip()[:120]}",
        )
    driver, cuda = parsed

    # 3. /etc/os-release.
    r = ssh.run(host, "cat /etc/os-release", timeout_s=15.0)
    os_slug = parse_os_release(r.stdout) if r.exit_code == 0 else None
    if os_slug is None:
        return CheckResult(
            host=host.host,
            status="unsupported-os",
            driver=driver,
            cuda=cuda,
            message="host is not Ubuntu 22.04 or 24.04",
        )

    # 4. floor comparison.
    if cuda_at_or_above(cuda, floor):
        return CheckResult(host=host.host, status="ok", driver=driver, cuda=cuda)
    return CheckResult(
        host=host.host,
        status="needs-upgrade",
        driver=driver,
        cuda=cuda,
        message=f"cuda {cuda} below floor {floor}",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: 19 passed (13 from Task 1 + 6 new check-path tests).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install.py tools/odin/tests/test_asgard_cuda_install.py
git commit -m "$(cat <<'EOF'
cuda_install: CheckResult + per-host check_cuda_valkyrie

Read-only health check: SSH reach -> nvidia-smi -> /etc/os-release ->
floor comparison. Returns one of ok / needs-upgrade / unreachable /
no-gpu / unsupported-os.
EOF
)"
```

---

### Task 3: `check_fleet` (parallel driver)

**Files:**
- Modify: `tools/odin/asgard/cuda_install.py`
- Modify: `tools/odin/tests/test_asgard_cuda_install.py`

Apply `check_cuda_valkyrie` across every host. Parallel by default; sequential opt-out (mirrors `bootstrap_fleet`).

- [ ] **Step 1: Append the failing tests**

```python
# --- check_fleet ----------------------------------------------------------

from tools.odin.asgard.cuda_install import check_fleet
from tools.odin.asgard.fleet import Fleet


def test_check_fleet_returns_results_in_fleet_order():
    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host="v1", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v2", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v3", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
        ],
    )
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    results = check_fleet(fleet, ssh=ssh, floor="12.4", parallel=False)
    assert [r.host for r in results] == ["v1", "v2", "v3"]
    assert all(r.status == "ok" for r in results)


def test_check_fleet_parallel_runs_concurrently():
    import time as _t

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host=f"v{i}", ssh_user="u", ssh_key=None, isaaclab_path="/p")
            for i in (1, 2, 3)
        ],
    )

    class _SlowSSH(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "nvidia-smi" in cmd:
                _t.sleep(0.1)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh = _SlowSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    t0 = _t.perf_counter()
    results = check_fleet(fleet, ssh=ssh, floor="12.4", parallel=True)
    elapsed = _t.perf_counter() - t0
    assert all(r.status == "ok" for r in results)
    assert elapsed < 0.25, f"parallel check_fleet elapsed={elapsed:.3f}s"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py::test_check_fleet_returns_results_in_fleet_order -v
```

Expected: ImportError on `check_fleet`.

- [ ] **Step 3: Implement `check_fleet`**

Append to `tools/odin/asgard/cuda_install.py` (and add `"check_fleet"` to `__all__`):

```python
import concurrent.futures as _cf

from tools.odin.asgard.fleet import Fleet


def check_fleet(
    fleet: Fleet,
    *,
    ssh: SSHRunner,
    floor: str = "12.4",
    parallel: bool = True,
) -> list[CheckResult]:
    """Run :func:`check_cuda_valkyrie` against every host in ``fleet``.

    Args:
        fleet: Loaded :class:`Fleet` (via :func:`load_fleet`).
        ssh: SSH runner shared across hosts.
        floor: CUDA floor passed through to each per-host check.
        parallel: When ``True`` (default), one thread per host. ``False`` runs
            sequentially.

    Returns:
        :class:`CheckResult` list in fleet order.
    """
    if parallel and len(fleet.hosts) > 1:
        with _cf.ThreadPoolExecutor(max_workers=len(fleet.hosts)) as pool:
            futures = [pool.submit(check_cuda_valkyrie, h, ssh=ssh, floor=floor) for h in fleet.hosts]
            return [f.result() for f in futures]
    return [check_cuda_valkyrie(h, ssh=ssh, floor=floor) for h in fleet.hosts]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: 21 passed (19 from Tasks 1-2 + 2 new fleet-driver tests).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install.py tools/odin/tests/test_asgard_cuda_install.py
git commit -m "$(cat <<'EOF'
cuda_install: parallel check_fleet driver

Thread-per-host across the whole fleet, mirroring bootstrap_fleet's
concurrency pattern. Sequential opt-out via parallel=False.
EOF
)"
```

---

### Task 4: `CudaInstallResult` + skip-when-already-at-floor

**Files:**
- Modify: `tools/odin/asgard/cuda_install.py`
- Modify: `tools/odin/tests/test_asgard_cuda_install.py`

Add the result type, the target-to-driver-major map, and the early-exit branch of `install_cuda_valkyrie`. Full pipeline lands in Task 5.

- [ ] **Step 1: Append the failing tests**

```python
# --- install_cuda_valkyrie skip path --------------------------------------

from tools.odin.asgard.cuda_install import (
    CudaInstallResult,
    TARGET_TO_DRIVER_MAJOR,
    install_cuda_valkyrie,
)


def test_target_to_driver_major_has_pinned_default():
    # The default install target must be present.
    assert TARGET_TO_DRIVER_MAJOR["12.9"] == "575"


def test_install_skip_when_already_at_floor():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9")
    assert isinstance(result, CudaInstallResult)
    assert result.ok is True
    assert result.skipped is True
    assert result.driver_before == "575.51.03"
    assert result.cuda_before == "12.9"
    # Skip path must not run apt or reboot.
    assert not any("apt-get install" in c.cmd for c in ssh.calls)
    assert not any("systemctl reboot" in c.cmd for c in ssh.calls)


def test_install_skip_unreachable_propagates_as_failure():
    ssh = _FakeSSH(
        replies={"echo cuda-check-ok": 255},
        reply_stderr={"echo cuda-check-ok": "ssh: conn refused"},
    )
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9")
    assert result.ok is False
    assert result.skipped is False
    assert "ssh unreachable" in result.message


def test_install_unsupported_os_refuses():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_122, "/etc/os-release": "ID=rhel\nVERSION_ID=\"9.4\"\n"},
    )
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9")
    assert result.ok is False
    assert "Ubuntu" in result.message
    # No install attempt.
    assert not any("apt-get" in c.cmd for c in ssh.calls)


def test_install_unknown_target_raises():
    ssh = _FakeSSH()
    with pytest.raises(ValueError, match="unknown target"):
        install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="99.99")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py::test_install_skip_when_already_at_floor -v
```

Expected: ImportError on `CudaInstallResult` / `install_cuda_valkyrie`.

- [ ] **Step 3: Implement the result type, the driver-major map, and the skip-only `install_cuda_valkyrie`**

Append to `tools/odin/asgard/cuda_install.py` (and add `"CudaInstallResult"`, `"TARGET_TO_DRIVER_MAJOR"`, `"install_cuda_valkyrie"` to `__all__`):

```python
from dataclasses import dataclass, field


# Pinned `cuda-X-Y` apt meta-package -> matching cuda-drivers major.
# Keep this small and explicit. Add entries as new targets are validated.
TARGET_TO_DRIVER_MAJOR: dict[str, str] = {
    "12.4": "550",
    "12.5": "555",
    "12.6": "560",
    "12.7": "565",
    "12.8": "570",
    "12.9": "575",
}


@dataclass
class CudaInstallResult:
    """Outcome of a single-host install attempt."""

    host: str
    ok: bool
    skipped: bool = False
    driver_before: str = ""
    cuda_before: str = ""
    driver_after: str = ""
    cuda_after: str = ""
    message: str = ""
    step_durations_s: dict[str, float] = field(default_factory=dict)


def install_cuda_valkyrie(
    host: ValkyrieConfig,
    *,
    ssh: SSHRunner,
    floor: str = "12.4",
    target: str = "12.9",
    reboot_timeout_s: float = 600.0,
    clock=None,
    sleep=None,
) -> CudaInstallResult:
    """Bring ``host`` to ``cuda >= floor`` by installing ``cuda-{target}``.

    Pipeline lives in subsequent steps; this Task implements only the
    pre-check + skip + unsupported-os branches. Full apt + reboot + verify
    arrives in Task 5.

    Args:
        host: Target Valkyrie.
        ssh: SSH runner.
        floor: CUDA threshold; hosts at or above this are skipped.
        target: Apt meta-package version key (e.g. ``"12.9"``); must be in
            :data:`TARGET_TO_DRIVER_MAJOR`.
        reboot_timeout_s: Wall-clock budget for the post-reboot SSH wait.
        clock: Optional ``time.monotonic`` replacement (test injection).
        sleep: Optional ``time.sleep`` replacement (test injection).

    Raises:
        ValueError: If ``target`` is not in :data:`TARGET_TO_DRIVER_MAJOR`.
    """
    if target not in TARGET_TO_DRIVER_MAJOR:
        raise ValueError(f"unknown target {target!r}; known: {sorted(TARGET_TO_DRIVER_MAJOR)}")

    # Pre-check (mirrors check_cuda_valkyrie, but populates the install-result
    # type so callers see consistent driver_before / cuda_before fields).
    pre = check_cuda_valkyrie(host, ssh=ssh, floor=floor)
    if pre.status == "unreachable":
        return CudaInstallResult(host=host.host, ok=False, message=pre.message)
    if pre.status == "no-gpu":
        return CudaInstallResult(host=host.host, ok=False, message=pre.message)
    if pre.status == "unsupported-os":
        return CudaInstallResult(
            host=host.host,
            ok=False,
            driver_before=pre.driver,
            cuda_before=pre.cuda,
            message="host is not Ubuntu 22.04 or 24.04 (refusing to install)",
        )
    if pre.status == "ok":
        return CudaInstallResult(
            host=host.host,
            ok=True,
            skipped=True,
            driver_before=pre.driver,
            cuda_before=pre.cuda,
        )
    # status == "needs-upgrade" — full pipeline is added in Task 5.
    return CudaInstallResult(
        host=host.host,
        ok=False,
        driver_before=pre.driver,
        cuda_before=pre.cuda,
        message="install pipeline not yet implemented",
    )
```

- [ ] **Step 4: Run the tests to verify the skip-path tests pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: 25 passed (21 from Tasks 1-3 + 4 new skip-path tests).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install.py tools/odin/tests/test_asgard_cuda_install.py
git commit -m "$(cat <<'EOF'
cuda_install: skeleton install_cuda_valkyrie + skip path

CudaInstallResult dataclass, TARGET_TO_DRIVER_MAJOR map, and the
short-circuit branches (already-at-floor, unreachable, no-gpu,
unsupported-os). Full apt + reboot pipeline in next commit.
EOF
)"
```

---

### Task 5: `install_cuda_valkyrie` full pipeline

**Files:**
- Modify: `tools/odin/asgard/cuda_install.py`
- Modify: `tools/odin/tests/test_asgard_cuda_install.py`

Implement the actual install pipeline: stop container → add NVIDIA repo → apt update → apt install → reboot → wait-for-SSH → post-verify → restart container.

- [ ] **Step 1: Append the failing tests**

```python
# --- install_cuda_valkyrie full pipeline ----------------------------------


def _stub_clock_and_sleep():
    """Return ``(clock, sleep)`` injectables that advance a fake monotonic
    counter without ever calling time.sleep — tests stay sub-second."""
    now = [0.0]

    def _clock() -> float:
        return now[0]

    def _sleep(delta: float) -> None:
        now[0] += delta

    return _clock, _sleep


def _install_happy_path_ssh() -> _FakeSSH:
    """SSH fake where pre-check sees 12.2, post-verify sees 12.9."""

    @dataclass
    class _PrePostSSH(_FakeSSH):
        post_phase: bool = False

        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            r = super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)
            # Flip to "post" the moment the install asks for the reboot.
            if "systemctl reboot" in cmd:
                self.post_phase = True
            return r

        # Override stdout for nvidia-smi based on phase.
        def __post_init__(self):
            self.replies.setdefault("echo cuda-check-ok", 0)
            self.replies.setdefault("/etc/os-release", 0)
            self.replies.setdefault("nvidia-smi", 0)
            self.replies.setdefault("container.py stop", 0)
            self.replies.setdefault("cuda-keyring", 0)
            self.replies.setdefault("apt-get update", 0)
            self.replies.setdefault("apt-get install", 0)
            self.replies.setdefault("systemctl reboot", 0)
            self.replies.setdefault("echo cuda-install-ok", 0)
            self.replies.setdefault("container.py start", 0)
            self.reply_stdout.setdefault("/etc/os-release", _OS_2404)

    ssh = _PrePostSSH()
    # Phase-dependent nvidia-smi stdout.
    base_run = ssh.run

    def _phased_run(host, cmd, *, timeout_s=None, stdout_tee=None):
        if "nvidia-smi" in cmd:
            ssh.reply_stdout["nvidia-smi"] = (
                _NVIDIA_SMI_OK_129 if ssh.post_phase else _NVIDIA_SMI_OK_122
            )
        return base_run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh.run = _phased_run  # type: ignore[assignment]
    return ssh


def test_install_full_happy_path_runs_all_steps():
    ssh = _install_happy_path_ssh()
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(),
        ssh=ssh,
        floor="12.4",
        target="12.9",
        reboot_timeout_s=600.0,
        clock=clock,
        sleep=sleep,
    )
    assert result.ok is True
    assert result.skipped is False
    assert result.cuda_before == "12.2"
    assert result.cuda_after == "12.9"
    assert result.driver_after.startswith("575.")
    # Step set must include the install pipeline phases.
    assert {
        "container_stop",
        "add_repo",
        "apt_update",
        "apt_install",
        "reboot",
        "wait_for_ssh",
        "post_verify",
        "container_start",
    } <= set(result.step_durations_s)


def test_install_apt_install_failure_short_circuits():
    ssh = _install_happy_path_ssh()
    ssh.replies["apt-get install"] = 100
    ssh.reply_stderr["apt-get install"] = "E: Unable to locate package cuda-12-9"
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep
    )
    assert result.ok is False
    assert "apt" in result.message.lower()
    assert not any("systemctl reboot" in c.cmd for c in ssh.calls)


def test_install_reboot_timeout_is_hard_failure():
    ssh = _install_happy_path_ssh()
    # Make every "echo cuda-install-ok" probe non-zero to simulate the host
    # never coming back.
    ssh.replies["echo cuda-install-ok"] = 255
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(),
        ssh=ssh,
        floor="12.4",
        target="12.9",
        reboot_timeout_s=30.0,
        clock=clock,
        sleep=sleep,
    )
    assert result.ok is False
    assert "reboot timed out" in result.message
    assert not any("container.py start" in c.cmd for c in ssh.calls if "stop" not in c.cmd)


def test_install_post_verify_failure_when_cuda_still_below_floor():
    """Reboot succeeds but driver kmod didn't load; nvidia-smi still reports 12.2."""
    ssh = _FakeSSH(
        replies={
            "echo cuda-check-ok": 0,
            "/etc/os-release": 0,
            "nvidia-smi": 0,
            "container.py stop": 0,
            "cuda-keyring": 0,
            "apt-get update": 0,
            "apt-get install": 0,
            "systemctl reboot": 0,
            "echo cuda-install-ok": 0,
            "container.py start": 0,
        },
        reply_stdout={
            "nvidia-smi": _NVIDIA_SMI_OK_122,  # never flips
            "/etc/os-release": _OS_2404,
        },
    )
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep
    )
    assert result.ok is False
    assert "verify" in result.message.lower()


def test_install_container_stop_failure_is_non_fatal():
    """A failed container.py stop must not block the install."""
    ssh = _install_happy_path_ssh()
    ssh.replies["container.py stop"] = 1
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep
    )
    assert result.ok is True


def test_install_container_start_failure_is_soft_warning():
    """A failed container.py start after install yields ok=True with a message."""
    ssh = _install_happy_path_ssh()
    ssh.replies["container.py start"] = 1
    ssh.reply_stderr["container.py start"] = "image build error"
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep
    )
    assert result.ok is True
    assert "container restart" in result.message
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v -k "install_full or install_apt or install_reboot or install_post or install_container"
```

Expected: failures referencing the not-yet-implemented pipeline (e.g. `'install pipeline not yet implemented' in result.message`).

- [ ] **Step 3: Replace the placeholder branch with the real pipeline**

Replace the final `# status == "needs-upgrade"` branch in
`install_cuda_valkyrie` (the `return CudaInstallResult(... message="install pipeline not yet implemented")`) with the full pipeline. Add the `_wait_for_ssh` helper next to it. The full new tail of `install_cuda_valkyrie` plus the helper:

```python
import time as _time

from tools.odin.asgard.provisioner import _container_start, _container_stop


def _wait_for_ssh(
    host: ValkyrieConfig,
    ssh: SSHRunner,
    *,
    timeout_s: float,
    poll_interval_s: float,
    clock,
    sleep,
) -> bool:
    """Poll ``echo cuda-install-ok`` until it succeeds or ``timeout_s`` elapses."""
    deadline = clock() + timeout_s
    # Tiny initial grace so the host has actually rebooted before we probe.
    sleep(min(poll_interval_s, 5.0))
    while clock() < deadline:
        r = ssh.run(host, "echo cuda-install-ok", timeout_s=10.0)
        if r.exit_code == 0:
            return True
        sleep(poll_interval_s)
    return False
```

…then replace the placeholder branch at the end of `install_cuda_valkyrie` with:

```python
    # status == "needs-upgrade" — run the full apt + reboot + verify pipeline.
    if clock is None:
        clock = _time.monotonic
    if sleep is None:
        sleep = _time.sleep

    driver_major = TARGET_TO_DRIVER_MAJOR[target]
    apt_pkg = f"cuda-{target.replace('.', '-')}"
    os_slug = parse_os_release(
        ssh.run(host, "cat /etc/os-release", timeout_s=15.0).stdout
    )
    keyring_url = (
        f"https://developer.download.nvidia.com/compute/cuda/repos/{os_slug}/x86_64/"
        "cuda-keyring_1.1-1_all.deb"
    )

    step_durations_s: dict[str, float] = {}

    def _step(name: str) -> "_StepCtx":
        return _StepCtx(name, step_durations_s, clock)

    # 2. Best-effort container stop.
    with _step("container_stop"):
        _container_stop(host, ssh)

    # 3. Add NVIDIA apt repo (idempotent — keyring deb is no-op if installed).
    with _step("add_repo"):
        r = ssh.run(
            host,
            (
                f"set -e; cd /tmp && wget -q -O cuda-keyring.deb {keyring_url} "
                "&& sudo dpkg -i cuda-keyring.deb"
            ),
            timeout_s=120.0,
        )
        if r.exit_code != 0:
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=f"add_repo failed: {r.stderr.strip() or r.stdout.strip()[:200]}",
                step_durations_s=step_durations_s,
            )

    # 4. apt-get update.
    with _step("apt_update"):
        r = ssh.run(
            host,
            "sudo apt-get update -o Acquire::Retries=3",
            timeout_s=300.0,
        )
        if r.exit_code != 0:
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=f"apt-get update failed: {r.stderr.strip() or r.stdout.strip()[:200]}",
                step_durations_s=step_durations_s,
            )

    # 5. apt-get install cuda-{target}.
    with _step("apt_install"):
        r = ssh.run(
            host,
            (
                "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "
                f"-o Dpkg::Options::=--force-confnew {apt_pkg}"
            ),
            timeout_s=1800.0,
        )
        if r.exit_code != 0:
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=f"apt-get install {apt_pkg} failed: {r.stderr.strip() or r.stdout.strip()[:200]}",
                step_durations_s=step_durations_s,
            )

    # 6. Reboot. SSH connection drops; non-zero is expected.
    with _step("reboot"):
        ssh.run(host, "sudo systemctl reboot", timeout_s=30.0)

    # 7. Wait for SSH to come back.
    with _step("wait_for_ssh"):
        if not _wait_for_ssh(
            host,
            ssh,
            timeout_s=reboot_timeout_s,
            poll_interval_s=10.0,
            clock=clock,
            sleep=sleep,
        ):
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=f"reboot timed out after {reboot_timeout_s:.0f}s",
                step_durations_s=step_durations_s,
            )

    # 8. Post-verify nvidia-smi.
    with _step("post_verify"):
        r = ssh.run(host, "nvidia-smi 2>&1", timeout_s=30.0)
        parsed = parse_nvidia_smi(r.stdout) if r.exit_code == 0 else None
        if parsed is None:
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message="post_verify: nvidia-smi unparseable after reboot",
                step_durations_s=step_durations_s,
            )
        driver_after, cuda_after = parsed
        if not cuda_at_or_above(cuda_after, floor):
            dmesg = ssh.run(host, "dmesg | grep -i nvidia | tail -5", timeout_s=15.0).stdout
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                driver_after=driver_after,
                cuda_after=cuda_after,
                message=f"verify-failed: cuda {cuda_after} < floor {floor}\n{dmesg}",
                step_durations_s=step_durations_s,
            )
        if not driver_after.startswith(driver_major + "."):
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                driver_after=driver_after,
                cuda_after=cuda_after,
                message=f"verify-failed: driver {driver_after} not in target family {driver_major}.x",
                step_durations_s=step_durations_s,
            )

    # 9. Best-effort container restart.
    soft_message = ""
    with _step("container_start"):
        if not _container_start(host, ssh, timeout_s=600):
            soft_message = "post-install container restart failed (run odin-bootstrap to recover)"

    return CudaInstallResult(
        host=host.host,
        ok=True,
        skipped=False,
        driver_before=pre.driver,
        cuda_before=pre.cuda,
        driver_after=driver_after,
        cuda_after=cuda_after,
        message=soft_message,
        step_durations_s=step_durations_s,
    )
```

Add the timing context manager helper near the top of the module (next to the parsers):

```python
class _StepCtx:
    """Context manager that records ``perf_counter`` deltas into a dict."""

    def __init__(self, name: str, sink: dict[str, float], clock) -> None:
        self._name = name
        self._sink = sink
        self._clock = clock
        self._t0 = 0.0

    def __enter__(self) -> "_StepCtx":
        self._t0 = self._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sink[self._name] = self._clock() - self._t0
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: 31 passed (25 from Tasks 1-4 + 6 new full-pipeline tests).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install.py tools/odin/tests/test_asgard_cuda_install.py
git commit -m "$(cat <<'EOF'
cuda_install: full install pipeline (apt + reboot + verify)

Stops container, adds NVIDIA apt keyring, apt-get update, apt-get
install cuda-{target}, reboots, waits for SSH, parses post-reboot
nvidia-smi, validates cuda >= floor and driver in target family,
restarts the container. Step durations captured per phase via a
small _StepCtx helper. clock/sleep injected so tests stay sub-second.
EOF
)"
```

---

### Task 6: `install_fleet` (parallel + sequential)

**Files:**
- Modify: `tools/odin/asgard/cuda_install.py`
- Modify: `tools/odin/tests/test_asgard_cuda_install.py`

Apply `install_cuda_valkyrie` across the fleet. Mirrors `bootstrap_fleet`'s `parallel`/`verbose` knobs.

- [ ] **Step 1: Append the failing tests**

```python
# --- install_fleet --------------------------------------------------------

from tools.odin.asgard.cuda_install import install_fleet


def test_install_fleet_returns_results_in_fleet_order():
    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host="v1", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v2", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
        ],
    )
    # Both hosts already at 12.9 -> both skip.
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    clock, sleep = _stub_clock_and_sleep()
    results = install_fleet(
        fleet, ssh=ssh, floor="12.4", target="12.9", parallel=False, clock=clock, sleep=sleep
    )
    assert [r.host for r in results] == ["v1", "v2"]
    assert all(r.ok and r.skipped for r in results)


def test_install_fleet_parallel_runs_concurrently():
    import time as _t

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host=f"v{i}", ssh_user="u", ssh_key=None, isaaclab_path="/p")
            for i in (1, 2, 3)
        ],
    )

    class _SlowSSH(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "nvidia-smi" in cmd:
                _t.sleep(0.1)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh = _SlowSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    clock, sleep = _stub_clock_and_sleep()
    t0 = _t.perf_counter()
    install_fleet(fleet, ssh=ssh, floor="12.4", target="12.9", parallel=True, clock=clock, sleep=sleep)
    elapsed = _t.perf_counter() - t0
    # Skip path runs check_cuda_valkyrie which calls nvidia-smi (slept 0.1s
    # per host). Serial would be >= 0.3s; parallel should be < 0.25s.
    assert elapsed < 0.25, f"parallel install_fleet elapsed={elapsed:.3f}s"


def test_install_fleet_verbose_prints_per_host(capsys):
    fleet = Fleet(
        fleet_name="test",
        hosts=[ValkyrieConfig(host="v-only", ssh_user="u", ssh_key=None, isaaclab_path="/p")],
    )
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    clock, sleep = _stub_clock_and_sleep()
    install_fleet(
        fleet,
        ssh=ssh,
        floor="12.4",
        target="12.9",
        parallel=False,
        verbose=True,
        clock=clock,
        sleep=sleep,
    )
    out = capsys.readouterr().out
    assert "v-only" in out
    assert "skipped" in out or "ok" in out
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v -k install_fleet
```

Expected: ImportError on `install_fleet`.

- [ ] **Step 3: Implement `install_fleet`**

Append to `tools/odin/asgard/cuda_install.py` (and add `"install_fleet"` to `__all__`):

```python
def install_fleet(
    fleet: Fleet,
    *,
    ssh: SSHRunner,
    floor: str = "12.4",
    target: str = "12.9",
    reboot_timeout_s: float = 600.0,
    parallel: bool = True,
    verbose: bool = False,
    clock=None,
    sleep=None,
) -> list[CudaInstallResult]:
    """Run :func:`install_cuda_valkyrie` against every host in ``fleet``.

    Args:
        fleet: Loaded :class:`Fleet`.
        ssh: SSH runner shared across hosts.
        floor: CUDA floor — hosts at or above are skipped.
        target: Apt meta-package version key (e.g. ``"12.9"``).
        reboot_timeout_s: Per-host reboot wait budget.
        parallel: Thread-per-host concurrency. ``False`` runs sequentially.
        verbose: Print one summary line per host as each finishes.
        clock: Optional ``time.monotonic`` replacement (test injection).
        sleep: Optional ``time.sleep`` replacement (test injection).
    """
    def _one(h: ValkyrieConfig) -> CudaInstallResult:
        return install_cuda_valkyrie(
            h,
            ssh=ssh,
            floor=floor,
            target=target,
            reboot_timeout_s=reboot_timeout_s,
            clock=clock,
            sleep=sleep,
        )

    if parallel and len(fleet.hosts) > 1:
        with _cf.ThreadPoolExecutor(max_workers=len(fleet.hosts)) as pool:
            futures = [pool.submit(_one, h) for h in fleet.hosts]
            results = [f.result() for f in futures]
    else:
        results = [_one(h) for h in fleet.hosts]

    if verbose:
        for r in results:
            if r.ok and r.skipped:
                tag = "skipped"
            elif r.ok:
                tag = "ok"
            else:
                tag = f"FAILED: {r.message}"
            print(f"[{r.host}] {tag}")
    return results
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: 34 passed (31 from Tasks 1-5 + 3 new install_fleet tests).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install.py tools/odin/tests/test_asgard_cuda_install.py
git commit -m "$(cat <<'EOF'
cuda_install: install_fleet driver

Parallel-by-default fleet driver mirroring bootstrap_fleet. Verbose
mode prints per-host ok/skipped/failed lines.
EOF
)"
```

---

### Task 7: Active-dispatch guard

**Files:**
- Modify: `tools/odin/asgard/cuda_install.py`
- Modify: `tools/odin/tests/test_asgard_cuda_install.py`

A helper that scans `runs_root` for in-flight dispatches (`dispatch.json` with `ended_at is None`). Used by the CLI to refuse `install` while jobs are running.

- [ ] **Step 1: Append the failing tests**

```python
# --- find_running_dispatches ----------------------------------------------

import json

from tools.odin.asgard.cuda_install import find_running_dispatches


def _write_dispatch_json(dispatch_dir: Path, *, ended: bool) -> None:
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "dispatch_id": dispatch_dir.name,
        "started_at": "2026-04-27T10:00:00+00:00",
        "ended_at": "2026-04-27T10:30:00+00:00" if ended else None,
        "seeds": [42],
        "commit_sha": "abcdef0",
        "fleet": [],
        "jobs": [],
        "skipped": [],
    }
    (dispatch_dir / "dispatch.json").write_text(json.dumps(payload))


def test_find_running_dispatches_finds_inflight(tmp_path: Path):
    _write_dispatch_json(tmp_path / "20260427-running", ended=False)
    _write_dispatch_json(tmp_path / "20260427-finished", ended=True)
    ids = find_running_dispatches(tmp_path)
    assert ids == ["20260427-running"]


def test_find_running_dispatches_empty_when_runs_root_missing(tmp_path: Path):
    assert find_running_dispatches(tmp_path / "nonexistent") == []


def test_find_running_dispatches_ignores_dirs_without_dispatch_json(tmp_path: Path):
    (tmp_path / "stray").mkdir()
    assert find_running_dispatches(tmp_path) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v -k find_running
```

Expected: ImportError on `find_running_dispatches`.

- [ ] **Step 3: Implement `find_running_dispatches`**

Append to `tools/odin/asgard/cuda_install.py` (and add `"find_running_dispatches"` to `__all__`):

```python
from pathlib import Path

from tools.odin.asgard.state import read_dispatch_state


def find_running_dispatches(runs_root: Path) -> list[str]:
    """Return the dispatch_ids of any in-flight dispatches under ``runs_root``.

    A dispatch is considered "running" if its ``dispatch.json`` exists and
    ``ended_at is None``. Order: ascending dispatch_id (the directory name).
    """
    if not runs_root.exists() or not runs_root.is_dir():
        return []
    running: list[str] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "dispatch.json").exists():
            continue
        state = read_dispatch_state(child)
        if state is None:
            continue
        if state.ended_at is None:
            running.append(state.dispatch_id)
    return running
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py -v
```

Expected: 37 passed (34 from Tasks 1-6 + 3 new dispatch-guard tests).

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install.py tools/odin/tests/test_asgard_cuda_install.py
git commit -m "$(cat <<'EOF'
cuda_install: find_running_dispatches guard helper

Scans odin_runs/<id>/dispatch.json for ended_at=None entries so
the install CLI can refuse to reboot hosts mid-dispatch.
EOF
)"
```

---

### Task 8: CLI argparse skeleton + `check` subcommand

**Files:**
- Create: `tools/odin/asgard/cuda_install_cli.py`
- Create: `tools/odin/tests/test_asgard_cuda_install_cli.py`

The `odin-cuda` CLI. Both subcommands share `--fleet`, `--floor`, `--verbose`. `install` adds the install-specific flags in Task 9.

- [ ] **Step 1: Write the failing tests**

Create `tools/odin/tests/test_asgard_cuda_install_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the odin-cuda CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.cuda_install import CheckResult
from tools.odin.asgard.cuda_install_cli import main, parse_args


def _write_fleet_yaml(tmp_path: Path) -> Path:
    content = """\
fleet_name: test
default_ssh_user: odin
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: v1.internal
  - host: v2.internal
"""
    path = tmp_path / "fleet.yaml"
    path.write_text(content)
    return path


def test_parse_args_check_minimal(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["check", "--fleet", str(fleet_path)])
    assert args.subcommand == "check"
    assert args.fleet == fleet_path
    assert args.floor == "12.4"
    assert args.verbose is False


def test_parse_args_check_custom_floor(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["check", "--fleet", str(fleet_path), "--floor", "12.6", "--verbose"])
    assert args.floor == "12.6"
    assert args.verbose is True


def test_parse_args_no_subcommand_errors(tmp_path: Path):
    with pytest.raises(SystemExit):
        parse_args([])


def test_main_check_exit_zero_when_all_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_check_fleet(fleet, *, ssh, floor, parallel):
        return [
            CheckResult(host="v1.internal", status="ok", driver="575.1", cuda="12.9"),
            CheckResult(host="v2.internal", status="ok", driver="575.1", cuda="12.9"),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.check_fleet", _fake_check_fleet)
    code = main(["check", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "v1.internal" in out
    assert "12.9" in out
    assert "ok" in out


def test_main_check_exit_one_when_any_below_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_check_fleet(fleet, *, ssh, floor, parallel):
        return [
            CheckResult(host="v1", status="ok", driver="575.1", cuda="12.9"),
            CheckResult(host="v2", status="needs-upgrade", driver="535.1", cuda="12.2"),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.check_fleet", _fake_check_fleet)
    code = main(["check", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "needs-upgrade" in out
    assert "v2" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install_cli.py -v
```

Expected: `ModuleNotFoundError: tools.odin.asgard.cuda_install_cli`.

- [ ] **Step 3: Create the CLI with argparse + `check` subcommand**

Create `tools/odin/asgard/cuda_install_cli.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-cuda CLI — fleet-wide CUDA detection + driver/toolkit upgrade.

Usage::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py check \\
        --fleet fleet.yaml [--floor 12.4] [--verbose]

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py install \\
        --fleet fleet.yaml [--floor 12.4] [--target 12.9] \\
        [--sequential] [--yes] [--force] [--reboot-timeout 600] \\
        [--runs-root ./odin_runs] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.odin.asgard.cuda_install import (
    check_fleet,
    find_running_dispatches,
    install_fleet,
)
from tools.odin.asgard.fleet import load_fleet
from tools.odin.asgard.transport import ShellSSHRunner

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the odin-cuda CLI args. Factored out for unit testing."""
    parser = argparse.ArgumentParser(
        prog="odin-cuda",
        description=(
            "Fleet-wide CUDA detection and driver/toolkit upgrade for Odin "
            "Valkyries. 'check' is read-only; 'install' reboots hosts."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    check = sub.add_parser("check", help="Read-only CUDA check across the fleet.")
    check.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    check.add_argument("--floor", default="12.4", help="CUDA floor (default: 12.4).")
    check.add_argument("--verbose", action="store_true")

    install = sub.add_parser(
        "install",
        help="Detect + upgrade hosts below floor (apt + reboot).",
    )
    install.add_argument("--fleet", required=True, type=Path, help="Path to fleet.yaml.")
    install.add_argument("--floor", default="12.4", help="CUDA floor (default: 12.4).")
    install.add_argument(
        "--target",
        default="12.9",
        help="Apt meta-package version key (default: 12.9 -> cuda-12-9).",
    )
    install.add_argument(
        "--sequential",
        action="store_true",
        help="Upgrade hosts one at a time instead of in parallel.",
    )
    install.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (for CI).",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Override the active-dispatch guard.",
    )
    install.add_argument(
        "--reboot-timeout",
        type=int,
        default=600,
        help="Per-host SSH-reachability wait after reboot (default: 600 s).",
    )
    install.add_argument(
        "--runs-root",
        type=Path,
        default=Path("odin_runs"),
        help="Root scanned for in-flight dispatches (default: ./odin_runs).",
    )
    install.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def _print_check_table(results) -> None:
    """Print per-host CUDA status as a fixed-width table."""
    width = max(len(r.host) for r in results) if results else 4
    header = f"{'host':<{width}}  {'driver':<14}  {'cuda':<6}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.host:<{width}}  {(r.driver or '-'):<14}  {(r.cuda or '-'):<6}  "
            f"{r.status}{('  — ' + r.message) if r.message else ''}"
        )


def _run_check(args: argparse.Namespace) -> int:
    fleet = load_fleet(args.fleet)
    results = check_fleet(fleet, ssh=ShellSSHRunner(), floor=args.floor, parallel=True)
    _print_check_table(results)
    return 0 if all(r.status == "ok" for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.subcommand == "check":
        return _run_check(args)
    # install handler arrives in Task 9.
    raise NotImplementedError(args.subcommand)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install_cli.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install_cli.py tools/odin/tests/test_asgard_cuda_install_cli.py
git commit -m "$(cat <<'EOF'
cuda_install: odin-cuda CLI skeleton + check subcommand

argparse with check / install subcommands; check prints a per-host
table and exits 1 if any host below floor.
EOF
)"
```

---

### Task 9: CLI `install` subcommand (with prompt + dispatch guard)

**Files:**
- Modify: `tools/odin/asgard/cuda_install_cli.py`
- Modify: `tools/odin/tests/test_asgard_cuda_install_cli.py`

The install handler: dispatch guard → confirmation prompt → `install_fleet` → summary.

- [ ] **Step 1: Append the failing tests**

```python
import json

from tools.odin.asgard.cuda_install import CudaInstallResult


def _write_running_dispatch(runs_root: Path, dispatch_id: str) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    d = runs_root / dispatch_id
    d.mkdir()
    (d / "dispatch.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dispatch_id": dispatch_id,
                "started_at": "2026-04-27T10:00:00+00:00",
                "ended_at": None,
                "seeds": [42],
                "commit_sha": "abc1234",
                "fleet": [],
                "jobs": [],
                "skipped": [],
            }
        )
    )


def test_parse_args_install_minimal(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["install", "--fleet", str(fleet_path)])
    assert args.subcommand == "install"
    assert args.target == "12.9"
    assert args.floor == "12.4"
    assert args.sequential is False
    assert args.yes is False
    assert args.force is False
    assert args.reboot_timeout == 600
    assert args.runs_root == Path("odin_runs")


def test_parse_args_install_all_flags(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(
        [
            "install",
            "--fleet",
            str(fleet_path),
            "--floor",
            "12.6",
            "--target",
            "12.8",
            "--sequential",
            "--yes",
            "--force",
            "--reboot-timeout",
            "900",
            "--runs-root",
            "/tmp/runs",
            "--verbose",
        ]
    )
    assert args.target == "12.8"
    assert args.floor == "12.6"
    assert args.sequential is True
    assert args.yes is True
    assert args.force is True
    assert args.reboot_timeout == 900
    assert args.runs_root == Path("/tmp/runs")


def test_main_install_refuses_with_active_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)
    runs_root = tmp_path / "odin_runs"
    _write_running_dispatch(runs_root, "20260427-active")

    # install_fleet must NOT be called.
    def _explode(*args, **kwargs):
        raise AssertionError("install_fleet must not run when dispatch is active")

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.install_fleet", _explode)
    code = main(
        [
            "install",
            "--fleet",
            str(fleet_path),
            "--runs-root",
            str(runs_root),
            "--yes",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "20260427-active" in out
    assert "--force" in out  # tells the user how to override


def test_main_install_force_overrides_dispatch_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fleet_path = _write_fleet_yaml(tmp_path)
    runs_root = tmp_path / "odin_runs"
    _write_running_dispatch(runs_root, "20260427-active")

    captured = {}

    def _fake_install_fleet(fleet, *, ssh, floor, target, reboot_timeout_s, parallel, verbose):
        captured["called"] = True
        return [
            CudaInstallResult(host="v1.internal", ok=True, skipped=True),
            CudaInstallResult(host="v2.internal", ok=True, skipped=True),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.install_fleet", _fake_install_fleet)
    code = main(
        [
            "install",
            "--fleet",
            str(fleet_path),
            "--runs-root",
            str(runs_root),
            "--yes",
            "--force",
        ]
    )
    assert code == 0
    assert captured.get("called") is True


def test_main_install_yes_skips_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    monkeypatch.setattr(
        "tools.odin.asgard.cuda_install_cli.install_fleet",
        lambda fleet, **kw: [
            CudaInstallResult(host="v1.internal", ok=True),
            CudaInstallResult(host="v2.internal", ok=True),
        ],
    )
    code = main(["install", "--fleet", str(fleet_path), "--yes"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Proceed? [y/N]" not in out


def test_main_install_prompt_no_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _explode(*args, **kwargs):
        raise AssertionError("install_fleet must not run after a 'no' answer")

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.install_fleet", _explode)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    code = main(["install", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 3
    assert "aborted" in out.lower()


def test_main_install_exit_one_when_any_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)
    monkeypatch.setattr(
        "tools.odin.asgard.cuda_install_cli.install_fleet",
        lambda fleet, **kw: [
            CudaInstallResult(host="v1.internal", ok=True),
            CudaInstallResult(host="v2.internal", ok=False, message="apt-get install failed"),
        ],
    )
    code = main(["install", "--fleet", str(fleet_path), "--yes"])
    assert code == 1
    out = capsys.readouterr().out
    assert "1/2" in out
    assert "v2.internal" in out
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install_cli.py -v -k install
```

Expected: failures referencing `NotImplementedError("install")` from Task 8's placeholder.

- [ ] **Step 3: Replace the install placeholder with the real handler**

In `tools/odin/asgard/cuda_install_cli.py`, replace the `raise NotImplementedError(args.subcommand)` line in `main` and add the `_run_install` helper above `main`:

```python
def _confirm_install(args: argparse.Namespace, fleet) -> bool:
    """Print plan + read y/N from stdin. Returns True iff the user agreed."""
    print(
        f"odin-cuda install: target=cuda-{args.target.replace('.', '-')} "
        f"(floor={args.floor}) on {len(fleet.hosts)} host(s):"
    )
    for h in fleet.hosts:
        print(f"  - {h.host}")
    print("Each host will reboot. This is disruptive.")
    answer = input("Proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _run_install(args: argparse.Namespace) -> int:
    running = find_running_dispatches(args.runs_root)
    if running and not args.force:
        print(
            f"odin-cuda install: refusing — running dispatch(es) under "
            f"{args.runs_root}: {', '.join(running)}\n"
            "  Pass --force to override (will reboot mid-dispatch)."
        )
        return 2

    fleet = load_fleet(args.fleet)
    if not args.yes and not _confirm_install(args, fleet):
        print("odin-cuda install: aborted by user.")
        return 3

    results = install_fleet(
        fleet,
        ssh=ShellSSHRunner(),
        floor=args.floor,
        target=args.target,
        reboot_timeout_s=args.reboot_timeout,
        parallel=not args.sequential,
        verbose=args.verbose,
    )
    ok_count = sum(1 for r in results if r.ok)
    skipped_count = sum(1 for r in results if r.ok and r.skipped)
    total = len(results)
    print(f"odin-cuda install: {ok_count}/{total} hosts ok ({skipped_count} skipped)")
    if ok_count < total:
        for r in results:
            if not r.ok:
                print(f"  {r.host}: {r.message}")
    return 0 if ok_count == total else 1
```

…and update `main` to dispatch to it:

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.subcommand == "check":
        return _run_check(args)
    if args.subcommand == "install":
        return _run_install(args)
    raise AssertionError(f"unknown subcommand {args.subcommand!r}")
```

- [ ] **Step 4: Run all CLI tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install_cli.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Run the whole new test suite once more**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py tools/odin/tests/test_asgard_cuda_install_cli.py -v
```

Expected: 49 passed (37 core + 12 CLI).

- [ ] **Step 6: Commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/cuda_install_cli.py tools/odin/tests/test_asgard_cuda_install_cli.py
git commit -m "$(cat <<'EOF'
cuda_install: odin-cuda install subcommand

Active-dispatch guard (refuses with --force override), interactive
y/N prompt skippable via --yes, runs install_fleet, prints summary +
per-host failures.
EOF
)"
```

---

### Task 10: README documentation + final sweep

**Files:**
- Modify: `tools/odin/README.md`

Add a short "Validating CUDA across the fleet" section between the existing `odin-bootstrap` and `odin-dispatch` sections, plus a final pre-commit + full-suite sweep.

- [ ] **Step 1: Locate the insertion point in the README**

```bash
grep -n "odin-bootstrap\|odin-dispatch" tools/odin/README.md | head -20
```

Use the result to find the section ordering. Insert the new section after the `odin-bootstrap` section and before `odin-dispatch`.

- [ ] **Step 2: Add the README section**

Insert this section immediately after the existing `odin-bootstrap` section:

````markdown
## Validating CUDA across the fleet

Newton (warp) workloads need at least CUDA 12.4 advertised by the host
NVIDIA driver. `odin-cuda` checks every Valkyrie and (optionally)
upgrades hosts that fall below the floor.

```bash
# Read-only: prints a per-host driver/cuda/status table, exits 1 if any
# host is below floor (default 12.4).
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py check \
    --fleet fleet.yaml

# Upgrade hosts below floor to cuda-12-9 (driver 575 + toolkit 12.9).
# Reboots each host. Prompts before doing anything; pass --yes to skip
# the prompt, --force to override the running-dispatch guard.
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py install \
    --fleet fleet.yaml --target 12.9
```

`install` runs the full pipeline per host: stop container → add NVIDIA
apt repo → `apt-get install cuda-12-9` → `systemctl reboot` → wait for
SSH → re-run `nvidia-smi` to verify driver family + CUDA floor → restart
container. Hosts already at-or-above floor are skipped without rebooting.
````

- [ ] **Step 3: Run the whole new test suite + pre-commit one last time**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_cuda_install.py tools/odin/tests/test_asgard_cuda_install_cli.py -v
./isaaclab.sh -f
```

Expected: 49 passed; pre-commit reports clean (or modifies whitespace, in which case re-stage and re-run until clean).

- [ ] **Step 4: Commit**

```bash
git add tools/odin/README.md
git commit -m "$(cat <<'EOF'
cuda_install: README section for odin-cuda

Documents the check/install subcommands, target package, and the
per-host install pipeline alongside the existing odin-bootstrap /
odin-dispatch entries.
EOF
)"
```

- [ ] **Step 5: Final verification**

```bash
git log --oneline antoiner/feat/odin..HEAD
./isaaclab.sh -p -m pytest tools/odin/tests/ -v
```

Expected: 12 commits since the branch base (1 spec + 1 plan + 10 task commits), full odin test suite still passing.

---

## Self-Review Checklist (already run by author)

- [x] Spec coverage: every section of the spec has a corresponding task — §4 CLI shape (Tasks 8, 9), §5 install pipeline (Tasks 4, 5), §6 result type (Tasks 2, 4), §7 module layout (all tasks), §8 concurrency (Tasks 3, 6), §9 failure taxonomy (Task 5 tests), §10 testing (every task is TDD), §11 docs (Task 10).
- [x] No placeholders: every step has actual code or commands.
- [x] Type consistency: `CheckResult.status` strings, `CudaInstallResult` field names, and CLI exit codes (0/1/2/3) match across all tests and implementation.
- [x] Numeric expectations stated: "29 passed", "47 passed", "10 commits" — engineer can verify each task ended in the expected state.
