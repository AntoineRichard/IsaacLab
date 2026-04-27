# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin CUDA detection + per-host upgrade helpers.

Public surface mirrors ``tools/odin/asgard/bootstrap.py``: a single-host
function and a fleet driver per phase (``check`` / ``install``).
"""

from __future__ import annotations

import concurrent.futures as _cf
import contextlib
import re
import time as _time
from dataclasses import dataclass, field

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.provisioner import _container_start, _container_stop
from tools.odin.asgard.transport import SSHRunner

__all__ = [
    "CheckResult",
    "CudaInstallResult",
    "TARGET_TO_DRIVER_MAJOR",
    "check_cuda_valkyrie",
    "check_fleet",
    "cuda_at_or_above",
    "install_cuda_valkyrie",
    "install_fleet",
    "parse_nvidia_smi",
    "parse_os_release",
]


_NVIDIA_SMI_HEADER_RE = re.compile(
    r"NVIDIA-SMI\s+\S+\s+Driver Version:\s+(?P<driver>\d+\.\d+(?:\.\d+)?)\s+CUDA Version:\s+(?P<cuda>\d+\.\d+)"
)


class _StepCtx:
    """Context manager that records ``perf_counter`` deltas into a dict."""

    def __init__(self, name: str, sink: dict[str, float], clock) -> None:
        self._name = name
        self._sink = sink
        self._clock = clock
        self._t0 = 0.0

    def __enter__(self) -> _StepCtx:
        self._t0 = self._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sink[self._name] = self._clock() - self._t0


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

    Args:
        measured: Observed CUDA version string (e.g. ``"12.2"``).
        floor: Minimum required CUDA version string (e.g. ``"12.4"``).
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
      2. ``nvidia-smi`` → ``no-gpu`` if missing or unparsable.
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
            message=f"nvidia-smi unavailable or unparsable: {r.stderr.strip() or r.stdout.strip()[:120]}",
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


def check_fleet(
    fleet: Fleet,
    *,
    ssh: SSHRunner,
    floor: str = "12.4",
    parallel: bool = True,
) -> list[CheckResult]:
    """Run :func:`check_cuda_valkyrie` against every host in ``fleet``.

    Args:
        fleet: Loaded :class:`Fleet` (via :func:`~tools.odin.asgard.fleet.load_fleet`).
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
    """Outcome of a single-host install attempt.

    Attributes:
        host: Host string from :class:`ValkyrieConfig`.
        ok: ``True`` when the host is at or above floor after this call.
        skipped: ``True`` when host was already at floor (no changes made).
        driver_before: Driver version seen at pre-check (``""`` if unreachable).
        cuda_before: CUDA version seen at pre-check (``""`` if unreachable).
        driver_after: Driver version after install (``""`` when skipped or failed).
        cuda_after: CUDA version after install (``""`` when skipped or failed).
        message: Human-readable diagnostic; populated for failures and placeholders.
        step_durations_s: Wall-clock seconds per named pipeline step.
    """

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
    # status == "needs-upgrade" — run the full apt + reboot + verify pipeline.
    if clock is None:
        clock = _time.monotonic
    if sleep is None:
        sleep = _time.sleep

    driver_major = TARGET_TO_DRIVER_MAJOR[target]
    apt_pkg = f"cuda-{target.replace('.', '-')}"
    os_slug = parse_os_release(ssh.run(host, "cat /etc/os-release", timeout_s=15.0).stdout)
    if os_slug is None:
        return CudaInstallResult(
            host=host.host,
            ok=False,
            driver_before=pre.driver,
            cuda_before=pre.cuda,
            message="os_slug refetch returned None (transient SSH error?)",
        )
    keyring_url = (
        f"https://developer.download.nvidia.com/compute/cuda/repos/{os_slug}/x86_64/cuda-keyring_1.1-1_all.deb"
    )

    step_durations_s: dict[str, float] = {}

    def _step(name: str) -> _StepCtx:
        return _StepCtx(name, step_durations_s, clock)

    _RECOVERY_HINT = "(container left stopped — run odin-bootstrap if recovery fails)"

    def _try_recover_container() -> None:
        """Best-effort container start on a failure path. Failure ignored."""
        with contextlib.suppress(Exception):
            _container_start(host, ssh, timeout_s=600)

    # 2. Best-effort container stop.
    with _step("container_stop"):
        _container_stop(host, ssh)

    # 3. Add NVIDIA apt repo (idempotent — keyring deb is no-op if installed).
    with _step("add_repo"):
        r = ssh.run(
            host,
            (f"set -e; cd /tmp && wget -q -O cuda-keyring.deb {keyring_url} && sudo dpkg -i cuda-keyring.deb"),
            timeout_s=120.0,
        )
        if r.exit_code != 0:
            _try_recover_container()
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=f"add_repo failed: {r.stderr.strip() or r.stdout.strip()[:200]} {_RECOVERY_HINT}",
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
            _try_recover_container()
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=f"apt-get update failed: {r.stderr.strip() or r.stdout.strip()[:200]} {_RECOVERY_HINT}",
                step_durations_s=step_durations_s,
            )

    # 5. apt-get install cuda-{target}.
    with _step("apt_install"):
        r = ssh.run(
            host,
            (f"sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::=--force-confnew {apt_pkg}"),
            timeout_s=1800.0,
        )
        if r.exit_code != 0:
            _try_recover_container()
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=(
                    f"apt-get install {apt_pkg} failed: {r.stderr.strip() or r.stdout.strip()[:200]} {_RECOVERY_HINT}"
                ),
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
            _try_recover_container()
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                message=f"post_verify: nvidia-smi unparsable after reboot {_RECOVERY_HINT}",
                step_durations_s=step_durations_s,
            )
        driver_after, cuda_after = parsed
        if not cuda_at_or_above(cuda_after, floor):
            dmesg = ssh.run(host, "dmesg | grep -i nvidia | tail -5", timeout_s=15.0).stdout
            _try_recover_container()
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                driver_after=driver_after,
                cuda_after=cuda_after,
                message=f"verify-failed: cuda {cuda_after} < floor {floor}\n{dmesg} {_RECOVERY_HINT}",
                step_durations_s=step_durations_s,
            )
        if not driver_after.startswith(driver_major + "."):
            _try_recover_container()
            return CudaInstallResult(
                host=host.host,
                ok=False,
                driver_before=pre.driver,
                cuda_before=pre.cuda,
                driver_after=driver_after,
                cuda_after=cuda_after,
                message=f"verify-failed: driver {driver_after} not in target family {driver_major}.x {_RECOVERY_HINT}",
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


def _wait_for_ssh(
    host: ValkyrieConfig,
    ssh: SSHRunner,
    *,
    timeout_s: float,
    poll_interval_s: float,
    clock,
    sleep,
) -> bool:
    """Poll ``echo cuda-install-ok`` until it succeeds or ``timeout_s`` elapses.

    Args:
        host: Target Valkyrie.
        ssh: SSH runner.
        timeout_s: Total budget in seconds.
        poll_interval_s: Seconds between probes.
        clock: Monotonic clock callable (injectable for tests).
        sleep: Sleep callable (injectable for tests).
    """
    deadline = clock() + timeout_s
    # Tiny initial grace so the host has actually rebooted before we probe.
    sleep(min(poll_interval_s, 5.0))
    while clock() < deadline:
        r = ssh.run(host, "echo cuda-install-ok", timeout_s=10.0)
        if r.exit_code == 0:
            return True
        sleep(poll_interval_s)
    return False
