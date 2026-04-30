# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-dispatch hygiene: kill orphan trainers left on a Valkyrie.

Legacy-PTY mode binds the remote ``docker exec``-launched training process
to the local SSH client via ``ssh -tt``. When the dispatcher's SSH dies
mid-run (network blip, dispatcher kill, host reboot from the operator's
end), the SIGHUP propagates and *usually* kills the trainer — but on
several occasions an Isaac Sim child process has survived and kept the
GPU pinned at 99%. The next dispatch then assigns a job to that host,
the new trainer collides with the orphan, and the host enters a tight
fail loop.

The fix is two-pronged: detached submit-and-poll (separate spec) avoids
creating the orphan in the first place, and this module sweeps any
existing orphans before each dispatch starts. Both legacy and detached
runs invoke the sweep so a fleet that ran a legacy dispatch yesterday
gets cleaned up before today's detached run.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.transport import SSHRunner

__all__ = ["SweepResult", "sweep_orphan_trainers"]


# Pattern matching every Hugin-launched trainer. ``benchmark_rsl_rl`` is the
# RSL-RL entrypoint and ``benchmark_skrl`` is the SKRL one; together they
# cover both supported frameworks and nothing else inside the container.
_PKILL_PATTERN = "benchmark_rsl_rl|benchmark_skrl"


@dataclass(frozen=True)
class SweepResult:
    """Outcome of a single host's orphan-trainer sweep.

    Attributes:
        host: Target host the sweep ran on.
        attempted: ``True`` when SSH ran (whether it succeeded or not).
        ok: ``True`` when the sweep completed without an infrastructure
            failure. ``pkill`` returning 1 (no match) counts as ``ok``.
        killed_count: Number of processes ``pkill`` reported killing
            (parsed from its ``-c`` stdout). ``0`` when the pattern
            matched nothing.
        message: Human-readable error detail when ``ok`` is ``False``.
    """

    host: str
    attempted: bool
    ok: bool
    killed_count: int
    message: str | None = None


def sweep_orphan_trainers(host: ValkyrieConfig, *, ssh: SSHRunner) -> SweepResult:
    """Run ``pkill -9 -c -f <pattern>`` inside ``host``'s container.

    The ``-c`` flag makes ``pkill`` print the number of matched processes
    to stdout, which we parse and surface in the result. ``-9`` is
    deliberate: orphan trainers from a network-blip cascade have been
    observed ignoring SIGTERM (Isaac Sim's signal handlers wedge during
    physics-state teardown), so the sweep skips the polite handshake.

    ``pkill`` returns 1 when no processes match — that is the steady-state
    case on a healthy host and is treated as ``ok=True``. SSH-level
    failures (exit 255, container missing, docker daemon down) propagate
    as ``ok=False`` so the runner can decide whether to drop the host.

    Args:
        host: Target Valkyrie configuration.
        ssh: SSH runner used to dispatch the cleanup command. Tests pass
            in a fake; production passes :class:`ShellSSHRunner`.

    Returns:
        Aggregated :class:`SweepResult`.
    """
    cmd = f"docker exec {host.container_name} pkill -9 -c -f '{_PKILL_PATTERN}'"
    r = ssh.run(host, cmd, timeout_s=30.0, pty=False)

    # pkill exit codes:
    #   0 → matched and signalled at least one process.
    #   1 → no matching processes (clean host).
    #   anything else → infrastructure error (SSH, docker, container).
    if r.exit_code in (0, 1):
        try:
            killed = int(r.stdout.strip() or "0")
        except ValueError:
            killed = 0
        return SweepResult(host=host.host, attempted=True, ok=True, killed_count=killed)

    detail = r.stderr.strip() or r.stdout.strip() or f"exit {r.exit_code}"
    return SweepResult(
        host=host.host,
        attempted=True,
        ok=False,
        killed_count=0,
        message=detail,
    )
