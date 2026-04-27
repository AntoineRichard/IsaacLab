# Odin CUDA Install — Design

**Status:** approved
**Date:** 2026-04-27
**Task covered:** Add a new Odin tool, `odin-cuda`, that detects each
Valkyrie's host-side CUDA support and, when needed, upgrades the host to a
pinned CUDA version (driver + toolkit) so newton workloads stop hitting
`CUDA driver version is insufficient` failures across the fleet.

## 1. Motivation

Newton (warp) workloads dispatched via Odin have started failing on the
production fleet. Inspection of the two current Valkyries
(`Odin-Runner-3` at 10.176.215.13 and `Odin-Runner-5` at 10.176.214.169)
shows both at NVIDIA driver `535.161.07`, which advertises a max-supported
CUDA of **12.2** — below the 12.4 floor newton requires. Neither host has
a CUDA toolkit installed outside the docker container. The fleet was
brought up by `odin-bootstrap`, which assumes the host is already at a
sufficient driver/CUDA combination and does nothing to validate that.

`nvidia-smi`'s "CUDA Version: X.Y" line reports the **driver-supported
maximum CUDA**, not an installed toolkit version — bumping it past 12.4
requires upgrading the NVIDIA kernel driver (≥ 550). On Ubuntu, the cleanest
path is the `cuda-X-Y` apt meta-package, which pulls a matched
`(cuda-drivers-NNN, cuda-toolkit-X-Y)` pair from NVIDIA's repo. Installing
the meta-package side-steps the question of "is the driver the issue or the
toolkit?" — both end up at the right version.

The fleet Valkyries do all have passwordless `sudo` for the `horde` SSH
user (verified on 10.176.214.169), so the upgrade can be driven entirely
over the existing `SSHRunner` transport without any additional
authorization scaffolding.

This spec adds a dedicated `odin-cuda` command that handles the detect /
upgrade / verify cycle, leaving the T3.1 dispatch path and `odin-bootstrap`
unchanged.

## 2. Goals

- A single command, `odin-cuda check --fleet fleet.yaml`, reports per-host
  driver version, advertised CUDA version, and a status verdict
  (`ok` / `needs-upgrade` / `no-gpu` / `unreachable` / `unsupported-os`).
  Read-only. Exits 0 iff every host is at `≥ floor` (default 12.4).
- A second command, `odin-cuda install --fleet fleet.yaml`, brings every
  below-floor host up to the pinned target (default `cuda-12-9` →
  driver 575 + toolkit 12.9), reboots, and verifies the result.
- Parallel across hosts by default; `--sequential` to opt out.
- Idempotent: re-running `install` against an already-up-to-date host is a
  no-op (`skipped=True`, no reboot).
- Refuse to run `install` while there is a running dispatch on the
  controller; `--force` to override.
- Hard refuse on non-Ubuntu hosts (`unsupported-os`). No silent best-effort.

## 3. Non-goals

- No changes to T3.1's `run_dispatch`, `preflight`, worker, or dispatch
  state-machine. CUDA validation does **not** become an automatic
  preflight step — running `odin-cuda` is a deliberate fleet-stand-up
  action, not a per-dispatch gate, so dispatches stay fast and never
  trigger surprise reboots.
- No container-side toolkit management. Newton's container ships its own
  CUDA toolkit; this tool only addresses the host-side driver/CUDA pair.
- No multi-GPU-per-host coordination beyond what `apt` and `systemctl
  reboot` do already.
- No new fleet-YAML schema. `ValkyrieConfig` and `Fleet` from
  `tools/odin/asgard/fleet.py` are sufficient.
- No CHANGELOG.rst entry (this lives under `tools/odin/`, not `source/`).

## 4. CLI shape

A new console entry, mirroring the `odin-bootstrap` / `odin-dispatch`
convention:

```text
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py check \
    --fleet fleet.yaml \
    [--floor 12.4] \
    [--verbose]

PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py install \
    --fleet fleet.yaml \
    [--floor 12.4] [--target 12.9] \
    [--sequential] [--yes] [--force] \
    [--reboot-timeout 600] \
    [--runs-root ./odin_runs] \
    [--verbose]
```

`check` prints a per-host table and exits 0 iff every host is at
`cuda ≥ floor`.

`install` first runs `check`; for hosts below floor it runs the install
pipeline (§5). `--floor` and `--target` are intentionally separate:

- `--floor` (default 12.4) is the "is this host acceptable?" threshold.
- `--target` (default 12.9) is the apt package family to install
  (`cuda-12-9`).

Keeping them separate means re-running `check --floor 12.4` after an
install reports `ok` regardless of whether the install pulled 12.9 or a
later target. `--yes` skips the interactive confirmation prompt (CI use).
`--force` overrides the active-dispatch refusal.

## 5. Per-host install pipeline

Steps short-circuit on first failure. Pattern matches `bootstrap.py`.

1. **Pre-check.** SSH probe, `nvidia-smi` parse, OS-release check
   (Ubuntu 22.04 or 24.04 only). If `cuda_before ≥ floor`, return
   `ok=True, skipped=True` immediately — no reboot.
2. **Stop container.** Best-effort
   `cd {isaaclab_path} && ./docker/container.py stop`. Non-zero exit is
   non-fatal (container may not exist).
3. **Add NVIDIA apt repo.** Idempotent:
   `wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{2204|2404}/x86_64/cuda-keyring_1.1-1_all.deb`
   then `sudo dpkg -i cuda-keyring_1.1-1_all.deb`. The keyring `.deb` is a
   no-op if already installed.
4. **`sudo apt-get update -o Acquire::Retries=3`.**
5. **`sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-{target} cuda-drivers-{driver_major}`.**
   Both packages are needed: on Ubuntu 24.04 `cuda-12-9` ships only the
   toolkit + userspace libs (`libnvidia-compute-575`), not the kernel
   module. `cuda-drivers-575` brings in `nvidia-dkms-575` which builds the
   matching kmod and updates the initramfs so the new module loads on the
   next reboot. Without it the host comes back with userspace 575 +
   kernel 535 and `nvidia-smi` reports "Driver/library version mismatch".
   Single apt transaction so partial failures roll back cleanly.
6. **Reboot.** `sudo systemctl reboot`. The SSH connection is expected to
   drop with a non-zero exit; that is treated as success for this step.
7. **Wait for SSH.** Poll `ssh host echo cuda-install-ok` every 10 s up
   to `--reboot-timeout` (default 600 s). Timeout is a hard failure
   (`reboot-timeout`).
8. **Post-verify.** Re-run `nvidia-smi`. The reported CUDA must be
   `≥ floor`, and the driver major must match the target's driver family
   (e.g. `target=12.9` → driver major `575`). Mismatch is a hard failure
   (`verify-failed`); the message includes the tail of `dmesg | grep -i
   nvidia` for triage.
9. **Restart container.** `./docker/container.py start`. Best-effort:
   non-zero is logged and emitted in `message` but does not flip
   `ok=False` — the user can re-run `odin-bootstrap` to recover.

## 6. Result type

```python
@dataclass
class CudaInstallResult:
    host: str
    ok: bool
    skipped: bool                          # True if already at floor
    driver_before: str                     # e.g. "535.161.07"
    cuda_before: str                       # e.g. "12.2"
    driver_after: str = ""
    cuda_after: str = ""
    message: str = ""
    step_durations_s: dict[str, float] = field(default_factory=dict)
```

`CheckResult` is a smaller variant — same `host`, `driver`, `cuda`,
`status`, `message` fields without timings.

## 7. Module layout

```text
tools/odin/asgard/
├── cuda_install.py        # core: CheckResult, CudaInstallResult,
│                          #       check_cuda_valkyrie(),
│                          #       install_cuda_valkyrie(),
│                          #       check_fleet(), install_fleet()
├── cuda_install_cli.py    # the `odin-cuda` CLI (check / install subcommands)
└── ... (existing modules unchanged)

tools/odin/tests/
├── test_cuda_install.py        # unit tests using FakeSSHRunner
└── test_cuda_install_cli.py    # argparse + main() coverage
```

Reuses existing primitives:

- `ValkyrieConfig` / `Fleet` / `load_fleet` from `fleet.py`.
- `SSHRunner` / `ShellSSHRunner` from `transport.py` — every host
  interaction routes through `ssh.run(host, cmd, timeout_s=...)`. Same
  fakes used by preflight / bootstrap tests.
- `_container_start` / `_container_stop` from `provisioner.py` — already
  re-imported by `bootstrap.py`; we follow the same pattern.
- `concurrent.futures.ThreadPoolExecutor` for parallel installs, mirroring
  `bootstrap_fleet`.

## 8. Concurrency and safety

- **Parallel default, sequential opt-out.** Each host's apt + reboot is
  local — no shared state, so parallelism is safe. `--sequential`
  remains for the case where a maintenance window dictates it (matches
  `odin-bootstrap`'s convention).
- **Active-dispatch guard.** Before any install work, scan
  `--runs-root` (default `./odin_runs`) for any `dispatch.json` whose
  `state` is `running`. If found, refuse with a message naming the
  dispatch_id. `--force` overrides.
- **Confirmation prompt.** Unless `--yes`, the CLI lists the hosts that
  will be upgraded and the target version, and reads `y/N` from stdin.

## 9. Failure taxonomy

Hard failures (ok=False; surfaces in CLI exit code):

| Status              | Trigger                                                              |
|---------------------|----------------------------------------------------------------------|
| `unreachable`       | SSH probe fails.                                                     |
| `no-gpu`            | `nvidia-smi` missing or non-zero. Hard fail for `install`.           |
| `unsupported-os`    | Not Ubuntu 22.04 / 24.04. No install attempt.                        |
| `apt-failed`        | Either `apt-get update` or `apt-get install cuda-{target} cuda-drivers-{driver_major}` non-zero. |
| `reboot-timeout`    | SSH did not return within `--reboot-timeout` after `reboot`.         |
| `verify-failed`     | Post-reboot `nvidia-smi` reports `cuda < floor`, or driver major     |
|                     | doesn't match target family.                                         |

Soft failures (logged, ok=True still possible):

- `container.py stop` non-zero before reboot — ignored.
- `container.py start` non-zero after reboot — `message` set,
  `ok=True` preserved.

## 10. Testing

- **Unit tests** with a `FakeSSHRunner` returning canned per-command
  stdout. One fixture per status in §9, plus the `skipped=True` happy path.
- **CLI tests** for both `check` and `install` subcommands: argparse
  surface, exit codes, the active-dispatch guard, the `--yes` prompt
  path.
- **Live-host integration test** (slow-marked, skipped by default): when
  `ODIN_CUDA_LIVE_HOST` is set, run the full pipeline against that host.
  Verifies the apt URL is reachable and the actual cuda-keyring deb
  resolves, but does not reboot — the test stops at step 5.

## 11. Documentation

Add a short "odin-cuda" section to `tools/odin/README.md` between the
existing `odin-bootstrap` and `odin-dispatch` entries. No `CHANGELOG.rst`
update — this code lives outside `source/<package>/`.

## 12. Open questions

None at design time. Items deferred to implementation:

- Exact polling cadence on the reboot wait (10 s default, may shorten if
  it feels slow in live testing).
- Whether to support `target=latest` as a non-default — kept out of v1
  for predictability; trivially addable later.
