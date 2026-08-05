# Resolved Preset Banner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain each active task preset choice and emit a compact INFO-level startup banner without adding a second configuration traversal.

**Architecture:** Extend the existing active preset walk with an optional private collector containing `(path, name, replacement_type)` tuples. `register_task()` freezes the collected entries on the environment config as `__resolved_presets__`, and `_run_hydra()` logs them only after all Hydra overrides have resolved successfully.

**Tech Stack:** Python, Isaac Lab configclasses, Hydra/OmegaConf, standard-library logging, pytest.

## Global Constraints

- Do not add a public API or change an existing public return signature.
- Do not add a second config traversal, registry load, backend import, or dependency.
- Add exactly one regression test; it must not launch Kit, initialize simulation, construct an environment, allocate GPU state, or run training.
- Preserve the traversal order of active presets and omit inactive branches.
- Keep `__resolved_presets__` out of Hydra and YAML serialization.
- Do not modify existing unrelated workspace changes.
- Run `./isaaclab.sh -f` before committing; review and restage any hook changes, then rerun it.

---

### Task 1: Capture and log resolved task presets

**Files:**

- Modify: `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py`
- Test: `source/isaaclab_tasks/test/core/test_hydra.py`
- Create: `source/isaaclab_tasks/changelog.d/resolved-preset-banner.rst`

**Interfaces:**

- Consumes: the existing `_pick_alternative()` and `_resolve_active_presets()` active-tree resolution path.
- Produces: private `env_cfg.__resolved_presets__: tuple[tuple[str, str, str], ...]` metadata and INFO records formatted as `path = preset_name -> ReplacementType`.
- Preserves: `resolve_task_config()` still returns `(env_cfg, agent_cfg)`, `hydra_task_config()` keeps its callback signature, and `resolve_presets()` remains behaviorally unchanged.

- [ ] **Step 1: Write the single failing Kit-free regression test**

Add `import logging` to `test_hydra.py`, then add this test after the existing typed-selector tests so it can reuse `_PhysxPhysicsCfg` and `_NewtonPhysicsCfg`:

```python
def test_resolve_task_config_logs_and_retains_resolved_newton_preset(monkeypatch, caplog):
    """Task resolution retains and logs its Newton preset without launching simulation."""

    @configclass
    class PhysicsPresetCfg(PresetCfg):
        default: _PhysxPhysicsCfg = _PhysxPhysicsCfg()
        newton_mjwarp: _NewtonPhysicsCfg = _NewtonPhysicsCfg()

    @configclass
    class EnvCfg:
        physics: PhysicsPresetCfg = PhysicsPresetCfg()

    from isaaclab_tasks.utils import parse_cfg as parse_cfg_mod

    monkeypatch.setattr(parse_cfg_mod, "load_cfg_from_registry", lambda _task, _entry: EnvCfg())
    monkeypatch.setattr("sys.argv", ["train.py", "physics=newton_mjwarp"])
    caplog.set_level(logging.INFO, logger=hydra_mod.__name__)

    env_cfg, agent_cfg = hydra_mod.resolve_task_config("Test-Resolved-Presets", "")

    assert agent_cfg is None
    assert getattr(env_cfg, "__resolved_presets__") == (
        ("env.physics", "newton_mjwarp", "_NewtonPhysicsCfg"),
    )
    assert "__resolved_presets__" not in env_cfg.to_dict()
    assert [record.getMessage() for record in caplog.records if record.name == hydra_mod.__name__] == [
        "---------------- Resolved task presets ----------------",
        "env.physics = newton_mjwarp -> _NewtonPhysicsCfg",
        "-------------------------------------------------------",
    ]
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_hydra.py::test_resolve_task_config_logs_and_retains_resolved_newton_preset \
  -q
```

Expected: FAIL at `getattr(env_cfg, "__resolved_presets__")` because the resolver does not retain preset metadata yet. The process must finish without launching Kit or creating a simulation environment.

- [ ] **Step 3: Add the minimal inline resolution collector**

In `hydra.py`, import `logging` and create the module logger:

```python
import logging

logger = logging.getLogger(__name__)
```

Add an optional collector to `_pick_alternative()`:

```python
def _pick_alternative(
    preset_obj: PresetCfg,
    selected,
    path: str = "",
    explicit_name: str | None = None,
    consumed_selected: set[str] | None = None,
    typed_hits: dict[str, set[PresetTarget]] | None = None,
    resolved_presets: list[tuple[str, str, str]] | None = None,
):
    fields = _preset_fields(preset_obj)

    def record(name: str, value):
        if resolved_presets is not None:
            resolved_presets.append((path, name, type(value).__name__))
        return value
```

Use `record()` at the three successful return points:

```python
if explicit_name in fields:
    return record(explicit_name, fields[explicit_name])

if match_name is not None:
    return record(match_name, match_value)
if "default" in fields:
    return record("default", fields["default"])
```

Add the same optional collector parameter to `_resolve_active_presets()` and pass it into every `_pick_alternative()` call. Leave `resolve_presets()` unchanged so callers that do not request capture incur only the existing resolution walk plus a `None` check at each selected node.

- [ ] **Step 4: Retain metadata and log it after final Hydra resolution**

In `register_task()`, allocate one collector before resolving the environment config and pass it to both the environment and agent `_resolve_active_presets()` calls:

```python
resolved_presets: list[tuple[str, str, str]] = []
```

After preset validation and non-preset override handling succeed, freeze it on the environment config before either return path:

```python
setattr(env_cfg, "__resolved_presets__", tuple(resolved_presets))
```

Add a private logging helper next to `_run_hydra()`:

```python
_PRESET_BANNER_HEADER = "---------------- Resolved task presets ----------------"
_PRESET_BANNER_FOOTER = "-" * len(_PRESET_BANNER_HEADER)


def _log_resolved_presets(env_cfg) -> None:
    resolved_presets = getattr(env_cfg, "__resolved_presets__", ())
    if not resolved_presets:
        return
    logger.info(_PRESET_BANNER_HEADER)
    for path, name, replacement_type in resolved_presets:
        logger.info("%s = %s -> %s", path, name, replacement_type)
    logger.info(_PRESET_BANNER_FOOTER)
```

Call `_log_resolved_presets(env_cfg)` in both `_run_hydra()` success paths after spaces, slices, agent values, and scalar Hydra overrides are restored, but immediately before invoking the user's callback. Do not log from `register_task()`, because subsequent Hydra application may still fail.

- [ ] **Step 5: Run the single regression test and verify GREEN**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/core/test_hydra.py::test_resolve_task_config_logs_and_retains_resolved_newton_preset \
  -q
```

Expected: `1 passed`, with no Kit or simulator startup output.

- [ ] **Step 6: Add the isaaclab_tasks changelog fragment**

Create `source/isaaclab_tasks/changelog.d/resolved-preset-banner.rst`:

```rst
Added
^^^^^

* Added INFO-level startup logging for resolved task presets, including physics and renderer backends.
```

- [ ] **Step 7: Run focused verification and formatting**

Run the same single pytest command from Step 5 once more after the changelog is present. Then run:

```bash
./isaaclab.sh -f
```

If pre-commit modifies files, review every changed path, retain only changes belonging to this task, stage the task files, and run `./isaaclab.sh -f` again. Confirm the final staged diff with:

```bash
git diff --cached --check
git diff --cached --stat
```

- [ ] **Step 8: Commit the focused implementation**

Stage only the implementation, one test, and changelog fragment:

```bash
git add \
  source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py \
  source/isaaclab_tasks/test/core/test_hydra.py \
  source/isaaclab_tasks/changelog.d/resolved-preset-banner.rst
git commit -m "Log resolved task presets"
```
