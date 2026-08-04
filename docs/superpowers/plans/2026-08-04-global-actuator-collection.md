# Global Actuator Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace articulation-local actuator storage with one simulation-scoped, clone-aware collection that preserves the develop API, exposes scoped group and exact-type views, aggregates built-in stateless actuator execution, and improves end-to-end runtime without adding steady-state allocation or synchronization.

**Architecture:** `SimulationContext` owns a lazily created `ActuatorCollection` service and eagerly registered lifecycle bridges. Articulations register resolved actuator metadata during their order-10 initialization callback; the order-20 bridge transactionally builds one generation of flat Warp-owned type and joint-domain stores, zero-copy Torch aliases, articulation facades, backend bindings, and immutable articulation-owned execution plans. Group views are stable strided views, exact-type views are compact contiguous views, legacy dense fields are lazy projections, solver properties remain articulation data, and deprecated topology mutation can only take effect across a STOP-to-PHYSICS_READY rebuild.

**Tech Stack:** Python 3.12, PyTorch, NVIDIA Warp, CUDA graphs, Isaac Lab `SimulationContext`/`ServiceLocator`/clone plans, PhysX, Newton, experimental OpenUSD PhysX, pytest, Sphinx, LaTeX, and the existing runtime benchmark command.

**Approved Design:** `docs/superpowers/specs/2026-08-04-global-actuator-collection-design.md`

## Global Constraints

- Work only in `/home/antoiner/Documents/IsaacLab/docs/superpowers/worktrees/actuator-global-storage` on branch `antoiner/actuators-collection-split-6248`.
- Preserve all public develop symbols. New behavior may deprecate an API with a once-only warning, but may not remove or rename it.
- Keep `ActuatorCollection` exported as the manager/documentation anchor. Expose only `ActuatorCollection.ArticulationView` through `Articulation.actuators`; do not add a scene-wide public accessor.
- Use exact actuator classes as `by_type` keys. Do not add string type aliases or a second convenience API.
- Keep `ArticulationData.joint_stiffness` and `joint_damping` as solver properties. Do not allocate solver-only fields in typed actuator storage.
- Own canonical buffers in Warp and expose Torch with `ProxyArray`/zero-copy aliases. Do not introduce a required or optional dependency.
- Use clone-plan rows and masks. Python construction cost and object count must not scale with the number of cloned worlds.
- Keep the ordinary selector path synchronization-free. Same-device Torch/Warp selector and value arrays are allocation-free; accepted Python `Sequence` inputs are an explicit compatibility conversion path that may allocate and, on CUDA, transfer once per call. Debug validation is opt-in and may synchronize in order to raise precise value-dependent errors.
- Keep actuator application owned by each articulation in this change. Shape the private plan so a future scene executor can schedule the same global ranges without redesigning storage.
- Preserve CPU/GPU semantics. CUDA graph capture is an optimization, not a separate behavior.
- All new files use the 2026 SPDX header. Public physical quantities use Google-style docstrings with SI units.
- Follow red-green-refactor for every behavior change: add the focused test, run it and record the expected failure, implement, then rerun it. A regression test is not complete until its pre-fix failure has been observed.
- Use `./isaaclab.sh -p` for Python, pytest, and benchmark helpers. Never invoke a bare `python` command.
- Before every commit: run the focused tests, run `./isaaclab.sh -f`, review any formatter edits, stage them, rerun `./isaaclab.sh -f`, and then commit. Never amend review-fix commits.
- Keep benchmark JSON, profiler traces, LaTeX, and PDF uncommitted under `/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/`.
- Do not add the previously rejected fake-remote-URL neural-checkpoint regression test; checkpoint resolution is outside this refactor.

## Frozen Public Contract

The implementation must converge on these names and signatures so later tasks do not invent parallel APIs:

```python
class ActuatorCollection:
    def register_articulation(
        self,
        *,
        key: object,
        cfgs: Mapping[str, ActuatorBaseCfg],
        control: ActuatorControl,
        replication_cfg_id: int,
        debug_validation: bool,
        debug_value_resolution: bool,
    ) -> ActuatorCollection.ArticulationView: ...

    def finalize(self) -> None: ...
    def clear_generation(self) -> None: ...
    def close(self) -> None: ...

    class ArticulationView(dict[str, ActuatorBase]):
        @property
        def by_type(self) -> Mapping[type[ActuatorBase], TypeView]: ...

        @property
        def command(self) -> Command: ...

        @property
        def joint_command(self) -> JointCommand: ...

        @property
        def computed_effort(self) -> ProxyArray: ...

        @property
        def applied_effort(self) -> ProxyArray: ...

        def reset(self, env_ids: Sequence[int] | slice | None = None) -> None: ...
        def compute(self, dt: float = 0.0) -> None: ...
        def submit_commands(self) -> None: ...

        class TypeView:
            @property
            def parameter_names(self) -> frozenset[str]: ...

            @property
            def parameters(self) -> Mapping[str, ProxyArray]: ...

            @property
            def joint_names(self) -> tuple[str, ...]: ...

            @property
            def joint_indices(self) -> torch.Tensor: ...

            @property
            def group_slices(self) -> Mapping[str, slice]: ...

            @property
            def num_instances(self) -> int: ...

            def set_parameter_index(
                self,
                name: str,
                value: float | torch.Tensor | wp.array(dtype=wp.float32),
                *,
                env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
                joint_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
            ) -> None: ...

            def set_parameter_mask(
                self,
                name: str,
                value: float | torch.Tensor | wp.array(dtype=wp.float32),
                *,
                env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
                joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            ) -> None: ...

class ActuatorBase:
    @property
    def parameter_names(self) -> frozenset[str]: ...

    @property
    def parameters(self) -> Mapping[str, ProxyArray]: ...

    def set_parameter_index(
        self,
        name: str,
        value: float | torch.Tensor | wp.array(dtype=wp.float32),
        *,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
        joint_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
    ) -> None: ...

    def set_parameter_mask(
        self,
        name: str,
        value: float | torch.Tensor | wp.array(dtype=wp.float32),
        *,
        env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
        joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
    ) -> None: ...
```

`joint_ids` always use articulation DOF coordinates, including on a compact type or group view. An explicit type-view joint ID fans out to every compact slot when same-type groups overlap; `joint_ids=None` selects compact slots directly in stable group/configuration order. Index setters form the Cartesian product of selected worlds and joints. Mask setters accept full articulation masks and apply them to every matching compact slot. The implementation details below preserve these semantics exactly.

---

### Task 1: Pin Develop Compatibility and the Current PR Baseline

**Files:**

- Modify: `source/isaaclab/test/actuators/test_actuator_collection.py`
- Modify: `source/isaaclab/test/assets/_articulation_iface_test_utils.py`
- Modify: `source/isaaclab/test/assets/test_articulation_ordering_iface.py`
- Create: `source/isaaclab/test/actuators/test_actuator_collection_compatibility.py`

- [ ] **Step 1: Record the comparison revisions and current focused-test result**

Use these immutable revisions throughout implementation:

```text
develop merge-base: 378225f8d2af0a9920e18a934ee7d044844e023e
intermediate staging snapshot: 77b77692aa78460d46759c032971dd74457f8521
pre-refactor PR head: 5c59a092be11e4c95d63195476f58e7d2b0b8084
final implementation: resolve with git rev-parse HEAD after Task 16
```

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection.py -q
```

Expected: the existing 24 tests pass. Save the terminal output in the implementation notes; do not commit generated output.

- [ ] **Step 2: Add characterization tests for APIs that must remain live**

Add tests with these exact names:

```python
def test_develop_actuator_gain_writers_update_groups_and_backend() -> None:
    articulation, control = make_initialized_articulation_fixture()
    values = torch.tensor([[11.0, 17.0]], device=articulation.device)
    articulation.write_actuator_stiffness_to_sim(
        stiffness=values,
        env_ids=torch.tensor([0], device=articulation.device),
        joint_ids=torch.tensor([0, 1], device=articulation.device),
    )
    torch.testing.assert_close(articulation.actuators["shoulder"].stiffness[:1], values)
    torch.testing.assert_close(control.native_stiffness[:1, :2], values)


def test_solver_gain_writers_remain_distinct_from_actuator_parameters() -> None:
    articulation, control = make_initialized_articulation_fixture()
    before = articulation.actuators["shoulder"].stiffness.clone()
    articulation.write_joint_stiffness_to_sim_index(
        torch.tensor([[31.0]], device=articulation.device),
        env_ids=torch.tensor([0], device=articulation.device),
        joint_ids=torch.tensor([0], device=articulation.device),
    )
    torch.testing.assert_close(articulation.actuators["shoulder"].stiffness, before)
    assert control.solver_stiffness[0, 0].item() == 31.0
```

Also characterize mapping lookup/iteration, in-place parameter writes, target setters, target/data alias identity, `computed_torque`, `applied_torque`, `soft_joint_vel_limits`, `gear_ratio`, and their current warning behavior. Use a focused fake that models the nested `ArticulationView` path; do not keep constructing `ActuatorCollection.Command` as a standalone class.

- [ ] **Step 3: Run the characterization set**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_actuator_collection_compatibility.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py -q
```

Expected: PASS on the pre-refactor PR head. These tests describe compatibility, not the new global topology.

- [ ] **Step 4: Format, stage, and commit the characterization tests**

```bash
./isaaclab.sh -f
git add source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_actuator_collection_compatibility.py \
  source/isaaclab/test/assets/_articulation_iface_test_utils.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py
./isaaclab.sh -f
git commit -m "Pin actuator compatibility behavior"
```

---

### Task 2: Define Exact-Class Schemas and Managed Group Bindings

**Files:**

- Create: `source/isaaclab/isaaclab/actuators/actuator_storage.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_base.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_pd.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_net.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py`
- Create: `source/isaaclab/test/actuators/test_actuator_collection_storage.py`

- [ ] **Step 1: Write failing schema and binding tests**

Add these tests:

```python
@pytest.mark.parametrize(
    ("actuator_type", "expected"),
    [
        (ImplicitActuator, frozenset({"stiffness", "damping", "effort_limit", "velocity_limit"})),
        (IdealPDActuator, frozenset({"stiffness", "damping", "effort_limit", "velocity_limit"})),
        (
            DCMotor,
            frozenset({"stiffness", "damping", "effort_limit", "velocity_limit", "saturation_effort"}),
        ),
        (ActuatorNetMLP, frozenset({"effort_limit", "velocity_limit", "saturation_effort"})),
        (ActuatorNetLSTM, frozenset({"effort_limit", "velocity_limit", "saturation_effort"})),
    ],
)
def test_exact_class_parameter_schema_excludes_solver_properties(actuator_type, expected) -> None:
    assert actuator_type._parameter_schema().parameter_names == expected
    assert "effort_limit_sim" not in expected
    assert "velocity_limit_sim" not in expected
    assert "armature" not in expected
    assert "friction" not in expected
    assert "dynamic_friction" not in expected
    assert "viscous_friction" not in expected


def test_managed_group_assignment_copies_without_rebinding() -> None:
    group, canonical = make_bound_ideal_pd_group(num_worlds=3, group_dofs=2, type_dofs=5, offset=1)
    held = group.stiffness
    group.stiffness = torch.full((3, 2), 7.0, device=held.device)
    assert group.stiffness.data_ptr() == held.data_ptr()
    assert group.stiffness.stride() == (5, 1)
    torch.testing.assert_close(canonical[:, 1:3], held, rtol=0.0, atol=0.0)


def test_neural_gains_are_lazy_deprecated_sidecars() -> None:
    group = make_bound_neural_group()
    assert group._deprecated_sidecars == {}
    with pytest.warns(DeprecationWarning, match="stiffness"):
        stiffness = group.stiffness
    assert stiffness.shape == (group.num_envs, group.num_joints)
    assert set(group._deprecated_sidecars) == {"stiffness"}


def test_solver_compatibility_fields_are_lazy_and_not_typed() -> None:
    group, store = make_bound_dc_group()
    assert "armature" not in store.allocated_fields(DCMotor)
    assert group._solver_compatibility_sidecars == {}
    armature = group.armature
    assert isinstance(armature, torch.Tensor)
    assert set(group._solver_compatibility_sidecars) == {"armature"}
```

- [ ] **Step 2: Run the tests and observe the red state**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  -k "schema or managed_group or neural_gains or solver_compatibility" -q
```

Expected: FAIL because schemas, canonical binding, per-slot `saturation_effort`, and neural sidecars do not exist.

- [ ] **Step 3: Add private storage/schema value types**

Implement these private types in `actuator_storage.py`:

```python
@dataclass(frozen=True)
class _FieldSpec:
    name: str
    dtype: type
    unit: str
    role: Literal["parameter", "output", "scratch", "state"]
    fill: float
    backend_side_effect: str | None


@dataclass(frozen=True)
class _ActuatorSchema:
    fields: tuple[_FieldSpec, ...]
    graphable: bool
    stateful: bool

    @property
    def parameter_names(self) -> frozenset[str]:
        return frozenset(field.name for field in self.fields if field.role == "parameter")


@dataclass(frozen=True)
class _GroupBinding:
    generation: int
    joint_indices: torch.Tensor
    joint_names: tuple[str, ...]
    type_slice: slice
    arrays: Mapping[str, ProxyArray]
```

Keep solver compatibility fields outside `_ActuatorSchema`. Add `_ManagedParameter` descriptors that return the bound Torch view and copy whole-attribute assignment into it. Existing group attributes for `effort_limit_sim`, `velocity_limit_sim`, armature, and all friction fields use lazy group-local compatibility buffers initialized from resolved solver values; they remain `torch.Tensor`, never enter typed execution storage, and are refreshed by existing solver-property writer hooks. They do not warn because they are existing public group fields. An unbound ordinary/custom actuator retains its existing owned tensor behavior.

- [ ] **Step 4: Declare exact built-in schemas and neural sidecars**

Replace `_EXECUTION_PARAMETER_NAMES` with class-owned schemas returned by the private exact-class hook `_parameter_schema()`. Treat a class as managed only when that exact class declares the hook in its own `__dict__`; an opaque subclass must not inherit a stateless storage/execution promise accidentally. Define `DCMotor.saturation_effort` as a real per-world, per-slot parameter and make clipping read that tensor. Neural exact schemas omit `stiffness` and `damping`; their inherited gain access allocates only the requested group-local sidecar and emits one warning per attribute per group.

Keep these ownership rules in code comments beside the schema definitions:

```text
typed actuator parameters: stiffness, damping, actuator effort/velocity limits, saturation_effort
typed outputs: computed_effort, applied_effort
solver compatibility only: effort_limit_sim, velocity_limit_sim, armature, all friction fields
structural/state: delay/history/recurrent buffers, network metadata, lookup tables
legacy fill only when no type declares it: gear_ratio = 1.0
```

For delayed and remotized exact classes, declare meaningful PD/DC numeric fields but keep delay/history/lookup data as structural metadata and state. Test that numeric differences do not alter the structural signature and that the state allocation remains owned by the eager group segment.

- [ ] **Step 5: Pass schema/storage tests and existing actuator model tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection.py -q
```

Expected: PASS, including exact DCMotor output equality before and after the per-slot conversion.

- [ ] **Step 6: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_storage.py \
  source/isaaclab/isaaclab/actuators/actuator_base.py \
  source/isaaclab/isaaclab/actuators/actuator_pd.py \
  source/isaaclab/isaaclab/actuators/actuator_net.py \
  source/isaaclab/isaaclab/actuators/actuator_base_cfg.py \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py
./isaaclab.sh -f
git commit -m "Define exact actuator parameter schemas"
```

---

### Task 3: Build Clone-Aware Flat Layouts and Zero-Copy Views

**Files:**

- Modify: `source/isaaclab/isaaclab/assets/asset_base.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_storage.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection_storage.py`
- Modify: `source/isaaclab/test/cloner/test_clone_plan_algebra.py`

- [ ] **Step 1: Write failing layout and clone-scaling tests**

Add pure builder tests with these names:

```python
def test_layout_uses_original_cfg_identity_after_asset_copy() -> None:
    cfg, copied_cfg, plan = make_copied_cfg_clone_plan()
    assert id(copied_cfg) not in plan.cfg_rows
    layout = _build_articulation_layout(
        replication_cfg_id=id(cfg), clone_plan=plan, registrations=make_variant_registrations()
    )
    assert layout.prototype_rows == plan.cfg_rows[id(cfg)]


@pytest.mark.parametrize("num_worlds", [1, 64, 4096])
def test_layout_python_object_count_is_clone_count_independent(num_worlds: int) -> None:
    layout, counters = build_instrumented_layout(num_worlds=num_worlds, num_prototypes=4)
    assert layout.num_worlds == num_worlds
    assert counters.group_records == 12
    assert counters.prototype_records == 4


def test_type_block_is_contiguous_and_group_block_is_strided_zero_copy() -> None:
    store, articulation, group = make_two_articulation_store()
    type_stiffness = articulation.type_proxy(IdealPDActuator, "stiffness").torch
    assert type_stiffness.is_contiguous()
    assert group.stiffness.stride() == (articulation.type_dofs(IdealPDActuator), 1)
    assert group.stiffness.untyped_storage().data_ptr() == store.stiffness.torch.untyped_storage().data_ptr()


def test_multiple_articulation_type_ranges_are_disjoint() -> None:
    store, first, second = make_two_articulation_store()
    first_stiffness = first.type_proxy(IdealPDActuator, "stiffness").torch
    second_stiffness = second.type_proxy(IdealPDActuator, "stiffness").torch
    first_stiffness.fill_(3.0)
    second_stiffness.fill_(9.0)
    torch.testing.assert_close(first_stiffness, torch.full_like(first_stiffness, 3.0))
    torch.testing.assert_close(second_stiffness, torch.full_like(second_stiffness, 9.0))


def test_overlapping_type_layout_builds_one_to_many_joint_fanout() -> None:
    layout = make_type_layout(group_joint_ids=((0, 2), (2, 3), (2,)))
    assert layout.compact_joint_indices == (0, 2, 2, 3, 2)
    assert csr_slots(layout, articulation_joint_id=2) == (1, 2, 4)
```

The heterogeneous test must assign four prototypes with different group parameters to 4096 worlds using `ClonePlan.clone_mask` and verify the expanded values exactly.

- [ ] **Step 2: Run the new tests and observe failure**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/cloner/test_clone_plan_algebra.py \
  -k "layout or type_block or articulation_type_ranges" -q
```

Expected: FAIL because original config identity is lost and global layouts/stores do not exist.

- [ ] **Step 3: Retain clone-plan identity without iterating worlds**

In `AssetBase.__init__`, store the original identity immediately before copying:

```python
queue_replication(cfg)
self._replication_cfg_id = id(cfg)
self.cfg = cfg.copy()
```

The layout builder must use `clone_plan.cfg_rows[replication_cfg_id]` plus tensor indexing into `clone_plan.clone_mask`. Do not call `iter_sources()`, `.tolist()`, or construct a Python object per world.

- [ ] **Step 4: Implement immutable layouts and flat Warp-owned stores**

Add private records and builders:

```python
@dataclass(frozen=True)
class _ArticulationLayout:
    registration_key: object
    num_worlds: int
    num_joints: int
    prototype_rows: tuple[int, ...]
    group_layouts: tuple[_GroupLayout, ...]
    type_layouts: Mapping[type[ActuatorBase], _TypeLayout]


@dataclass(frozen=True)
class _TypeLayout:
    actuator_type: type[ActuatorBase]
    global_slice: slice
    articulation_offset: int
    num_worlds: int
    num_dofs: int
    compact_joint_indices: tuple[int, ...]
    articulation_to_compact_offsets: tuple[int, ...]
    articulation_to_compact_slots: tuple[int, ...]


class _TypedStore:
    def allocate(self, layouts: Sequence[_ArticulationLayout], *, device: str) -> None: ...
    def type_proxy(self, layout: _TypeLayout, field: str) -> ProxyArray: ...
    def group_proxy(self, layout: _GroupLayout, field: str) -> ProxyArray: ...
```

Allocate exactly one flat one-dimensional Warp array per `(exact type, field)`. Its length is the sum of `layout.num_worlds * layout.num_dofs` over articulation blocks of that exact type, in registration order. Interpret each articulation's pointer-offset segment as a contiguous `[num_worlds, num_type_dofs]` view. Construct group aliases from that segment with shape `[num_worlds, group_dofs]` and byte strides `[num_type_dofs * sizeof(float), sizeof(float)]`. Cache a single zero-copy Torch alias in each `ProxyArray`.

`compact_joint_indices` has one articulation-DOF entry per compact slot and may contain duplicates when same-exact-type groups overlap. Build a deterministic CSR fanout in configuration/compact order: `articulation_to_compact_offsets` has length `num_joints + 1`, and `articulation_to_compact_slots` contains every matching compact slot. Finalization copies these tables to immutable device arrays once. Never collapse an overlapping type layout into a one-to-one lookup.

Initialize prototype-varying numeric fields with one generic expansion launch per field, using a device prototype-assignment array derived from clone masks. Homogeneous values may use a fill launch. No host loop may scale with `num_worlds`.

- [ ] **Step 5: Run storage, clone, and pointer tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/cloner/test_clone_plan_algebra.py -q
```

Expected: PASS on CPU and CUDA when CUDA is available.

- [ ] **Step 6: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/assets/asset_base.py \
  source/isaaclab/isaaclab/actuators/actuator_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/cloner/test_clone_plan_algebra.py
./isaaclab.sh -f
git commit -m "Build clone-aware actuator storage layouts"
```

---

### Task 4: Add the Simulation-Scoped Transactional Lifecycle

**Files:**

- Modify: `source/isaaclab/isaaclab/sim/simulation_context.py`
- Modify: `source/isaaclab/isaaclab/assets/asset_base.py`
- Rewrite: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_control.py`
- Create: `source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py`
- Modify: `source/isaaclab/test/sim/test_simulation_context.py`

- [ ] **Step 1: Write failing callback-order and lazy-service tests**

Use a fake `PhysicsManager` whose dispatch snapshots callbacks before invocation. Add:

```python
def test_context_registers_actuator_bridges_before_ready_dispatch() -> None:
    sim = make_context_with_fake_physics_manager()
    callbacks = fake_callbacks(PhysicsEvent.PHYSICS_READY)
    assert callbacks[5].name == "render_context_initialize"
    assert callbacks[20].name == "actuator_collection_finalize"
    assert sim.services[ActuatorCollection] is None


def test_ready_finalizes_every_order_ten_registration() -> None:
    sim = make_context_with_fake_physics_manager()
    first = make_deferred_asset(sim, name="first")
    second = make_deferred_asset(sim, name="second")
    dispatch_ready()
    assert first.actuators.is_ready
    assert second.actuators.is_ready
    assert first.is_initialized
    assert second.is_initialized
    assert sim.services[ActuatorCollection].registration_keys == (first, second)


def test_callback_added_during_dispatch_does_not_hide_finalization_bug() -> None:
    sim = make_context_with_fake_physics_manager()
    make_deferred_asset(sim, name="first")
    ready_snapshot = snapshot_callbacks(PhysicsEvent.PHYSICS_READY)
    invoke_snapshot(ready_snapshot)
    assert sim.services[ActuatorCollection].is_finalized
```

- [ ] **Step 2: Write failing transactional and generation tests**

Add:

```python
def test_failed_finalization_publishes_no_partial_generation() -> None:
    collection = make_manager()
    good = collection.register_articulation(**valid_registration("good"))
    bad = collection.register_articulation(**invalid_registration("bad"))
    with pytest.raises(ValueError, match="bad.*IdealPDActuator.*wheel"):
        collection.finalize()
    assert collection.generation is None
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = good.command.position
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = bad.command.position


def test_second_completion_failure_rolls_back_first_completed_asset() -> None:
    collection, first, second = make_two_pending_articulations()
    second.control.complete_error = RuntimeError("second completion")
    with pytest.raises(RuntimeError, match="second completion"):
        collection.finalize()
    assert collection.generation is None
    assert not first.is_initialized and not first.data.is_primed
    assert not second.is_initialized and not second.data.is_primed
    assert first.control.invalidate_count == second.control.invalidate_count == 1
    assert_candidate_storage_released(collection)
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = first.actuators.command.position


def test_stop_replay_invalidates_old_facade_and_builds_new_generation() -> None:
    sim, asset = make_ready_asset()
    old = asset.actuators
    old_generation = old.generation
    dispatch_stop()
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = old.command.position
    dispatch_ready()
    assert asset.actuators.generation == old_generation + 1
    assert asset.actuators is not old


def test_clear_instance_closes_collection_before_second_context() -> None:
    first_sim, first_asset = make_ready_asset()
    first_view = first_asset.actuators
    SimulationContext.clear_instance()
    with pytest.raises(RuntimeError, match="closed"):
        _ = first_view.command.position
    second_sim = make_context_with_fake_physics_manager()
    assert second_sim.services[ActuatorCollection] is None


def test_late_registration_marks_active_generation_dirty() -> None:
    manager, first = make_finalized_manager()
    late = manager.register_articulation(**valid_registration("late"))
    assert not late.is_ready
    with pytest.raises(RuntimeError, match="late registration.*rebuild"):
        first.compute()
    manager.clear_generation()
    register_replay_assets(manager, first, late)
    manager.finalize()
    assert first.is_ready and late.is_ready
```

- [ ] **Step 3: Run lifecycle tests and observe failure**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py \
  source/isaaclab/test/sim/test_simulation_context.py \
  -k "actuator or failed_finalization or completion_failure or stop_replay or second_context" -q
```

Expected: FAIL because the service, bridge callbacks, pending facade state, and generation invalidation are absent.

- [ ] **Step 4: Add a generic deferred-initialization hook to `AssetBase`**

Keep existing assets synchronous. Only an articulation that calls the protected hook is deferred:

```python
def _initialize_callback(self, event) -> None:
    if self._is_initialized:
        return
    self._backend = SimulationContext.instance().physics_manager.get_backend()
    self._device = SimulationContext.instance().physics_manager.get_device()
    self._initialization_deferred = False
    self._initialize_impl()
    if not self._initialization_deferred:
        self._is_initialized = True


def _defer_initialization(self) -> None:
    self._initialization_deferred = True


def _complete_deferred_initialization(self) -> None:
    if not self._initialization_deferred:
        raise RuntimeError("Asset initialization was not deferred.")
    self._initialization_deferred = False
    self._is_initialized = True
```

Reset `_initialization_deferred` during STOP invalidation. This hook is protected and does not alter public API.

- [ ] **Step 5: Register eager lifecycle bridges and keep the service lazy**

Construct `self._services = ServiceLocator()` before registering callbacks in `SimulationContext.__init__`. Register named callbacks at order 20:

```python
self._actuator_finalize_handle = self.physics_manager.register_callback(
    lambda payload: PhysicsManager.safe_callback_invoke(
        self._finalize_actuator_collection,
        payload,
        physics_manager=self.physics_manager,
    ),
    PhysicsEvent.PHYSICS_READY,
    order=20,
    name="actuator_collection_finalize",
)
self._actuator_stop_handle = self.physics_manager.register_callback(
    lambda payload: PhysicsManager.safe_callback_invoke(
        self._clear_actuator_collection_generation,
        payload,
        physics_manager=self.physics_manager,
    ),
    PhysicsEvent.STOP,
    order=20,
    name="actuator_collection_stop",
)
```

The common `PhysicsManager.register_callback()` already accepts `name`; also name the existing order-5 render callback so the ordering test can inspect it. `safe_callback_invoke()` is required so Kit/PhysX stores and re-raises finalization errors after its external event bus returns instead of swallowing them. Implement a private lazy accessor:

```python
def _get_actuator_collection(self) -> ActuatorCollection:
    from isaaclab.actuators import ActuatorCollection

    collection = self._services[ActuatorCollection]
    if collection is None:
        collection = ActuatorCollection(self)
        self._services[ActuatorCollection] = collection
    return collection
```

Use `TYPE_CHECKING` plus the local import shown above to avoid a `simulation_context` ↔ `actuator_collection` import cycle.

The finalization/STOP bridge checks the locator without creating an unused service. `clear_instance()` continues to close all services and also deregisters the two bridge handles through normal physics-manager closure.

- [ ] **Step 6: Implement pending registration and transactional publication**

Turn `ActuatorCollection` into the manager. Registration creates metadata and a pending `ArticulationView`, but no runtime arrays:

```python
@dataclass(frozen=True)
class _ArticulationRegistration:
    key: object
    cfgs: Mapping[str, ActuatorBaseCfg]
    control: ActuatorControl
    replication_cfg_id: int
    debug_validation: bool
    debug_value_resolution: bool


def finalize(self) -> None:
    candidate = _CollectionGeneration.build(self._registrations, self._sim_context)
    try:
        candidate.validate()
        candidate.bind_facade_storage()
    except Exception as error:
        candidate.close()
        self._invalidate_pending(error)
        raise
    self._publish(candidate)
```

At this task boundary the candidate binds only the storage and group/type records introduced in Tasks 2-3; Task 10 extends the same pre-publication transaction with execution-plan construction after joint-domain storage exists. Private builders receive an unguarded `_ArticulationBinding` owned by the candidate generation. They must never call the public pending `ArticulationView`, whose getters intentionally raise until publication.

Publication has four explicit phases: prepare and reversibly bind every private candidate record, atomically swap the validated candidate into the active-generation slot, call `control.bind_actuator_view()` with the now-live public facade for every registration, then call `control.complete_articulation_initialization()` for every registration. If any public bind or completion fails, unpublish the candidate, call `control.invalidate_actuator_view()` for every registration (which also clears `data.is_primed` and deferred asset readiness), close the candidate, keep `generation is None`, and re-raise with articulation/group/type context. No public facade is dereferenced before the active-generation swap, and the synchronous callback provides no external execution window between swap and rollback.

The two-articulation rollback regression must prove that when the first control completes and the second `complete_articulation_initialization()` raises, the candidate is unpublished, both controls are invalidated, the first asset returns to uninitialized/unprimed state, both facades reject access, all candidate allocations are released, and no active generation remains.

Registration after publication creates a pending facade, marks the active generation dirty, and cannot trigger an immediate rebuild. `clear_generation()` invalidates all views, plans, cached launches, and graph handles, clears registrations, and increments the generation counter used by the next publish. It retains only deprecated staged topology overrides until the matching asset registers during replay. `close()` clears those overrides too, is idempotent, and permanently rejects registration.

- [ ] **Step 7: Split control discovery from final binding**

Add these hooks to `ActuatorControl` and default no-op implementations where valid:

```python
def discover_native_actuators(self, cfgs: Mapping[str, ActuatorBaseCfg]) -> set[str]: ...
def prepare_actuator_binding(self, binding: _ArticulationBinding) -> None: ...
def bind_actuator_view(self, view: ActuatorCollection.ArticulationView) -> None: ...
def invalidate_actuator_view(self) -> None: ...
def complete_articulation_initialization(self) -> None: ...
def write_actuator_parameter(self, name: str, write: _ActuatorParameterWrite) -> None: ...
```

`_ActuatorParameterWrite` is a private immutable record containing the value array and either signed `env_ids`/`joint_ids` or full boolean `env_mask`/`joint_mask`, plus the group lookup or type CSR articulation-to-compact fanout and the backend-owner slots used by the canonical write. It lets every backend reuse identical overlap/selection semantics without host compaction.

Registration-time discovery and solver-property resolution remain synchronous because backend validation still runs during the order-10 phase. Command, runtime parameter, telemetry, and Newton-controller pointers prepare from the private order-20 candidate binding after it validates. Data objects receive the guarded public facade only after the active-generation swap; completion then marks the asset ready.

- [ ] **Step 8: Pass lifecycle and existing simulation-context tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py \
  source/isaaclab/test/sim/test_simulation_context.py -q
```

Expected: PASS. Confirm the lifecycle test explicitly demonstrates why lazy callback registration would miss the ready dispatch.

- [ ] **Step 9: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/sim/simulation_context.py \
  source/isaaclab/isaaclab/assets/asset_base.py \
  source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/isaaclab/actuators/actuator_control.py \
  source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py \
  source/isaaclab/test/sim/test_simulation_context.py
./isaaclab.sh -f
git commit -m "Add the global actuator lifecycle"
```

---

### Task 5: Publish Scoped Group and Exact-Type Facades

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_storage.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_base.py`
- Modify: `source/isaaclab/isaaclab/assets/articulation/base_articulation.py`
- Create: `source/isaaclab/test/actuators/test_actuator_collection_facade.py`

- [ ] **Step 1: Write failing facade tests**

Add:

```python
def test_facade_preserves_mapping_and_exact_group_classes() -> None:
    robot = make_finalized_robot(
        groups={"hip": ideal_pd_cfg(), "knee": dc_motor_cfg(), "ankle": ideal_pd_cfg()}
    )
    assert list(robot.actuators) == ["hip", "knee", "ankle"]
    assert isinstance(robot.actuators["hip"], IdealPDActuator)
    assert type(robot.actuators["knee"]) is DCMotor
    assert list(robot.actuators.keys()) == ["hip", "knee", "ankle"]
    assert len(robot.actuators.items()) == 3
    assert isinstance(robot.actuators, dict)
    assert type(robot.actuators.copy()) is dict


def test_facade_preserves_dict_copy_union_reverse_and_fromkeys_behavior() -> None:
    actuators = make_finalized_robot().actuators
    extra = sentinel_actuator()
    assert actuators.copy() == dict(actuators)
    assert actuators | {"extra": extra} == dict(actuators) | {"extra": extra}
    assert {"extra": extra} | actuators == {"extra": extra} | dict(actuators)
    assert list(reversed(actuators)) == list(reversed(dict(actuators)))
    assert type(actuators.fromkeys(("a", "b"), 1)) is dict


def test_by_type_uses_exact_classes_and_returns_compact_contiguous_views() -> None:
    robot = make_finalized_robot_with_subclasses()
    pd = robot.actuators.by_type[IdealPDActuator]
    dc = robot.actuators.by_type[DCMotor]
    assert pd.joint_names == ("hip", "ankle")
    assert pd.joint_indices.tolist() == [0, 2]
    assert pd.parameters["stiffness"].torch.is_contiguous()
    assert "saturation_effort" not in pd.parameter_names
    assert "saturation_effort" in dc.parameter_names
    with pytest.raises(KeyError):
        _ = robot.actuators.by_type[ActuatorBase]
    with pytest.raises(KeyError):
        _ = robot.actuators.by_type["ideal_pd"]


def test_type_view_exposes_group_slices_without_dense_projection() -> None:
    view = make_finalized_robot().actuators.by_type[IdealPDActuator]
    assert view.group_slices == {"hip": slice(0, 2), "ankle": slice(2, 3)}
    assert view.parameters["damping"].torch.shape == (view.num_instances, 3)
    assert not hasattr(view, "compact")
```

Cover group `joint_names`, group articulation `joint_ids`, type metadata, multiple articulation isolation, unknown group keys, unsupported custom groups, and generation checks on every facade operation.

- [ ] **Step 2: Run the facade tests and observe failure**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection_facade.py -q
```

Expected: FAIL because the current collection itself is the per-articulation mapping and has no exact-class facade.

- [ ] **Step 3: Implement the nested facade and exact-class view map**

Define `ArticulationView` and `TypeView` inside `ActuatorCollection`. `ArticulationView` subclasses `dict[str, ActuatorBase]` so `isinstance(robot.actuators, dict)` and develop's dict surface remain valid. Populate the actual base dictionary with `dict.__init__` during construction. Read methods validate the bound generation before delegating to the base dictionary; `by_type` is a read-only mapping keyed by `type(group)`, never `isinstance`.

Preserve snapshot operations exactly: `copy()`, `__or__`, and `__ror__` return ordinary `dict` objects without dirtying topology; `__reversed__` preserves insertion order; and `fromkeys()` returns an ordinary `dict` rather than trying to construct a manager-bound facade. Explicitly override every mutating dict entry point—`__setitem__`, `__delitem__`, `clear`, `pop`, `popitem`, `setdefault`, `update`, and `__ior__`—because C-level `dict` methods are not required to dispatch through the primitive overrides. They all use the one staged-mutation helper and once-only warning. Characterize repository uses of `.copy()`, unions, reverse iteration, and `isinstance(..., dict)` before implementation. Exact identity `type(robot.actuators) is dict` necessarily becomes false to expose the scoped facade; document this concrete-type change in migration/release notes and do not treat exact builtin identity as a supported semantic contract.

`TypeView.parameters[name]` returns a compact `[num_worlds, num_type_dofs]` `ProxyArray` whose Torch alias is contiguous. `TypeView.joint_indices` and `joint_names` contain one entry per compact slot in group/configuration order, including repeated articulation DOFs for overlapping groups; `group_slices` identifies each occurrence. Group direct attributes remain Torch tensors with regular stride `(num_type_dofs, 1)`. `group.parameters[name]` also returns a `ProxyArray` so callers that need Warp and Torch can use one lightweight object. Both parameter mappings are read-only mappings of names to mutable arrays; assigning a mapping entry raises, while mutating the returned array or using a setter remains supported. Do not add dense type projections or a `compact` argument.

Custom types without a declared managed schema remain exact-class group objects in the named mapping, do not appear in `by_type`, and execute per group. A `by_type[CustomActuator]` lookup therefore raises `KeyError` instead of exposing a misleading empty typed-storage view.

- [ ] **Step 4: Implement deprecated topology mutation and generation safety**

Route `__setitem__` and `__delitem__` through one helper:

```python
def _mutate_topology(self, operation: str, name: str, actuator: ActuatorBase | None) -> None:
    self._require_current_generation()
    self._warnings.warn_once(
        "mapping_mutation",
        "Mutating Articulation.actuators is deprecated; rebuild the simulation from ArticulationCfg instead.",
    )
    self._manager.stage_deprecated_mutation(self, operation, name, actuator)
```

Mutation updates the facade's logical ordered mapping immediately but marks execution dirty. For insertion/replacement, stage `actuator.cfg.copy()` rather than retaining the old runtime arrays; for deletion, stage the removed key. The manager keeps the resulting ordered config override keyed by the articulation object across STOP, applies it when that same asset registers during the next PHYSICS_READY event, and clears it after successful publication. `compute()`, `submit_commands()`, parameter setters, and command setters raise between mutation and rebuild. Every new generation gets fresh facades; retained facade/group/type objects check generation and raise. Document that raw retained Torch/Warp tensors cannot be guarded and must be reacquired after a rebuild.

Test every explicitly overridden dict mutator, including `|=`, once and verify it warns, stages copied configuration, and dirties execution without bypassing generation checks.

- [ ] **Step 5: Update the base articulation annotation**

Use `ActuatorCollection.ArticulationView` for `BaseArticulation.actuators`. Keep the manager itself exported, but do not expose it from `SimulationContext.services` in user documentation.

- [ ] **Step 6: Pass facade, storage, and lifecycle tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_facade.py \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py -q
```

- [ ] **Step 7: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/isaaclab/actuators/actuator_storage.py \
  source/isaaclab/isaaclab/actuators/actuator_base.py \
  source/isaaclab/isaaclab/assets/articulation/base_articulation.py \
  source/isaaclab/test/actuators/test_actuator_collection_facade.py
./isaaclab.sh -f
git commit -m "Expose scoped actuator facade views"
```

---

### Task 6: Implement Synchronization-Free Parameter Selectors

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_kernels.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_base.py`
- Modify: `source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection_facade.py`

- [ ] **Step 1: Add a parameterized selector contract test**

Cover group and type views on CPU and CUDA with these cases:

```python
@pytest.mark.parametrize("scope", ["group", "type"])
@pytest.mark.parametrize("device", available_devices())
def test_parameter_index_uses_cartesian_articulation_selectors(scope: str, device: str) -> None:
    view = make_parameter_view(scope=scope, device=device)
    before = view.parameters["stiffness"].torch.clone()
    view.set_parameter_index(
        "stiffness",
        torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], device=device),
        env_ids=torch.tensor([3, 1], device=device),
        joint_ids=torch.tensor([5, 0, 2], device=device),
    )
    assert_parameter_cells(view, before, {(3, 5): 10.0, (3, 2): 12.0, (1, 5): 20.0, (1, 2): 22.0})


def test_parameter_index_filters_value_columns_for_out_of_scope_joints() -> None:
    view = make_group_view(articulation_joint_ids=(1, 4))
    view.set_parameter_index(
        "damping",
        torch.tensor([10.0, 20.0, 30.0]),
        joint_ids=torch.tensor([4, 3, 1]),
    )
    torch.testing.assert_close(view.damping, expected_columns(joint_4=10.0, joint_1=30.0))


def test_parameter_index_duplicate_ids_use_last_value() -> None:
    view = make_group_view()
    view.set_parameter_index(
        "damping",
        torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
        env_ids=torch.tensor([0, 0]),
        joint_ids=torch.tensor([1, 1]),
    )
    assert view_value(view, env_id=0, joint_id=1) == 7.0


def test_type_index_fans_out_to_every_overlapping_compact_slot() -> None:
    view = make_overlapping_type_view(group_joint_ids=((0, 2), (2, 3), (2,)))
    view.set_parameter_index("stiffness", torch.tensor([13.0]), joint_ids=torch.tensor([2]))
    assert_compact_slots_equal(view, slots=(1, 2, 4), value=13.0)


def test_type_mask_values_remain_compact_across_overlapping_slots() -> None:
    view = make_overlapping_type_view(group_joint_ids=((0, 2), (2, 3), (2,)))
    values = torch.tensor([1.0, 11.0, 22.0, 3.0, 44.0])
    view.set_parameter_mask("stiffness", values, joint_mask=torch.tensor([False, False, True, False]))
    assert_compact_slots_equal(view, slots=(1, 2, 4), values=(11.0, 22.0, 44.0))


def test_overlapping_implicit_side_effect_uses_configuration_owner_slot() -> None:
    group_first, group_last, control = make_overlapping_implicit_groups()
    group_first.set_parameter_index("stiffness", 5.0, joint_ids=[2])
    assert control.solver_stiffness_for(2) == initial_last_group_value()
    group_last.set_parameter_index("stiffness", 17.0, joint_ids=[2])
    assert control.solver_stiffness_for(2) == 17.0
```

Also test scalar, one-dimensional per-joint, and two-dimensional world-by-joint values; `None`; negative and out-of-range IDs; empty index arrays; full boolean masks; scalar/compact 1D/compact 2D mask values; all-false masks; unknown fields; malformed rank/shape/dtype/device failures in normal mode; value-dependent failures in debug mode; and no value/backend-state change for empty selections.

- [ ] **Step 2: Observe the selector tests fail**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_facade.py \
  -k "parameter_index or parameter_mask or debug_validation or overlapping_implicit" -q
```

Expected: FAIL because group/type parameter setters and their filtering rules do not exist.

- [ ] **Step 3: Add the opt-in configuration flag**

Add to `ArticulationCfg`:

```python
actuator_debug_validation: bool = False
"""Whether actuator selector writes perform synchronized diagnostic validation. Defaults to False."""
```

Do not reuse `disable_shape_checks` and do not conflate this flag with `actuator_value_resolution_debug_print`.

- [ ] **Step 4: Implement common selector normalization without device-to-host reads**

Unknown parameter names raise before any launch. In both modes, synchronously validate only host-visible metadata: tensor/array rank, shape, dtype category, device, and whether the value shape is one of the documented broadcast forms. Unsupported rank/dtype/device and non-broadcastable values raise `TypeError` or `ValueError` before reaching a kernel; reading this metadata does not synchronize. In normal mode, keep selector contents on device. Group setters map through their one-to-one preallocated articulation-to-group lookup. Type setters traverse the immutable CSR `articulation_to_compact_offsets`/`articulation_to_compact_slots` fanout and write every compact slot matching the selected articulation DOF. Invalid world IDs and empty fanout ranges return without writing.

Index values support exactly:

```text
scalar                         -> broadcast over env_ids × joint_ids
[len(joint_ids)]               -> broadcast across env_ids
[len(env_ids), len(joint_ids)] -> one value per Cartesian pair
```

Filtering an invalid/out-of-scope joint leaves the value at its original selector column. To make duplicates deterministic without an atomic race, a thread writes only when no later selected environment ID and no later selected joint ID produce the same destination; this makes the last Cartesian occurrence win identically on CPU and CUDA.

For an explicit `joint_ids` selector, one supplied value fans out unchanged to all matching slots of an overlapping type view. For `joint_ids=None`, the selector addresses compact slots directly in stable group/configuration order, so accepted non-scalar values have `num_scope_dofs` columns and may set duplicate articulation DOFs differently. Last-occurrence rules apply per destination compact slot.

Mask inputs must be boolean full-articulation arrays `[num_worlds]` and `[num_joints]`. Values support scalar, compact `[num_scope_dofs]`, and compact `[num_worlds, num_scope_dofs]`. The kernel indexes compact values by the view's stable compact column, not by count of selected `True` entries; all duplicate slots whose articulation joint mask is true are eligible and may receive distinct compact values.

- [ ] **Step 5: Add synchronized debug validation only on the debug branch**

Debug mode adds value-dependent checks for bounds, ownership, and duplicates and raises `ValueError` with the field and facade scope. Instrument the normal tests by monkeypatching `torch.Tensor.cpu`, `torch.Tensor.tolist`, `torch.cuda.synchronize`, and Warp synchronization helpers to raise; well-formed normal selector calls must still succeed, while malformed metadata must fail without invoking a kernel.

Use `_WarpLaunchCache` only with manager-owned pointer-stable selector/value/output staging. Arbitrary user-owned arrays use ordinary `wp.launch`; do not key an unbounded launch cache by transient user pointers or retain user tensors through recorded launch arguments. Python sequences are converted to device arrays for compatibility and therefore are outside the allocation-free promise.

After the canonical write, call one facade-owned side-effect router. It uses the same device selectors/mapping as the canonical setter, so ignored entries remain ignored without host compaction. Finalization precomputes a configuration-order backend-owner compact slot for each `(parameter, articulation_joint)` that has a side effect. A group setter updates backend state only where that group's slot is the owner; a type setter routes the value now stored in the owner slot after fanout. Thus overlapping implicit groups cannot make canonical and solver-drive values disagree. The router is a no-op for ordinary explicit parameters, writes PhysX/OVPhysX solver drives for implicit `stiffness`/`damping`, and delegates native-controller effects through `ActuatorControl.write_actuator_parameter()`. A statically zero-length index selection returns before any launch. An arbitrary all-false device mask cannot be detected without synchronization, so it may execute canonical and side-effect masked no-op kernels; it must change neither canonical values nor backend state.

- [ ] **Step 6: Run selector tests on both devices**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_facade.py \
  -k "parameter_index or parameter_mask or debug_validation or unknown_parameter or overlapping_implicit" -q
```

Expected: PASS with identical asserted results on CPU and CUDA.

- [ ] **Step 7: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_kernels.py \
  source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/isaaclab/actuators/actuator_base.py \
  source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py \
  source/isaaclab/test/actuators/test_actuator_collection_facade.py
./isaaclab.sh -f
git commit -m "Add scoped actuator parameter setters"
```

---

### Task 7: Allocate Global Joint-Domain Commands and Telemetry

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_storage.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_kernels.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection_storage.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection.py`

- [ ] **Step 1: Write failing joint-store and command tests**

Add:

```python
def test_joint_domain_views_are_disjoint_live_and_pointer_stable() -> None:
    manager, first, second = make_finalized_two_articulation_manager()
    first_pointer = first.command.position.torch.data_ptr()
    second.command.position.torch.fill_(8.0)
    assert first.command.position.torch.data_ptr() == first_pointer
    assert torch.count_nonzero(first.command.position.torch) == 0
    assert torch.all(second.command.position.torch == 8.0)


def test_raw_and_processed_commands_have_separate_stable_storage() -> None:
    view = make_finalized_view()
    assert view.command.position.torch.data_ptr() != view.joint_command.position.torch.data_ptr()
    pointers = command_pointers(view)
    view.command.set_position_index(
        value=torch.tensor([[1.0, 2.0]], device=view.device),
        env_ids=torch.tensor([0], device=view.device),
        joint_ids=torch.tensor([1, 3], device=view.device),
    )
    assert command_pointers(view) == pointers


def test_effort_telemetry_is_persistent_and_in_articulation_order() -> None:
    view = make_finalized_reordered_view()
    pointers = (view.computed_effort.torch.data_ptr(), view.applied_effort.torch.data_ptr())
    publish_fake_type_outputs(view)
    torch.testing.assert_close(view.computed_effort.torch, expected_computed_in_joint_order())
    torch.testing.assert_close(view.applied_effort.torch, expected_applied_in_joint_order())
    assert pointers == (view.computed_effort.torch.data_ptr(), view.applied_effort.torch.data_ptr())
```

Keep and adapt the existing position/velocity/effort index and mask command tests, including signed `int64` selectors and the `full_data` compatibility argument.

- [ ] **Step 2: Run the tests and observe failure**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  -k "joint_domain or command or telemetry" -q
```

Expected: FAIL because command and telemetry buffers are still articulation-local fields on the old collection implementation.

- [ ] **Step 3: Add the joint-domain store and nested command facades**

Implement one articulation-major flat store for each field:

```python
class _JointDomainStore:
    raw_position: ProxyArray
    raw_velocity: ProxyArray
    raw_effort: ProxyArray
    processed_position: ProxyArray
    processed_velocity: ProxyArray
    processed_effort: ProxyArray
    computed_effort: ProxyArray
    applied_effort: ProxyArray

    def articulation_proxy(self, field: str, layout: _ArticulationLayout) -> ProxyArray: ...
```

Each articulation alias has shape `[num_worlds, num_joints]` and is contiguous. Keep raw and processed commands separate. Type staging is derived fixed storage and must never become authoritative command storage.

Nest `Command` and `JointCommand` beneath `ActuatorCollection.ArticulationView`. Preserve the existing index/mask setter names and `full_data` behavior. All writes validate generation and dirty state before using the cached command setter kernels.

- [ ] **Step 4: Allocate output storage in the typed and joint domains**

`computed_effort` and `applied_effort` exist in each supported exact-type store and in stable articulation-order publication buffers. The execution task will fill them; this task supplies the fixed pointers and a tested scatter helper. Missing/non-effort groups leave their joint-domain cells zero.

- [ ] **Step 5: Pass joint-domain and pre-existing command tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection.py -q
```

- [ ] **Step 6: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_storage.py \
  source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/isaaclab/actuators/actuator_kernels.py \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection.py
./isaaclab.sh -f
git commit -m "Add global actuator command storage"
```

---

### Task 8: Register and Bind All Articulation Backends

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_control.py`
- Modify: `source/isaaclab/isaaclab/assets/articulation/base_articulation.py`
- Modify: `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/benchmark/assets/runtime.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/benchmark/assets/runtime.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/benchmark/assets/runtime.py`
- Modify: `source/isaaclab/test/assets/_articulation_iface_test_utils.py`
- Modify: `source/isaaclab/test/assets/test_articulation_ordering_iface.py`
- Modify: `source/isaaclab_physx/test/assets/test_articulation.py`
- Modify: `source/isaaclab_ovphysx/test/assets/test_articulation.py`
- Modify: `source/isaaclab_newton/test/assets/test_articulation.py`

- [ ] **Step 1: Add failing shared-backend interface tests**

Extend the ordering-interface fake so the same tests cover each backend adapter:

```python
def test_actuator_registration_applies_solver_properties_before_global_finalize(articulation) -> None:
    articulation._process_actuators_cfg()
    assert articulation._actuator_control.solver_property_write_count > 0
    assert not articulation.actuators.is_ready


def test_backend_data_binds_only_after_transactional_publish(articulation, manager) -> None:
    pending = articulation.actuators
    assert articulation.data._actuator_view is None
    manager.finalize()
    assert articulation.data._actuator_view is pending
    assert articulation.data.joint_pos_target.torch.data_ptr() == pending.command.position.torch.data_ptr()


def test_backend_completion_marks_articulation_ready_after_publish(articulation, manager) -> None:
    assert not articulation.is_initialized
    assert not articulation.data.is_primed
    manager.finalize()
    assert articulation.is_initialized
    assert articulation.data.is_primed
```

Add backend integration checks for reset routing, user-order submission, signed selectors, and failed binding rollback. The PhysX, OVPhysX, and Newton test files should assert the same facade contract rather than backend-specific storage ownership.

- [ ] **Step 2: Run focused interface tests and observe failure**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py \
  -k "actuator or completion_marks" -q
```

Expected: FAIL because every backend still constructs and binds an articulation-local collection inside `_process_actuators_cfg()`.

- [ ] **Step 3: Resolve registration metadata at source-prototype granularity**

During `register_articulation`, resolve joint regexes and default USD/solver values once per used clone source row. Run numeric resolution once per `(source prototype, logical group)`, but create exactly one eventual runtime group object per logical config group. Use a private resolution path that runs the concrete actuator class's existing parsing rules while directing per-prototype numeric results into registration metadata instead of allocating per-world runtime tensors:

```python
@classmethod
def _resolve_managed_registration(
    cls,
    *,
    cfg: ActuatorBaseCfg,
    joint_names: list[str],
    joint_indices: slice | torch.Tensor,
    defaults_by_source: Sequence[ActuatorJointProperties],
) -> _ResolvedManagedGroup: ...
```

`_ResolvedManagedGroup` contains the copied config, metadata, structural signature, and per-source numeric field rows. After canonical storage exists, construct one exact-class group shell and bind it through `_bind_managed_runtime(_GroupBinding)`; that is the object exposed in the facade. Custom classes that do not implement managed resolution use their existing constructor once per logical group and become opaque eager groups. They do not share typed storage. Preserve the exact configured object class in both cases.

The registration record must contain: articulation/control identity, world/joint counts, configuration order, exact class, schema, articulation joint map, joint names, source-row numeric values, clone-row assignment, native ownership, structural signature, and backend binding metadata. Validate dtype, device, schema, joint map, clone assignment, and backend compatibility with errors naming the articulation, exact class, and group.

- [ ] **Step 4: Apply solver properties synchronously and once per property**

Refactor `ActuatorJointProperties`, `get_default_joint_properties`, and `write_resolved_joint_properties` so registration obtains source-prototype rows without slicing/copying a dense buffer once per logical group. PhysX/OVPhysX may read the source USD rows directly; native Newton may gather representative source rows on device. Resolve every group first, expand prototype values with bounded generic kernels, then perform one backend write per solver property. This must complete before `_validate_cfg()` runs. Do not call `.cpu()`, `.item()`, `.tolist()`, or a per-world query while forming source assignments.

Typed fields and solver fields follow the ownership table in Task 2. Implicit actuator stiffness/damping have both a typed canonical value and a solver-drive side effect; `effort_limit_sim`, `velocity_limit_sim`, armature, and friction remain solver-only.

Implement `write_actuator_parameter()` in all three controls. PhysX and OVPhysX route implicit gain writes to solver drives using the supplied device-side selection. PhysX-hosted and native Newton bind or route every meaningful native parameter. Explicit Lab-only fields require no backend call. The control must not allocate a compact selection or synchronize to the host.

- [ ] **Step 5: Replace local construction with pending registration in all backends**

Each `_process_actuators_cfg()` must:

```python
control = BackendArticulationActuatorControl(self)
manager = SimulationContext.instance()._get_actuator_collection()
self.actuators = manager.register_articulation(
    key=self,
    cfgs=self.cfg.actuators,
    control=control,
    replication_cfg_id=self._replication_cfg_id,
    debug_validation=self.cfg.actuator_debug_validation,
    debug_value_resolution=self.cfg.actuator_value_resolution_debug_print,
)
self._defer_initialization()
```

Move first `update(0.0)`, logging, `data.is_primed = True`, and `_complete_deferred_initialization()` into `complete_articulation_initialization()`, after the manager publishes and `bind_actuator_view()` succeeds. Backend validation may run in order 10 after synchronous solver resolution, but it must not mark the asset ready.

Remove PhysX/Newton `_joint_pos_target_sim`, `_joint_vel_target_sim`, and `_joint_effort_target_sim` allocations that duplicate finalized global joint storage. Remove equivalent ownership from OVPhysX. Route `reset`, `write_data_to_sim`, and command submission through `ArticulationView`.

- [ ] **Step 6: Bind data and runtime benchmark mocks to the nested view**

Change all three `bind_actuator_collection` methods to:

```python
def bind_actuator_collection(self, actuators: ActuatorCollection.ArticulationView) -> None:
    self._actuator_view = actuators
```

The bind method must not read a lazy compatibility projection. Update each runtime benchmark mock/control signature to accept an `ArticulationView`, and update `_MockActuatorCollection` to model nested command/joint-command access without allocating old dense fields.

- [ ] **Step 7: Verify all three backends**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering_iface.py -q
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -k actuator -q
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_articulation.py -k actuator -q
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_articulation.py \
  -k "actuator or gain or write_data_to_sim" -q
```

Expected: PASS with real backend bindings where their fixtures are available. Record explicit skips with their reason; do not report skipped integration as verified.

- [ ] **Step 8: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_control.py \
  source/isaaclab/isaaclab/assets/articulation/base_articulation.py \
  source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py \
  source/isaaclab_physx/isaaclab_physx/assets/articulation \
  source/isaaclab_newton/isaaclab_newton/assets/articulation \
  source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation \
  source/isaaclab_physx/isaaclab_physx/benchmark/assets/runtime.py \
  source/isaaclab_newton/isaaclab_newton/benchmark/assets/runtime.py \
  source/isaaclab_ovphysx/isaaclab_ovphysx/benchmark/assets/runtime.py \
  source/isaaclab/test/assets/_articulation_iface_test_utils.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py \
  source/isaaclab_physx/test/assets/test_articulation.py \
  source/isaaclab_ovphysx/test/assets/test_articulation.py \
  source/isaaclab_newton/test/assets/test_articulation.py
./isaaclab.sh -f
git commit -m "Bind articulation backends to global actuators"
```

---

### Task 9: Preserve Commands, Telemetry, and Legacy Data Views

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_storage.py`
- Modify: `source/isaaclab/isaaclab/assets/articulation/base_articulation.py`
- Modify: `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py`
- Modify: `source/isaaclab/isaaclab/envs/mdp/events.py`
- Modify: `source/isaaclab/isaaclab/envs/mdp/rewards.py`
- Modify: `source/isaaclab/isaaclab/envs/mdp/observations.py`
- Modify: `source/isaaclab/isaaclab/envs/mdp/terminations.py`
- Modify: `tools/actuator_parameters.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/mdp/observations.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/mdp/events.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/mdp/rewards.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection_compatibility.py`
- Create: `source/isaaclab/test/envs/test_actuator_randomization.py`
- Create: `source/isaaclab_tasks/test/core/test_actuator_api_migration.py`

- [ ] **Step 1: Add failing alias and lazy-projection tests**

Add:

```python
def test_develop_command_aliases_share_canonical_memory_and_warn_once() -> None:
    robot = make_finalized_robot()
    with pytest.warns(DeprecationWarning, match="actuators.command.position"):
        first = robot.data.joint_pos_target
    with warnings.catch_warnings(record=True) as caught:
        second = robot.data.joint_pos_target
    assert not caught
    assert first.data_ptr() == second.data_ptr() == robot.actuators.command.position.torch.data_ptr()


def test_torque_aliases_redirect_to_effort_storage() -> None:
    robot = make_finalized_robot()
    with pytest.warns(DeprecationWarning, match="computed_effort"):
        computed = robot.data.computed_torque
    with pytest.warns(DeprecationWarning, match="applied_effort"):
        applied = robot.data.applied_torque
    assert computed.data_ptr() == robot.actuators.computed_effort.torch.data_ptr()
    assert applied.data_ptr() == robot.actuators.applied_effort.torch.data_ptr()


def test_legacy_projections_are_lazy_stable_and_use_legacy_fill() -> None:
    robot = make_robot_with_partial_velocity_and_no_gearing()
    assert robot.actuators._compatibility_allocations == {}
    velocity = robot.data.soft_joint_vel_limits
    gear = robot.data.gear_ratio
    assert robot.actuators._compatibility_allocations.keys() == {"soft_joint_vel_limits", "gear_ratio"}
    assert velocity.data_ptr() == robot.data.soft_joint_vel_limits.data_ptr()
    assert gear.data_ptr() == robot.data.gear_ratio.data_ptr()
    assert_missing_joint_values(velocity, fill=0.0)
    assert_missing_joint_values(gear, fill=1.0)
```

Also test held projection refresh after a parameter setter and after `compute()`, no allocation when never accessed, once-only warnings, live target aliasing, and unchanged/no-warning `data.joint_stiffness`/`joint_damping`.

Add `test_solver_property_writer_refreshes_only_activated_group_compatibility_sidecars`: access one group's `armature`, leave another solver field untouched, call the existing joint-armature writer, and assert the held armature tensor refreshes while no sidecar is allocated for the untouched field.

- [ ] **Step 2: Add failing writer and capability-aware randomizer tests**

```python
def test_deprecated_actuator_writer_delegates_to_capable_type_views() -> None:
    robot = make_robot_with_pd_dc_and_neural_groups()
    with pytest.warns(DeprecationWarning, match="set_parameter_index"):
        robot.write_actuator_stiffness_to_sim(
            stiffness=torch.tensor([[4.0, 5.0, 6.0]], device=robot.device),
            env_ids=torch.tensor([0], device=robot.device),
            joint_ids=torch.tensor([0, 1, 2], device=robot.device),
        )
    assert_pd_and_dc_stiffness(robot, (4.0, 5.0))
    assert robot.actuators["neural"]._deprecated_sidecars == {}


def test_randomize_actuator_gains_skips_types_without_gain_capability() -> None:
    robot = make_robot_with_pd_and_neural_groups()
    randomize_actuator_gains(make_env(robot), env_ids=torch.tensor([0]), asset_cfg=make_asset_cfg())
    assert robot.actuators["neural"]._deprecated_sidecars == {}
    assert "stiffness" in robot.actuators["pd"].parameter_names
```

Solver `write_joint_stiffness_to_sim_index()` and damping variants remain unchanged and emit no deprecation.

In `source/isaaclab_tasks/test/core/test_actuator_api_migration.py`, add failing call-site regressions for Anymal-C, trocar, Spot, and Cartpole using fakes whose deprecated public properties raise. Assert that torque consumers request the replacement effort field and dense soft-limit consumers request the private compatibility helper without emitting `DeprecationWarning`. Add a source-audit assertion that no first-party runtime source outside explicit compatibility/migration tests reads `applied_torque`, `computed_torque`, or public `soft_joint_vel_limits`.

- [ ] **Step 3: Run the compatibility tests and observe failure**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_compatibility.py \
  source/isaaclab/test/envs/test_actuator_randomization.py \
  source/isaaclab_tasks/test/core/test_actuator_api_migration.py -q
```

Expected: FAIL because data binding currently eagerly owns dense fields and gain randomization assumes inherited stiffness/damping.

- [ ] **Step 4: Implement a shared once-only warning registry and live aliases**

Use one per-facade warning registry for target setters/properties, torque terminology, lazy projections, topology mutation, neural sidecars, and actuator gain writers. Keep warning messages actionable and name the replacement public path.

Map:

```text
set_joint_position_target* -> actuators.command.set_position_*
set_joint_velocity_target* -> actuators.command.set_velocity_*
set_joint_effort_target*   -> actuators.command.set_effort_*
data.joint_*_target        -> actuators.command.* live aliases
data.computed_torque       -> actuators.computed_effort live alias
data.applied_torque        -> actuators.applied_effort live alias
```

The three backends delegate their existing actuator stiffness/damping writers to every exact-type view declaring that field, using articulation-coordinate selectors. Warn once in favor of group/type `set_parameter_index()`. Do not redirect or deprecate solver joint-property writers.

Remove the unshipped PR-only dense collection properties `actuator_stiffness`, `actuator_damping`, collection-level `soft_joint_vel_limits`, collection-level `gear_ratio`, `computed_torque`, and `applied_torque`. They never appeared in a release and are replaced by typed parameters, lazy `ArticulationData` compatibility projections, `computed_effort`, and `applied_effort`; do not create a second deprecation layer for them.

- [ ] **Step 5: Implement lazy dense compatibility projections**

On first access, allocate one stable articulation-order array, fill missing `soft_joint_vel_limits` entries with `0.0` or missing `gear_ratio` entries with `1.0`, then scatter capable typed values according to configuration-order ownership. Later property access refreshes before returning. Parameter setters refresh an already-active projection, and the post-compute publication boundary refreshes held projections. If a projection was never accessed, neither allocation nor refresh launch exists.

Neural gain sidecars are group-local Torch arrays, warn once, and are never registered with storage, backend side effects, randomization, execution, or projections.

- [ ] **Step 6: Make gain randomization capability-based and migrate all first-party consumers**

In `randomize_actuator_gains`, inspect `parameter_names` and call group/type setters. Never use inherited `isinstance` behavior to infer gain capability. Migrate every core and `isaaclab_tasks` use of deprecated computed/applied torque aliases in rewards, observations, terminations, and `tools/actuator_parameters.py` to effort terminology. The plotting tool may retain serialized result keys for backward-compatible plot input, but its live articulation reads use `computed_effort`/`applied_effort`.

Cartpole and Spot currently require an articulation-dense soft-limit tensor. Route those two first-party algorithms through one private, warning-free `ArticulationData._get_actuator_compatibility_projection("soft_joint_vel_limits")` helper; it activates the same lazy projection as the deprecated public property but does not create a new public non-compact API. Public `data.soft_joint_vel_limits` continues to warn once. Make the failing task-package call-site and source-audit regressions from Step 2 pass.

- [ ] **Step 7: Pass compatibility and backend side-effect tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection_compatibility.py \
  source/isaaclab/test/envs/test_actuator_randomization.py \
  source/isaaclab_tasks/test/core/test_actuator_api_migration.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py -q
./isaaclab.sh -p -m pytest \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py \
  -k "gain or implicit_storage or implicit_subset" -q
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_articulation.py \
  -k "gain or stiffness or damping" -q
```

- [ ] **Step 8: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators \
  source/isaaclab/isaaclab/assets/articulation \
  source/isaaclab/isaaclab/envs/mdp/events.py \
  source/isaaclab/isaaclab/envs/mdp/rewards.py \
  source/isaaclab/isaaclab/envs/mdp/observations.py \
  source/isaaclab/isaaclab/envs/mdp/terminations.py \
  tools/actuator_parameters.py \
  source/isaaclab/test/actuators/test_actuator_collection_compatibility.py \
  source/isaaclab/test/envs/test_actuator_randomization.py \
  source/isaaclab_physx/isaaclab_physx/assets/articulation \
  source/isaaclab_newton/isaaclab_newton/assets/articulation \
  source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation \
  source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py \
  source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env.py \
  source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/mdp/observations.py \
  source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/mdp/events.py \
  source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/mdp/rewards.py \
  source/isaaclab_tasks/test/core/test_actuator_api_migration.py
./isaaclab.sh -f
git commit -m "Preserve actuator API compatibility"
```

---

### Task 10: Build Articulation-Owned Typed Execution Plans

**Files:**

- Create: `source/isaaclab/isaaclab/actuators/actuator_execution.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_kernels.py`
- Create: `source/isaaclab/test/actuators/test_actuator_execution.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection.py`

- [ ] **Step 1: Write failing plan-shape and aggregation tests**

Add:

```python
def test_plan_builds_one_range_per_exact_stateless_type() -> None:
    plan = make_plan(groups=three_pd_three_dc_two_implicit_groups())
    assert [execution_range.actuator_type for execution_range in plan.stateless_ranges] == [
        IdealPDActuator,
        DCMotor,
        ImplicitActuator,
    ]
    assert tuple(len(execution_range.group_names) for execution_range in plan.stateless_ranges) == (3, 3, 2)


def test_plan_does_not_split_ranges_for_different_numeric_parameters() -> None:
    plan = make_plan(groups=twelve_ideal_pd_groups_with_unique_gains())
    assert len(plan.stateless_ranges) == 1
    assert plan.stateless_ranges[0].group_names == tuple(f"group_{index}" for index in range(12))


@pytest.mark.parametrize("actuator_type", [ImplicitActuator, IdealPDActuator, DCMotor])
def test_aggregated_range_matches_independent_groups_exactly(actuator_type) -> None:
    independent, aggregated = make_literal_execution_pair(actuator_type)
    independent.compute(dt=0.005)
    aggregated.compute(dt=0.005)
    torch.testing.assert_close(aggregated.joint_command, independent.joint_command, rtol=0.0, atol=0.0)
    torch.testing.assert_close(aggregated.computed_effort, independent.computed_effort, rtol=0.0, atol=0.0)
    torch.testing.assert_close(aggregated.applied_effort, independent.applied_effort, rtol=0.0, atol=0.0)
```

The independent expected path must construct and run separate ordinary actuators with literal deterministic commands, states, and parameters. It may not call the new range helper on both sides.

- [ ] **Step 2: Write failing overlap, fallback, and stale-plan tests**

```python
def test_overlapping_stateless_groups_preserve_last_writer_order() -> None:
    view = make_overlapping_plan(group_order=("first", "second", "third"))
    view.compute()
    assert_joint_outputs_equal_literal_last_group(view, joint_id=2, owner="third")


def test_mixed_overlap_uses_last_writer_per_present_output_field() -> None:
    view = make_implicit_then_dc_overlap()
    view.compute()
    assert_position_and_velocity_owned_by(view, "implicit")
    assert_effort_and_telemetry_owned_by(view, "dc")


def test_stateful_neural_and_custom_groups_remain_ordered_eager_segments() -> None:
    plan = make_mixed_plan()
    assert [segment.group_name for segment in plan.eager_segments] == ["delayed", "neural", "custom"]
    assert all(len(segment.group_names) == 1 for segment in plan.eager_segments)


def test_plan_rejects_stale_or_dirty_generation() -> None:
    manager, view = make_finalized_manager()
    plan = view._execution_plan
    manager.stage_deprecated_mutation(view, "delete", "hip", None)
    with pytest.raises(RuntimeError, match="dirty"):
        plan.compute()
    manager.clear_generation()
    with pytest.raises(RuntimeError, match="stale"):
        plan.compute()
```

- [ ] **Step 3: Run the execution tests and observe failure**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_execution.py -q
```

Expected: FAIL because the current `_ExecutionBatch` builder splits unsafe overlap and still owns aggregation inside the local collection.

- [ ] **Step 4: Implement immutable ranges and plans**

Use these private records:

```python
@dataclass(frozen=True)
class _ExecutionRange:
    actuator_type: type[ActuatorBase]
    group_names: tuple[str, ...]
    group_slices: tuple[slice, ...]
    joint_indices: wp.array(dtype=wp.int32)
    owner_slots_by_field: Mapping[str, wp.array(dtype=wp.int32)]
    graphable: bool


class _ArticulationExecutionPlan:
    @classmethod
    def build(
        cls,
        *,
        binding: _ArticulationBinding,
        control: ActuatorControl,
        generation: int,
    ) -> _ArticulationExecutionPlan: ...

    def compute(self, dt: float = 0.0) -> None: ...
    def reset(self, env_ids: Sequence[int] | slice) -> None: ...
    def invalidate(self) -> None: ...
```

Build one range for each exact supported stateless class present in an articulation, excluding native-Newton-owned groups. Numeric values never enter a range signature. Allocate one typed slot for every logical group DOF, including overlap. Gather raw commands and current joint position/velocity once into fixed type-order staging. Execute every Lab-owned stateless range once. Precompute a separate owner-slot table for position command, velocity command, effort command, computed effort, and applied effort. A group is eligible for a field only when its ordinary `compute()` result provides that field; configuration-order last-writer selection then happens independently per field during one fused scatter. This preserves cases such as an implicit group followed by a DC group, where position/velocity remain implicit-owned while effort/telemetry become DC-owned.

Create one ordered eager segment per neural, delayed, remotized, stateful, or opaque custom group. These segments preserve their existing constructor, `compute`, `reset`, state, and output behavior. Their outputs land in preallocated segment staging and participate in the same field-specific configuration-order owner tables. Do not merge neural models.

Extend the Task 4 finalization transaction here: build each plan from the private candidate `_ArticulationBinding` before publication, install it only after all plans validate, and roll every plan/binding back on failure. Never dereference the guarded public facade during candidate construction. `ArticulationView.compute()` validates generation/dirty state and delegates to the published plan. `write_data_to_sim()` remains the backend lifecycle entry point and calls Lab-owned computation then submission exactly once; native Newton groups contribute staging/ranges to `NewtonManager` but are never physically stepped by this plan.

- [ ] **Step 5: Pass execution equality and existing collection tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_actuator_collection.py -q
```

- [ ] **Step 6: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_execution.py \
  source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/isaaclab/actuators/actuator_kernels.py \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_actuator_collection.py
./isaaclab.sh -f
git commit -m "Build typed actuator execution plans"
```

---

### Task 11: Bind Newton Controllers Directly to Canonical Storage

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_control.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/actuators/__init__.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/actuators/adapter.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/actuators/kernels.py`
- Modify: `source/isaaclab_physx/test/assets/test_newton_actuators_physx.py`
- Modify: `source/isaaclab_newton/test/assets/test_newton_actuators_newton.py`
- Modify: `source/isaaclab_newton/test/assets/test_articulation.py`

- [ ] **Step 1: Write failing canonical-pointer tests for PhysX-hosted Newton**

Add:

```python
def test_newton_controller_parameters_alias_canonical_type_storage() -> None:
    robot, adapter = make_physx_newton_robot_with_multiple_pd_groups()
    typed = robot.actuators.by_type[IdealPDActuator]
    assert adapter.controllers[0].kp.ptr == typed.parameters["stiffness"].warp.ptr
    assert adapter.controllers[0].kd.ptr == typed.parameters["damping"].warp.ptr


def test_newton_parameter_write_requires_no_copy_synchronization(monkeypatch) -> None:
    robot, adapter = make_physx_newton_robot()
    monkeypatch.setattr(wp, "copy", fail_if_called)
    robot.actuators.by_type[IdealPDActuator].set_parameter_index("stiffness", 19.0)
    assert read_controller_kp(adapter) == 19.0
```

Also assert that adapter computed/applied effort arrays alias canonical typed output storage.

- [ ] **Step 2: Write failing native-Newton range tests**

```python
def test_native_newton_controllers_alias_global_collection_storage() -> None:
    robot, adapter = make_native_newton_robot()
    assert_controller_parameter_pointers_match(robot, adapter)


def test_two_articulations_bind_disjoint_canonical_newton_ranges() -> None:
    first, second, adapter = make_two_native_newton_articulations()
    assert first.actuators.by_type[IdealPDActuator].parameters["stiffness"].warp.ptr != (
        second.actuators.by_type[IdealPDActuator].parameters["stiffness"].warp.ptr
    )
    assert_adapter_ranges_match_facades(adapter, first, second)


def test_native_newton_group_is_physically_stepped_only_by_manager(monkeypatch) -> None:
    robot, manager = make_native_newton_robot()
    lab_compute = spy_on_lab_owned_range_compute(monkeypatch, robot)
    native_step = spy_on_newton_manager_actuator_step(monkeypatch, manager)
    robot.write_data_to_sim()
    manager.step_actuators()
    assert lab_compute.call_count == 0
    assert native_step.call_count == 1


def test_exported_defaults_helper_remains_compatible_and_warns_once() -> None:
    args = make_defaults_helper_arguments()
    with pytest.warns(DeprecationWarning, match="canonical actuator storage"):
        first = build_newton_actuator_defaults(**args)
    with warnings.catch_warnings(record=True) as caught:
        second = build_newton_actuator_defaults(**args)
    assert not caught
    assert_defaults_results_equal(first, second)
```

- [ ] **Step 3: Run the Newton tests and observe failure**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py \
  -k "canonical_type_storage or no_copy_synchronization" -q
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py \
  -k "global_collection_storage or disjoint_canonical or physically_stepped or defaults_helper" -q
```

Expected: FAIL because `bind_articulation()` currently returns copied gain snapshots and adapter buffers own duplicate outputs.

- [ ] **Step 4: Make Newton signatures structural and aggregate stateless controllers**

Rewrite `_actuator_signature()` so numeric per-DOF parameters never split an actuator. Retain only exact controller/clamping/delay/state structure that changes kernel behavior or state shape. Aggregate compatible stateless Newton controllers into the same exact-type ordering as the canonical compact block; keep neural/stateful controllers separate when their structural signatures differ.

Validate controller index order against the type layout. A stateless controller's flattened parameter order must match canonical world-major/type-DOF-major storage before pointer binding; raise with articulation/type/groups context rather than silently copying or reordering.

- [ ] **Step 5: Bind controller arrays and outputs instead of snapshots**

Change the binding contract to:

```python
def bind_native_actuators(self, binding: _ArticulationBinding) -> None: ...


def bind_articulation(
    self,
    binding: _ArticulationBinding,
    *,
    dof_offset: int,
    joint_user_to_backend_indices: Sequence[int] | None = None,
) -> NewtonActuatorAdapter.ArticulationBinding: ...
```

Pass private candidate bindings—not guarded pending facades—to native and hosted-Newton builders. Remove stiffness/damping snapshots from `ArticulationBinding`, stop all internal use of `build_newton_actuator_defaults()`, and delete duplicated `newton_default_stiffness`/`newton_default_damping` manager storage. Because `build_newton_actuator_defaults()` is already exported, retain it as a deprecated on-demand compatibility utility with its existing signature/results and a once-only migration warning; do not remove it from `isaaclab_newton.actuators.__all__` in this release. Make the Step 2 warning-and-result regression pass. Bind controller `kp`, `kd`, meaningful limits, computed effort, and applied effort directly to final canonical arrays/ranges. Parameter setters therefore need no gain-copy synchronization.

Native Newton remains physically stepped once by `NewtonManager`. The articulation plan owns its scoped command staging and telemetry and registers canonical ranges with that manager; its Lab-owned stateless/eager range lists explicitly exclude native-owned groups, and it must not call a competing physical actuator step. Preserve the existing alternating graph behavior for stateful Newton actuators.

- [ ] **Step 6: Run full Newton actuator tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_newton_actuators_physx.py -q
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_newton_actuators_newton.py -q
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_articulation.py \
  -k "gain or actuator or rebind" -q
```

Expected: PASS, including existing reversed-order, two-articulation, gain-randomization, reset-isolation, ping-pong graph, capture-fallback, and outer-capture-rejection tests.

- [ ] **Step 7: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_control.py \
  source/isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py \
  source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py \
  source/isaaclab_newton/isaaclab_newton/actuators/__init__.py \
  source/isaaclab_newton/isaaclab_newton/actuators/adapter.py \
  source/isaaclab_newton/isaaclab_newton/actuators/kernels.py \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py \
  source/isaaclab_newton/test/assets/test_articulation.py
./isaaclab.sh -f
git commit -m "Bind Newton actuators to typed storage"
```

---

### Task 12: Make Stateless Staging and Compute Allocation-Free

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_base.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_pd.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_execution.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_kernels.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_execution.py`
- Modify: `source/isaaclab/test/actuators/test_ideal_pd_actuator.py`
- Modify: `source/isaaclab/test/actuators/test_dc_motor.py`

- [ ] **Step 1: Write failing pointer-stability tests**

```python
def test_ideal_pd_execution_preserves_output_pointers() -> None:
    actuator, action, state = make_ideal_pd_execution_fixture()
    pointers = execution_pointers(actuator, action)
    for command_scale in (1.0, 2.0, -3.0):
        fill_action(action, command_scale)
        actuator._compute_execution(action, state.position, state.velocity)
        assert execution_pointers(actuator, action) == pointers


def test_dc_motor_execution_preserves_output_and_scratch_pointers() -> None:
    actuator, action, state = make_dc_execution_fixture()
    pointers = execution_and_scratch_pointers(actuator, action)
    for command_scale in (1.0, 2.0, -3.0):
        fill_action(action, command_scale)
        actuator._compute_execution(action, state.position, state.velocity)
        assert execution_and_scratch_pointers(actuator, action) == pointers


def test_parameter_updates_preserve_all_plan_pointers() -> None:
    view = make_finalized_stateless_view()
    pointers = all_plan_pointers(view._execution_plan)
    view.by_type[DCMotor].set_parameter_index("saturation_effort", 33.0)
    view.compute()
    assert all_plan_pointers(view._execution_plan) == pointers
```

- [ ] **Step 2: Run pointer tests and observe failure**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py \
  source/isaaclab/test/actuators/test_dc_motor.py \
  -k "preserves_output or preserves_all_plan_pointers" -q
```

Expected: FAIL because built-in Torch arithmetic replaces tensors and DCMotor clipping creates temporaries.

- [ ] **Step 3: Add private in-place execution hooks without changing public `compute()`**

Implement:

```python
def _compute_execution(
    self,
    control_action: ArticulationActions,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> ArticulationActions: ...


def _clip_effort_into(self, effort: torch.Tensor, out: torch.Tensor) -> None: ...
```

`IdealPDActuator` uses finalization-owned error/output scratch with `out=` and in-place operations. `DCMotor` additionally owns fixed clipped-velocity, torque-speed upper/lower, min-effort, and max-effort scratch. Public `compute()` keeps its develop signature and delegates through a compatibility path. Managed range execution calls `_compute_execution()` directly.

Preallocate command, joint-state, output, DCMotor scratch, routing, backend staging, and compatible projection staging during collection finalization. Never replace an `ArticulationActions` tensor field after publication.

- [ ] **Step 4: Record every pointer-stable Warp launch**

Extend `_WarpLaunchCache` only as needed to expose recorded `wp.Launch` lifetime/clear operations. Create recorded launches after all buffers are final and warm. Use cached launches for prototype expansion, command setters backed by manager-owned stable staging, gather, implicit compute, scatter/telemetry, active projection refresh, and backend staging. Pointer-changing user arrays always use the ordinary launch path; never retain transient user tensors in an unbounded pointer-keyed cache.

Do not add `wp.printf`. Use a standalone reproduction script if a kernel needs diagnosis, then remove it.

- [ ] **Step 5: Pass pointer and model-equality tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_implicit_actuator.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py \
  source/isaaclab/test/actuators/test_dc_motor.py \
  source/isaaclab/test/envs/test_actuator_randomization.py -q
```

- [ ] **Step 6: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_base.py \
  source/isaaclab/isaaclab/actuators/actuator_pd.py \
  source/isaaclab/isaaclab/actuators/actuator_execution.py \
  source/isaaclab/isaaclab/actuators/actuator_kernels.py \
  source/isaaclab/isaaclab/utils/warp/launch_cache.py \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py \
  source/isaaclab/test/actuators/test_dc_motor.py
./isaaclab.sh -f
git commit -m "Make stateless actuator execution stable"
```

---

### Task 13: Capture Complete Graphable Actuator Sequences

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/actuator_execution.py`
- Modify: `source/isaaclab/isaaclab/utils/warp/launch_cache.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_execution.py`
- Create: `source/isaaclab/test/actuators/test_actuator_execution_cuda.py`

- [ ] **Step 1: Write the real Torch/Warp stream interoperability test**

```python
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_warp_torch_interop_capture_uses_changed_pointer_stable_inputs() -> None:
    plan = make_cuda_plan(actuator_types=(ImplicitActuator, IdealPDActuator, DCMotor))
    plan.warmup_and_capture()
    first_pointers = all_plan_pointers(plan)
    fill_canonical_inputs(plan, command=1.0, position=0.25, velocity=-0.5)
    plan.compute()
    first = plan.articulation.applied_effort.torch.clone()
    fill_canonical_inputs(plan, command=3.0, position=-0.75, velocity=0.5)
    plan.compute()
    second = plan.articulation.applied_effort.torch.clone()
    assert not torch.equal(first, second)
    assert all_plan_pointers(plan) == first_pointers
    assert_matches_eager_literal(second, command=3.0, position=-0.75, velocity=0.5)
```

This must run real Torch and Warp operations on the same CUDA stream. Mock call counts are insufficient.

- [ ] **Step 2: Add failing full-graph, mixed-plan, and fallback tests**

```python
def test_fully_graphable_plan_replays_one_complete_graph(monkeypatch) -> None:
    plan = make_cuda_plan(actuator_types=(ImplicitActuator, IdealPDActuator, DCMotor))
    plan.warmup_and_capture()
    assert plan._full_graph is not None
    assert plan._prefix_graph is None
    replayed = spy_on_capture_launch(monkeypatch)
    eager_scatter = spy_on_eager_scatter(monkeypatch, plan)
    plan.compute()
    assert replayed.graphs == [plan._full_graph]
    assert eager_scatter.call_count == 0


def test_mixed_plan_captures_prefix_then_runs_eager_and_cached_scatter(monkeypatch) -> None:
    plan = make_cuda_plan(actuator_types=(IdealPDActuator, ActuatorNetMLP))
    plan.warmup_and_capture()
    assert plan._full_graph is None
    assert plan._prefix_graph is not None
    replayed = spy_on_capture_launch(monkeypatch)
    eager_groups = spy_on_eager_groups(monkeypatch, plan)
    cached_scatter = spy_on_cached_scatter(monkeypatch, plan)
    plan.compute()
    assert replayed.graphs == [plan._prefix_graph]
    assert eager_groups.call_count == 1
    assert cached_scatter.call_count == 1


def test_capture_failure_falls_back_for_the_generation(monkeypatch) -> None:
    plan = make_cuda_plan(actuator_types=(DCMotor,))
    capture_attempts = fail_and_count_capture_begin(monkeypatch)
    eager_steps = spy_on_cached_eager_step(monkeypatch, plan)
    with pytest.warns(RuntimeWarning, match="cached eager execution"):
        plan.warmup_and_capture()
    plan.compute()
    plan.compute()
    assert plan._graph_capture_failed
    assert capture_attempts.call_count == 1
    assert eager_steps.call_count == 2


def test_projection_activated_after_capture_refreshes_outside_graph() -> None:
    plan = make_cuda_plan(actuator_types=(DCMotor,))
    plan.warmup_and_capture()
    assert plan._post_graph_projection_launches == ()
    held = plan.articulation.data.gear_ratio
    change_canonical_gear_ratio(plan, 7.0)
    plan.compute()
    assert torch.equal(held, torch.full_like(held, 7.0))
    assert len(plan._post_graph_projection_launches) == 1
    assert plan._full_graph is not None
```

Also add `test_graph_replay_matches_eager_all_stateless_types_exactly`, `test_plan_invalidation_releases_graphs_and_recorded_launches`, and a CPU test proving identical routing/equality without graph objects.

- [ ] **Step 3: Run the CUDA tests and observe failure**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_execution_cuda.py -q
```

Expected: FAIL because there is no articulation-level full/prefix capture state.

- [ ] **Step 4: Implement warm-up, capture, replay, and teardown**

Use this private state:

```python
class _ArticulationExecutionPlan:
    _full_graph: wp.Graph | None
    _prefix_graph: wp.Graph | None
    _graph_capture_failed: bool

    def warmup_and_capture(self) -> None: ...
    def clear_graph(self) -> None: ...
```

On CUDA, warm kernels/modules only after every execution pointer is final. If all segments are graphable, capture gather, every stateless type computation, scatter/telemetry, compatibility refreshes already activated at capture time, and backend staging in one graph. `compute()` replays that graph once when there are no later-activated compatibility projections.

A lazy compatibility projection first accessed after capture allocates its stable pointer and registers one cached post-graph refresh launch. It does not invalidate or recapture the command graph, because reading a deprecated data view must not trigger capture. Subsequent `compute()` calls replay the existing graph and then execute those cached refresh launches; this is reported as a full actuator graph plus compatibility epilogue, never as a one-replay complete sequence. Generation teardown clears both graph and epilogue launches.

For a mixed plan, capture only gather plus independent graphable stateless computations. Then run neural/stateful/custom segments eagerly and use one cached final scatter/telemetry/backend-staging launch sequence. Never label this a full capture. On CPU, run the identical storage/routing plan eagerly.

Capture failure emits one warning, releases partial graph state, marks the generation, and permanently uses recorded eager launches until rebuild. Generation invalidation calls `clear_graph()` and clears recorded launches. Stateful graphable Newton code may retain its existing alternating ping-pong graphs.

- [ ] **Step 5: Pass eager and graph equality tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_actuator_execution_cuda.py -q
```

Expected: PASS with zero-tolerance equality for every supported stateless actuator. Preserve develop's arithmetic order inside fused kernels; if a class cannot be fused without changing it, keep that class/range on the cached eager path. Do not relax this contract without explicit maintainer approval.

- [ ] **Step 6: Format and commit**

```bash
./isaaclab.sh -f
git add source/isaaclab/isaaclab/actuators/actuator_execution.py \
  source/isaaclab/isaaclab/utils/warp/launch_cache.py \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_actuator_execution_cuda.py
./isaaclab.sh -f
git commit -m "Capture graphable actuator execution"
```

---

### Task 14: Close Cross-Cutting Correctness and Error-Path Coverage

**Files:**

- Modify: `source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection_storage.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection_facade.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_collection_compatibility.py`
- Modify: `source/isaaclab/test/actuators/test_actuator_execution.py`
- Modify: `source/isaaclab/test/envs/test_actuator_randomization.py`
- Modify: `source/isaaclab/test/assets/test_articulation_iface.py`
- Modify: `source/isaaclab_tasks/test/core/test_actuator_api_migration.py`
- Modify: `source/isaaclab_contrib/test/assets/test_multirotor.py`

- [ ] **Step 1: Audit the focused test inventory against the approved specification**

The following named behaviors must each have a direct assertion by the end of this task. Add a focused failing test for any missing behavior before fixing it:

```text
homogeneous and four-source heterogeneous clone expansion
non-contiguous clone environment IDs
two articulations with disjoint type and joint ranges
exact concrete group type and exact-class by_type lookup
dict copy/union/reverse/mutator transition compatibility
strided group alias and contiguous type alias pointer equality
whole-attribute copy and in-place mutation
every scalar/vector/matrix index and mask broadcast form
one-to-many type selector fanout and overlapping backend-owner side effects
normal invalid-selector ignore with no synchronization
debug invalid-selector and duplicate-selector raises
implicit parameter backend side effects
neural sidecar warning and execution inertness
lazy projection shape, fill, pointer, warning, and refresh
all three stateless aggregate equality cases
configuration-order overlap ownership
stateful, neural, delayed, remotized, and custom fallback
Newton canonical pointer binding with no copy synchronization
eager versus graph exact equality
failed finalize rollback, including second-control completion failure after first completion
STOP/replay and close/second-context cleanup
deprecated mutation dirty rejection and generation invalidation
capability-based actuator gain randomization
first-party task callers avoid deprecated actuator data views
```

Use the tests created in earlier tasks rather than duplicating a case under another name.

- [ ] **Step 2: Add explicit negative/error-context tests**

```python
def test_finalization_error_names_articulation_type_and_groups() -> None:
    manager = make_manager_with_schema_mismatch(
        articulation_name="left_arm", actuator_type=DCMotor, group_names=("wrist", "fingers")
    )
    with pytest.raises(ValueError, match="left_arm.*DCMotor.*wrist.*fingers"):
        manager.finalize()


def test_unknown_group_type_and_parameter_keys_raise() -> None:
    view = make_finalized_view()
    with pytest.raises(KeyError, match="missing"):
        _ = view["missing"]
    with pytest.raises(KeyError, match="DCMotor"):
        _ = view.by_type[DCMotor]
    with pytest.raises(KeyError, match="made_up"):
        view["pd"].set_parameter_index("made_up", 1.0)


def test_normal_selector_path_does_not_synchronize(monkeypatch) -> None:
    view = make_cuda_group_view(debug_validation=False)
    forbid_host_or_device_sync(monkeypatch)
    view.set_parameter_index(
        "stiffness",
        torch.tensor([2.0, 3.0], device="cuda:0"),
        env_ids=torch.tensor([0, 999999], device="cuda:0"),
        joint_ids=torch.tensor([0, -1], device="cuda:0"),
    )
```

- [ ] **Step 3: Prove the contrib multirotor collection is unaffected**

Add a regression that initializes `_ThrusterCollection`, exercises named lookup/iteration/update, and asserts it is not registered with the joint-actuator manager. Do not change the multirotor's public `actuators` mapping or force thrusters into joint-domain storage.

- [ ] **Step 4: Scan production kernels and public ownership**

Run:

```bash
rg -n "wp\.printf" \
  source/isaaclab/isaaclab/actuators \
  source/isaaclab_newton/isaaclab_newton/actuators
rg -n "effort_limit_sim|velocity_limit_sim|armature|friction|dynamic_friction|viscous_friction" \
  source/isaaclab/isaaclab/actuators/actuator_storage.py
```

Expected: the first command prints nothing. The second may find the explicit solver-ownership exclusion table/validation, but no typed field allocation.

- [ ] **Step 5: Run the complete focused correctness suite**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection_facade.py \
  source/isaaclab/test/actuators/test_actuator_collection_compatibility.py \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_actuator_execution_cuda.py \
  source/isaaclab/test/actuators/test_implicit_actuator.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py \
  source/isaaclab/test/actuators/test_dc_motor.py \
  source/isaaclab/test/envs/test_actuator_randomization.py -q
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/sim/test_simulation_context.py \
  source/isaaclab/test/assets/test_articulation_iface.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py \
  source/isaaclab_tasks/test/core/test_actuator_api_migration.py \
  source/isaaclab_contrib/test/assets/test_multirotor.py -q
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_newton_actuators_physx.py -q
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_newton_actuators_newton.py -q
```

Run the changed PhysX, OVPhysX, and Newton articulation selectors too. Do not call the entire repository test suite unless a focused failure indicates broader impact.

- [ ] **Step 6: Format and commit only if this audit added missing tests**

If a behavior defect is exposed, return to the owning implementation task, observe the new test fail, fix it there, and rerun this audit before committing this test-only task.

```bash
./isaaclab.sh -f
git add source/isaaclab/test/actuators \
  source/isaaclab/test/envs/test_actuator_randomization.py \
  source/isaaclab/test/assets/test_articulation_iface.py \
  source/isaaclab_tasks/test/core/test_actuator_api_migration.py \
  source/isaaclab_contrib/test/assets/test_multirotor.py
./isaaclab.sh -f
git commit -m "Complete actuator collection regressions"
```

If no file changed, record the audit as complete and do not create an empty commit.

---

### Task 15: Add a Reproducible Actuator Benchmark Driver

**Files:**

- Create: `scripts/benchmarks/benchmark_actuator_collection.py`
- Create: `scripts/benchmarks/summarize_actuator_collection.py`
- Create: `scripts/benchmarks/test/test_actuator_collection_benchmark.py`
- Reuse: `source/isaaclab/isaaclab/benchmark/micro.py`
- Reuse: `source/isaaclab/isaaclab/benchmark/formatters.py`
- Reuse: `source/isaaclab/isaaclab/benchmark/recorders/`

- [ ] **Step 1: Write failing CLI, capability, and matrix-definition tests**

```python
def test_build_matrix_covers_every_contract_case() -> None:
    cases = {case.name: case for case in build_matrix()}
    assert set(cases) == {"B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"}
    assert cases["B1"].world_counts == (1, 64, 4096)
    assert cases["B3"].num_sources == 4
    assert cases["B5"].group_counts == (1, 3, 12)


def test_empty_lifecycle_case_is_candidate_only_and_has_no_command_boundary() -> None:
    case = {case.name: case for case in build_matrix()}["B0"]
    assert case.supported_revisions == frozenset({"global"})
    assert case.terminal_boundary == "clear"


def test_runtime_matrix_records_revision_capabilities_without_fake_rows() -> None:
    expected_support = {
        "develop": {"cached_eager"},
        "current": {"cached_eager"},
        "global": {"cached_eager", "graph"},
    }
    for revision, supported_modes in expected_support.items():
        rows = runtime_matrix(revision)
        assert {(row.actuator_type, row.groups, row.requested_execution) for row in rows} == {
            (actuator_type, groups, execution)
            for actuator_type in ("implicit", "ideal_pd", "dc_motor")
            for groups in (1, 3, 12)
            for execution in ("cached_eager", "graph")
        }
        for row in rows:
            assert row.requested_execution in {"cached_eager", "graph"}
            assert row.supported == (row.requested_execution in supported_modes)
            assert row.effective_execution == (row.requested_execution if row.supported else None)


def test_all_build_cases_reject_scalar_dimension_overrides() -> None:
    for flag, value in (
        ("--num_worlds", "4096"),
        ("--num_sources", "4"),
        ("--num_articulations", "2"),
        ("--groups", "12"),
    ):
        with pytest.raises(SystemExit):
            parse_args(["--mode", "build", "--case", "all", flag, value])


def test_cold_matrix_coordinator_launches_one_fresh_child_per_row(monkeypatch) -> None:
    launched = spy_on_subprocess_run(monkeypatch)
    coordinate_build_matrix(revision="global", phase="cold", repetitions=2)
    assert len(launched.calls) == 2 * len(expand_build_matrix())
    assert all("--child_row" in call.args for call in launched.calls)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--num_worlds", "0"],
        ["--groups", "0"],
        ["--warmup_iterations", "-1"],
        ["--num_iterations", "0"],
    ],
)
def test_cli_rejects_nonpositive_dimensions_and_iterations(arguments) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)
```

Add `test_all_runtime_dimensions_expand_the_exact_supported_and_unsupported_rows`, `test_schema_records_revision_sha_requested_and_effective_execution`, `test_unsupported_graph_row_has_reason_and_no_timing`, `test_attempt_writer_never_overwrites_an_existing_attempt`, `test_attempt_allocator_uses_next_free_number_atomically`, `test_acceptance_manifest_requires_six_balanced_pair_ids`, `test_preflight_records_samples_and_rejection_reasons`, `test_structural_counter_formulas_on_literal_fake_generation`, and a CPU smoke that performs one build/application without wall-clock thresholds.

- [ ] **Step 2: Run the benchmark-driver tests and observe failure**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_actuator_collection_benchmark.py -q
```

Expected: FAIL because the driver and case definitions do not exist.

- [ ] **Step 3: Implement a stable coordinator/child CLI**

Support these invocations:

```bash
./isaaclab.sh -p scripts/benchmarks/benchmark_actuator_collection.py \
  --mode build \
  --case B3 \
  --num_worlds 4096 \
  --num_sources 4 \
  --num_articulations 1 \
  --groups 3 \
  --actuator_types implicit \
  --warmup_iterations 10 \
  --num_iterations 100 \
  --device cuda:0 \
  --benchmark_formatter schema \
  --output_path /tmp/actuator-build-benchmark

./isaaclab.sh -p scripts/benchmarks/benchmark_actuator_collection.py \
  --mode runtime \
  --actuator_types dc_motor \
  --groups 12 \
  --execution cached_eager \
  --num_worlds 4096 \
  --warmup_iterations 100 \
  --num_iterations 10000 \
  --device cuda:0 \
  --benchmark_formatter schema \
  --output_path /tmp/actuator-runtime-benchmark
```

Use feature-detected adapters so the candidate script can import packages from and benchmark all three pinned revisions. The adapter choice must be recorded in output:

```python
class _DevelopAdapter: ...
class _CurrentPrAdapter: ...
class _GlobalCollectionAdapter: ...


def select_adapter() -> _BenchmarkAdapter:
    try:
        from isaaclab.actuators import ActuatorCollection as collection_type
    except ImportError:
        return _DevelopAdapter()
    if hasattr(collection_type, "register_articulation"):
        return _GlobalCollectionAdapter()
    if hasattr(collection_type, "_build_execution_batches"):
        return _CurrentPrAdapter()
    return _DevelopAdapter()
```

Import `ActuatorCollection` inside a guarded `try/except ImportError` so develop revisions that do not define it can reach `_DevelopAdapter`. Each adapter measures the same externally visible boundary: resolved actuator construction through first command application. Candidate-only output additionally decomposes registration and finalization. `--case all`, `--actuator_types all`, `--groups all`, and `--execution all` expand the complete deterministic matrices for unattended final runs.

The child driver requires `--revision {develop,current,global}`, `--revision_sha`, `--candidate_sha`, `--observation_key`, `--attempt_id`, and `--phase {cold,warm,runtime}` in final-run mode. Every record contains both `requested_execution` and `effective_execution`. Develop and current support only `cached_eager`; global supports `cached_eager` and `graph`. Build cases also declare revision capability: B0 is global-only, while B1-B8 use all three adapters. An unsupported requested row emits a schema-valid record with `supported=false`, a stable reason, `effective_execution=null`, and no timing samples. It must never execute eager work labeled as graph work or invent an empty-collection command boundary.

`--case all` is coordinator mode, not an in-process loop. It expands the frozen B0-B8 dimensions and starts a fresh child process with one `--child_row` for every cold row and repetition. The child imports the target revision and performs exactly one first collection construction. Module/kernel compilation is prewarmed in a separate process and shared only through that revision's on-disk Warp cache; no collection is constructed in the cold child before measurement. Warm construction uses separate children, performs 10 unmeasured collection constructions, then 100 measured constructions. `--case all` rejects `--num_worlds`, `--num_sources`, `--num_articulations`, and scalar `--groups`; a single named case accepts only dimensions that the case declares overridable. Runtime `all` selectors expand the exact frozen matrix and reject any contradictory scalar selector.

Add top-level `--mode coordinate --matrix {build,runtime}` for the final three-revision run. It accepts the three `(worktree, exact SHA)` pairs, candidate SHA, immutable run root/batch ID, and repetition counts, then invokes the single-revision child interface through each worktree's own `isaaclab.sh` and `.venv`. Build rows use fresh children. Runtime rows use the balanced process-pair schedule in Step 5. Keep scheduling in the candidate process so run order, telemetry, rejection, and accepted-attempt identity share one schema. For each observation, allocate `attempt-XX` atomically with exclusive directory creation using the next free integer; contamination retries increment only that observation's attempt number and cannot collide with another matrix or batch.

- [ ] **Step 4: Implement the construction matrix and metrics**

Define:

```text
B0: candidate-only empty collection finalize/clear lifecycle (no command timing)
B1: homogeneous source, 1/64/4096 worlds, three implicit groups
B2: two articulations sharing the manager
B3: four sources × 1024 worlds with distinct values
B4: mixed Implicit/IdealPD/DCMotor types
B5: 1/3/12 same-exact-type groups with distinct values
B6: compatibility projection untouched/first access/repeated access/all projections
B7: neural/stateful/custom fallback registration
B8: finalize/clear/re-register/finalize lifecycle
```

Record cold and warm construction; registration CPU time; finalization CPU, GPU, and synchronized total time; time to first application; Python descriptor/object count; device allocation count and bytes by type/field; initialization launches; H2D bytes; D2H synchronization count; host/GPU peak and steady bytes; projection bytes/launches; pointer replacements; and owned storage after clear.

For cross-revision B1-B8 rows, use the same external timing boundary: immediately before the adapter begins resolved actuator construction/registration through completion of the first externally visible command application and required CUDA-event synchronization. Cross-revision gates use only these end-to-end timings. B0 is explicitly candidate-only and measures manager creation, empty finalize, and clear without inventing a command application or a develop collection adapter; D/C emit capability-declared unsupported B0 records with no timing. Candidate-only decomposition and structural counters use these exact definitions:

```text
python_descriptor_count = number of unique manager-owned instances reachable from
  the candidate generation in these classes: _ArticulationRegistration,
  _ResolvedManagedGroup, _GroupBinding, _TypeStore, _ArticulationBinding,
  ArticulationView, _ArticulationExecutionPlan, _ExecutionRange, and eager segment.
  Config/source objects and Torch/Warp aliases are excluded.

canonical_allocation_count = number of unique owning Warp arrays listed by typed
  stores and the joint-domain store, deduplicated by (device, ptr); group/type/
  articulation aliases and lazy projections are excluded.

canonical_allocated_bytes = sum(capacity * dtype_itemsize) over those unique arrays.
analytical_canonical_bytes = sum over every declared type field of
  (num_worlds * compact_type_dofs * dtype_itemsize), plus every declared
  joint-domain field's (articulation_worlds * articulation_dofs * dtype_itemsize).

analytical_owned_layout_bytes = sum(prod(spec.shape) * spec.dtype_itemsize) over
  the immutable _CollectionLayout.owned_array_specs created before allocation.
  The specs enumerate canonical type/joint fields, clone assignments, group and
  articulation-to-compact maps, one owner-slot map per output field, execution
  staging/scratch, and backend staging; aliases and lazy projections are separate.
  measured_owned_layout_bytes deduplicates every corresponding owner by
  (device, ptr), then sums capacity * dtype_itemsize.

initialization_launch_count = wp.launch/wp.launch_tiled calls observed by a
  benchmark-scoped wrapper between the adapter boundary markers, classified by
  kernel key; cached replays count as launches, not allocations.

h2d_bytes = bytes observed by benchmark-scoped wp.copy/Torch conversion wrappers
  where source is host and destination is the measured CUDA device.
d2h_sync_count = explicit synchronization/readback APIs observed by benchmark-
  scoped wrappers inside the boundary; final benchmark-timing synchronization is
  tagged harness-owned and excluded.

pointer_replacements = changes in the ordered (field, device, ptr) snapshot from
  post-finalize through every warm compute; steady allocations use allocator deltas
  over the same interval.
```

Expose these through a private benchmark adapter/introspection seam, not production hot-path counters. Older adapters emit unavailable structural values as `null`; structural gates apply only to the global candidate and are kept separate from cross-revision end-to-end timing gates.

For each B1-B8 cross-revision construction row and phase, schedule six independent current/global process pairs in balanced C-G/G-C order and six analogous develop/global pairs. Apply the fixed telemetry/acceptance protocol from Step 5 around each pair. Candidate-only structural counts, including B0, need one accepted singleton observation per matrix row; construction latency confidence intervals use only the paired B1-B8 process observations.

- [ ] **Step 5: Implement the actuator-only runtime matrix and metrics**

For every built-in stateless exact class and group count 1/3/12, run supported execution modes in a fresh process with 100 warm-ups and 10,000 CUDA-event-timed applications. Record latency distribution, throughput, steady allocations, computation/range count, recorded-launch count, graph replay count, eager segment count, pointer replacements, and exact comparison with independent execution.

For every actuator-type/group row, execute six independent current/global process pairs: three in C-G order and three in G-C order. The primary comparison is global graph versus current cached-eager; the secondary comparison is global cached-eager versus current cached-eager. Execute six analogous balanced develop/global pairs for the historical merge-base comparison. Treat the process-level median as the independent observation and bootstrap paired process-median ratios. The 10,000 within-process steps characterize the latency distribution but are not independent bootstrap units.

Before and after each process pair, sample GPU temperature, utilization, SM/memory clocks, throttle reasons, and compute-process PIDs for a fixed five-second window at 250 ms cadence. Accept a pair only when there is no competing compute process, pre-run utilization stays below 5%, no throttle reason is active, and the two revisions' start/end temperature envelopes differ by at most 5 degrees Celsius. Always write the samples, `accepted` flag, and explicit rejection reasons. A rejected pair remains immutable and a new attempt ID is scheduled; it is never deleted or silently excluded.

Collect counters through benchmark-side allocator/launch wrappers, plan introspection, and profiler APIs. Do not add an always-on production instrumentation branch to `compute()`. Do not assert timing thresholds in pytest. Emit machine-readable schema JSON plus a concise terminal table.

- [ ] **Step 6: Implement deterministic result aggregation**

`summarize_actuator_collection.py` requires `--run_root` for one `runs/<G_SHA>` directory, `--candidate_sha`, and its `accepted-attempts.json`. A paired observation/attempt key is `(matrix, row_key, comparison, mode_pair, pair_id, order)`; it intentionally excludes revision because one atomic attempt owns both compared members and their shared telemetry. `mode_pair` names both sides, for example `current-cached_eager__global-graph`. Each member record beneath the attempt is keyed by `(revision, requested_execution, effective_execution)` and stores its own timing. The manifest must contain pair IDs `01` through `06`, three in each order, and each selected pair attempt must contain exactly the two required member records. Candidate-only structural/B0 observations and capability-declared unsupported rows use distinct singleton keys that include their revision/mode. Duplicate rejection applies to the complete pair or singleton key, not to the logical benchmark row, so six independent pairs are representable without scheduling meaningless unsupported pairs.

The summarizer rejects a dirty candidate, any record whose candidate SHA differs, duplicate complete observation keys, repeated/missing pair IDs, an unbalanced order schedule, accepted references to missing attempt directories, missing supported rows, timing data on unsupported rows, or an unrecorded capability decision. It never fabricates graph rows for D/C and never combines attempts from different G SHAs. It computes medians/means/p95/dispersion, paired process-median revision ratios, and percentile-bootstrap 95% confidence intervals with seed 42, evaluates every named gate without dropping outliers, and writes `benchmark-summary.json`, `.csv`, and `.md`.

Add `test_summary_uses_paired_process_medians_and_fixed_bootstrap_seed` with a literal six-pair fixture whose expected ratios and interval endpoints are asserted. Add tests that mixed SHAs, duplicate complete observation keys, missing/repeated pair IDs, unbalanced orders, absent supported rows, fabricated unsupported timing, and missing attempt IDs are rejected, while six distinct pair keys and complete capability-declared unsupported graph rows are accepted.

The command is:

```bash
ACTUATOR_G_SHA=$(git rev-parse HEAD)
ACTUATOR_RUN_ROOT="/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/runs/${ACTUATOR_G_SHA}"
./isaaclab.sh -p scripts/benchmarks/summarize_actuator_collection.py \
  --run_root "${ACTUATOR_RUN_ROOT}" \
  --candidate_sha "${ACTUATOR_G_SHA}" \
  --bootstrap_seed 42 \
  --output_dir /home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report
```

- [ ] **Step 7: Pass driver tests and a CPU smoke**

```bash
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_actuator_collection_benchmark.py -q
./isaaclab.sh -p scripts/benchmarks/benchmark_actuator_collection.py \
  --mode build --case B1 --num_worlds 1 --num_sources 1 --num_articulations 1 \
  --groups 3 --actuator_types implicit --warmup_iterations 1 --num_iterations 1 \
  --device cpu --benchmark_formatter schema --output_path /tmp/actuator-benchmark-smoke
```

- [ ] **Step 8: Format and commit**

```bash
./isaaclab.sh -f
git add scripts/benchmarks/benchmark_actuator_collection.py \
  scripts/benchmarks/summarize_actuator_collection.py \
  scripts/benchmarks/test/test_actuator_collection_benchmark.py
./isaaclab.sh -f
git commit -m "Add actuator collection benchmarks"
```

---

### Task 16: Update the Public Surface, Documentation, and Release Notes

**Files:**

- Modify: `source/isaaclab/isaaclab/actuators/__init__.py`
- Modify: `source/isaaclab/isaaclab/actuators/__init__.pyi`
- Modify: `docs/source/api/lab/isaaclab.actuators.rst`
- Modify: `docs/source/overview/core-concepts/actuators.rst`
- Modify: `docs/source/migration/migrating_to_isaaclab_3-0.rst`
- Modify: `docs/source/tutorials/01_assets/run_articulation.rst`
- Modify: `docs/source/tutorials/03_envs/create_manager_base_env.rst`
- Modify: `scripts/tutorials/01_assets/run_articulation.py`
- Modify: `source/isaaclab/changelog.d/actuator-collection.minor.rst`
- Modify: `source/isaaclab_physx/changelog.d/actuator-collection.minor.rst`
- Modify: `source/isaaclab_newton/changelog.d/actuator-collection.minor.rst`
- Modify: `source/isaaclab_ovphysx/changelog.d/actuator-collection.minor.rst`
- Modify: `source/isaaclab_contrib/changelog.d/actuator-collection.skip`
- Create: `source/isaaclab_tasks/changelog.d/actuator-collection.skip`

- [ ] **Step 1: Audit exported symbols and references before editing**

Run:

```bash
rg -n "ActuatorCollection(\.Command|\.JointCommand)?|computed_torque|applied_torque|soft_joint_vel_limits|gear_ratio|write_actuator_(stiffness|damping)_to_sim|set_joint_(position|velocity|effort)_target" \
  docs/source scripts/tutorials source/isaaclab/isaaclab/actuators/__init__.pyi
```

Classify every hit as current public API, deprecated compatibility, migration example, or internal implementation. Do not mechanically rewrite solver-property uses of `joint_stiffness`/`joint_damping`.

- [ ] **Step 2: Update exports and the type stub**

Keep `ActuatorCollection` in `isaaclab.actuators.__all__`. Model `ArticulationView` as a nested `dict[str, ActuatorBase]` subclass and the other nested public types in `__init__.pyi` so this is valid and discoverable:

```python
robot.actuators: ActuatorCollection.ArticulationView
robot.actuators["legs"]
robot.actuators.by_type[IdealPDActuator]
robot.actuators.by_type[IdealPDActuator].parameters["stiffness"]
```

Expose no manager getter and no string aliases. Ensure every public parameter and command signature uses `snake_case`, PEP 604 unions, `torch.Tensor`, and concrete Warp dtypes. Ensure all physical docstrings state SI units and solver-dependent conventions.

- [ ] **Step 3: Rewrite the actuator concept page around ownership and scope**

Explain, with executable examples:

1. One simulation-global canonical store versus `robot.actuators` scoped access.
2. Logical named groups versus exact-type execution ranges.
3. Strided writable group views versus compact contiguous type views.
4. Exact-class lookup with `robot.actuators.by_type[IdealPDActuator]`.
5. Articulation-coordinate index and mask setters, including normal/debug behavior.
6. Raw actuator `command`, processed `joint_command`, `computed_effort`, and `applied_effort`.
7. Solver `data.joint_stiffness`/`joint_damping` versus actuator gains.
8. Lazy soft-velocity/gear projections and their zero/one fills.
9. Neural gain sidecars during deprecation.
10. Topology mutation, STOP-to-ready rebuild, generation invalidation, and retained-tensor rule.
11. Dict-transition behavior: `isinstance(..., dict)`, snapshots/unions, and mutators remain compatible, while exact `type(...) is dict` identity changes to the nested facade subclass.

Describe aggregation, recorded launches, and CUDA graphs as implementation details rather than user contracts.

- [ ] **Step 4: Add an explicit migration table**

The migration page must map every deprecated develop API:

```text
set_joint_position_target*       -> actuators.command.set_position_*
set_joint_velocity_target*       -> actuators.command.set_velocity_*
set_joint_effort_target*         -> actuators.command.set_effort_*
data.joint_pos_target            -> actuators.command.position
data.joint_vel_target            -> actuators.command.velocity
data.joint_effort_target         -> actuators.command.effort
data.computed_torque             -> actuators.computed_effort
data.applied_torque              -> actuators.applied_effort
write_actuator_stiffness_to_sim  -> group/type set_parameter_index("stiffness", ...)
write_actuator_damping_to_sim    -> group/type set_parameter_index("damping", ...)
data.soft_joint_vel_limits       -> capable group/type velocity_limit parameters
data.gear_ratio                  -> capable group/type gear_ratio parameters
```

State explicitly that solver `write_joint_*_to_sim*`, `data.joint_stiffness`, and `data.joint_damping` are unchanged. Also document that `isaaclab_newton.actuators.build_newton_actuator_defaults()` remains callable but is deprecated because collection-managed integrations bind canonical controller storage directly.

- [ ] **Step 5: Update API docs, tutorials, and executable examples**

Change `ActuatorCollection.Command` references to `ActuatorCollection.ArticulationView.Command`, document `TypeView`, and demonstrate group/type reads and setters. Update tutorial code to new names where it is teaching the new API; keep deprecated names only in a clearly labeled migration block.

- [ ] **Step 6: Update one changelog fragment per touched package**

Use past tense under `Added`, `Changed`, and `Deprecated`, with public Sphinx roles and direct migration guidance. Mention the scoped facade, type access, global storage, aggregation, and deprecations. Do not edit any generated `CHANGELOG.rst`, version, or `config/extension.toml`. Do not create a second fragment for a package that already has the `actuator-collection` fragment; create the listed `.skip` fragment only for the newly touched `isaaclab_tasks` package.

- [ ] **Step 7: Regenerate and verify documentation**

```bash
./isaaclab.sh -d
./isaaclab.sh -f
git diff --check
```

Review every generated modification. Stage all intended source/docs/fragment changes, rerun `./isaaclab.sh -f`, and ensure no benchmark artifact, TeX file, or PDF is staged.

- [ ] **Step 8: Commit public documentation**

```bash
git add source/isaaclab/isaaclab/actuators/__init__.py \
  source/isaaclab/isaaclab/actuators/__init__.pyi \
  docs/source/api/lab/isaaclab.actuators.rst \
  docs/source/overview/core-concepts/actuators.rst \
  docs/source/migration/migrating_to_isaaclab_3-0.rst \
  docs/source/tutorials/01_assets/run_articulation.rst \
  docs/source/tutorials/03_envs/create_manager_base_env.rst \
  scripts/tutorials/01_assets/run_articulation.py \
  source/isaaclab/changelog.d/actuator-collection.minor.rst \
  source/isaaclab_physx/changelog.d/actuator-collection.minor.rst \
  source/isaaclab_newton/changelog.d/actuator-collection.minor.rst \
  source/isaaclab_ovphysx/changelog.d/actuator-collection.minor.rst \
  source/isaaclab_contrib/changelog.d/actuator-collection.skip \
  source/isaaclab_tasks/changelog.d/actuator-collection.skip
./isaaclab.sh -f
git commit -m "Document global actuator storage"
```

---

### Task 17: Run the Final Benchmark, Regenerate the PDF, Push, and Update PR 6839

**Repository files:** none unless verification uncovers a defect; fixes require a new focused red-green commit before restarting the affected measurements.

**Uncommitted artifact root:** `/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04`

- [ ] **Step 1: Establish a clean, immutable candidate**

Run the verification-before-completion workflow:

```bash
git status --short --branch
git diff --check
./isaaclab.sh -f
git rev-parse HEAD
git rev-parse 378225f8d2af0a9920e18a934ee7d044844e023e
git rev-parse 5c59a092be11e4c95d63195476f58e7d2b0b8084
```

The working tree must be clean. Record the final candidate SHA as `G`; do not benchmark a dirty tree.

- [ ] **Step 2: Run final focused verification**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py \
  source/isaaclab/test/actuators/test_actuator_collection_storage.py \
  source/isaaclab/test/actuators/test_actuator_collection_facade.py \
  source/isaaclab/test/actuators/test_actuator_collection_compatibility.py \
  source/isaaclab/test/actuators/test_actuator_execution.py \
  source/isaaclab/test/actuators/test_actuator_execution_cuda.py \
  source/isaaclab/test/actuators/test_implicit_actuator.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py \
  source/isaaclab/test/actuators/test_dc_motor.py \
  source/isaaclab/test/envs/test_actuator_randomization.py -q
./isaaclab.sh -p -m pytest source/isaaclab/test/sim/test_simulation_context.py -q
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/assets/test_articulation_iface.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py \
  source/isaaclab_tasks/test/core/test_actuator_api_migration.py -q
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_newton_actuators_physx.py -q
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_newton_actuators_newton.py -q
./isaaclab.sh -p -m pytest scripts/benchmarks/test/test_actuator_collection_benchmark.py -q
```

Also rerun the focused changed PhysX, OVPhysX, Newton, and contrib articulation selectors from Tasks 8 and 14. Record pass/fail/skip counts and skip reasons in `report/benchmark-summary.md`.

- [ ] **Step 3: Create clean comparison worktrees and isolated Isaac Sim environments**

Use `superpowers:using-git-worktrees` at execution time. Create:

```bash
git worktree add /tmp/isaaclab-actuator-bench-develop 378225f8d2af0a9920e18a934ee7d044844e023e
git worktree add /tmp/isaaclab-actuator-bench-current 5c59a092be11e4c95d63195476f58e7d2b0b8084
```

In all three worktrees, create or refresh a worktree-local environment from that revision's lockfile and install Isaac Sim:

```bash
uv sync --extra isaacsim
```

If the environment lacks only the test runner needed by the benchmark helper, install it into that environment:

```bash
uv pip install --python .venv/bin/python pytest pytest-mock
```

The candidate may reuse its existing worktree-local `.venv`, but still run the sync so its manifest matches G. Never share a `.venv` between revisions.

- [ ] **Step 4: Create the artifact layout and reproducibility manifest**

Create these ignored paths:

```text
tools/tectonic-0.16.9/
cache/{D_SHA,C_SHA,G_SHA}/
runs/{G_SHA}/accepted-attempts.json
runs/{G_SHA}/selection-history/accepted-attempts-{timestamp}.json
runs/{G_SHA}/batches/{build-01,runtime-01,franka-01,startup-01,nsys-01}/manifest.json
runs/{G_SHA}/observations/{observation-key}/attempt-{01..NN}/manifest.json
runs/{G_SHA}/observations/{observation-key}/attempt-{01..NN}/preflight/
runs/{G_SHA}/observations/{observation-key}/attempt-{01..NN}/members/{develop,current,global}/
report/benchmark-summary.json
report/benchmark-summary.csv
report/benchmark-summary.md
report/pr-6839-body.md
report/isaaclab-actuator-refactoring.tex
report/isaaclab-actuator-refactoring.pdf
```

Batch manifests record coordinator invocations but never own result identity. Each filesystem-safe paired observation key encodes matrix, logical row, comparison/mode pair, and pair ID/order—never one member revision. Its `members/{revision}` records carry requested/effective modes. Singleton keys encode their revision/mode directly. Never overwrite an `attempt-XX` directory: the coordinator atomically creates the next free number within that observation. Rejected attempts and their telemetry remain in place; reruns cannot collide with another row or batch. A Franka observation attempt similarly owns the complete three-revision triplet rather than three unrelated attempts.

`accepted-attempts.json` selects six complete paired attempts for every paired logical row and one attempt for every singleton structural row. Every update is copied to append-only `selection-history/`. The canonical `report/` is generated atomically from that selection manifest and records `G_SHA`; raw results never live directly under stable `build/` or `actuator_runtime/` paths.

Record exact D/C/G SHA and dirty flag; lockfile hash; task/benchmark configuration hash; selected compatibility adapter; requested/effective execution; supported/unsupported reason; GPU model; driver; CUDA; Isaac Sim; Newton; Warp; Torch; Python 3.12 patch version; run order; timestamps; all fixed-window GPU telemetry samples; acceptance decision; and competing compute processes. Give each revision SHA a separate `WARP_CACHE_PATH`, prewarm compilation once in an unmeasured process, and keep compilation outside measured collection construction.

- [ ] **Step 5: Run the full B0-B8 construction and actuator-runtime matrix**

Set the invariant paths:

```bash
ACTUATOR_CANDIDATE_WORKTREE=/home/antoiner/Documents/IsaacLab/docs/superpowers/worktrees/actuator-global-storage
ACTUATOR_ARTIFACT_ROOT=/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04
ACTUATOR_D_SHA=378225f8d2af0a9920e18a934ee7d044844e023e
ACTUATOR_C_SHA=5c59a092be11e4c95d63195476f58e7d2b0b8084
ACTUATOR_G_SHA=$(git -C "${ACTUATOR_CANDIDATE_WORKTREE}" rev-parse HEAD)
ACTUATOR_RUN_ROOT="${ACTUATOR_ARTIFACT_ROOT}/runs/${ACTUATOR_G_SHA}"
```

Invoke the final coordinator from the candidate environment. It launches every revision through that revision's own wrapper and `.venv`, refuses an existing batch ID, atomically allocates per-observation attempts, writes preflight/acceptance metadata even on failure, and records capability-declared unsupported rows:

```bash
"${ACTUATOR_CANDIDATE_WORKTREE}/isaaclab.sh" -p \
  "${ACTUATOR_CANDIDATE_WORKTREE}/scripts/benchmarks/benchmark_actuator_collection.py" \
  --mode coordinate \
  --matrix build \
  --develop_worktree /tmp/isaaclab-actuator-bench-develop \
  --develop_sha "${ACTUATOR_D_SHA}" \
  --current_worktree /tmp/isaaclab-actuator-bench-current \
  --current_sha "${ACTUATOR_C_SHA}" \
  --global_worktree "${ACTUATOR_CANDIDATE_WORKTREE}" \
  --global_sha "${ACTUATOR_G_SHA}" \
  --candidate_sha "${ACTUATOR_G_SHA}" \
  --run_root "${ACTUATOR_RUN_ROOT}" \
  --batch_id build-01 \
  --cold_repetitions 6 \
  --pair_repetitions 6 \
  --warmup_iterations 10 \
  --num_iterations 100 \
  --device cuda:0 \
  --benchmark_formatter schema

"${ACTUATOR_CANDIDATE_WORKTREE}/isaaclab.sh" -p \
  "${ACTUATOR_CANDIDATE_WORKTREE}/scripts/benchmarks/benchmark_actuator_collection.py" \
  --mode coordinate \
  --matrix runtime \
  --develop_worktree /tmp/isaaclab-actuator-bench-develop \
  --develop_sha "${ACTUATOR_D_SHA}" \
  --current_worktree /tmp/isaaclab-actuator-bench-current \
  --current_sha "${ACTUATOR_C_SHA}" \
  --global_worktree "${ACTUATOR_CANDIDATE_WORKTREE}" \
  --global_sha "${ACTUATOR_G_SHA}" \
  --candidate_sha "${ACTUATOR_G_SHA}" \
  --run_root "${ACTUATOR_RUN_ROOT}" \
  --batch_id runtime-01 \
  --pair_repetitions 6 \
  --warmup_iterations 100 \
  --num_iterations 10000 \
  --device cuda:0 \
  --benchmark_formatter schema
```

The build coordinator expands the frozen case dimensions internally; it receives no conflicting `--num_worlds`, `--num_sources`, `--num_articulations`, or `--groups` override. Every cold observation is the first construction in a fresh child after only cache prewarm. Warm observations occur in separate children after 10 unmeasured constructions. The runtime coordinator runs every stateless type x 1/3/12 group row, the supported mode schedule, and the six balanced process pairs defined in Task 15.

Evaluate these candidate-only structural gates:

```text
Python work/object count is O(prototypes × groups + articulations), never worlds × DOFs.
B1 descriptor/object count is identical for 1, 64, and 4096 worlds.
Allocation and initialization-launch counts are identical for 1, 64, and 4096 worlds; bytes may scale.
Adding same-type groups adds metadata/views, not canonical allocations or initialization launches.
Internal D2H synchronization count is zero.
Post-finalize allocation and pointer replacement counts are zero.
Untouched compatibility projection bytes are zero; repeated access allocates zero bytes.
STOP/clear releases owned storage and registrations to zero.
G registration at 4096 worlds <= 1.2 × G registration at 64 worlds.
Four-source B3 time <= 5 × one-source B1 time.
Allocated bytes are within 1% of the analytical field/layout total.
```

Evaluate these cross-revision build timing gates from the common external boundary:

```text
Warm G construction-through-first-application - warm C <= max(5% of C, 0.25 ms).
G construction-through-first-application p95 <= C p95 + 10%.
```

Evaluate these actuator-runtime gates:

```text
Aggregated output equals independent execution exactly for all supported stateless classes.
Steady-state allocation count is zero.
Each exact stateless class computes once independent of logical group count.
Global graph with no compatibility epilogue performs one complete graph replay.
Primary: G-graph/C-cached-eager one-group ratio <= 1.00 and paired-process-bootstrap upper 95% bound <= 1.02.
Primary: G-graph/C-cached-eager three- and twelve-group ratio < 1.00 and paired-process-bootstrap upper 95% bound < 1.00.
Secondary: report G-cached-eager/C-cached-eager for every row; its aggregate median ratio must be <= 1.02.
Reference: report G-graph/D-cached-eager with the same paired-process method.
Unsupported D/C graph records have no timing and are excluded by declared capability, never by observed performance.
```

- [ ] **Step 6: Run twelve counterbalanced Franka Reach trials**

Use two blocks of all six revision permutations:

```text
D-C-G, D-G-C, C-D-G, C-G-D, G-D-C, G-C-D
G-C-D, G-D-C, C-G-D, C-D-G, D-G-C, D-C-G
```

Before and after each scheduled triplet, use the same fixed five-second/250 ms telemetry sampler and acceptance criteria as the actuator-only matrix. Reject the whole triplet if any revision is contaminated or if cross-revision temperature envelopes differ by more than 5 degrees Celsius; retain it with reasons and atomically allocate the next attempt under that same triplet observation key. Expand D/C/G to the exact `(ACTUATOR_LABEL, ACTUATOR_REVISION_WORKTREE, SHA)` triples `(develop, /tmp/isaaclab-actuator-bench-develop, D_SHA)`, `(current, /tmp/isaaclab-actuator-bench-current, C_SHA)`, and `(global, /home/antoiner/Documents/IsaacLab/docs/superpowers/worktrees/actuator-global-storage, G_SHA)`. Set `ACTUATOR_RUN_ID` to `01` through `12` and execute the three labels in that row's order.

Before the first member, set `ACTUATOR_ATTEMPT_ROOT` to the atomically allocated `runs/G/observations/franka-order-NN/attempt-XX` directory. All three sub-runs write below that one attempt's `members/`; set `ACTUATOR_REVISION_SHA` from each triple. The attempt manifest records order and pre/post telemetry, and only complete accepted triplets enter `accepted-attempts.json`.

For each scheduled sub-run, use:

```bash
VIRTUAL_ENV="${ACTUATOR_REVISION_WORKTREE}/.venv" \
OMNI_KIT_ACCEPT_EULA=yes \
WARP_CACHE_PATH="${ACTUATOR_ARTIFACT_ROOT}/cache/${ACTUATOR_REVISION_SHA}" \
"${ACTUATOR_REVISION_WORKTREE}/isaaclab.sh" benchmark runtime \
  --task Isaac-Reach-Franka \
  --num_envs 4096 \
  --warmup_steps 100 \
  --num_steps 1000 \
  --seed 42 \
  --visualizer none \
  --benchmark_formatter schema \
  --output_path "${ACTUATOR_ATTEMPT_ROOT}/members/${ACTUATOR_LABEL}" \
  physics=isaacsim_physx
```

Report all accepted runs, median, mean, p95, dispersion, throughput, paired G/C and G/D ratios, and bootstrap 95% confidence intervals. Do not mix the historical three-run numbers into the new statistics. The final gates are:

```text
G median <= C median.
The paired-bootstrap 95% upper confidence bound for G/C <= 1.02.
G/D median latency ratio <= 0.90.
```

Keep the historical values only as clearly labeled context: D median 14.92 ms, intermediate `77b77692aa` median 12.91 ms, C median 14.08 ms, and C/D three-run mean improvement 8.35%.

- [ ] **Step 7: Measure startup separately and capture Nsight evidence**

Run at least three startup trials per revision after setting the same exact label/worktree/SHA triple, atomically allocating `ACTUATOR_ATTEMPT_ROOT` under the `startup-{label}-run-NN` observation, and setting `ACTUATOR_RUN_ID` to `01`, `02`, then `03`:

```bash
VIRTUAL_ENV="${ACTUATOR_REVISION_WORKTREE}/.venv" \
OMNI_KIT_ACCEPT_EULA=yes \
WARP_CACHE_PATH="${ACTUATOR_ARTIFACT_ROOT}/cache/${ACTUATOR_REVISION_SHA}" \
"${ACTUATOR_REVISION_WORKTREE}/isaaclab.sh" benchmark startup \
  --task Isaac-Reach-Franka \
  --num_envs 4096 \
  --seed 42 \
  --visualizer none \
  --benchmark_formatter schema \
  --output_path "${ACTUATOR_ATTEMPT_ROOT}/members/${ACTUATOR_LABEL}" \
  physics=isaacsim_physx
```

Environment-creation and first-step medians should not exceed C by more than 3%; GPU memory should not exceed C by more than 5%.

Capture one accepted C and G profile with 200 measured steps. Atomically allocate one `nsys-current-global` observation attempt, then set the label/worktree/SHA triple first to current and then to global; both profiles belong to that attempt:

```bash
VIRTUAL_ENV="${ACTUATOR_REVISION_WORKTREE}/.venv" \
OMNI_KIT_ACCEPT_EULA=yes \
WARP_CACHE_PATH="${ACTUATOR_ARTIFACT_ROOT}/cache/${ACTUATOR_REVISION_SHA}" \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="${ACTUATOR_ATTEMPT_ROOT}/members/${ACTUATOR_LABEL}/capture" \
  "${ACTUATOR_REVISION_WORKTREE}/isaaclab.sh" benchmark runtime \
    --task Isaac-Reach-Franka \
    --num_envs 4096 \
    --warmup_steps 100 \
    --num_steps 200 \
    --seed 42 \
    --visualizer none \
    --benchmark_formatter schema \
    --output_path "${ACTUATOR_ATTEMPT_ROOT}/members/${ACTUATOR_LABEL}/schema" \
    physics=isaacsim_physx

nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum \
  --format csv \
  --output "${ACTUATOR_ATTEMPT_ROOT}/members/${ACTUATOR_LABEL}/stats" \
  "${ACTUATOR_ATTEMPT_ROOT}/members/${ACTUATOR_LABEL}/capture.nsys-rep"
```

`nsys stats --output` writes explicit `stats_cuda_api_sum.csv` and `stats_cuda_gpu_kern_sum.csv` artifacts rather than relying on captured terminal output. The trace must show no added hot-loop allocation or D2H synchronization and no increase in actuator dispatch count from C to G. Profiler wall time is attribution evidence, not primary latency evidence.

- [ ] **Step 8: Summarize results and handle missed gates honestly**

Run:

```bash
./isaaclab.sh -p scripts/benchmarks/summarize_actuator_collection.py \
  --run_root "${ACTUATOR_RUN_ROOT}" \
  --candidate_sha "${ACTUATOR_G_SHA}" \
  --bootstrap_seed 42 \
  --output_dir /home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report
```

If a correctness or structural gate fails, diagnose it with `superpowers:systematic-debugging`, add a focused failing test, fix it in a new commit, rerun formatting/verification, assign a new G SHA, and restart affected measurements. If a statistical performance target misses while correctness and structural gates pass, report the miss and confidence interval rather than selecting favorable runs.

- [ ] **Step 9: Recreate the external LaTeX engineering document and PDF**

The prior temporary TeX/PDF are no longer present. Recreate the detailed document from the approved specification and final evidence at:

```text
/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report/isaaclab-actuator-refactoring.tex
/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report/isaaclab-actuator-refactoring.pdf
```

Include: motivation; old/new ownership diagrams; lifecycle sequence; flat layout formula and overlapping one-to-many example; clone-plan construction; group/type/setter API; parameter ownership table; dict/compatibility/deprecation table; execution and graph boundaries; native Newton binding; failure/rebuild semantics; test evidence; immutable-attempt and accepted-pair protocol; complete build/runtime/Franka tables; confidence intervals; Nsight attribution; limitations; and future scene-global execution.

Provision the PDF compiler once in the persistent artifact tool cache. Pin both the official release asset and its GitHub-published SHA-256 digest; refuse extraction on mismatch and record URL, version, and digest in the final manifest:

```bash
TECTONIC_TOOL_ROOT=/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/tools/tectonic-0.16.9
TECTONIC_ARCHIVE="${TECTONIC_TOOL_ROOT}/tectonic-0.16.9-x86_64-unknown-linux-gnu.tar.gz"
TECTONIC_URL=https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-unknown-linux-gnu.tar.gz
TECTONIC_SHA256=f3c825128095dc3399ea11c08c18035b33050a216930c295c79e8eb11bd21de4
mkdir -p "${TECTONIC_TOOL_ROOT}"
curl --fail --location --retry 3 --output "${TECTONIC_ARCHIVE}" "${TECTONIC_URL}"
test "$(sha256sum "${TECTONIC_ARCHIVE}" | awk '{print $1}')" = "${TECTONIC_SHA256}"
tar --extract --gzip --file "${TECTONIC_ARCHIVE}" --directory "${TECTONIC_TOOL_ROOT}"
"${TECTONIC_TOOL_ROOT}/tectonic" --version
```

Compile outside the repository commit:

```bash
"${TECTONIC_TOOL_ROOT}/tectonic" -X compile \
  --outdir /home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report \
  /home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report/isaaclab-actuator-refactoring.tex
pdfinfo /home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report/isaaclab-actuator-refactoring.pdf
pdftoppm -png -r 120 \
  /home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report/isaaclab-actuator-refactoring.pdf \
  /tmp/isaaclab-actuator-refactoring-page
```

Inspect every rendered page, correct clipping/overflow/table issues, recompile, and confirm the TeX/PDF remain untracked and unstaged.

- [ ] **Step 10: Push only to the fork and update the upstream PR**

Run once more immediately before push:

```bash
./isaaclab.sh -f
git diff --check
git status --short --branch
git log --oneline --decorate -20
```

Push only:

```bash
git push antoine antoiner/actuators-collection-split-6248
```

Build `report/pr-6839-body.md` from the final report. Include architecture, public API, migration/deprecation behavior, exact verification commands/results, D/C/G protocol and table, confidence intervals, construction gates, Nsight evidence, limitations, and the final candidate SHA. Then update and verify the upstream PR:

```bash
gh pr edit 6839 --repo isaac-sim/IsaacLab --body-file \
  /home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report/pr-6839-body.md
gh pr view 6839 --repo isaac-sim/IsaacLab \
  --json url,headRefName,headRefOid,baseRefName,body
```

Confirm `baseRefName` is `develop`, the head SHA is G, and no push occurred to `origin`.

- [ ] **Step 11: Hand off the evidence**

Report the PR URL, final SHA, focused test results, any skips, construction/runtime gate table, Franka paired result, profiler conclusion, artifact root, and this absolute PDF path:

```text
/home/antoiner/Documents/IsaacLab/benchmarks/actuator-collection-refactor/2026-08-04/report/isaaclab-actuator-refactoring.pdf
```

Do not claim a performance win that is outside the measured confidence interval.

---

## Completion Definition

The work is complete only when:

- the public compatibility contract passes on every available backend;
- global construction is clone-count independent in Python and transactional across lifecycle events;
- group/type views and selectors match the frozen contract;
- dict transition behavior and one-to-many overlapping type selectors preserve develop semantics;
- supported stateless actuator aggregation is exact and allocation-free;
- native Newton aliases canonical storage without gain snapshots;
- full graphable plans replay one graph and mixed plans use the documented prefix/eager/scatter boundary;
- all focused tests and formatting pass;
- documentation, stubs, tutorials, and changelog fragments are current;
- the pinned benchmark matrix has six accepted balanced process pairs per paired row, immutable rejected attempts, and Nsight attribution under the persistent artifact root;
- the LaTeX source and regenerated PDF exist at the stated uncommitted path;
- the branch is pushed only to `antoine`; and
- PR 6839 targets `develop` and contains the final evidence-backed description.
