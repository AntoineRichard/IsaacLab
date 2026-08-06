# Direct Physics Engine API Access Documentation Design

## Purpose

Isaac Lab exposes a unified high-level asset and sensor API across physics
backends, but advanced users also need access to each engine's native low-level
data interfaces. Those interfaces are intentionally different:

- PhysX provides typed Tensor API views with explicit reads and writes.
- Newton exposes live `Model`, `State`, `Control`, and `Contacts` arrays, with a
  generic selection API layered over them.
- OvPhysX provides tensor bindings with explicit pull and push operations;
  Isaac Lab's `OvPhysxView` is a convenience manager over those bindings.

The documentation must explain these native models accurately instead of
presenting `root_view` as a portable cross-backend abstraction.

## Goals

- Give users a high-level comparison of the three native access models.
- Explain why Isaac Lab does not provide one low-level "view to rule them all."
- Show how to reuse low-level handles created by Isaac Lab.
- Show how to construct raw native accessors when an Isaac Lab object does not
  own the desired selection.
- Teach runtime discovery and link to authoritative upstream references instead
  of mirroring volatile method, attribute, or enum inventories.
- Document ownership, lifetime, synchronization, device, dtype, and mutation
  consequences for each backend.
- Cover PhysX sensor-related view families and hand method-level details to the
  upstream Tensor API reference.
- Make the stable Isaac Lab manager and convenience-view entry points easy to
  find in the generated API reference.

## Non-goals

- Creating a new common runtime abstraction.
- Making native low-level code portable between backends.
- Reproducing complete upstream API references.
- Maintaining Newton attribute lists, OvPhysX `TensorType` inventories, or
  every PhysX Tensor API method in hand-written documentation.
- Adding dependencies, changing simulation behavior, or changing public APIs.
- Documenting private sensor fields as supported public access points.

## Implementation Baseline

The documentation change will be implemented against the latest `develop`
baseline, not the currently checked-out feature branch, because the backend
interfaces and physical-backend guides have continued to evolve. Examples and
cross-references must be checked again after the implementation worktree is
created and before any documentation is written.

## Design Principle: Preserve Native Power

The guide will state explicitly that a unified low-level view would hide
important engine semantics and collapse the interfaces to a least-common
denominator.

PhysX and OvPhysX use explicit pull/push operations. PhysX organizes those
operations around typed views for different physics object families. OvPhysX
organizes them around generic bindings selected by tensor type, with
`OvPhysxView` supplying guarded convenience operations.

Newton instead exposes live Warp arrays owned by `Model`, `State`, `Control`,
and `Contacts`. Its selection API describes subsets and stable batched layouts;
it does not introduce a separate copy or push/pull ownership layer. Mutating a
live array, writing a Tensor API view, and writing a TensorBinding therefore
have different synchronization and correctness requirements.

Isaac Lab will document and expose these distinctions so advanced users retain
the performance, expressiveness, and engine-specific capabilities of the
native APIs.

## Documentation Architecture

Create one overview and three backend-specific pages:

```text
docs/source/overview/core-concepts/physical-backends/
└── direct-api-access/
    ├── index.rst
    ├── physx.rst
    ├── newton.rst
    └── ovphysx.rst
```

The overview owns comparison and decision guidance. Each backend page owns its
native mental model, access paths, examples, and hazards. This keeps the main
page concise while giving each backend enough space to explain its genuinely
different semantics.

### Overview page

Title the page **Direct Physics Engine API Access**. It will contain:

1. A warning that these APIs are backend-specific escape hatches that bypass
   parts of Isaac Lab's buffering and validation.
2. A short explanation of when direct access is appropriate.
3. The rationale for not creating a unified low-level view.
4. A comparison table with these stable conceptual columns:
   - native entry point;
   - selection model;
   - data ownership;
   - read/write model;
   - synchronization responsibility;
   - invalidation boundary.
5. Guidance for choosing between the unified Isaac Lab API, an Isaac Lab-owned
   native handle, and a caller-created raw accessor.
6. A toctree linking the PhysX, Newton, and OvPhysX pages.

The table will compare concepts rather than enumerate methods or attributes.

### Shared backend-page skeleton

All three pages will use the following navigational skeleton where it fits the
engine:

1. Mental model.
2. Lifecycle prerequisites.
3. Reuse an accessor created by Isaac Lab.
4. Construct raw native access.
5. Runtime discovery.
6. Read and write examples.
7. Ownership, synchronization, and invalidation.
8. Authoritative references.

The wording and examples will remain engine-native. Symmetry means answering
the same user questions, not forcing identical APIs or section content.

## PhysX Page

The PhysX page will explain the typed Tensor API view model.

### Access paths

- Obtain Isaac Lab's active `omni.physics.tensors.SimulationView` through
  `isaaclab_physx.physics.PhysxManager.get_physics_sim_view()` after physics
  initialization.
- Reuse an Isaac Lab asset's typed `root_view` where one already exists.
- Create a raw typed view from `SimulationView` for an independently selected
  set of prims.

### Discovery and examples

- Demonstrate runtime discovery of `create_*_view` factories through Python
  introspection rather than copying the complete factory list.
- Show a representative typed view creation from a prim glob.
- Show one representative read and one explicit setter write using the Warp
  frontend used by Isaac Lab.
- Identify the typed view families used by Isaac Lab's PhysX sensors, including
  rigid-body, articulation, and rigid-contact access. This is a local behavior
  map, not a reproduction of every upstream method.
- Link method-level details and the authoritative view inventory to the Omni
  Physics Tensor API documentation.

### Required cautions

- A typed view is bound to a physics-object family and selection pattern.
- Tensor API selection globs and shape/order conventions must be preserved.
- Writes require the corresponding setter; modifying a returned or staging
  tensor does not necessarily publish state by itself.
- The caller must respect the frontend, device, dtype, and expected tensor
  shape.
- Views become invalid across hard reset or physics teardown and must be
  reacquired.
- The documentation must not present private sensor view attributes as stable
  public API.

## Newton Page

The Newton page will explain live native data ownership and generic selection.

### Access paths

- Obtain the authoritative `Model`, current `State`, `Control`, and optional
  `Contacts` from `isaaclab_newton.physics.NewtonManager` after model
  finalization.
- Reuse an Isaac Lab-created `newton.selection.ArticulationView` through
  `root_view` where convenient, while stating that the selection is generic and
  is not the storage or ownership model for a particular Isaac Lab asset type.
- Construct a caller-owned `ArticulationView` from the Newton model and label
  pattern when a separate selection is needed.

### Discovery and examples

- Explain the roles of `Model` for static properties, `State` for evolving
  quantities, `Control` for actuation inputs, and `Contacts` for collision
  output.
- Demonstrate discovery through model counts and labels, selection metadata,
  and Python introspection.
- Show direct access to a representative live array.
- Show a representative selection read and write with stable batched shapes.
- Link to Newton's generated API reference and articulation/selection guide
  rather than maintaining an attribute inventory locally.

### Required cautions

- Arrays are live engine-owned pointers, not pulled snapshots.
- Direct writes can bypass Isaac Lab caches and validation.
- Model-property changes may require `NewtonManager.add_model_change(...)` so
  the solver can refresh dependent data.
- Generalized-coordinate writes may require forward kinematics before body
  transforms are consistent.
- State layout and required synchronization can depend on the active Newton
  solver.
- Manager-owned objects and selections must not be retained past physics
  teardown or model rebuild.

## OvPhysX Page

The OvPhysX page will explain raw `TensorBinding` access and Isaac Lab's optional
convenience view.

### Access paths

- Obtain the active `ovphysx.PhysX` instance through
  `isaaclab_ovphysx.physics.OvPhysxManager.get_physx_instance()` after runtime
  initialization.
- Create a raw binding through the runtime's `create_tensor_binding(...)` API.
- Reuse an Isaac Lab-owned `OvPhysxView` through `root_view` where one exists.
- Construct an independent `OvPhysxView` over a pattern or explicit prim-path
  list when its validation and discoverability are useful.

### Discovery and examples

- Discover the installed runtime's `TensorType` members dynamically.
- For `OvPhysxView`, demonstrate `attribute_names`, availability checks,
  `get_attribute`, reusable-buffer `read_into`, and `set_attribute`.
- Show the raw binding `read()` and `write()` flow with caller-owned buffers.
- Explain that `binding_for` is an advanced escape hatch that bypasses the
  convenience view's device, dtype, shape, and read-only guards.
- Link to upstream OvPhysX documentation only when the repository's pinned
  runtime metadata identifies a versioned authoritative reference. Otherwise,
  link to generated Isaac Lab references for `OvPhysxView` and teach runtime
  discovery.

### Required cautions

- Tensor bindings use explicit pull and push operations.
- State bindings may be device-resident while some property bindings are
  CPU-only.
- Buffer shape and scalar dtype must match binding-reported DLPack metadata.
- Structured Warp values may require reinterpretation of the binding's flat
  scalar representation.
- Availability is selection-dependent; a valid `TensorType` need not be
  available for every matched prim set.
- Access mode may be read/write, read-only, or write-only, and runtime metadata
  remains authoritative.
- Bindings and convenience views must be reacquired after physics teardown or
  stage reload.

## Example Policy

Examples will be concise, copyable fragments that assume the simulation has
completed initialization/reset. The guide will not duplicate three complete
environment setup scripts.

Every example will state:

- the lifecycle prerequisite;
- who owns the returned handle and memory;
- whether the operation observes live data, pulls into a buffer, or pushes a
  buffer;
- whether a follow-up publish, notification, or kinematic synchronization step
  is required;
- when the handle becomes invalid.

Examples will use one representative operation per concept. Runtime discovery
and authoritative links will cover API growth without requiring edits to this
guide whenever upstream adds or removes a field.

## Navigation and Existing Documentation

Update these documentation entry points:

- `docs/source/overview/core-concepts/physical-backends/index.rst`: add the new
  overview to the toctree and introduce direct engine access alongside backend
  selection.
- Each backend landing page: add a short link to its detailed low-level access
  page.
- `docs/source/overview/core-concepts/multi_backend_architecture.rst`: link from
  the asset and sensor interface discussion and distinguish the unified API
  from native escape hatches.
- `docs/source/migration/migrating_to_isaaclab_3-0.rst`: mark the existing
  `root_view.get_masses()` example as PhysX-specific and link to the new guide.

The detailed pages will cross-link back to the overview and to the relevant
backend landing page.

## Generated API Reference

Improve generated reference discoverability without hand-maintaining upstream
inventories:

- Expand `docs/source/api/lab_physx/isaaclab_physx.physics.rst` to list and
  document `PhysxManager` and `PhysxCfg` explicitly.
- Preserve Newton's existing explicit `NewtonManager` reference.
- Expand `docs/source/api/lab_ovphysx/isaaclab_ovphysx.physics.rst` to list and
  document `OvPhysxManager` and `OvPhysxCfg` explicitly.
- Expand `docs/source/api/lab_ovphysx/isaaclab_ovphysx.sim.views.rst` to list and
  document `OvPhysxView` explicitly.

Generated pages derive their method lists from source docstrings. Native
PhysX and Newton method inventories remain owned by upstream documentation.

## Verification

Implementation will be verified by:

1. Comparing every snippet against the current backend implementation and its
   existing tests.
2. Running the existing pure `OvPhysxView` unit tests when the installed
   OvPhysX dependency permits them.
3. Building the public documentation with `./isaaclab.sh -d` and resolving all
   Sphinx warnings introduced by the change.
4. Running `./isaaclab.sh -f`, reviewing any formatter changes, staging only
   the intended documentation files, and running `./isaaclab.sh -f` again.
5. Checking the final diff for broken cross-references, stale PhysX-only claims,
   private sensor APIs presented as public, and copied upstream inventories.

No new runtime tests are required because the design changes documentation and
generated API-reference coverage only. Existing backend tests remain the
behavioral source of truth for the demonstrated interfaces.

## Success Criteria

- A reader can explain why the three low-level access models are not unified.
- A reader can find both the Isaac Lab-owned and caller-created raw entry path
  for each backend.
- Each backend page includes accurate discovery, read, write, synchronization,
  ownership, and invalidation guidance.
- PhysX sensor view families are covered without documenting private sensor
  members as stable API.
- Volatile native attributes and methods are discovered at runtime or delegated
  to authoritative references.
- The new guide is reachable from the backend hub, each backend page, the
  architecture guide, and the relevant migration example.
- Local Isaac Lab manager and convenience-view APIs have generated reference
  pages suitable for Sphinx cross-references.
- The documentation build and repository pre-commit checks pass.
