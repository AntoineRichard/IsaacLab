<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Global Actuator Collection and Typed Storage

## Status

This design was approved interactively on 2026-08-04. It supersedes the
articulation-local storage and execution-batch design currently implemented on
the actuator-collection pull request. The implementation must preserve the
develop API through deprecation shims while establishing a Newton-style global
storage model.

## Context

Isaac Lab currently treats each named actuator configuration group as both a
user-facing access boundary and an execution boundary. The pull request improves
that model by introducing an articulation-local `ActuatorCollection`, combining
compatible stateless groups, recording Warp launches, and separating actuator
commands from processed joint commands. However, it still owns universal dense
actuator fields such as stiffness, damping, and gear ratio. Those fields imply
that every actuator supports every parameter, which is false for models such as
neural actuators. The local collection also duplicates storage owned by Newton
controllers and cannot coordinate allocations across a complete simulation.

Newton provides the architectural north star: build a complete model from
registered metadata, allocate canonical flat arrays once, expose scoped views
into them, and step structurally compatible actuators together. Isaac Lab should
mirror this architecture while retaining logical group objects and letting each
articulation own actuator application for now.

## Goals

- Create one simulation-scoped actuator collection with a complete view of every
  registered articulation.
- Allocate canonical flat storage by concrete actuator type so a type owns only
  parameters that are meaningful to it.
- Use clone-plan metadata so Python construction cost scales with source
  prototypes and configured groups, never cloned worlds times DOFs.
- Preserve logical named actuator groups as concrete `ActuatorBase` objects with
  group-shaped parameter views.
- Add articulation-scoped type access keyed by exact actuator class.
- Support index and mask parameter writes using articulation-level world and
  joint-DOF selectors.
- Aggregate stateless actuators aggressively even when per-DOF parameters differ.
- Keep actuator application owned by each articulation while making future
  scene-global execution a plan-only change.
- Make staging allocation-free and pointer-stable and capture the complete
  graphable actuator sequence in one CUDA graph per articulation.
- Preserve develop behavior through real deprecation shims and avoid hot-path
  cost for compatibility surfaces that are never accessed.
- Demonstrate construction and runtime performance with pinned, repeatable
  benchmarks.

## Non-goals

- Do not provide scene-wide public parameter reads or writes in this change.
- Do not provide articulation-dense projections for new type-scoped parameters.
- Do not aggregate neural actuator execution.
- Do not introduce a new required or optional dependency.
- Do not move ownership of actuator application from articulations to a scene
  executor yet.
- Do not promise aggregation for third-party actuator classes that have not
  declared a compatible storage and execution schema.
- Do not remove develop public symbols in this release.

## Terminology

### Logical actuator group

A named entry in `ArticulationCfg.actuators`, such as `legs` or `gripper`. It is
the configuration and user-access boundary and retains its concrete actuator
class, configuration, joint names, articulation joint indices, and group-shaped
attributes.

### Actuator type

An exact concrete actuator class. Type grouping never uses inheritance. For
example, `DCMotor` and `ActuatorNetMLP` are separate types even though the latter
currently inherits common implementation from the former.

### Typed store

The canonical flat storage for one exact actuator type. Its schema contains only
the parameters, state, scratch, and output fields that the type consumes or
produces.

### Joint-domain store

Canonical flat storage whose values are inherently defined once per articulation
joint rather than once per actuator type. Raw position, velocity, and effort
commands and processed joint commands belong here. Dense effort telemetry is a
pointer-stable publication of canonical typed outputs into articulation order.

### Execution plan

An immutable articulation-owned plan containing typed ranges, joint mappings,
staging buffers, cached launches, and CUDA graph metadata. It references global
storage and owns when actuator commands are applied.

## Object Model

### Simulation-scoped collection

`ActuatorCollection` becomes a simulation-scoped internal manager associated
with the active `SimulationContext`. It owns registration, finalization, global
typed stores, joint-domain stores, and the metadata shared by articulation
execution plans. It is cleared with the simulation on stop or close.

`SimulationContext` registers collection lifecycle bridge callbacks before any
`PHYSICS_READY` event can be dispatched. Existing asset callbacks run at order
10 and register articulation metadata; the bridge finalizes the collection at
order 20. The bridge is registered eagerly even though the collection service
itself is created lazily, because event dispatch snapshots its callback list
before invoking any asset. Registration from inside an asset callback must
therefore never be relied upon to add a finalizer to the event currently being
dispatched.

The collection is not a new scene-wide public read/write API. Its public-facing
object is a nested articulation-scoped facade, conceptually
`ActuatorCollection.ArticulationView`, exposed at `articulation.actuators`.
Nested view types are appropriate because they have no useful identity outside
the owning collection.

`ActuatorCollection` remains exported from `isaaclab.actuators` as the lifecycle
manager type and extension/documentation anchor, but simulation-wide stores are
not exposed through `SimulationContext`. The public articulation annotation is
`ActuatorCollection.ArticulationView`; ordinary users obtain only that scoped
view from `Articulation.actuators`.

### Articulation-scoped facade

The facade preserves mapping behavior for logical group lookup and iteration:

```python
legs = robot.actuators["legs"]
for name, actuator in robot.actuators.items():
    ...
```

Each value remains the configured concrete `ActuatorBase` subclass. Group
membership, concrete types, configurations, joint names, joint indices, and
group-shaped attributes remain observable.

Develop currently exposes this object as a mutable `dict`, even though dynamic
membership is not a documented actuator lifecycle. Replacing it immediately
with a read-only mapping is nevertheless a behavioral break. During the
deprecation period, mutation operations emit one `DeprecationWarning`, mark the
simulation collection dirty, and schedule a full safe rebuild at the next
collection boundary. The rebuilt plan preserves configuration-order semantics.
The facade is a `dict` subclass during this transition: `isinstance(..., dict)`,
lookup/iteration/equality, `copy()`, `fromkeys()`, reverse iteration, and `|`/
reverse-`|` snapshot operations preserve dict behavior. Snapshot operations
return ordinary dictionaries. Every mutating entry point, including `|=`, is
overridden explicitly so C-level dict methods cannot bypass staged mutation.
Exact builtin identity (`type(robot.actuators) is dict`) necessarily changes in
order to expose the scoped facade and is called out in migration notes; no
first-party or documented API may depend on that identity.
The facade's logical mapping changes immediately for lookup and iteration, while
the supplied group's copied configuration is staged by articulation identity.
That staged ordered configuration survives STOP, is consumed when the same asset
registers during the following PHYSICS_READY event, and is cleared after a
successful publication. Runtime arrays from the supplied group are never carried
across generations.
After the deprecation period, membership becomes read-only and topology changes
require rebuilding the simulation explicitly. No mutation checks run on the
ordinary steady-state path.

Mutation while the simulation is active makes subsequent execution raise until
the next `STOP` to `PHYSICS_READY` boundary rebuilds the collection. A topology
rebuild advances the facade generation and invalidates previously retained group,
type, parameter, and data views; pointer stability is guaranteed within one
finalized generation, not across deprecated topology mutation. Facade operations
on an old generation raise. A raw Torch or Warp tensor retained across the
documented rebuild boundary cannot be guarded and must be reacquired.

Whole-attribute replacement on a managed group, for example
`actuator.stiffness = values`, copies into canonical storage through a descriptor
instead of detaching the group object. Existing in-place writes continue to
modify canonical storage directly.

### Type-scoped facade

The articulation facade exposes a class-keyed mapping:

```python
pd = robot.actuators.by_type[IdealPDActuator]
```

Only exact concrete class keys are accepted. There is no string alias registry:
it would introduce naming policy, collisions, and maintenance work for custom
classes. `by_type.keys()` exposes the available managed classes.

A type view exposes a contiguous compact parameter block, logical group slices,
articulation joint names, articulation joint indices, and parameter mutation
methods. Joint names/indices have one entry per compact group slot and therefore
retain duplicates when same-type groups overlap. A custom actuator without a storage schema remains available as a
logical group and follows its existing per-group execution path; it is not
silently granted a misleading typed parameter API.

## Registration and Finalization

### Registration phase

Every articulation registers metadata during its existing physics-ready
initialization. Registration records:

- articulation identity and backend binding;
- number of worlds and joints;
- logical group names and configuration order;
- exact actuator classes and declared schemas;
- group-to-articulation joint mappings;
- source-prototype parameter values;
- clone-plan source and clone-row assignments;
- structural execution signatures for stateful or native models; and
- existing backend or Newton controller bindings.

Registration does not allocate per-group runtime arrays, create Python objects
per cloned world, traverse every cloned USD prim, or synchronize the device.

### Clone-aware metadata

For a homogeneous 4096-world scene, Python processes one source prototype and
its configured groups. A device-resident assignment vector expands those values
to the cloned world rows. For a heterogeneous scene, Python processes each
source prototype once and uses the clone plan's source-to-clone assignments.

Regular expression resolution and USD property resolution happen at the source
prototype level. Generic device kernels expand resolved prototype values into
canonical storage. Kernel specialization must not depend on the number of
groups, worlds, or prototypes.

### Finalization phase

The global collection registers a deterministic physics-ready callback after
asset initialization. Finalization:

1. Validates schemas, group mappings, clone assignments, devices, dtypes, and
   backend bindings.
2. Groups registrations by exact concrete actuator class.
3. Computes prefix sums for typed and joint-domain offsets.
4. Allocates every canonical field, mapping, staging array, scratch array, and
   output array once.
5. Expands source-prototype values with a bounded number of device launches.
6. Constructs or rebinds logical group objects to their canonical views.
7. Constructs articulation and type facades.
8. Binds Newton controller arrays and backend interfaces to canonical storage.
9. Builds immutable articulation execution plans and recorded-launch objects.
10. Freezes topology and warms graphable modules before CUDA capture.

Finalization is transactional: allocations, offsets, facades, backend bindings,
and execution plans are prepared off to the side from private candidate-binding
records and published only after every registration validates. Candidate builders
never dereference a guarded pending public facade. After the atomic publication,
controls bind their data objects to the now-live facades and then complete
articulation initialization; failure in any bind or completion
unpublishes the candidate, invalidates every control including ones that already
completed, restores their assets/data to uninitialized and unprimed state, and
releases the whole candidate generation before the `PHYSICS_READY` callback
raises. Pending and rolled-back facades remain unusable.
An asset is not considered actuator-ready merely because its order-10 asset
callback completed; command access and execution require the successfully
published order-20 collection generation.

Late registration or deprecated mapping mutation sets a rebuild requirement.
The rebuild happens only at an explicit safe collection boundary; attempting to
use a dirty plan before that boundary raises rather than executing stale
pointers. A late registration receives a pending facade and joins the next
generation; it never triggers an in-place allocation or publication while an
active generation may be executing.

## Storage Layout

### Typed parameter storage

Each exact actuator type declares a field schema containing name, dtype, units,
mutability, initialization rule, backend side effect, and execution use. A
neural type therefore does not allocate stiffness or damping merely because a
legacy base class currently exposes those names.

Schema fields distinguish four storage roles:

- per-world/per-DOF values, which use the flat layout below;
- per-world state, such as a delay cursor, owned by a stateful execution segment;
- immutable structural data, such as a network, checkpoint, or lookup-table
  shape, which participates in a stateful execution signature; and
- pointer-stable scratch and output arrays, which are allocated at finalization
  but are not user parameters.

Existing fields have the following ownership. This table is exhaustive for the
built-in Lab actuator hierarchy and prevents solver properties from leaking back
into universal actuator storage:

| Existing field | Canonical owner |
| --- | --- |
| `stiffness`, `damping` | Typed parameters for implicit and analytical PD actuators; implicit writes also update solver drives. Neural access uses deprecated group-local sidecars. |
| `effort_limit` | Typed parameter for actuator classes that clip or expose actuator effort limits. |
| `velocity_limit` | Typed parameter for classes that expose rated/soft velocity or consume it in clipping. |
| `saturation_effort` | Typed parameter for `DCMotor` and neural models that use DC-motor clipping. |
| Type-specific lookup tables, network metadata, delay state, and recurrent/history state | Structural metadata or state owned by the corresponding eager execution segment. |
| `computed_effort`, `applied_effort` | Typed output storage. |
| `effort_limit_sim`, `velocity_limit_sim`, `armature`, `friction`, `dynamic_friction`, `viscous_friction` | Solver joint properties. Existing group attributes remain compatibility views but these fields are never allocated in typed actuator execution storage. |
| `gear_ratio` | Typed only for an exact actuator class that declares and consumes it; otherwise supplied solely by the deprecated dense projection's `1.0` fill. |

The existing group attributes for solver-only fields remain `torch.Tensor`
compatibility values outside typed execution storage. They allocate lazily per
group on first access, initialize from resolved solver values, and are refreshed
by the existing solver-property writer hooks. They do not participate in actuator
execution and do not warn; this retains the existing group surface without
reintroducing a universal eager actuator buffer.

Numeric values that can be evaluated elementwise are expanded into per-DOF
arrays and never split a stateless execution range. Only non-flattenable
structural data may split a stateful range. This distinction prevents a scalar
configuration detail from accidentally becoming an execution signature.

Each field uses one global flat allocation. Articulation blocks are placed in
registration order. Within one articulation block, storage is row-major by
world and then by compact type-local DOF:

```text
[Art1 world0: group1 | group2 | group3]
[Art1 world1: group1 | group2 | group3]
...
[Art2 world0: group1 | group2 | group3]
...
```

An articulation/type block is a contiguous two-dimensional view with shape
`[num_worlds, num_type_dofs]`. Each logical group occupies a contiguous column
range inside that block. A group view is zero-copy and regularly strided; it is
not guaranteed Torch- or Warp-contiguous when other type-compatible group
columns follow it. Execution uses contiguous articulation/type blocks and never
relies on public strided group views.

Each type layout stores one compact articulation-joint ID per slot plus an
immutable CSR articulation-to-compact fanout. The offset array has
`num_joints + 1` entries and the slot array contains every matching compact slot
in configuration order. Same-type overlap is therefore represented one-to-many,
not collapsed into an ambiguous lookup.

This is physical flat storage even when exposed through shaped Torch, Warp, or
`ProxyArray` views. There are no Python lists proportional to world × DOF slots.
Canonical device allocations are Warp-owned so native Newton controllers and
Warp kernels bind them directly. Torch aliases are created once through
zero-copy interoperation for existing Lab actuator code; neither representation
is synchronized through copies at runtime.

### Joint-domain storage

Commands are articulation-joint concepts, not actuator-specific parameters.
The global collection therefore owns flat articulation-major stores for:

- actuator input position, velocity, and effort commands;
- processed position, velocity, and effort joint commands;
- published computed effort; and
- published applied effort.

Each articulation receives a contiguous `[num_worlds, num_joints]` view. This
preserves the develop contract that `data.joint_*_target` is a live mutable
`ProxyArray`; arbitrary mutation of a raw Torch or Warp view cannot be
intercepted reliably. The develop data fields and the new
`robot.actuators.command` facade alias the same canonical memory.

Typed execution staging is separate, preallocated memory populated by a fused
gather. It is not another authoritative command store. Type-scoped public
parameter views are contiguous, but the initial API does not expose type-scoped
command views whose freshness would depend on staging. Processed joint commands
remain joint-domain data because they are submitted to a joint-oriented backend.

Per-type computed and applied effort buffers are the canonical actuator outputs.
The final fused scatter publishes them into persistent joint-domain telemetry
buffers, preserving existing dense reward and observation consumers and the
stable pointers they retain. These publication buffers are derived execution
outputs, not universal actuator-parameter storage.

### Lazy compatibility projections

Actuator-specific dense develop fields that have no universal meaning are not
part of joint-domain canonical storage:

- `data.soft_joint_vel_limits` is projected from each compatible group's
  `velocity_limit`; uncovered columns retain develop's zero fill.
- `data.gear_ratio` is projected only from actuator types that own gear ratio;
  unsupported columns retain develop's `1.0` fill.

These buffers allocate on first access, retain a stable pointer thereafter, and
emit a once-only `DeprecationWarning`. Canonical setters update activated
projections, and execution refreshes activated dynamic projections at the next
collection boundary. A projection activated before CUDA capture is refreshed in
that graph; one activated afterward gets a pointer-stable cached refresh launch
after graph replay without invalidating or recapturing the actuator graph. Direct
group-tensor mutation becomes visible in a retained projection at that boundary,
matching the synchronization limits of today's duplicated buffers without
charging users who never access them.

## Parameter API

### Group access

Managed concrete group classes expose their meaningful fields directly and
through a generic mapping:

```python
legs = robot.actuators["legs"]
legs.parameter_names
legs.stiffness
legs.parameters["stiffness"]
```

The direct field descriptors retain their develop types, including
`torch.Tensor` for existing group attributes. Generic `parameters[...]` entries
return a lightweight dual Warp/Torch view. Both representations address the
same group-shaped storage and preserve existing in-place and whole-attribute
mutation behavior. Group metadata includes articulation joint names and
group-to-articulation indices.

### Type access

```python
pd = robot.actuators.by_type[IdealPDActuator]
pd.parameters["stiffness"]
pd.joint_names
pd.joint_indices
pd.group_slices
```

All type parameter reads use the explicit `parameters` mapping and return
compact, contiguous dual Warp/Torch views. This avoids pretending that a generic
type-view class has statically discoverable attributes for every custom schema;
`parameter_names` makes supported fields inspectable, while the concrete logical
group remains the autocomplete-friendly typed API. The type API intentionally
omits a `compact` flag and articulation-dense return. Develop has no such
actuator parameter getter to preserve, heterogeneous-learning workflows benefit
from flat compact data, and a dense return would require allocation, scatter,
fill semantics, and a validity mask. A future explicit copying utility can be
added without changing this contract if a real use case emerges.

### Index and mask setters

Group and type views provide parallel string-based setter families:

```python
pd.set_parameter_index(
    "stiffness",
    value,
    env_ids=env_ids,
    joint_ids=joint_ids,
)

pd.set_parameter_mask(
    "stiffness",
    value,
    env_mask=env_mask,
    joint_mask=joint_mask,
)
```

Selectors use articulation coordinates:

- `env_ids` and `env_mask` address articulation-local worlds;
- `joint_ids` and `joint_mask` address articulation DOFs;
- `None` selects every valid world, and for joints selects every compact slot in
  the group/type scope;
- index values correspond to the supplied selector order, including entries
  outside the current scope, which are ignored in the normal path; and
- mask values are scalar/broadcastable or full compact scope-shaped arrays.

Index setters form the Cartesian product of `env_ids` and `joint_ids`. Values
may be a scalar, a one-dimensional value with one entry per supplied joint ID,
or a two-dimensional value with shape `[len(env_ids), len(joint_ids)]`; the
one-dimensional form broadcasts across selected worlds. Filtering out a joint
that is outside the group or type scope also filters the matching value column,
so the remaining values never shift position. Negative and out-of-range world
or articulation-joint IDs are ignored in the normal path. Duplicate IDs use the
last supplied value deterministically on both CPU and GPU; debug mode rejects
duplicates instead.

For a type view, each explicit articulation joint ID fans its corresponding value
out to every matching compact slot. With `joint_ids=None`, compact slots are
selected directly in stable group/configuration order and non-scalar values have
one column per compact slot, allowing overlapping slots to receive different
values. Group mappings remain one-to-one.

Masks are boolean arrays with full articulation shapes `[num_worlds]` and
`[num_joints]`. Values may be scalar, compact one-dimensional
`[num_scope_dofs]`, or compact two-dimensional
`[num_worlds, num_scope_dofs]` arrays and broadcast in the missing dimension.
The articulation joint mask gates every matching compact slot; compact value
columns remain slot-specific even when several columns name the same joint.
An empty index selection or an all-false mask is a no-op and performs no backend
state change. A statically empty index selection returns before launch. An
all-false device mask may execute masked no-op canonical and backend kernels
because detecting it on the host would synchronize. Both modes synchronously
validate host-visible rank, shape, dtype category, device, and documented value
broadcastability before launching. Debug mode additionally validates selector
bounds, ownership, and duplicate-free index values; the normal path retains the
same selection and broadcasting semantics without synchronizing for diagnostic
reporting.

Unknown parameter names always raise synchronously. Unsupported ranks or dtypes,
wrong-device arrays, and non-broadcastable value shapes also raise from metadata
alone and never reach a kernel. The normal fast path ignores out-of-scope
articulation DOFs without warning or device synchronization. An optional debug
mode, off by default, synchronizes only for value-dependent selector bounds,
ownership, and duplicate diagnostics. CPU and GPU use the same resolution
semantics. Same-device Torch/Warp arrays are allocation-free; accepted Python
sequences are a compatibility conversion path that may allocate and transfer.

Setters mutate canonical storage in place. A field whose value is also consumed
by a backend performs the required side effect in the same operation. In
particular, implicit stiffness and damping updates write solver drive properties.
Deprecated develop write methods delegate to these setters.

For overlapping backend-owned parameters, finalization precomputes the last
configuration-order owner slot per `(field, articulation joint)`. A group setter
changes backend state only when it addresses that owner; a type setter fans out
first and routes the owner slot's resulting value. Canonical overlapping slots
may differ without leaving the solver drive inconsistent with the effective
configuration-order owner.

## Public Terminology and Compatibility

The new API uses actuator terminology for actuator inputs and effort terminology
for quantities that may be either linear force or torque:

```python
robot.actuators.command.set_position_index(...)
robot.actuators.command.position
robot.actuators.joint_command.effort
robot.actuators.computed_effort
robot.actuators.applied_effort
```

The following develop APIs remain functional and issue a once-only
`DeprecationWarning` with migration guidance:

- `set_joint_position_target*`, `set_joint_velocity_target*`, and
  `set_joint_effort_target*`;
- `data.joint_pos_target`, `data.joint_vel_target`, and
  `data.joint_effort_target`;
- `data.computed_torque` and `data.applied_torque`;
- `data.soft_joint_vel_limits`; and
- `data.gear_ratio`.

The develop articulation writers `write_actuator_stiffness_to_sim()` and
`write_actuator_damping_to_sim()` also remain functional. They delegate to the
canonical parameter setters, preserve native-controller side effects, and warn
in favor of group- or type-scoped `set_parameter_index()`. Solver-property
writers such as `write_joint_stiffness_to_sim_index()` remain distinct and are
not deprecated or redirected.

The target and effort aliases point at canonical joint-domain storage, so held
references remain live. The actuator-specific limit and gearing properties use
the lazy projection behavior described above.

First-party rewards, observations, tools, and tasks migrate from deprecated
torque aliases to effort names. The Cartpole and Spot algorithms that genuinely
need articulation-dense soft limits use one private warning-free data helper for
the same lazy compatibility projection; this is not a new public non-compact
actuator API. Normal first-party execution therefore does not emit deprecation
warnings merely because internal code retained an old spelling.

`data.joint_stiffness` and `data.joint_damping` remain solver joint-drive
properties. They are neither deprecated nor reinterpreted as explicit actuator
gains.

The pull-request-only collection-wide `actuator_stiffness`,
`actuator_damping`, `soft_joint_vel_limits`, `gear_ratio`, `computed_torque`,
and `applied_torque` names have not shipped. They may be removed or replaced by
the semantically correct API without a deprecation cycle.

Neural actuators currently inherit unused stiffness and damping attributes from
`DCMotor`. Removing those public attributes is breaking. During the deprecation
period, accessing them creates a group-local compatibility sidecar, emits a
once-only warning, and has no execution effect. Capability-aware utilities such
as gain randomization do not access or allocate these sidecars. A later release
may remove the meaningless fields after the required deprecation period.

The already-exported Newton helper `build_newton_actuator_defaults()` remains
callable with its existing result contract and becomes deprecated. Collection-
managed backend integration no longer uses its duplicated snapshots and instead
binds canonical storage directly; removing the exported helper requires a later
release.

## Execution Architecture

### Articulation-owned plan

Every finalized articulation owns one immutable execution plan that references
global storage. `write_data_to_sim()` remains the lifecycle entry point and runs:

```text
joint-domain commands + backend joint state
                    |
             fused Warp gather
                    |
       contiguous typed execution ranges
        |              |             |
   Implicit Warp   IdealPD Torch   DCMotor Torch
        |              |             |
        +--------------+-------------+
                    |
             fused Warp scatter
                    |
 joint commands + effort telemetry + backend submit
```

The gather reads raw commands and current joint position and velocity into
fixed, type-ordered execution slots. Typed ranges execute their models. The
scatter routes processed commands and effort outputs back to articulation joint
order, updates activated compatibility projections, and prepares backend
submission.

### Stateless aggregation

All groups of the same exact supported stateless class execute in one typed
range per articulation regardless of numeric parameter differences. Stiffness,
damping, effort and velocity limits, saturation effort, gear ratio, and similar
values are per-slot arrays and do not participate in the execution signature.
Logical group names affect configuration and access only.

The initial supported policies are:

- `ImplicitActuator`: one allocation-free Warp path over all articulation
  implicit-actuator slots;
- `IdealPDActuator`: one allocation-free Torch range per articulation;
- `DCMotor`: one allocation-free Torch range per articulation; and
- neural, delayed, remotized, custom, and other stateful actuators: one eager
  logical-group segment initially.

Overlapping stateless logical groups may still share computation. The builder
precomputes separate articulation-joint owner maps for position command,
velocity command, effort command, computed effort, and applied effort. Each map
considers only groups whose ordinary result supplies that field and then applies
develop's configuration-order last-writer semantics. This preserves mixed
overlaps where a later explicit actuator replaces effort but does not supply or
erase an earlier implicit position/velocity target. Stateful or otherwise
order-dependent overlaps remain separate ordered segments.

Neural execution is never merged in this change. A future opt-in structural
descriptor may allow compatible stateful or neural batching, but absence of
that descriptor must preserve the existing per-group path.

### Allocation-free staging

Finalization allocates command/state staging, output staging, type-specific
scratch, routing maps, and backend staging once. Built-in Torch actuator
operations must use stable output and scratch buffers rather than replacing
tensors or allocating temporaries on every step. Logical group output attributes
remain views into current canonical output storage.

Every Warp launch has stable input and output pointers and a cached `wp.Launch`
object. Parameter updates mutate values without rebinding arrays. Only topology,
schema, exact class, device, or joint-map changes require rebuilding execution
plans.

That recorded-launch guarantee applies to manager-owned execution/staging arrays.
Parameter setters receiving arbitrary user arrays use ordinary `wp.launch` unless
the inputs are first copied into existing manager-owned staging; transient user
pointers are never used as keys in an unbounded launch cache or retained by a
recorded launch.

### CUDA graph boundary

After module warm-up and complete allocation, CUDA captures the full graphable
actuator sequence for an articulation:

```text
gather -> all graphable stateless type computations -> scatter/telemetry
```

Pure Warp paths use Warp graph capture. Mixed Torch/Warp paths use the same CUDA
stream and must pass an explicit interoperability test before capture is enabled.
Cached launches remain the eager fallback. A fully graphable articulation uses
one graph containing gather, every stateless computation, and the final scatter.
Compatibility projections already active at capture time may be included. A
projection activated later appends a cached refresh epilogue; that execution is
reported as a full actuator graph plus compatibility epilogue, not as a complete
single-replay sequence.
For a mixed plan, one prefix graph may contain gather and all independent
graphable stateless computation, followed by eager neural/stateful segments and
a cached final scatter. The mixed plan therefore does not claim a completely
captured sequence. Stateful graphable implementations may use alternating
graphs for ping-pong state, following the existing Newton adapter pattern.

The CPU uses the same storage, routing, and execution plan eagerly; CUDA graph
replay is an optimization layer, not a different semantic strategy.

### Newton and backend ownership

For PhysX, OVPhysX, and PhysX-hosted Newton actuators, the articulation plan
computes and submits commands through the backend bridge. Newton controller
arrays bind to canonical typed storage, eliminating Lab/controller/dense-snapshot
duplication and runtime synchronization.

Native Newton remains physically stepped by its simulation-global manager. The
articulation still owns scoped command staging, bindings, and telemetry; its
plan registers the relevant ranges with Newton, excludes those ranges from its
Lab-owned physical compute lists, and never creates a competing per-articulation
physical executor.

Future scene-global execution replaces iteration over articulation plans with
iteration over global typed ranges and existing range tables. No storage,
logical group, facade, or public API migration is required.

## Error Handling and Debugging

- Unknown group names and exact type keys raise `KeyError`.
- Unknown parameter names raise immediately.
- Schema mismatch, device mismatch, invalid clone metadata, incompatible backend
  binding, or failed built-in aggregation raises during finalization with the
  articulation, type, and logical groups identified.
- A dirty or stale topology raises before execution; it never silently executes
  old pointers.
- Unsupported custom types follow their preserved per-group path rather than
  silently pretending to use unified typed storage.
- Selector validation is synchronization-free and permissive in normal mode;
  optional debug mode synchronizes only when required to report and raise a
  precise error.
- Production Warp kernels contain no `wp.printf` calls.

## Construction Performance Contract

Construction benchmarks compare the pull-request merge-base
`378225f8d2af0a9920e18a934ee7d044844e023e`, the current optimized pull-request
head `5c59a092be11e4c95d63195476f58e7d2b0b8084`, and the final implementation.

The matrix covers:

- 1, 64, and 4096 worlds;
- one homogeneous clone source and four heterogeneous sources;
- 1, 3, and 12 logical groups with deliberately different parameters;
- homogeneous stateless types and mixed implicit, ideal-PD, and DC-motor types;
- one and multiple articulations sharing the global collection; and
- cold first construction and warm repeated construction.

The empty-collection lifecycle row is candidate-only: it terminates after empty
finalize/clear and has no fabricated command application or develop/current
timing. Cross-revision construction comparisons start with rows that contain an
articulation and use the common construction-through-first-application boundary.

Every cold construction is the first collection construction in a fresh child
process after a separate compile-only cache prewarm; an `all` matrix coordinator
must not measure several nominally cold rows in one process. Warm construction
uses separate children with ten unmeasured constructions followed by 100 measured
constructions. Scalar dimension overrides are rejected when the frozen `all`
matrix is requested.

Metrics include Python registration time, finalization wall time, CPU and GPU
allocation counts and bytes, initialization launch count, host-to-device bytes,
peak memory, and time to first actuator application.

Structural counters have fixed candidate-only definitions: manager descriptors
count the unique registration/resolution/binding/store/facade/plan/range/segment
objects reachable from the generation; canonical allocations deduplicate owning
Warp arrays by `(device, ptr)` and exclude aliases/projections; canonical bytes
sum allocation capacity times dtype size; initialization launches are benchmark-
wrapped `wp.launch`/`wp.launch_tiled` calls inside the construction boundary; and
pointer replacement compares ordered field pointers after finalization and every
warm compute. Older revisions may report unavailable structural fields as null.
Cross-revision performance uses only the common construction-through-first-
application timing boundary.

Hard candidate structural gates are:

- no Python loop or object count proportional to cloned worlds or world × DOF
  slots;
- no device-to-host synchronization;
- allocation count determined by storage types and fields, not logical groups
  or worlds;
- source-prototype registration cost independent of clone count;
- a bounded number of generic initialization launches; and
- no allocation or pointer replacement after finalization.

Isolated finalization must not regress relative to the current pull-request head
by more than `max(5%, 0.25 ms)` in a primary matrix case. Full-scene startup is
reported separately so USD, physics, or shader startup variance cannot hide a
collection regression.

## Runtime Benchmark Contract

### Actuator-only benchmark

A focused temporary benchmark uses 1, 3, and 12 logical groups with different
parameters for each supported stateless class. It runs 100 warm-up applications
and 10,000 measured applications, uses CUDA events, compares eager cached-launch
and CUDA-graph modes, and records steady-state allocations and dispatch counts.

Execution capability is revision-specific. Develop and the current PR snapshot
provide cached-eager rows; only the global candidate provides the new full graph
row. Every record stores requested and effective mode. Unsupported graph rows
for older revisions contain a reason and no timing rather than relabeled eager
measurements. The primary comparison is global graph versus current cached-eager;
global cached-eager versus current cached-eager is a secondary attribution, and
global graph versus develop cached-eager is historical context.

Each actuator-type/group comparison uses six independent process-level pairs in
balanced order. Confidence intervals bootstrap paired process medians; the 10,000
steps within one process describe its distribution but are not treated as 10,000
independent observations. A fixed five-second, 250 ms cadence telemetry window
before and after each pair records temperature, utilization, clocks, throttle
reasons, and compute PIDs. Contaminated attempts are retained with rejection
reasons and rerun under a new immutable attempt ID.

Success requires:

- zero-tolerance equality with independent execution for supported stateless
  actuators;
- zero steady-state allocation;
- one computation per exact stateless type independent of group count;
- one graph replay for the complete sequence of a fully graphable articulation;
- no regression for the single-group case; and
- a measurable improvement over the current pull-request head for the 3- and
  12-group cases.

Artifacts are stored under
`runs/<candidate-sha>/observations/<observation-key>/attempt-XX`; attempts are
allocated atomically per observation and are never overwritten or deleted.
Coordinator batch IDs are separate from result attempts, so a contaminated row
cannot collide with a later matrix. A paired observation key includes matrix,
logical row, comparison/mode pair, and pair ID/order, but no member-specific
revision. One atomic attempt owns both members and shared telemetry; member keys
beneath it carry revision plus requested/effective mode. The accepted-attempt
manifest requires six distinct balanced pair IDs per paired row and one attempt
for singleton structural or declared-unsupported rows. Summarization rejects dirty or
mixed candidate SHAs, duplicate complete observation keys, missing/repeated pair
IDs, unbalanced order, missing supported rows, timing on unsupported rows, or
missing attempt references. Stable report files are generated only from this
manifest, so rejected attempts remain available without contaminating results.

### Franka Reach task benchmark

The end-to-end task benchmark uses PhysX, 4096 environments, seed 42, headless
execution, the low-overhead schema formatter, 100 warm-up steps, and 1000
measured steps. At least six runs execute every permutation of the three
revisions once; twelve runs in two complete counterbalanced blocks are preferred.
The same fixed-window telemetry and immutable rejection protocol guards every
counterbalanced triplet. Median step latency and throughput are primary; means
and dispersion are also reported.

The previous three-run comparison measured a 14.92 ms median step at the
merge-base, 12.91 ms at the intermediate graph-staging snapshot
`77b77692aa78460d46759c032971dd74457f8521`, and 14.08 ms at the current
pull-request head. The current head's three-run mean was 13.64 ms versus 14.88 ms
at the merge-base, an 8.35% mean latency reduction; the small sample variance is
too high for a final claim. The final report reruns all pinned comparison
revisions under identical conditions rather than combining historical and new
measurements.

The final implementation must not regress against the current pull-request head
and must preserve at least a 10% median latency improvement over the merge-base.
Any additional gain is reported even if small. One Nsight Systems capture
verifies that changes in dispatch, staging, or actuator kernels explain the
result.

## Verification Strategy

Focused tests cover:

1. Clone-aware registration produces the same canonical values for homogeneous
   and heterogeneous source assignments.
2. Multiple articulations receive disjoint correct views into global storage.
3. Group views retain exact concrete classes, metadata, shapes, writable aliasing,
   and whole-attribute copy behavior.
4. Type views are class-keyed, compact, contiguous, and omit unsupported fields.
5. Index and mask setters resolve both world and articulation joint selectors;
   type setters fan out to every overlapping compact slot and backend side
   effects use the effective configuration-order owner.
6. Normal out-of-range selector values are ignored without synchronization;
   malformed metadata raises before launch, while debug mode also raises for
   value-dependent invalidity and duplicates.
7. Implicit backend side effects and explicit direct writes update the correct
   canonical storage.
8. Neural compatibility fields warn and remain execution-inert.
9. Legacy command, telemetry, soft-limit, and gear-ratio properties preserve
   shape, fill, pointer, and warning behavior.
10. Ideal-PD, DC-motor, and implicit aggregation is exactly equal to independent
    execution with deliberately different parameters.
11. Field-specific overlap owner maps preserve configuration-order output
    behavior for mixed implicit/explicit results.
12. Stateful, neural, and opaque custom actuators retain per-group execution.
13. Newton controllers bind canonical arrays without duplicate synchronization,
    and native groups are physically stepped only once by `NewtonManager`.
14. Eager cached launches and captured graphs produce exactly identical outputs.
15. Allocation and dispatch instrumentation confirms the structural performance
    gates.
16. Physics-ready finalization, rollback after a later control-completion failure,
    stop/restart, close, and second-context creation leave no initialized asset,
    stale registration, or usable stale view behind.
17. Deprecated mapping mutation dirties execution, rebuilds only at the safe
    boundary, advances the facade generation, and documents invalid held tensors.

Regression tests must be observed failing before the corresponding implementation
fix and passing afterward. Performance thresholds remain in the final benchmark
rather than introducing flaky wall-clock unit tests.

Before every commit and push, `./isaaclab.sh -f` runs and any modifications are
reviewed and restaged. Focused unit and backend tests run through
`./isaaclab.sh -p`. Public documentation is regenerated with
`./isaaclab.sh -d` after API documentation changes.

## Documentation and Release Notes

The actuator concept documentation will explain:

- global canonical storage versus articulation-scoped access;
- logical groups versus type execution ranges;
- compact type access and articulation-indexed setters;
- actuator commands versus processed joint commands;
- solver joint properties versus actuator parameters;
- deprecation migrations; and
- graphing and aggregation as implementation details.

Public API references and stubs are updated for the nested scoped facade and
type view. Each touched package receives a changelog fragment using the correct
bump tier and migration guidance. `CHANGELOG.rst` and package versions are not
edited directly.

The existing external LaTeX presentation document is updated with the final
architecture, benchmark tables, and profiler evidence, and its PDF is regenerated
at a persistent workstation path. The LaTeX source and PDF remain uncommitted as
previously requested. The pull-request description receives the same concise
architecture and performance summary.

## Alternatives Considered

### Keep one articulation-local collection

This is the smallest continuation of the current pull request, but it duplicates
Newton controller storage, repeats allocations and Python work, and cannot support
future heterogeneous scene-global execution without moving data again.

### Keep universal articulation-dense actuator parameters

Dense stiffness, damping, and gear-ratio arrays are convenient for legacy access
but assert capabilities that some actuator types do not have. They also require
fill semantics and synchronization. Typed canonical stores plus lazy deprecated
projections preserve compatibility without making the semantic lie permanent.

### Expose both compact and articulation-dense type reads

An articulation-dense type return has no develop contract, is never zero-copy,
requires validity and fill policy, and adds substantial synchronization surface.
Compact-only type access is sufficient and additive utilities can be introduced
later if justified.

### Aggregate only groups with identical numeric parameters

This leaves most performance on the table. Stateless actuator math is elementwise
and its numeric parameters naturally live in per-slot arrays. Exact concrete type
and structural state requirements determine compatibility; numeric values do not.

### Move execution to the scene immediately

Scene-global execution is the eventual optimum, but changing lifecycle ownership
and storage simultaneously raises integration risk. Articulation-owned immutable
plans deliver the storage and aggregation gains now and deliberately preserve a
straightforward future handoff.
