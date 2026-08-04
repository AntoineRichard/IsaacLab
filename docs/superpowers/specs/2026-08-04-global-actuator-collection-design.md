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

The collection is not a new scene-wide public read/write API. Its public-facing
object is a nested articulation-scoped facade, conceptually
`ActuatorCollection.ArticulationView`, exposed at `articulation.actuators`.
Nested view types are appropriate because they have no useful identity outside
the owning collection.

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
After the deprecation period, membership becomes read-only and topology changes
require rebuilding the simulation explicitly. No mutation checks run on the
ordinary steady-state path.

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
methods. A custom actuator without a storage schema remains available as a
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

Late registration or deprecated mapping mutation sets a rebuild requirement.
The rebuild happens only at an explicit safe collection boundary; attempting to
use a dirty plan before that boundary raises rather than executing stale
pointers.

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
collection boundary. Direct group-tensor mutation becomes visible in a retained
projection at that boundary, matching the synchronization limits of today's
duplicated buffers without charging users who never access them.

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
- `None` selects every valid world or DOF in the group/type scope;
- index values correspond to the supplied selector order, including entries
  outside the current scope, which are ignored in the normal path; and
- mask values are scalar/broadcastable or full compact scope-shaped arrays.

Unknown parameter names always raise synchronously. The normal fast path ignores
out-of-scope articulation DOFs without warning or device synchronization. An
optional debug mode, off by default, validates shape, dtype, device, selector
bounds, ownership, and parameter capability and raises on any violation. CPU and
GPU use the same resolution semantics.

Setters mutate canonical storage in place. A field whose value is also consumed
by a backend performs the required side effect in the same operation. In
particular, implicit stiffness and damping updates write solver drive properties.
Deprecated develop write methods delegate to these setters.

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

The target and effort aliases point at canonical joint-domain storage, so held
references remain live. The actuator-specific limit and gearing properties use
the lazy projection behavior described above.

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
precomputes a per-output articulation-joint owner map that reproduces develop's
configuration-order last-writer semantics in the fused scatter. Stateful or
otherwise order-dependent overlaps remain separate ordered segments.

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

### CUDA graph boundary

After module warm-up and complete allocation, CUDA captures the full graphable
actuator sequence for an articulation:

```text
gather -> all graphable stateless type computations -> scatter/telemetry
```

Pure Warp paths use Warp graph capture. Mixed Torch/Warp paths use the same CUDA
stream and must pass an explicit interoperability test before capture is enabled.
Cached launches remain the eager fallback. Non-graphable neural or stateful
segments execute eagerly without disabling capture for independent stateless
segments. Stateful graphable implementations may use alternating graphs for
ping-pong state, following the existing Newton adapter pattern.

The CPU uses the same storage, routing, and execution plan eagerly; CUDA graph
replay is an optimization layer, not a different semantic strategy.

### Newton and backend ownership

For PhysX, OVPhysX, and PhysX-hosted Newton actuators, the articulation plan
computes and submits commands through the backend bridge. Newton controller
arrays bind to canonical typed storage, eliminating Lab/controller/dense-snapshot
duplication and runtime synchronization.

Native Newton remains physically stepped by its simulation-global manager. The
articulation still owns scoped command staging, bindings, and telemetry; its
plan registers the relevant ranges with Newton rather than creating a competing
per-articulation physical executor.

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

Metrics include Python registration time, finalization wall time, CPU and GPU
allocation counts and bytes, initialization launch count, host-to-device bytes,
peak memory, and time to first actuator application.

Hard structural gates are:

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

Success requires:

- zero-tolerance equality with independent execution for supported stateless
  actuators;
- zero steady-state allocation;
- one computation per exact stateless type independent of group count;
- one graph replay for the complete graphable articulation sequence;
- no regression for the single-group case; and
- a measurable improvement over the current pull-request head for the 3- and
  12-group cases.

### Franka Reach task benchmark

The end-to-end task benchmark uses PhysX, 4096 environments, seed 42, headless
execution, the low-overhead schema formatter, 100 warm-up steps, and 1000
measured steps. Seven counterbalanced runs per revision reduce order and thermal
bias. Idle and temperature guards reject contaminated runs. Median step latency
and throughput are primary; means and dispersion are also reported.

The previous three-run comparison measured 14.92 ms per step at the merge-base
and 12.91 ms per step at the current pull-request head, a 13.4% latency reduction
and 15.5% throughput improvement. The final report reruns all three pinned
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
5. Index and mask setters resolve both world and articulation joint selectors.
6. Normal invalid selectors are ignored, while debug mode raises.
7. Implicit backend side effects and explicit direct writes update the correct
   canonical storage.
8. Neural compatibility fields warn and remain execution-inert.
9. Legacy command, telemetry, soft-limit, and gear-ratio properties preserve
   shape, fill, pointer, and warning behavior.
10. Ideal-PD, DC-motor, and implicit aggregation is exactly equal to independent
    execution with deliberately different parameters.
11. Overlap owner maps preserve configuration-order output behavior.
12. Stateful, neural, and opaque custom actuators retain per-group execution.
13. Newton controllers bind canonical arrays without duplicate synchronization.
14. Eager cached launches and captured graphs produce exactly identical outputs.
15. Allocation and dispatch instrumentation confirms the structural performance
    gates.

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
