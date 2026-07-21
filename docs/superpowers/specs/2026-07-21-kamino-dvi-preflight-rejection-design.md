# Kamino DVI Preflight Rejection Compatibility Design

## Scope and invariant

The analyzer must treat a failed exact preflight as immutable terminal rejection
evidence when the runner correctly withheld the corresponding measured seed-42
screening run. This is an in-memory evidence projection only: no artifacts are
modified or backfilled and no GPU work is launched.

Derivation is limited to missing seed-42 Wave 1 and Wave 2 slots. Measured
evidence always wins. Baseline and every multi-seed later stage retain strict
measured coverage; no seed-43 or seed-44 record is synthesized.

## Loader architecture

`load_tuning_records` becomes a two-stream strict loader. Measured attempts and
relevant preflight attempts pass through shared checks for typed manifest
identity, directory/run identity, artifact root, locked revisions and schema,
4096 environments, configuration hash, exact command and command hash, exact
source HEAD, recorded artifact hashes, and terminal lifecycle state.

A normal preflight identity has stage `preflight`, seed 42, and exactly five
iterations. The existing canonical five-iteration special case remains
canonical-only and is never projected into a measured rejection. Preflight
attempts are grouped by candidate, configuration, seed, environment count, and
iteration count. Attempt zero has no parent; later attempts must be contiguous
and name the preceding run ID. The highest terminal attempt is authoritative.
Nonterminal, gapped, mis-parented, duplicate, stale, ambiguous, or expected-
configuration-mismatched preflight evidence raises rather than being ignored.

After both streams are validated and their highest terminal attempts selected,
reconciliation fills a missing Wave 1 or Wave 2 seed-42 measured slot only when
the authoritative exact preflight is failed. A completed preflight with no
measured record leaves a hard missing-record error. A present measured record
wins over a completed preflight and no projection is added.

## Derived record and provenance

A projected `TuningRecord` uses the actual immutable preflight `run_id`,
manifest path and hash, configuration and hash, and source HEAD. Its metrics use
the expected measured stage and seed solely for coverage and selection, contain
empty series, and carry `preflight:<failure_category>` as the rejection reason.
`bundle_git_dirty` is `None` because no completed schema bundle was consumed.

The record explicitly sets `derived_from_preflight=true` and retains the typed
original preflight identity. Shared provenance serialization exposes the
projection flag, identity, actual run ID, manifest/hash, config hash, source
HEAD, event fields, and bundle dirty value.

## Consumers and audit output

Wave 1 and Wave 2 decision computation receives one terminal record per
expected candidate. Existing metrics failure-first validation excludes the
projected failures before any 40-sample access, so `resolve_wave2` ranks only
measured successes.

Decision `source_manifests` include and strictly validate derived-preflight
provenance. Validation JSON includes deterministic derived-preflight rejection
count and source entries alongside terminal counts and rejection reasons.
Report summary source evidence and coverage retain the same fields. Funnel rows
include a derived-preflight rejection count, and Markdown/PDF render the count,
actual preflight run IDs, failure reasons, and source provenance.

## Testing and real gate

TDD covers failed preflight with no measured record, completed preflight with no
measured record, measured evidence superseding a completed preflight, exact
configuration mismatches, nonterminal evidence, retry gaps, retry parent
mismatches, Wave 2 seed-42 generalization, and refusal to synthesize later-stage
seeds. Decision, validation, summary, Markdown/PDF, and funnel propagation are
covered with deterministic assertions.

The final read-only real command validates baseline and Wave 1 together. It
must report baseline expected/terminal/valid 3 and rejected 0; Wave 1 expected
and terminal 18, valid 15, rejected 3, with the exact retained preflight
categories and provenance. `resolve-wave2` must consume those records without
requiring samples from the three failures. Focused tests, the full Kamino DVI
suite, repository hooks, and the Task 4 evidence report complete the handoff.
