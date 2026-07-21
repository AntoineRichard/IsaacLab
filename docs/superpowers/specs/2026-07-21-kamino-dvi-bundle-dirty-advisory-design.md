# Kamino DVI Bundle Dirty Advisory Design

## Scope

Align the strict tuning evidence consumer with the producer's broad
`versions.git_dirty` definition. Completed bundles must contain an actual JSON
boolean, but either value is accepted. All exact commit, command, config,
artifact, event, trace, and runner provenance checks remain unchanged.

## Record and audit schema

`TuningRecord` preserves `bundle_git_dirty` as `bool | None`; failed terminal
manifests use `None` because no completed schema bundle is consumed.

Every validate stage, adaptive decision, and final report contains this
structured disclosure:

```json
{
  "bundle_git_dirty": {
    "count": 1,
    "run_ids": ["sorted-run-id"],
    "advisory": "..."
  }
}
```

The advisory states that the bundle flag is broad because it is produced by
plain `git status --porcelain` and includes untracked paths; the runner
separately enforced tracked-only cleanliness before launch; and a true flag
does not prove that only untracked paths differed. Decisions also retain the
per-record flag in `source_manifests`. Baseline-prefix and canonical provenance
retain it per record. Markdown and its paginated PDF render the same advisory
and affected count/run IDs.

## Failure behavior and tests

A non-boolean bundle value is corrupt evidence and raises. Boolean false keeps
existing behavior; boolean true loads and validates while being disclosed.
TDD covers loader acceptance/rejection, validate JSON, decision JSON, report
summary/Markdown/PDF, and the real read-only Task 5 baseline gate. No artifacts
are changed and no GPU work is run.
