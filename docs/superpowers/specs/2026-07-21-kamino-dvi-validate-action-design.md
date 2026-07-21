# Kamino DVI Validate Action Design

## Scope

Add a read-only `validate` subcommand to `benchmarks.kamino_dvi.analyze_tuning`.
It validates one or more measured tuning stages through the existing strict
manifest loader and exact coverage checker. It does not run training or mutate
artifacts or decisions.

## CLI

```text
python -m benchmarks.kamino_dvi.analyze_tuning validate \
  --stages baseline wave1 [wave2 halve final canonical] \
  --artifact-root PATH [--logs-root PATH] [--decision-root PATH] \
  [--matrix PATH] [--tuning-matrix PATH]
```

`--stages` and `--artifact-root` are required. `--logs-root` defaults to
`logs`. Baseline and Wave 1 require no decision files. Adaptive stages resolve
candidate names from the exact upstream decision: `wave2.json` for Wave 2,
`stage2.json` for halve, and `finalists.json` for final. Canonical expects
`canonical_winner`. Decision parsing uses the existing strict decision parser.

## Data flow and output

For each stage in requested order:

1. Resolve exact `(candidate, seed)` identities.
2. Load terminal records through `load_tuning_records`.
3. Require exact coverage through `validate_tuning_records`.
4. Count expected, terminal, valid, and rejected records.
5. Sort selected terminal run IDs and rejection entries.

After all stages pass, write one standard deterministic JSON document to
stdout with schema version `1.0` and a `stages` list. Each stage record contains
`stage`, `expected`, `terminal`, `valid`, `rejected`, sorted `run_ids`, and
sorted `rejection_reasons`.

Requested stage order is preserved. JSON uses sorted keys and rejects NaN and
Infinity. Missing, unexpected, duplicate, corrupt, or provenance-invalid
evidence raises through the existing CLI exception path, emits no success
JSON, and exits nonzero.

## Tests and documentation

TDD coverage will prove parser/help behavior, baseline-only success, missing
Wave 1 failure, baseline plus all 18 Wave 1 terminal records success, failed
terminal counting, deterministic ordering, and adaptive decision resolution.
The README will document the exact Task 5 command and output behavior. No GPU
work is in scope.
