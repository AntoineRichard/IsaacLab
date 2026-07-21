# ANYmal-D DVI Tuning Design

## Objective

Tune Kamino's DVI solver specifically for `Isaac-Velocity-Flat-AnymalD` and minimize steady-state RSL-RL training
iteration time subject to the explicit learning-quality and stability gates below.

The final comparison uses 4096 environments, 300 training iterations, and seeds 42–44. A candidate qualifies only
when its three-seed mean reward and success rate meet the Stage 3 non-degradation bounds. Capacity failures, crashes,
non-finite traces, incomplete bundles, or missing required metrics disqualify a candidate.
Episode length is retained as a diagnostic guardrail.

## Scope and Fairness

The campaign may tune DVI per task, matching the precedent established by the task-specific MJWarp settings. It keeps
the following fixed:

- the ANYmal-D task and RSL-RL agent configuration;
- the physics timestep and one simulation substep;
- 4096 environments for every tuning and validation run;
- the task's contact-capacity setting;
- the final seed set and 300-iteration protocol; and
- reward, success-rate, episode-length, and runtime definitions used by the existing Kamino report.

The search may vary only Kamino/DVI controls: the integrator, constrained linear solver budget, DVI block and contact
iteration budgets, bilateral solve period, global and contact relaxation, dynamics and contact-block preconditioning,
and warm-start mode. The canonical `newton_kamino_dvi` ANYmal-D preset remains unchanged until a finalist passes the
full validation gate.

## Search Strategy

### Stage 0: Clean Baseline

Rerun the current ANYmal-D DVI preset through the hardened benchmark harness. The baseline must have clean tracked
source provenance, exact IsaacLab and Newton revisions, a complete schema v1.1 bundle, and a retained TensorBoard event
path and SHA-256 hash. Tuning does not begin if this baseline fails.

### Stage 1: Runtime Screening

Evaluate exactly 24 deliberately selected configurations for 40 iterations with seed 42. Exclude iterations
1–10 from runtime estimates. Use a two-wave structured search rather than a Cartesian product. Wave 1 contains these
18 one-field changes from the current DVI baseline:

- Euler integration;
- CR iteration budgets of 3, 5, and 7;
- DVI block iteration budgets of 4, 8, and 12;
- one contact iteration;
- a bilateral solve period of 4;
- a global DVI update factor of 0.5;
- contact-Jacobi update factors of 0.3 and 0.6;
- contact-Jacobi relaxation factors of 0.7 and 1.0;
- dynamics preconditioning enabled and contact block preconditioning enabled; and
- internal and disabled warm-start modes.

Rank Wave 1 by steady-state runtime. Wave 2 contains six cumulative configurations formed by applying the fastest two,
three, four, five, six, and seven compatible one-field changes to the baseline. Each field can appear only once. Persist
the resolved overrides and their hashes before running Wave 2, so the adaptive choice is reproducible.

Candidates are selected to isolate important effects first and then combine compatible reductions. A candidate is
immediately rejected on a capacity, process, numerical, artifact, or metric failure.

### Stage 2: Successive Halving

Promote the eight fastest valid Stage 1 candidates to 100 iterations with seeds 42 and 43. If fewer than eight
candidates are valid, promote every valid candidate and record the shortfall. Compare each candidate's final-20 reward,
success rate, and episode length with the matching seed and iteration window from the clean baseline. Reject a candidate
when either seed has reward below 80% of baseline, success more than 0.10 below baseline, or episode length more than
20% away from baseline. Promote the three fastest remaining candidates to Stage 3.

### Stage 3: Final Validation

Run the best three candidates and the clean baseline for 300 iterations with seeds 42–44. A candidate qualifies when:

- all three runs complete at 4096 environments;
- all required series are finite, aligned, and complete;
- its mean reward is no lower than the baseline mean minus the baseline reward 95% CI half-width;
- its mean success rate is no lower than the baseline mean minus the baseline success 95% CI half-width; and
- no individual seed exhibits a stability failure.

Select the fastest qualifying candidate. When its runtime 95% confidence interval overlaps that of another qualifying
candidate, break the tie lexicographically by higher contact iterations, higher block iterations, higher CR iterations,
more frequent bilateral solves, and finally the smallest absolute relaxation-setting distance from the baseline.

## Tuning Harness

Add a separate declarative ANYmal-D tuning matrix with stable candidate names and exact solver overrides. Do not add
dozens of public physics presets. The tuning runner reuses the hardened benchmark execution, provenance, manifest,
failure-classification, trace-parsing, and statistics components while extending the declared command grammar to the
exact candidate overrides.

Every run records:

- the candidate name and canonical configuration hash;
- the exact task, seed, environment count, and iteration count;
- the IsaacLab HEAD and locked Newton revision;
- the exact command and command hash;
- schema-bundle artifact hashes; and
- the matched TensorBoard event path and SHA-256 hash.

Resume behavior must match the exact candidate configuration and provenance. An undeclared Hydra override or a stale
manifest cannot enter aggregation.

## Data Flow and Outputs

The tuning matrix expands into immutable run identities. The runner executes them sequentially on one GPU and writes
raw artifacts atomically. The analyzer validates the complete expected candidate/seed set before ranking it. Promotion
decisions are written as compact JSON so each stage can be audited and resumed.

Intermediate traces, rejected candidates, and diagnostic rankings remain under ignored tuning-artifact directories.
Committed outputs consist of:

- the declarative candidate matrix and tuning utilities;
- unit tests for expansion, validation, promotion, and qualification;
- the winning ANYmal-D DVI preset values;
- a compact JSON result summary; and
- a Markdown/PDF addendum comparing the clean baseline, tuned DVI, MJWarp, and PhysX.

## Failure Handling

The tuning campaign never lowers the environment count. A capacity failure at 4096 disqualifies that candidate.
Crashes, timeouts, non-finite values, incomplete schema bundles, missing TensorBoard success, missing event hashes, and
command/provenance mismatches are retained and reported but excluded from promotion. The runner stops the affected
candidate; it does not silently substitute schema success values or stale artifacts.

## Verification

Before GPU execution, unit tests cover:

- deterministic candidate expansion and configuration hashing;
- exact command construction and rejection of undeclared overrides;
- resume matching and artifact integrity;
- runtime-window and learning-window calculations;
- Stage 1 and Stage 2 promotion rules;
- final reward/success qualification boundaries; and
- failure exclusion and reporting.

Run a five-iteration construction preflight for each candidate before its first measured stage. After selecting the
winner, update only the ANYmal-D preset and its focused preset tests, run the complete Kamino tuning/harness tests, and
rerun the final three-seed validation from the committed clean source state.

## Expected Cost

The expected single-GPU cost is 1.5–3 hours: 24 one-seed 40-iteration screens, eight two-seed
100-iteration runs, and four three-seed 300-iteration final configurations. The exact duration depends on process
startup and how many candidates survive construction and stability checks.
