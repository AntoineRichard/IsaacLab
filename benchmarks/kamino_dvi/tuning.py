# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and resolve the declarative ANYmal-D Kamino DVI tuning matrix."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from .matrix import DEFAULT_MATRIX_PATH, load_matrix
from .models import TaskName, Variant
from .statistics import Estimate, mean_ci95

SolverValue = str | int | float | bool
DEFAULT_TUNING_MATRIX_PATH = Path(__file__).with_name("tuning_matrix.yaml")
HYDRA_PREFIX = "env.sim.physics.solver_cfg."

_BASELINE_KEYS = {
    "integrator",
    "dynamics_linear_solver_max_iterations",
    "dvi_block_iterations",
    "dvi_contact_iterations",
    "dvi_bilateral_solve_period",
    "dvi_omega",
    "dvi_contact_jacobi_omega",
    "dvi_contact_jacobi_relaxation",
    "dynamics_preconditioning",
    "dvi_contact_block_preconditioner",
    "dvi_warmstart_mode",
}


@dataclass(frozen=True)
class TuningCandidate:
    """One named solver configuration change to screen."""

    name: str
    overrides: dict[str, SolverValue]


@dataclass(frozen=True)
class TuningMatrix:
    """Validated immutable configuration for the ANYmal-D tuning campaign."""

    task: str
    variant: Variant
    preset: str
    num_envs: int
    seeds: tuple[int, ...]
    preflight_iterations: int
    screen_iterations: int
    halve_iterations: int
    final_iterations: int
    warmup_iterations: int
    learning_window: int
    baseline: dict[str, SolverValue]
    wave1: tuple[TuningCandidate, ...]

    def candidate(self, name: str) -> TuningCandidate:
        """Return a Wave 1 candidate by name.

        Args:
            name: Candidate name to find.

        Returns:
            The matching tuning candidate.

        Raises:
            KeyError: If no Wave 1 candidate has the requested name.
        """
        for candidate in self.wave1:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


@dataclass(frozen=True)
class TuningRunMetrics:
    """Terminal runtime and learning traces for one tuning run."""

    candidate: str
    stage: str
    seed: int
    num_envs: int
    iteration_time_s: tuple[float, ...]
    reward: tuple[float, ...]
    success_rate: tuple[float, ...]
    ep_length: tuple[float, ...]
    failure: str | None = None

    def steady_time(self, warmup: int) -> float:
        """Return the mean iteration time after the warmup window.

        Args:
            warmup: Number of leading iterations to exclude.

        Returns:
            Mean steady-state iteration time [s].

        Raises:
            ValueError: If the run failed or has no samples after warmup.
        """
        if self.failure is not None or len(self.iteration_time_s) <= warmup:
            raise ValueError(f"{self.candidate} is not a valid runtime record")
        return statistics.mean(self.iteration_time_s[warmup:])

    def final_mean(self, values: tuple[float, ...], window: int) -> float:
        """Return the mean over the final learning window.

        Args:
            values: Learning trace to summarize.
            window: Number of final samples to include.

        Returns:
            Arithmetic mean over the final window.

        Raises:
            ValueError: If the trace has fewer samples than the window.
        """
        if len(values) < window:
            raise ValueError(f"{self.candidate} has fewer than {window} learning points")
        return statistics.mean(values[-window:])


@dataclass(frozen=True)
class PromotionDecision:
    """Deterministic candidate selections and rejection reasons for one stage."""

    source_stage: str
    selected: tuple[str, ...]
    rejected: dict[str, str]
    resolved_candidates: tuple[TuningCandidate, ...] = ()


@dataclass(frozen=True)
class FinalQualification:
    """Three-seed final estimates and learning-gate outcome for one candidate."""

    candidate: str
    qualified: bool
    reason: str | None
    runtime: Estimate
    reward: Estimate
    success_rate: Estimate
    ep_length: Estimate


def config_hash(config: Mapping[str, SolverValue]) -> str:
    """Return the canonical SHA-256 hash of a resolved solver configuration.

    Args:
        config: Resolved solver configuration to hash.

    Returns:
        The lowercase hexadecimal SHA-256 digest.
    """
    payload = json.dumps(dict(sorted(config.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_config(matrix: TuningMatrix, candidate: TuningCandidate) -> dict[str, SolverValue]:
    """Overlay one candidate on the baseline solver configuration.

    Args:
        matrix: Tuning matrix containing the baseline configuration.
        candidate: Candidate overrides to apply.

    Returns:
        A resolved copy of the solver configuration.

    Raises:
        ValueError: If the candidate contains fields outside the baseline schema.
    """
    unknown = candidate.overrides.keys() - matrix.baseline.keys()
    if unknown:
        raise ValueError(f"candidate {candidate.name} has unknown fields: {sorted(unknown)}")
    return {**matrix.baseline, **candidate.overrides}


def resolve_wave2(matrix: TuningMatrix, results: Sequence[TuningRunMetrics]) -> tuple[TuningCandidate, ...]:
    """Resolve cumulative Wave 2 candidates from successful Wave 1 results.

    Args:
        matrix: Tuning matrix containing all declared Wave 1 candidates.
        results: Complete terminal seed-42 Wave 1 result set.

    Returns:
        Six cumulative candidates containing the fastest two through seven
        distinct-field changes.

    Raises:
        ValueError: If the terminal result set is incomplete or fewer than
            seven successful distinct-field changes remain.
    """
    terminal = {result.candidate: result for result in results if result.seed == 42}
    expected = {candidate.name for candidate in matrix.wave1}
    if len(terminal) != len(results) or set(terminal) != expected:
        raise ValueError("Wave 1 requires exactly 18 terminal candidate results")

    valid = {name: result for name, result in terminal.items() if result.failure is None}
    if len(valid) < 7:
        raise ValueError("Wave 2 requires at least seven valid one-field changes")
    ordered = sorted(
        (candidate for candidate in matrix.wave1 if candidate.name in valid),
        key=lambda candidate: (valid[candidate.name].steady_time(matrix.warmup_iterations), candidate.name),
    )
    compatible: list[TuningCandidate] = []
    selected_fields: set[str] = set()
    for candidate in ordered:
        field = next(iter(candidate.overrides))
        if field not in selected_fields:
            compatible.append(candidate)
            selected_fields.add(field)
    if len(compatible) < 7:
        raise ValueError("Wave 2 requires seven distinct valid one-field changes")

    resolved: list[TuningCandidate] = []
    for count in range(2, 8):
        overrides = {field: value for candidate in compatible[:count] for field, value in candidate.overrides.items()}
        resolved.append(TuningCandidate(f"combined_top_{count:02d}", overrides))
    return tuple(resolved)


def _hydra_value(value: SolverValue) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def hydra_overrides(matrix: TuningMatrix, candidate: TuningCandidate) -> tuple[str, ...]:
    """Return deterministic Hydra overrides for a tuning candidate.

    Args:
        matrix: Tuning matrix containing the baseline schema.
        candidate: Candidate overrides to serialize.

    Returns:
        Canonically ordered Hydra override strings.

    Raises:
        ValueError: If the candidate contains fields outside the baseline schema.
    """
    resolve_config(matrix, candidate)
    return tuple(f"{HYDRA_PREFIX}{name}={_hydra_value(value)}" for name, value in sorted(candidate.overrides.items()))


def load_tuning_matrix(path: Path = DEFAULT_TUNING_MATRIX_PATH) -> TuningMatrix:
    """Load and validate the declarative ANYmal-D tuning matrix.

    Args:
        path: Tuning matrix YAML file to load.

    Returns:
        The validated tuning matrix.

    Raises:
        ValueError: If the matrix violates the locked tuning protocol.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("tuning matrix root must be a mapping")

    benchmark_matrix = load_matrix(DEFAULT_MATRIX_PATH)
    if data["task"] != TaskName.ANYMAL_D.value:
        raise ValueError("tuning task must be the ANYmal-D velocity task")
    variant = Variant(data["variant"])
    if variant is not Variant.KAMINO_PR_DVI:
        raise ValueError("tuning variant must be kamino_pr_dvi")
    if data["preset"] != benchmark_matrix.variant(variant).preset:
        raise ValueError("tuning preset must match the locked benchmark matrix")

    num_envs = data["num_envs"]
    if not isinstance(num_envs, int) or isinstance(num_envs, bool):
        raise ValueError("tuning environment count must be an integer")
    if num_envs != 4096:
        raise ValueError("tuning matrix must use exactly 4096 environments")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in data["seeds"]):
        raise ValueError("tuning matrix seeds must be integers")
    seeds = tuple(data["seeds"])
    if seeds != (42, 43, 44):
        raise ValueError("tuning matrix seeds must be exactly (42, 43, 44)")

    iterations = data["iterations"]
    protocol_values = (
        int(iterations["preflight"]),
        int(iterations["screen"]),
        int(iterations["halve"]),
        int(iterations["final"]),
        int(data["warmup_iterations"]),
        int(data["learning_window"]),
    )
    if any(value <= 0 for value in protocol_values):
        raise ValueError("iteration and window values must be positive")

    baseline = dict(data["baseline"])
    missing_baseline_keys = _BASELINE_KEYS - baseline.keys()
    if missing_baseline_keys:
        raise ValueError(f"baseline is missing required keys: {sorted(missing_baseline_keys)}")

    wave1 = tuple(TuningCandidate(name=item["name"], overrides=dict(item["overrides"])) for item in data["wave1"])
    names = tuple(candidate.name for candidate in wave1)
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate names")
    if len(wave1) != 18:
        raise ValueError("Wave 1 must contain exactly 18 candidates")
    if any(len(candidate.overrides) != 1 for candidate in wave1):
        raise ValueError("each Wave 1 candidate must override exactly one field")

    matrix = TuningMatrix(
        task=data["task"],
        variant=variant,
        preset=data["preset"],
        num_envs=num_envs,
        seeds=seeds,
        preflight_iterations=protocol_values[0],
        screen_iterations=protocol_values[1],
        halve_iterations=protocol_values[2],
        final_iterations=protocol_values[3],
        warmup_iterations=protocol_values[4],
        learning_window=protocol_values[5],
        baseline=baseline,
        wave1=wave1,
    )
    for candidate in matrix.wave1:
        resolve_config(matrix, candidate)
    return matrix


def _group_required_seeds(
    results: Sequence[TuningRunMetrics], required_seeds: set[int], description: str
) -> dict[str, dict[int, TuningRunMetrics]]:
    grouped: dict[str, dict[int, TuningRunMetrics]] = {}
    counts: dict[str, int] = {}
    for result in results:
        counts[result.candidate] = counts.get(result.candidate, 0) + 1
        grouped.setdefault(result.candidate, {})[result.seed] = result
    for candidate, by_seed in grouped.items():
        if counts[candidate] != len(required_seeds) or set(by_seed) != required_seeds:
            separator = "and" if len(required_seeds) == 2 else "through"
            raise ValueError(
                f"{description} requires {candidate} seeds {min(required_seeds)} {separator} {max(required_seeds)}"
            )
    return grouped


def _baseline_by_seed(
    matrix: TuningMatrix, results: Sequence[TuningRunMetrics], required_seeds: set[int], description: str
) -> dict[int, TuningRunMetrics]:
    grouped = _group_required_seeds(results, required_seeds, description)
    if len(grouped) != 1:
        raise ValueError(f"{description} requires exactly one baseline candidate")
    baseline = next(iter(grouped.values()))
    for result in baseline.values():
        if result.failure is not None:
            raise ValueError(f"{description} baseline failed: {result.failure}")
        if result.num_envs != matrix.num_envs:
            raise ValueError(f"{description} baseline requires {matrix.num_envs} environments")
        result.steady_time(matrix.warmup_iterations)
        for values in (result.reward, result.success_rate, result.ep_length):
            result.final_mean(values, matrix.learning_window)
    return baseline


def _record_failure(matrix: TuningMatrix, result: TuningRunMetrics) -> str | None:
    if result.failure is not None:
        return result.failure
    if result.num_envs != matrix.num_envs:
        return f"requires {matrix.num_envs} environments"
    try:
        result.steady_time(matrix.warmup_iterations)
        for values in (result.reward, result.success_rate, result.ep_length):
            result.final_mean(values, matrix.learning_window)
    except ValueError as error:
        return str(error)
    return None


def promote_stage2(matrix: TuningMatrix, results: Sequence[TuningRunMetrics]) -> PromotionDecision:
    """Select up to eight fastest successful Stage 1 candidates.

    Args:
        matrix: Tuning protocol defining runtime warmup and environment count.
        results: Terminal Stage 1 candidate metrics.

    Returns:
        Deterministic promotion decision ordered by steady-state runtime.

    Raises:
        ValueError: If candidate names are duplicated or a record uses a seed
            other than 42.
    """
    if len({result.candidate for result in results}) != len(results) or any(result.seed != 42 for result in results):
        raise ValueError("Stage 1 requires one seed-42 result per candidate")
    rejected: dict[str, str] = {}
    ranked: list[tuple[float, str]] = []
    for result in results:
        failure = _record_failure(matrix, result)
        if failure is not None:
            rejected[result.candidate] = failure
        else:
            ranked.append((result.steady_time(matrix.warmup_iterations), result.candidate))
    ranked.sort()
    return PromotionDecision(
        source_stage="stage1",
        selected=tuple(candidate for _, candidate in ranked[:8]),
        rejected=rejected,
    )


def promote_finalists(
    matrix: TuningMatrix,
    baseline_results: Sequence[TuningRunMetrics],
    candidate_results: Sequence[TuningRunMetrics],
) -> PromotionDecision:
    """Apply per-seed learning gates and select the three fastest finalists.

    Args:
        matrix: Tuning protocol defining runtime and learning windows.
        baseline_results: Successful Stage 2 baseline results for seeds 42 and 43.
        candidate_results: Stage 2 candidate results for seeds 42 and 43.

    Returns:
        Deterministic finalist promotion decision.

    Raises:
        ValueError: If baseline or candidate seed sets are incomplete.
    """
    required_seeds = {42, 43}
    baseline = _baseline_by_seed(matrix, baseline_results, required_seeds, "Stage 2")
    grouped = _group_required_seeds(candidate_results, required_seeds, "Stage 2")
    rejected: dict[str, str] = {}
    ranked: list[tuple[float, str]] = []
    for candidate, by_seed in grouped.items():
        reason: str | None = None
        for seed in sorted(required_seeds):
            result = by_seed[seed]
            failure = _record_failure(matrix, result)
            if failure is not None:
                reason = f"seed {seed}: {failure}"
                break
            reference = baseline[seed]
            candidate_reward = result.final_mean(result.reward, matrix.learning_window)
            baseline_reward = reference.final_mean(reference.reward, matrix.learning_window)
            if candidate_reward < 0.8 * baseline_reward:
                reason = f"seed {seed}: reward below 80% of baseline"
                break
            candidate_success = result.final_mean(result.success_rate, matrix.learning_window)
            baseline_success = reference.final_mean(reference.success_rate, matrix.learning_window)
            if candidate_success < baseline_success - 0.10:
                reason = f"seed {seed}: success below baseline by more than 0.10"
                break
            candidate_ep_length = result.final_mean(result.ep_length, matrix.learning_window)
            baseline_ep_length = reference.final_mean(reference.ep_length, matrix.learning_window)
            if baseline_ep_length == 0.0:
                raise ValueError("Stage 2 baseline episode length must be nonzero")
            ep_length_ratio = abs(candidate_ep_length / baseline_ep_length)
            if not 0.8 <= ep_length_ratio <= 1.2:
                reason = f"seed {seed}: episode length ratio outside [0.8, 1.2]"
                break
        if reason is not None:
            rejected[candidate] = reason
        else:
            runtime = statistics.mean(result.steady_time(matrix.warmup_iterations) for result in by_seed.values())
            ranked.append((runtime, candidate))
    ranked.sort()
    return PromotionDecision(
        source_stage="stage2",
        selected=tuple(candidate for _, candidate in ranked[:3]),
        rejected=rejected,
    )


def _estimate_final(matrix: TuningMatrix, by_seed: Mapping[int, TuningRunMetrics]) -> tuple[Estimate, ...]:
    ordered = [by_seed[seed] for seed in sorted(by_seed)]
    return (
        mean_ci95([result.steady_time(matrix.warmup_iterations) for result in ordered]),
        mean_ci95([result.final_mean(result.reward, matrix.learning_window) for result in ordered]),
        mean_ci95([result.final_mean(result.success_rate, matrix.learning_window) for result in ordered]),
        mean_ci95([result.final_mean(result.ep_length, matrix.learning_window) for result in ordered]),
    )


def _invalid_estimate() -> Estimate:
    return Estimate(mean=float("nan"), half_width=float("nan"), n=0)


def qualify_finalists(
    matrix: TuningMatrix,
    baseline_results: Sequence[TuningRunMetrics],
    candidate_results: Sequence[TuningRunMetrics],
) -> dict[str, FinalQualification]:
    """Qualify finalists against three-seed baseline confidence lower bounds.

    Args:
        matrix: Tuning protocol defining runtime and learning windows.
        baseline_results: Successful final baseline results for seeds 42–44.
        candidate_results: Final candidate results for seeds 42–44.

    Returns:
        Qualifications keyed by candidate name.

    Raises:
        ValueError: If baseline or candidate seed sets are incomplete.
    """
    required_seeds = {42, 43, 44}
    baseline = _baseline_by_seed(matrix, baseline_results, required_seeds, "Final qualification")
    baseline_estimates = _estimate_final(matrix, baseline)
    reward_floor = baseline_estimates[1].mean - baseline_estimates[1].half_width
    success_floor = baseline_estimates[2].mean - baseline_estimates[2].half_width
    grouped = _group_required_seeds(candidate_results, required_seeds, "Final qualification")
    qualifications: dict[str, FinalQualification] = {}
    for candidate, by_seed in grouped.items():
        reason: str | None = None
        for seed in sorted(required_seeds):
            failure = _record_failure(matrix, by_seed[seed])
            if failure is not None:
                reason = f"seed {seed}: {failure}"
                break
        if reason is not None:
            invalid = _invalid_estimate()
            qualifications[candidate] = FinalQualification(candidate, False, reason, invalid, invalid, invalid, invalid)
            continue

        runtime, reward, success_rate, ep_length = _estimate_final(matrix, by_seed)
        if reward.mean < reward_floor:
            reason = "reward mean below baseline 95% confidence lower bound"
        elif success_rate.mean < success_floor:
            reason = "success mean below baseline 95% confidence lower bound"
        qualifications[candidate] = FinalQualification(
            candidate=candidate,
            qualified=reason is None,
            reason=reason,
            runtime=runtime,
            reward=reward,
            success_rate=success_rate,
            ep_length=ep_length,
        )
    return qualifications


def _intervals_overlap(left: Estimate, right: Estimate) -> bool:
    return (
        left.mean - left.half_width <= right.mean + right.half_width
        and right.mean - right.half_width <= left.mean + left.half_width
    )


def _configuration_tie_key(
    matrix: TuningMatrix, candidate: str, config: Mapping[str, SolverValue]
) -> tuple[float | int | str, ...]:
    if set(config) != set(matrix.baseline):
        raise ValueError(f"{candidate} does not have a complete resolved configuration")
    relaxation_distance = sum(
        abs(float(config[field]) - float(matrix.baseline[field]))
        for field in ("dvi_omega", "dvi_contact_jacobi_omega", "dvi_contact_jacobi_relaxation")
    )
    return (
        -int(config["dvi_contact_iterations"]),
        -int(config["dvi_block_iterations"]),
        -int(config["dynamics_linear_solver_max_iterations"]),
        int(config["dvi_bilateral_solve_period"]),
        relaxation_distance,
        candidate,
    )


def select_winner(
    matrix: TuningMatrix,
    qualifications: Mapping[str, FinalQualification],
    resolved_configs: Mapping[str, Mapping[str, SolverValue]],
) -> str:
    """Select the fastest qualified finalist with deterministic overlap ties.

    Args:
        matrix: Tuning matrix containing the baseline solver configuration.
        qualifications: Final three-seed qualifications keyed by candidate.
        resolved_configs: Complete resolved solver configurations keyed by candidate.

    Returns:
        Winning candidate name.

    Raises:
        ValueError: If no candidate qualified or a qualified configuration is missing.
    """
    qualified = sorted(
        (qualification for qualification in qualifications.values() if qualification.qualified),
        key=lambda qualification: (qualification.runtime.mean, qualification.candidate),
    )
    if not qualified:
        raise ValueError("no qualified tuning candidates")
    missing = [
        qualification.candidate for qualification in qualified if qualification.candidate not in resolved_configs
    ]
    if missing:
        raise ValueError(f"missing resolved configuration for {missing}")

    fastest = qualified[0]
    overlapping = [
        qualification for qualification in qualified if _intervals_overlap(fastest.runtime, qualification.runtime)
    ]
    if len(overlapping) == 1:
        return fastest.candidate
    return min(
        overlapping,
        key=lambda qualification: _configuration_tie_key(
            matrix,
            qualification.candidate,
            resolved_configs[qualification.candidate],
        ),
    ).candidate
