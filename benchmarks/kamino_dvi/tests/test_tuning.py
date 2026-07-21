# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the declarative ANYmal-D Kamino DVI tuning matrix."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from benchmarks.kamino_dvi.statistics import Estimate, mean_ci95
from benchmarks.kamino_dvi.tuning import (
    DEFAULT_TUNING_MATRIX_PATH,
    FinalQualification,
    TuningRunMetrics,
    config_hash,
    hydra_overrides,
    load_tuning_matrix,
    promote_finalists,
    promote_stage2,
    qualify_finalists,
    resolve_config,
    resolve_wave2,
    select_winner,
)


@pytest.fixture
def matrix():
    """Return the locked ANYmal-D tuning matrix."""
    return load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)


def make_screen_metric(candidate: str, index: int, *, failure: str | None = None) -> TuningRunMetrics:
    """Build one deterministic Wave 1 terminal metric."""
    steady_time = 0.10 + index / 100
    return TuningRunMetrics(
        candidate=candidate,
        stage="wave1",
        seed=42,
        num_envs=4096,
        iteration_time_s=(1.0,) * 10 + (steady_time,) * 30,
        reward=(20.0,) * 40,
        success_rate=(1.0,) * 40,
        ep_length=(980.0,) * 40,
        failure=failure,
    )


def test_wave2_uses_cumulative_prefixes_of_fastest_compatible_changes(matrix):
    """Wave 2 combines the fastest distinct-field changes in cumulative prefixes."""
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]

    wave2 = resolve_wave2(matrix, results)

    assert [candidate.name for candidate in wave2] == [
        "combined_top_02",
        "combined_top_03",
        "combined_top_04",
        "combined_top_05",
        "combined_top_06",
        "combined_top_07",
    ]
    assert len(wave2[0].overrides) == 2
    assert len(wave2[-1].overrides) == 7


def test_wave2_skips_faster_repeated_fields_for_next_distinct_changes(matrix):
    """Repeated CR and block candidates do not consume compatible prefix slots."""
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]

    largest = resolve_wave2(matrix, results)[-1]

    assert largest.overrides["dynamics_linear_solver_max_iterations"] == 3
    assert largest.overrides["dvi_block_iterations"] == 4
    assert set(largest.overrides) == {
        "integrator",
        "dynamics_linear_solver_max_iterations",
        "dvi_block_iterations",
        "dvi_contact_iterations",
        "dvi_bilateral_solve_period",
        "dvi_omega",
        "dvi_contact_jacobi_omega",
    }


def test_wave2_excludes_failed_candidates_but_requires_every_terminal_record(matrix):
    """Failed changes are excluded only after the complete terminal set is validated."""
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]
    results[-1] = dataclasses.replace(results[-1], failure="numerical")

    wave2 = resolve_wave2(matrix, results)

    assert all(candidate.overrides.get("dvi_warmstart_mode") != "none" for candidate in wave2)
    with pytest.raises(ValueError, match="18 terminal"):
        resolve_wave2(matrix, results[:-1])


def test_wave2_requires_seven_valid_changes(matrix):
    """Wave 2 cannot be formed without seven successful one-field records."""
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]
    results = [dataclasses.replace(result, failure="numerical") for result in results[:12]] + results[12:]

    with pytest.raises(ValueError, match="at least seven valid"):
        resolve_wave2(matrix, results)


def test_wave2_requires_seven_distinct_valid_fields(matrix):
    """Seven successful records are insufficient when they cover fewer than seven fields."""
    valid_names = {
        "cr_iterations_3",
        "cr_iterations_5",
        "cr_iterations_7",
        "block_iterations_4",
        "block_iterations_8",
        "block_iterations_12",
        "contact_iterations_1",
    }
    results = [
        make_screen_metric(candidate.name, index, failure=None if candidate.name in valid_names else "numerical")
        for index, candidate in enumerate(matrix.wave1)
    ]

    with pytest.raises(ValueError, match="seven distinct valid"):
        resolve_wave2(matrix, results)


def test_wave2_rejects_duplicate_or_wrong_seed_terminal_records(matrix):
    """Each declared Wave 1 candidate must have exactly one seed-42 terminal record."""
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]

    with pytest.raises(ValueError, match="18 terminal"):
        resolve_wave2(matrix, [*results, results[0]])
    with pytest.raises(ValueError, match="18 terminal"):
        resolve_wave2(matrix, [dataclasses.replace(results[0], seed=43), *results[1:]])


def test_tuning_matrix_declares_exact_anymal_protocol_and_wave1():
    """The tuning matrix must lock the approved ANYmal-D campaign."""
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    assert matrix.task == "Isaac-Velocity-Flat-AnymalD"
    assert matrix.num_envs == 4096
    assert matrix.seeds == (42, 43, 44)
    assert (matrix.preflight_iterations, matrix.screen_iterations) == (5, 40)
    assert (matrix.halve_iterations, matrix.final_iterations) == (100, 300)
    assert len(matrix.wave1) == 18
    assert all(len(candidate.overrides) == 1 for candidate in matrix.wave1)


def test_resolved_hash_is_order_independent_and_hydra_values_are_canonical():
    """Resolved configurations must hash canonically and emit canonical Hydra values."""
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = matrix.candidate("dynamics_preconditioning_true")
    resolved = resolve_config(matrix, candidate)
    assert resolved["dynamics_preconditioning"] is True
    assert config_hash(dict(reversed(tuple(resolved.items())))) == config_hash(resolved)
    assert hydra_overrides(matrix, candidate) == ("env.sim.physics.solver_cfg.dynamics_preconditioning=true",)


def _write_matrix(tmp_path: Path, mutate) -> Path:
    """Write a mutated copy of the checked-in tuning matrix."""
    data = yaml.safe_load(DEFAULT_TUNING_MATRIX_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "tuning_matrix.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda data: data["wave1"].append(data["wave1"][0]), "duplicate candidate names"),
        (lambda data: data.update(task="Isaac-Ant-Direct"), "ANYmal"),
        (lambda data: data.update(num_envs=2048), "4096"),
        (lambda data: data.update(num_envs=4096.5), "integer"),
        (lambda data: data.update(seeds=[42, 43]), "seeds"),
        (lambda data: data.update(seeds=[42.9, 43, 44]), "integers"),
        (lambda data: data["wave1"].pop(), "18"),
        (lambda data: data["wave1"][0]["overrides"].update(dvi_omega=0.5), "exactly one"),
        (lambda data: data["baseline"].pop("dvi_omega"), "baseline"),
        (lambda data: data["iterations"].update(screen=0), "positive"),
        (lambda data: data.update(learning_window=0), "positive"),
    ),
)
def test_tuning_matrix_rejects_invalid_protocol(tmp_path: Path, mutate, message: str):
    """Invalid changes to locked tuning dimensions must fail during loading."""
    with pytest.raises(ValueError, match=message):
        load_tuning_matrix(_write_matrix(tmp_path, mutate))


def test_resolve_config_rejects_unknown_override_fields():
    """Candidate fields outside the baseline schema must be rejected."""
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    candidate = type(matrix.wave1[0])("unknown", {"not_a_solver_field": 1})

    with pytest.raises(ValueError, match="unknown fields"):
        resolve_config(matrix, candidate)


def make_metric(
    candidate: str,
    stage: str,
    seed: int,
    iterations: int,
    *,
    runtime: float = 0.1,
    reward: float = 10.0,
    success: float = 0.8,
    ep_length: float = 100.0,
    failure: str | None = None,
) -> TuningRunMetrics:
    """Build a deterministic terminal metric for promotion tests."""
    return TuningRunMetrics(
        candidate=candidate,
        stage=stage,
        seed=seed,
        num_envs=4096,
        iteration_time_s=(1.0,) * 10 + (runtime,) * (iterations - 10),
        reward=(reward,) * iterations,
        success_rate=(success,) * iterations,
        ep_length=(ep_length,) * iterations,
        failure=failure,
    )


@pytest.fixture
def baseline_100():
    """Return the two-seed Stage 2 baseline."""
    return tuple(make_metric("baseline", "halve", seed, 100, runtime=0.2) for seed in (42, 43))


def make_two_seed_metrics(
    *, reward_scale: float, success_delta: float, ep_length_scale: float, candidate: str = "candidate"
) -> tuple[TuningRunMetrics, ...]:
    """Build two-seed metrics relative to the constant Stage 2 baseline."""
    return tuple(
        make_metric(
            candidate,
            "halve",
            seed,
            100,
            runtime=0.1,
            reward=10.0 * reward_scale,
            success=0.8 + success_delta,
            ep_length=100.0 * ep_length_scale,
        )
        for seed in (42, 43)
    )


def test_stage1_promotes_eight_fastest_successful_candidates(matrix):
    """Stage 1 ranks successful candidates by steady time and caps promotion at eight."""
    results = [make_screen_metric(f"candidate_{index:02d}", 9 - index) for index in range(10)]
    results[8] = dataclasses.replace(results[8], failure="capacity")

    decision = promote_stage2(matrix, results)

    assert decision.source_stage == "stage1"
    assert decision.selected == (
        "candidate_09",
        "candidate_07",
        "candidate_06",
        "candidate_05",
        "candidate_04",
        "candidate_03",
        "candidate_02",
        "candidate_01",
    )
    assert decision.rejected == {"candidate_08": "capacity"}


def test_stage1_promotes_every_valid_candidate_when_fewer_than_eight(matrix):
    """Stage 1 records failures without requiring a full eight promotions."""
    results = [make_screen_metric(f"candidate_{index}", index) for index in range(4)]
    results.append(make_screen_metric("failed", 4, failure="numerical"))

    decision = promote_stage2(matrix, results)

    assert decision.selected == ("candidate_0", "candidate_1", "candidate_2", "candidate_3")
    assert decision.rejected == {"failed": "numerical"}


def test_stage2_rejects_reward_below_eighty_percent(matrix, baseline_100):
    """Either seed falling below 80 percent of baseline rejects the candidate."""
    candidate = make_two_seed_metrics(reward_scale=0.799, success_delta=0.0, ep_length_scale=1.0)

    decision = promote_finalists(matrix, baseline_100, candidate)

    assert decision.selected == ()
    assert "reward" in decision.rejected[candidate[0].candidate]


@pytest.mark.parametrize("ep_length_scale", (0.8, 1.2))
def test_stage2_accepts_exact_learning_boundaries(matrix, baseline_100, ep_length_scale: float):
    """Reward, success, and episode-length thresholds are inclusive."""
    candidate = make_two_seed_metrics(
        reward_scale=0.8,
        success_delta=-0.10,
        ep_length_scale=ep_length_scale,
    )

    assert promote_finalists(matrix, baseline_100, candidate).selected == (candidate[0].candidate,)


@pytest.mark.parametrize(
    ("reward_scale", "success_delta", "ep_length_scale", "reason"),
    (
        (1.0, -0.101, 1.0, "success"),
        (1.0, 0.0, 0.799, "episode length"),
        (1.0, 0.0, 1.201, "episode length"),
    ),
)
def test_stage2_rejects_learning_values_outside_each_boundary(
    matrix, baseline_100, reward_scale: float, success_delta: float, ep_length_scale: float, reason: str
):
    """Every Stage 2 learning guard rejects a value just outside its boundary."""
    candidate = make_two_seed_metrics(
        reward_scale=reward_scale,
        success_delta=success_delta,
        ep_length_scale=ep_length_scale,
    )

    decision = promote_finalists(matrix, baseline_100, candidate)

    assert decision.selected == ()
    assert reason in decision.rejected[candidate[0].candidate]


def test_stage2_requires_matching_seeds_and_selects_three_fastest(matrix, baseline_100):
    """Stage 2 requires seeds 42 and 43 and caps qualified runtime ranking at three."""
    candidates = tuple(
        dataclasses.replace(metric, candidate=f"candidate_{index}", iteration_time_s=(1.0,) * 10 + (runtime,) * 90)
        for index, runtime in enumerate((0.4, 0.3, 0.2, 0.1))
        for metric in make_two_seed_metrics(reward_scale=1.0, success_delta=0.0, ep_length_scale=1.0)
    )

    decision = promote_finalists(matrix, baseline_100, candidates)

    assert decision.selected == ("candidate_3", "candidate_2", "candidate_1")
    with pytest.raises(ValueError, match="seeds 42 and 43"):
        promote_finalists(matrix, baseline_100, candidates[:-1])


@pytest.fixture
def baseline_final():
    """Return a varying three-seed final baseline with nonzero confidence widths."""
    return tuple(
        make_metric(
            "baseline",
            "baseline",
            seed,
            300,
            runtime=runtime,
            reward=reward,
            success=success,
            ep_length=ep_length,
        )
        for seed, runtime, reward, success, ep_length in (
            (42, 0.19, 9.0, 0.7, 90.0),
            (43, 0.20, 10.0, 0.8, 100.0),
            (44, 0.21, 11.0, 0.9, 110.0),
        )
    )


@pytest.fixture
def candidate_final(baseline_final):
    """Return a candidate exactly on the baseline reward and success lower bounds."""
    reward_floor = mean_ci95([9.0, 10.0, 11.0]).mean - mean_ci95([9.0, 10.0, 11.0]).half_width
    success_floor = mean_ci95([0.7, 0.8, 0.9]).mean - mean_ci95([0.7, 0.8, 0.9]).half_width
    return tuple(
        make_metric(
            "candidate",
            "final",
            seed,
            300,
            runtime=0.1,
            reward=reward_floor,
            success=success_floor,
            ep_length=100.0,
        )
        for seed in (42, 43, 44)
    )


def test_final_gate_uses_baseline_ci_lower_bounds(matrix, baseline_final, candidate_final):
    """Stage 3 reward and success floors include their exact confidence boundaries."""
    result = qualify_finalists(matrix, baseline_final, candidate_final)

    assert result[candidate_final[0].candidate].qualified is True
    assert result[candidate_final[0].candidate].reason is None


@pytest.mark.parametrize(("metric", "reason"), (("reward", "reward"), ("success_rate", "success")))
def test_final_gate_rejects_mean_below_baseline_ci_floor(
    matrix, baseline_final, candidate_final, metric: str, reason: str
):
    """A candidate mean just below either Stage 3 learning floor is rejected."""
    changed = tuple(
        dataclasses.replace(result, **{metric: tuple(value - 1e-6 for value in getattr(result, metric))})
        for result in candidate_final
    )

    qualification = qualify_finalists(matrix, baseline_final, changed)["candidate"]

    assert qualification.qualified is False
    assert reason in qualification.reason


def test_final_gate_requires_three_seeds_and_disqualifies_terminal_failures(matrix, baseline_final, candidate_final):
    """Stage 3 rejects incomplete seed sets and retains terminal failure reasons."""
    with pytest.raises(ValueError, match="seeds 42 through 44"):
        qualify_finalists(matrix, baseline_final, candidate_final[:-1])

    failed = (dataclasses.replace(candidate_final[0], failure="numerical"), *candidate_final[1:])
    qualification = qualify_finalists(matrix, baseline_final, failed)["candidate"]
    assert qualification.qualified is False
    assert qualification.reason == "seed 42: numerical"


def make_qualification(candidate: str, runtime: Estimate, *, qualified: bool = True) -> FinalQualification:
    """Build a compact final qualification for winner-selection tests."""
    estimate = Estimate(mean=1.0, half_width=0.0, n=3)
    return FinalQualification(
        candidate, qualified, None if qualified else "rejected", runtime, estimate, estimate, estimate
    )


def resolved(matrix, **overrides):
    """Return a complete resolved configuration with selected field changes."""
    return {**matrix.baseline, **overrides}


def test_winner_uses_runtime_when_confidence_intervals_do_not_overlap(matrix):
    """A strictly separated fastest runtime wins without invoking configuration preferences."""
    qualifications = {
        "fast": make_qualification("fast", Estimate(0.10, 0.001, 3)),
        "slow": make_qualification("slow", Estimate(0.12, 0.001, 3)),
    }
    configs = {
        "fast": resolved(matrix, dvi_contact_iterations=1),
        "slow": resolved(matrix, dvi_contact_iterations=99),
    }

    assert select_winner(matrix, qualifications, configs) == "fast"


@pytest.mark.parametrize(
    ("left_overrides", "right_overrides", "expected"),
    (
        ({"dvi_contact_iterations": 3}, {"dvi_contact_iterations": 2, "dvi_block_iterations": 99}, "left"),
        (
            {"dvi_block_iterations": 17},
            {"dvi_block_iterations": 16, "dynamics_linear_solver_max_iterations": 99},
            "left",
        ),
        (
            {"dynamics_linear_solver_max_iterations": 10},
            {"dynamics_linear_solver_max_iterations": 9, "dvi_bilateral_solve_period": 1},
            "left",
        ),
        ({"dvi_bilateral_solve_period": 1}, {"dvi_bilateral_solve_period": 2}, "left"),
        ({"dvi_omega": 0.4}, {"dvi_omega": 0.5}, "left"),
    ),
)
def test_overlapping_runtime_uses_approved_lexicographic_tie_break(
    matrix, left_overrides: dict, right_overrides: dict, expected: str
):
    """Overlapping intervals compare each approved configuration dimension in order."""
    qualifications = {
        "left": make_qualification("left", Estimate(0.100, 0.010, 3)),
        "right": make_qualification("right", Estimate(0.101, 0.010, 3)),
    }
    configs = {
        "left": resolved(matrix, **left_overrides),
        "right": resolved(matrix, **right_overrides),
    }

    assert select_winner(matrix, qualifications, configs) == expected


def test_overlapping_runtime_uses_candidate_name_as_final_fallback(matrix):
    """Identical overlapping configurations fall back to candidate name deterministically."""
    qualifications = {
        "zeta": make_qualification("zeta", Estimate(0.100, 0.010, 3)),
        "alpha": make_qualification("alpha", Estimate(0.101, 0.010, 3)),
    }
    configs = {name: dict(matrix.baseline) for name in qualifications}

    assert select_winner(matrix, qualifications, configs) == "alpha"


def test_winner_rejects_empty_qualified_set_and_missing_config(matrix):
    """Winner selection requires a configuration for at least one qualified candidate."""
    rejected = {"candidate": make_qualification("candidate", Estimate(0.1, 0.0, 3), qualified=False)}
    with pytest.raises(ValueError, match="no qualified"):
        select_winner(matrix, rejected, {"candidate": dict(matrix.baseline)})

    qualified = {"candidate": make_qualification("candidate", Estimate(0.1, 0.0, 3))}
    with pytest.raises(ValueError, match="resolved configuration"):
        select_winner(matrix, qualified, {})


_PROTOCOL_MUTATIONS = (
    ("num_envs", "4096 environments"),
    ("stage", "stage"),
    ("iteration_time_s", "iteration_time_s"),
    ("reward", "reward"),
    ("success_rate", "success_rate"),
    ("ep_length", "ep_length"),
)


def invalidate_protocol(record: TuningRunMetrics, field: str) -> TuningRunMetrics:
    """Return a record with one protocol identity field made invalid."""
    if field == "num_envs":
        return dataclasses.replace(record, num_envs=2048)
    if field == "stage":
        return dataclasses.replace(record, stage="wrong_stage")
    values = getattr(record, field)
    return dataclasses.replace(record, **{field: values[:-1]})


@pytest.mark.parametrize(("field", "reason"), _PROTOCOL_MUTATIONS)
def test_wave2_excludes_records_with_invalid_protocol_identity(matrix, field: str, reason: str):
    """Wave 2 excludes wrong environments, stages, and non-exact traces."""
    results = [make_screen_metric(candidate.name, index) for index, candidate in enumerate(matrix.wave1)]
    results[0] = invalidate_protocol(results[0], field)

    largest = resolve_wave2(matrix, results)[-1]

    assert "integrator" not in largest.overrides, reason


@pytest.mark.parametrize(("field", "reason"), _PROTOCOL_MUTATIONS)
def test_stage1_rejects_records_with_invalid_protocol_identity(matrix, field: str, reason: str):
    """Stage 1 rejects wrong environments, stages, and non-exact traces."""
    invalid = invalidate_protocol(make_screen_metric("invalid", 0), field)
    valid = make_screen_metric("valid", 1)

    decision = promote_stage2(matrix, (invalid, valid))

    assert decision.selected == ("valid",)
    assert reason in decision.rejected["invalid"]


@pytest.mark.parametrize(("field", "reason"), _PROTOCOL_MUTATIONS)
def test_stage2_rejects_records_with_invalid_protocol_identity(matrix, baseline_100, field: str, reason: str):
    """Stage 2 rejects wrong environments, stages, and non-exact traces."""
    candidate = list(make_two_seed_metrics(reward_scale=1.0, success_delta=0.0, ep_length_scale=1.0))
    candidate[0] = invalidate_protocol(candidate[0], field)

    decision = promote_finalists(matrix, baseline_100, candidate)

    assert decision.selected == ()
    assert reason in decision.rejected["candidate"]


def test_stage2_retains_terminal_failure_reason(matrix, baseline_100):
    """Stage 2 retains an explicit terminal failure instead of treating it as malformed data."""
    candidate = list(make_two_seed_metrics(reward_scale=1.0, success_delta=0.0, ep_length_scale=1.0))
    candidate[0] = dataclasses.replace(candidate[0], failure="capacity")

    decision = promote_finalists(matrix, baseline_100, candidate)

    assert decision.rejected == {"candidate": "seed 42: capacity"}


@pytest.mark.parametrize(("field", "reason"), _PROTOCOL_MUTATIONS)
def test_final_gate_rejects_records_with_invalid_protocol_identity(
    matrix, baseline_final, candidate_final, field: str, reason: str
):
    """Final qualification rejects wrong environments, stages, and non-exact traces."""
    candidate = list(candidate_final)
    candidate[0] = invalidate_protocol(candidate[0], field)

    qualification = qualify_finalists(matrix, baseline_final, candidate)["candidate"]

    assert qualification.qualified is False
    assert reason in qualification.reason


@pytest.mark.parametrize(("field", "reason"), _PROTOCOL_MUTATIONS)
def test_final_gate_requires_valid_baseline_protocol_identity(
    matrix, baseline_final, candidate_final, field: str, reason: str
):
    """Final qualification refuses a malformed baseline protocol identity."""
    baseline = list(baseline_final)
    baseline[0] = invalidate_protocol(baseline[0], field)

    with pytest.raises(ValueError, match=reason):
        qualify_finalists(matrix, baseline, candidate_final)


def test_stage1_accepts_wave2_records_and_rejects_non_screen_seed(matrix):
    """Stage 1 promotion accepts both waves but only the screening seed."""
    wave2 = dataclasses.replace(make_screen_metric("wave2", 0), stage="wave2")
    wrong_seed = dataclasses.replace(make_screen_metric("wrong_seed", 1), seed=43)

    decision = promote_stage2(matrix, (wave2, wrong_seed))

    assert decision.selected == ("wave2",)
    assert decision.rejected == {"wrong_seed": "requires seed 42"}


def test_stage2_requires_halve_stage_for_baseline(matrix, baseline_100):
    """Stage 2 refuses a baseline record from a different stage."""
    baseline = (dataclasses.replace(baseline_100[0], stage="baseline"), baseline_100[1])
    candidate = make_two_seed_metrics(reward_scale=1.0, success_delta=0.0, ep_length_scale=1.0)

    with pytest.raises(ValueError, match="stage halve"):
        promote_finalists(matrix, baseline, candidate)
