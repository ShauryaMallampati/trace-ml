import math

import pytest

from trace_ml.verification import failure_types as FT
from trace_ml.verification.verify_claim import verify


def make_runs(values=(0.7, 0.8, 0.9), **overrides):
    return [
        {
            "run_id": f"run-{index}",
            "dataset": "dataset-a",
            "method": "method-a",
            "seed": index,
            "metric_name": "accuracy",
            "metric_split": "test",
            "metric_value": value,
            "status": "completed",
            **overrides,
        }
        for index, value in enumerate(values)
    ]


def make_claim(**overrides):
    claim = {
        "dataset": "dataset-a",
        "method": "method-a",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "mean",
        "claimed_value": 0.8,
    }
    claim.update(overrides)
    return claim


@pytest.mark.parametrize(
    ("aggregation", "claimed_value", "extra"),
    [
        ("mean", 0.8, {}),
        ("best", 0.9, {}),
        ("max", 0.9, {}),
        ("worst", 0.7, {}),
        ("min", 0.7, {}),
        ("median", 0.8, {}),
        ("std", math.sqrt(0.02 / 3), {}),
        ("standard_deviation", math.sqrt(0.02 / 3), {}),
        ("range", None, {"claimed_low_value": 0.7, "claimed_high_value": 0.9}),
        ("min_max_range", None, {"claimed_low_value": 0.7, "claimed_high_value": 0.9}),
        (
            "mean_plus_minus_std",
            0.8,
            {"uncertainty_value": math.sqrt(0.02 / 3)},
        ),
        ("single_seed", 0.8, {"claimed_seed_id": 1}),
        ("baseline_value", 0.8, {}),
        ("delta", 0.2, {"true_baseline_value": 0.6}),
        ("absolute_improvement", 0.2, {"true_baseline_value": 0.6}),
        (
            "relative_improvement",
            (0.8 - 0.6) / 0.6,
            {"true_baseline_value": 0.6},
        ),
    ],
)
def test_supported_aggregation_matrix(aggregation, claimed_value, extra):
    claim = make_claim(
        claimed_aggregation=aggregation,
        claimed_value=claimed_value,
        **extra,
    )
    assert verify(claim, make_runs())["verdict"] == FT.VERDICT_SUPPORTED


@pytest.mark.parametrize(
    ("aggregation", "extra"),
    [
        ("median", {"claimed_value": 0.75}),
        ("std", {"claimed_value": 0.5}),
        ("range", {"claimed_value": None, "claimed_low_value": 0.6, "claimed_high_value": 0.9}),
        ("mean_plus_minus_std", {"claimed_value": 0.8, "uncertainty_value": 0.5}),
        ("single_seed", {"claimed_value": 0.75, "claimed_seed_id": 1}),
        ("baseline_value", {"claimed_value": 0.75}),
        ("delta", {"claimed_value": 0.1, "true_baseline_value": 0.6}),
        ("relative_improvement", {"claimed_value": 0.1, "true_baseline_value": 0.6}),
    ],
)
def test_wrong_aggregation_values_are_violations(aggregation, extra):
    result = verify(make_claim(claimed_aggregation=aggregation, **extra), make_runs())
    assert result["verdict"] == FT.VERDICT_VIOLATION
    assert result["failure_type"] == FT.AGGREGATION_MISMATCH


@pytest.mark.parametrize(
    "claim",
    [
        make_claim(claimed_aggregation="range", claimed_value=None),
        make_claim(claimed_aggregation="mean_plus_minus_std", uncertainty_value=None),
        make_claim(claimed_aggregation="single_seed", claimed_seed_id=None),
        make_claim(claimed_aggregation="delta", true_baseline_value=None),
        make_claim(claimed_aggregation="relative_improvement", true_baseline_value=0.0),
        make_claim(claimed_aggregation="unknown_role"),
    ],
)
def test_uncheckable_aggregations_return_insufficient_evidence(claim):
    result = verify(claim, make_runs())
    assert result["verdict"] == FT.VERDICT_INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    "claim",
    [
        make_claim(claimed_count_type="seeds", claimed_seed_count=3),
        make_claim(claimed_count_type="runs", claimed_run_count=3),
        make_claim(claimed_count_type="trials", claimed_trial_count=3),
    ],
)
def test_explicit_count_types_can_be_supported(claim):
    assert verify(claim, make_runs())["verdict"] == FT.VERDICT_SUPPORTED


def test_missing_seed_is_reported():
    result = verify(make_claim(claimed_count_type="seeds", claimed_seed_count=4), make_runs())
    assert result["failure_type"] == FT.SEED_OMISSION


def test_omitted_run_is_reported():
    result = verify(make_claim(claimed_count_type="runs", claimed_run_count=2), make_runs())
    assert result["failure_type"] == FT.RETROSPECTIVE_SEED_FILTERING


def test_duplicate_seed_makes_trial_count_ambiguous():
    runs = make_runs()
    runs[2]["seed"] = 1
    result = verify(make_claim(claimed_count_type="trials", claimed_trial_count=3), runs)
    assert result["verdict"] == FT.VERDICT_INSUFFICIENT_EVIDENCE
    assert result["failure_type"] == FT.AMBIGUOUS_SEED_COUNT


@pytest.mark.parametrize(
    ("run_override", "expected"),
    [
        ({"status": "failed"}, FT.INSUFFICIENT_EVIDENCE),
        ({"dataset": "dataset-b"}, FT.SPLIT_MISMATCH),
        ({"method": "method-b"}, FT.STALE_BASELINE),
        ({"metric_name": "f1"}, FT.METRIC_MISMATCH),
        ({"metric_split": "validation"}, FT.SPLIT_MISMATCH),
    ],
)
def test_run_identity_and_completion_fail_closed(run_override, expected):
    result = verify(make_claim(), make_runs(**run_override))
    assert result["failure_type"] == expected


def test_comparison_baseline_identity_is_checked():
    claim = make_claim(
        claimed_aggregation="delta",
        claimed_value=0.2,
        baseline="baseline-a",
        claimed_baseline_method="baseline-b",
        true_baseline_value=0.6,
    )
    assert verify(claim, make_runs())["failure_type"] == FT.STALE_BASELINE


def test_empty_evidence_is_not_a_violation():
    result = verify(make_claim(), [])
    assert result["verdict"] == FT.VERDICT_INSUFFICIENT_EVIDENCE
    assert result["failure_type"] == FT.INSUFFICIENT_EVIDENCE
