from copy import deepcopy

from trace_ml.verification.verify_claim import verify

CLAIM = {
    "dataset": "dataset_a",
    "method": "method_a",
    "metric": "accuracy",
    "metric_split": "test",
    "claimed_seed_count": 2,
    "claimed_aggregation": "mean",
    "claimed_value": 0.75,
}

RUNS = [
    {
        "run_id": "r0",
        "dataset": "dataset_a",
        "method": "method_a",
        "seed": 0,
        "metric_name": "accuracy",
        "metric_split": "test",
        "metric_value": 0.7,
        "status": "completed",
    },
    {
        "run_id": "r1",
        "dataset": "dataset_a",
        "method": "method_a",
        "seed": 1,
        "metric_name": "accuracy",
        "metric_split": "test",
        "metric_value": 0.8,
        "status": "completed",
    },
]


def test_missing_optional_legacy_count_does_not_crash():
    claim = deepcopy(CLAIM)
    claim.pop("claimed_seed_count")
    assert verify(claim, deepcopy(RUNS))["verdict"] == "supported"


def test_explicit_count_type_requires_corresponding_count():
    claim = {**CLAIM, "claimed_count_type": "runs"}
    result = verify(claim, deepcopy(RUNS))
    assert "claimed_run_count" in result["rationale"]


def test_unknown_count_type_is_nonjudgment_not_silent_pass():
    claim = {**CLAIM, "claimed_count_type": "folds"}
    assert verify(claim, deepcopy(RUNS))["verdict"] == "insufficient_evidence"


def test_invalid_range_bound_is_nonjudgment():
    claim = {
        **CLAIM,
        "claimed_aggregation": "range",
        "claimed_value": None,
        "claimed_low_value": "0.7",
        "claimed_high_value": 0.8,
    }
    assert verify(claim, deepcopy(RUNS))["verdict"] == "insufficient_evidence"


def test_nonfinite_uncertainty_is_nonjudgment():
    claim = {
        **CLAIM,
        "claimed_aggregation": "mean_plus_minus_std",
        "uncertainty_value": float("inf"),
    }
    assert verify(claim, deepcopy(RUNS))["verdict"] == "insufficient_evidence"


def test_boolean_claimed_value_is_rejected():
    claim = {**CLAIM, "claimed_value": True}
    assert verify(claim, deepcopy(RUNS))["verdict"] == "insufficient_evidence"


def test_mixed_seed_identifier_types_are_rejected_safely():
    runs = deepcopy(RUNS)
    runs[1]["seed"] = "1"
    assert verify(deepcopy(CLAIM), runs)["verdict"] == "insufficient_evidence"


def test_duplicate_run_id_is_rejected():
    runs = deepcopy(RUNS)
    runs[1]["run_id"] = "r0"
    result = verify(deepcopy(CLAIM), runs)
    assert "duplicate run_id" in result["rationale"]


def test_duplicate_seed_for_seed_aggregation_is_ambiguous():
    runs = deepcopy(RUNS)
    runs[1]["seed"] = 0
    claim = {
        **CLAIM,
        "claimed_count_type": "seeds",
        "claimed_seed_count": 1,
    }
    assert verify(claim, runs)["failure_type"] == "ambiguous_seed_count"


def test_default_pipeline_checks_named_baseline_identity():
    claim = {
        **CLAIM,
        "claimed_aggregation": "delta",
        "claimed_value": 0.05,
        "claimed_seed_count": None,
        "baseline": "baseline_a",
        "claimed_baseline_method": "baseline_b",
        "true_baseline_value": 0.70,
    }
    assert verify(claim, deepcopy(RUNS))["failure_type"] == "stale_baseline"


def test_repeated_verification_is_deterministic():
    first = verify(deepcopy(CLAIM), deepcopy(RUNS))
    second = verify(deepcopy(CLAIM), list(reversed(deepcopy(RUNS))))
    assert first["evidence"]["computed_mean"] == second["evidence"]["computed_mean"]
