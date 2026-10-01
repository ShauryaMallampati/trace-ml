from copy import deepcopy

import pytest

from trace_ml.verification.verify_claim import verify

METHOD_RUNS = [
    {
        "run_id": f"run_method_{i}",
        "dataset": "ds_a",
        "method": "random_forest",
        "metric_name": "accuracy",
        "metric_split": "test",
        "status": "completed",
        "seed": i,
        "metric_value": 0.9325,
    }
    for i in range(5)
]

BASELINE_RUNS = [
    {
        "run_id": f"run_baseline_{i}",
        "dataset": "ds_a",
        "method": "logistic_regression",
        "metric_name": "accuracy",
        "metric_split": "test",
        "status": "completed",
        "seed": i,
        "metric_value": 0.88,
    }
    for i in range(5)
]


def base_claim(**overrides):
    claim = {
        "claim_id": "test_delta_claim",
        "dataset": "ds_a",
        "method": "random_forest",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "delta",
        "claimed_value": 0.0525,
        "claimed_count_type": None,
        "claimed_seed_count": None,
        "claimed_run_count": None,
        "claimed_trial_count": None,
        "baseline": "logistic_regression",
        "claimed_baseline_method": "logistic_regression",
        "candidate_run_ids": [run["run_id"] for run in METHOD_RUNS],
        "baseline_run_ids": [run["run_id"] for run in BASELINE_RUNS],
    }
    claim.update(overrides)
    return claim


def ledger(method_runs=None, baseline_runs=None):
    return list(method_runs or METHOD_RUNS) + list(
        BASELINE_RUNS if baseline_runs is None else baseline_runs
    )


def test_correct_absolute_delta_is_supported():
    result = verify(base_claim(), ledger())
    assert result["verdict"] == "supported"
    assert result["evidence"]["computed_baseline_mean"] == pytest.approx(0.88)
    assert result["evidence"]["aggregation_computed"] == "delta"


def test_wrong_absolute_delta_is_aggregation_mismatch():
    result = verify(base_claim(claimed_value=0.10), ledger())
    assert result["failure_type"] == "aggregation_mismatch"


def test_correct_negative_delta_is_supported():
    method_runs = [{**run, "metric_value": 0.8275} for run in METHOD_RUNS]
    result = verify(base_claim(claimed_value=-0.0525), ledger(method_runs=method_runs))
    assert result["verdict"] == "supported"


def test_missing_baseline_manifest_is_insufficient_evidence():
    claim = base_claim()
    claim.pop("baseline_run_ids")
    result = verify(claim, ledger())
    assert result["verdict"] == "insufficient_evidence"


def test_missing_referenced_baseline_run_is_insufficient_evidence():
    result = verify(base_claim(), ledger(baseline_runs=BASELINE_RUNS[:-1]))
    assert result["verdict"] == "insufficient_evidence"
    assert "missing" in result["rationale"].lower()


def test_wrong_baseline_method_is_stale_baseline():
    result = verify(
        base_claim(claimed_baseline_method="decision_tree"),
        ledger(),
    )
    assert result["failure_type"] == "stale_baseline"


def test_baseline_aliases_are_not_silently_normalized():
    result = verify(
        base_claim(claimed_baseline_method="Logistic Regression"),
        ledger(),
    )
    assert result["failure_type"] == "stale_baseline"


def test_baseline_run_method_must_match_configured_baseline():
    baseline_runs = deepcopy(BASELINE_RUNS)
    baseline_runs[0]["method"] = "decision_tree"
    result = verify(base_claim(), ledger(baseline_runs=baseline_runs))
    assert result["failure_type"] == "stale_baseline"


def test_stated_baseline_value_must_match_baseline_runs():
    result = verify(base_claim(true_baseline_value=0.87), ledger())
    assert result["failure_type"] == "stale_baseline"
