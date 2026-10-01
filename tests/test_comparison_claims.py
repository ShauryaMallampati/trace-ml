from trace_ml.verification.seedset_checks import check_seed_count
from trace_ml.verification.verify_claim import CHECKS, verify

CHECKS_NO_SEED_COUNT = [check for check in CHECKS if check is not check_seed_count]

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


def base_claim(**overrides):
    claim = {
        "claim_id": "test_delta_claim",
        "dataset": "ds_a",
        "method": "random_forest",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "delta",
        "claimed_count_type": None,
        "claimed_seed_count": None,
        "claimed_run_count": None,
        "claimed_trial_count": None,
        "claimed_low_value": None,
        "claimed_high_value": None,
        "uncertainty_value": None,
        "claimed_seed_id": None,
        "baseline": "logistic_regression",
        "claimed_baseline_method": "logistic_regression",
        "true_baseline_value": 0.8800,
    }
    claim.update(overrides)
    return claim


def test_correct_absolute_delta_is_supported():
    result = verify(base_claim(claimed_value=0.0525), METHOD_RUNS, checks=CHECKS_NO_SEED_COUNT)
    assert result["verdict"] == "supported"


def test_wrong_absolute_delta_is_aggregation_mismatch():
    result = verify(base_claim(claimed_value=0.10), METHOD_RUNS, checks=CHECKS_NO_SEED_COUNT)
    assert result["failure_type"] == "aggregation_mismatch"


def test_correct_negative_delta_is_supported():
    runs = [{**run, "metric_value": 0.8275} for run in METHOD_RUNS]
    result = verify(base_claim(claimed_value=-0.0525), runs, checks=CHECKS_NO_SEED_COUNT)
    assert result["verdict"] == "supported"


def test_missing_baseline_evidence_is_insufficient_evidence():
    result = verify(
        base_claim(claimed_value=0.0525, true_baseline_value=None),
        [],
        checks=CHECKS_NO_SEED_COUNT,
    )
    assert result["verdict"] == "insufficient_evidence"


def test_wrong_baseline_method_is_stale_baseline():
    result = verify(
        base_claim(claimed_value=0.0525, claimed_baseline_method="decision_tree"),
        METHOD_RUNS,
        checks=CHECKS_NO_SEED_COUNT,
    )
    assert result["failure_type"] == "stale_baseline"
