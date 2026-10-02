import pytest

from trace_ml.verification.verify_claim import verify

BASE_CLAIM = {
    "dataset": "d",
    "method": "m",
    "metric": "accuracy",
    "metric_split": "test",
    "claimed_seed_count": 1,
    "claimed_aggregation": "mean",
    "claimed_value": 0.8,
}

BASE_RUN = {
    "run_id": "r0",
    "dataset": "d",
    "method": "m",
    "seed": 0,
    "metric_name": "accuracy",
    "metric_split": "test",
    "metric_value": 0.8,
    "status": "completed",
}


def test_missing_run_metric_value_is_insufficient_evidence():
    run = dict(BASE_RUN)
    del run["metric_value"]
    result = verify(dict(BASE_CLAIM), [run])
    assert result["verdict"] == "insufficient_evidence"


def test_nonfinite_claim_value_is_insufficient_evidence():
    claim = dict(BASE_CLAIM, claimed_value=float("nan"))
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"


def test_missing_claim_identity_is_insufficient_evidence():
    claim = dict(BASE_CLAIM)
    del claim["metric"]
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"


def test_empty_string_seed_is_insufficient_evidence():
    run = dict(BASE_RUN, seed="")
    result = verify(dict(BASE_CLAIM), [run])
    assert result["verdict"] == "insufficient_evidence"


def test_boolean_claimed_seed_id_is_insufficient_evidence():
    claim = dict(
        BASE_CLAIM,
        claimed_aggregation="single_seed",
        claimed_value=0.8,
        claimed_seed_id=True,
    )
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"


def test_duplicate_candidate_run_ids_are_insufficient_evidence():
    claim = dict(BASE_CLAIM, candidate_run_ids=["r0", "r0"])
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"


def test_whitespace_only_identity_is_insufficient_evidence():
    claim = dict(BASE_CLAIM, method="   ")
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"


def test_unknown_run_status_is_insufficient_evidence():
    run = dict(BASE_RUN, status="done")
    result = verify(dict(BASE_CLAIM), [run])
    assert result["verdict"] == "insufficient_evidence"


def test_null_claimed_value_is_insufficient_evidence():
    claim = dict(BASE_CLAIM, claimed_value=None)
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"
    assert "claimed_value" in result["rationale"]


def test_missing_claimed_aggregation_is_insufficient_evidence():
    claim = dict(BASE_CLAIM)
    del claim["claimed_aggregation"]
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"
    assert "claimed_aggregation" in result["rationale"]


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "dataset",
        "method",
        "seed",
        "metric_name",
        "metric_split",
        "metric_value",
        "status",
    ],
)
def test_null_required_run_fields_are_insufficient_evidence(field):
    run = dict(BASE_RUN)
    run[field] = None
    result = verify(dict(BASE_CLAIM), [run])
    assert result["verdict"] == "insufficient_evidence"
    assert field in result["rationale"]


@pytest.mark.parametrize("field", ["claimed_low_value", "claimed_high_value"])
def test_range_requires_non_null_bounds(field):
    claim = {
        **BASE_CLAIM,
        "claimed_aggregation": "range",
        "claimed_value": None,
        "claimed_low_value": 0.8,
        "claimed_high_value": 0.8,
    }
    claim[field] = None
    result = verify(claim, [dict(BASE_RUN)])
    assert result["verdict"] == "insufficient_evidence"
    assert field in result["rationale"]
