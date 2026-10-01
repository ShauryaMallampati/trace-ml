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
