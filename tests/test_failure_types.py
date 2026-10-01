from trace_ml.verification import failure_types as FT

EXPECTED_FAILURE_TYPES = {
    "supported",
    "fabricated_metric",
    "dataset_mismatch",
    "method_mismatch",
    "metric_mismatch",
    "split_mismatch",
    "stale_baseline",
    "seed_omission",
    "seed_cherry_pick",
    "retrospective_seed_filtering",
    "aggregation_mismatch",
    "insufficient_evidence",
    "ambiguous_seed_count",
    "unsupported_claim_type",
}


def test_expected_failure_type_count():
    assert len(FT.ALL_FAILURE_TYPES) == 14


def test_no_duplicate_failure_types():
    assert len(set(FT.ALL_FAILURE_TYPES)) == len(FT.ALL_FAILURE_TYPES)


def test_expected_failure_type_labels_present():
    assert set(FT.ALL_FAILURE_TYPES) == EXPECTED_FAILURE_TYPES


def test_nonjudgment_failures_map_to_insufficient_evidence():
    assert FT.INSUFFICIENT_EVIDENCE in FT.INSUFFICIENT_VERDICT_TYPES
    assert FT.AMBIGUOUS_SEED_COUNT in FT.INSUFFICIENT_VERDICT_TYPES
    assert FT.UNSUPPORTED_CLAIM_TYPE in FT.INSUFFICIENT_VERDICT_TYPES


def test_verdict_constants():
    assert FT.VERDICT_SUPPORTED == "supported"
    assert FT.VERDICT_VIOLATION == "violation"
    assert FT.VERDICT_INSUFFICIENT_EVIDENCE == "insufficient_evidence"
