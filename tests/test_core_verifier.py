import statistics

from tests.fixtures.synthetic import make_claim, make_runs, run_ids
from trace_ml.verification import failure_types as FT
from trace_ml.verification.verify_claim import verify

VALUES = [0.80, 0.85, 0.90, 0.95, 1.00]


def _verify(aggregation, value, **extra):
    runs = make_runs(VALUES)
    claim = make_claim(
        claimed_value=value,
        claimed_aggregation=aggregation,
        candidate_run_ids=run_ids(runs),
        **extra,
    )
    return verify(claim, runs)


def test_mean_supported():
    assert _verify("mean", 0.9)["verdict"] == FT.VERDICT_SUPPORTED


def test_best_supported():
    assert _verify("best", 1.0)["verdict"] == FT.VERDICT_SUPPORTED


def test_worst_supported():
    assert _verify("worst", 0.8)["verdict"] == FT.VERDICT_SUPPORTED


def test_median_supported():
    expected = statistics.median(VALUES)
    assert _verify("median", expected)["verdict"] == FT.VERDICT_SUPPORTED


def test_std_supported():
    expected = statistics.pstdev(VALUES)
    assert _verify("std", expected)["verdict"] == FT.VERDICT_SUPPORTED


def test_range_supported_without_scalar_claimed_value():
    runs = make_runs(VALUES)
    claim = make_claim(
        claimed_aggregation="range",
        candidate_run_ids=run_ids(runs),
        claimed_low_value=0.8,
        claimed_high_value=1.0,
    )
    claim.pop("claimed_value")

    assert verify(claim, runs)["verdict"] == FT.VERDICT_SUPPORTED


def test_seed_omission():
    result = _verify("mean", 0.9, claimed_seed_count=10)
    assert result["failure_type"] == FT.SEED_OMISSION


def test_retrospective_filtering():
    result = _verify("mean", 0.9, claimed_seed_count=3)
    assert result["failure_type"] == FT.RETROSPECTIVE_SEED_FILTERING


def test_metric_mismatch():
    result = _verify("mean", 0.9, metric="f1")
    assert result["failure_type"] == FT.METRIC_MISMATCH


def test_incomplete_is_insufficient():
    runs = make_runs(VALUES)
    runs[0]["status"] = "failed"
    claim = make_claim(
        claimed_value=0.9,
        candidate_run_ids=run_ids(runs),
    )
    assert verify(claim, runs)["verdict"] == FT.VERDICT_INSUFFICIENT_EVIDENCE
