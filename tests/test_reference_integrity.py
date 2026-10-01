import math

from trace_ml.verification.verify_claim import verify


def run(run_id, value, *, method="m", seed=0):
    return {
        "run_id": run_id,
        "dataset": "d",
        "method": method,
        "seed": seed,
        "metric_name": "accuracy",
        "metric_split": "test",
        "metric_value": value,
        "status": "completed",
    }


BASE = {
    "dataset": "d",
    "method": "m",
    "metric": "accuracy",
    "metric_split": "test",
}


def test_candidate_manifest_must_match_supplied_records_exactly():
    claim = {
        **BASE,
        "claimed_aggregation": "mean",
        "claimed_value": 0.9,
        "candidate_run_ids": ["r0"],
    }
    result = verify(claim, [run("r0", 0.8), run("r1", 1.0, seed=1)])
    assert result["verdict"] == "insufficient_evidence"
    assert "unreferenced" in result["rationale"].lower()


def test_missing_referenced_candidate_is_not_silently_ignored():
    claim = {
        **BASE,
        "claimed_aggregation": "mean",
        "claimed_value": 0.8,
        "candidate_run_ids": ["missing"],
    }
    result = verify(claim, [run("r0", 0.8)])
    assert result["verdict"] == "insufficient_evidence"
    assert "missing" in result["rationale"].lower()


def test_comparison_requires_independent_baseline_records():
    claim = {
        **BASE,
        "claimed_aggregation": "delta",
        "claimed_value": 0.1,
        "baseline": "baseline",
        "true_baseline_value": 0.7,
        "candidate_run_ids": ["r0"],
    }
    result = verify(claim, [run("r0", 0.8)])
    assert result["verdict"] == "insufficient_evidence"
    assert "baseline_run_ids" in result["rationale"]


def test_four_decimal_precision_does_not_accept_four_ten_thousandths_error():
    claim = {
        **BASE,
        "claimed_aggregation": "mean",
        "claimed_value": 0.8004,
    }
    result = verify(claim, [run("r0", 0.8)])
    assert result["verdict"] == "violation"


def test_result_is_invariant_to_input_run_order():
    claim = {
        **BASE,
        "claimed_aggregation": "mean",
        "claimed_value": 0.8,
    }
    runs = [run("b", 0.9, seed=1), run("a", 0.7, seed=0)]
    forward = verify(claim, runs)
    reverse = verify(claim, list(reversed(runs)))
    assert forward == reverse
    assert forward["supporting_run_ids"] == ["a", "b"]


def test_overlapping_method_and_baseline_manifests_fail_closed():
    claim = {
        **BASE,
        "claimed_aggregation": "delta",
        "claimed_value": 0.1,
        "baseline": "baseline",
        "candidate_run_ids": ["r0"],
        "baseline_run_ids": ["r0"],
    }
    result = verify(claim, [run("r0", 0.8)])
    assert result["verdict"] == "insufficient_evidence"
    assert "overlap" in result["rationale"].lower()


def test_large_finite_metrics_do_not_overflow_mean():
    claim = {
        **BASE,
        "claimed_aggregation": "mean",
        "claimed_value": 1e308,
    }
    result = verify(
        claim,
        [run("r0", 1e308, seed=0), run("r1", 1e308, seed=1)],
    )
    assert result["verdict"] == "supported"
    assert result["evidence"]["computed_mean"] == 1e308


def test_relative_improvement_avoids_overflowing_subtraction():
    method = run("method", 1e308, seed=0)
    baseline = run("baseline", -1e308, method="baseline", seed=0)
    claim = {
        **BASE,
        "claimed_aggregation": "relative_improvement",
        "claimed_value": -2.0,
        "baseline": "baseline",
        "candidate_run_ids": ["method"],
        "baseline_run_ids": ["baseline"],
    }
    result = verify(claim, [method, baseline])
    assert result["verdict"] == "supported"


def test_seed_level_mean_rejects_duplicate_retry_records_even_when_values_match():
    claim = {
        **BASE,
        "claimed_count_type": "seeds",
        "claimed_seed_count": 2,
        "claimed_aggregation": "mean",
        "claimed_value": 1 / 3,
    }
    runs = [
        run("a", 0.0, seed=0),
        run("b", 0.0, seed=0),
        run("c", 1.0, seed=1),
    ]
    result = verify(claim, runs)
    assert result["verdict"] == "insufficient_evidence"
    assert result["failure_type"] == "ambiguous_seed_count"


def test_run_level_mean_can_include_repeated_seed_records():
    claim = {
        **BASE,
        "claimed_count_type": "runs",
        "claimed_run_count": 3,
        "claimed_aggregation": "mean",
        "claimed_value": 1 / 3,
    }
    runs = [
        run("a", 0.0, seed=0),
        run("b", 0.0, seed=0),
        run("c", 1.0, seed=1),
    ]
    result = verify(claim, runs)
    assert result["verdict"] == "supported"


def test_extreme_standard_deviation_remains_finite():
    claim = {
        **BASE,
        "claimed_aggregation": "std",
        "claimed_value": 1e308,
    }
    result = verify(
        claim,
        [run("low", -1e308, seed=0), run("high", 1e308, seed=1)],
    )
    assert result["verdict"] == "supported"
    assert result["evidence"]["computed_std"] == 1e308


def test_single_seed_claim_rejects_conflicting_retries():
    claim = {
        **BASE,
        "claimed_aggregation": "single_seed",
        "claimed_value": 0.8,
        "claimed_seed_id": 7,
    }
    runs = [
        run("retry-a", 0.8, seed=7),
        run("retry-b", 0.9, seed=7),
    ]
    result = verify(claim, runs)
    assert result["verdict"] == "insufficient_evidence"
    assert result["failure_type"] == "ambiguous_seed_count"


def test_unrepresentable_absolute_delta_does_not_leak_infinity():
    method = run("method", 1e308, seed=0)
    baseline = run("baseline", -1e308, method="baseline", seed=0)
    claim = {
        **BASE,
        "claimed_aggregation": "delta",
        "claimed_value": 1e308,
        "baseline": "baseline",
        "candidate_run_ids": ["method"],
        "baseline_run_ids": ["baseline"],
    }
    result = verify(claim, [method, baseline])
    assert result["verdict"] == "insufficient_evidence"
    assert result["evidence"]["computed_comparison_value"] is None


def test_subnormal_mean_preserves_representable_value():
    value = 5e-324
    claim = {
        **BASE,
        "claimed_aggregation": "mean",
        "claimed_value": value,
    }
    result = verify(
        claim,
        [run("a", value, seed=0), run("b", value, seed=1)],
    )
    assert result["verdict"] == "supported"
    assert result["evidence"]["computed_mean"] == value


def test_unrepresentable_json_number_returns_insufficient_instead_of_crashing():
    claim = {
        **BASE,
        "claimed_aggregation": "mean",
        "claimed_value": 0.0,
    }
    result = verify(claim, [run("huge", 10**400)])
    assert result["verdict"] == "insufficient_evidence"
    assert "finite number" in result["rationale"]


def test_extreme_asymmetric_standard_deviation_remains_finite():
    values = [1.79e308, -1.79e308, 1.79e308]
    expected = 1.79e308 * ((8.0 / 9.0) ** 0.5)
    claim = {
        **BASE,
        "claimed_aggregation": "std",
        "claimed_value": expected,
    }
    result = verify(
        claim,
        [
            run("a", values[0], seed=0),
            run("b", values[1], seed=1),
            run("c", values[2], seed=2),
        ],
    )
    assert result["verdict"] == "supported"
    assert math.isclose(result["evidence"]["computed_std"], expected, rel_tol=2e-16)
