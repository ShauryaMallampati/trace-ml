"""Core TRACE-ML verification orchestration and evidence computation."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

from trace_ml.verification import failure_types as FT
from trace_ml.verification.seedset_checks import (
    TOLERANCE,
    check_aggregation_correctness,
    check_baseline_method_identity,
    check_consistent_dataset_method_metric,
    check_runs_completed,
    check_runs_exist,
    check_seed_count,
)

ValidationCheck = Callable[[dict, list], dict | None]

CHECKS: list[ValidationCheck] = [
    check_runs_exist,
    check_runs_completed,
    check_consistent_dataset_method_metric,
    check_seed_count,
    check_baseline_method_identity,
    check_aggregation_correctness,
]


def _singleton_or_list(values):
    ordered = sorted(values, key=lambda value: (type(value).__name__, repr(value)))
    if len(ordered) == 1:
        return ordered[0]
    if not ordered:
        return None
    return ordered


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _population_std(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _is_finite_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_nonnegative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_inputs(claim: dict, runs: list) -> list[str]:
    if not isinstance(claim, dict):
        return ["claim must be a dictionary"]
    if not isinstance(runs, list):
        return ["runs must be a list"]

    problems: list[str] = []

    for field in ("dataset", "method", "metric", "metric_split"):
        value = claim.get(field)
        if field not in claim or value in (None, ""):
            problems.append(f"claim missing required field '{field}'")
        elif not isinstance(value, str):
            problems.append(f"claim field '{field}' must be a non-empty string")

    aggregation = claim.get("claimed_aggregation", "mean")
    if not isinstance(aggregation, str) or not aggregation:
        problems.append("claim field 'claimed_aggregation' must be a non-empty string")

    if aggregation not in ("range", "min_max_range") and "claimed_value" not in claim:
        problems.append("claim missing required field 'claimed_value'")

    claimed_value = claim.get("claimed_value")
    if claimed_value is not None and not _is_finite_number(claimed_value):
        problems.append("claim field 'claimed_value' must be a finite number or null")

    numeric_optional_fields = (
        "claimed_low_value",
        "claimed_high_value",
        "uncertainty_value",
        "true_baseline_value",
    )
    for field in numeric_optional_fields:
        value = claim.get(field)
        if value is not None and not _is_finite_number(value):
            problems.append(f"claim field '{field}' must be a finite number or null")

    count_type = claim.get("claimed_count_type")
    if count_type not in (None, "seeds", "runs", "trials"):
        problems.append(
            "claim field 'claimed_count_type' must be one of: seeds, runs, trials, or null"
        )

    required_count_field = {
        "seeds": "claimed_seed_count",
        "runs": "claimed_run_count",
        "trials": "claimed_trial_count",
    }.get(count_type)
    if required_count_field is not None:
        count_value = claim.get(required_count_field)
        if count_value is None:
            problems.append(
                f"claim missing required field '{required_count_field}' for count type {count_type}"
            )
        elif not _is_nonnegative_int(count_value):
            problems.append(f"claim field '{required_count_field}' must be a non-negative integer")

    for field in ("claimed_seed_count", "claimed_run_count", "claimed_trial_count"):
        value = claim.get(field)
        if value is not None and not _is_nonnegative_int(value):
            problems.append(f"claim field '{field}' must be a non-negative integer")

    candidate_run_ids = claim.get("candidate_run_ids")
    if candidate_run_ids is not None:
        valid_ids = isinstance(candidate_run_ids, list) and all(
            isinstance(run_id, str) and run_id for run_id in candidate_run_ids
        )
        if not valid_ids:
            problems.append("claim field 'candidate_run_ids' must be a list of non-empty strings")

    required_run_fields = (
        "run_id",
        "dataset",
        "method",
        "seed",
        "metric_name",
        "metric_split",
        "metric_value",
        "status",
    )
    string_run_fields = (
        "run_id",
        "dataset",
        "method",
        "metric_name",
        "metric_split",
        "status",
    )

    seen_run_ids: set[str] = set()
    seed_types: set[type] = set()

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            problems.append(f"run[{index}] must be a dictionary")
            continue

        for field in required_run_fields:
            if field not in run:
                problems.append(f"run[{index}] missing required field '{field}'")

        for field in string_run_fields:
            value = run.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                problems.append(f"run[{index}] field '{field}' must be a non-empty string")

        run_id = run.get("run_id")
        if isinstance(run_id, str) and run_id:
            if run_id in seen_run_ids:
                problems.append(f"duplicate run_id '{run_id}'")
            seen_run_ids.add(run_id)

        seed = run.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, str)):
                problems.append(f"run[{index}] field 'seed' must be an integer or string")
            else:
                seed_types.add(type(seed))

        value = run.get("metric_value")
        if value is not None and not _is_finite_number(value):
            problems.append(f"run[{index}] field 'metric_value' must be a finite number")

    if len(seed_types) > 1:
        problems.append("run seed identifiers must use one consistent type")

    return problems


def _sort_key(value):
    return (type(value).__name__, repr(value))


def compute_evidence(claim: dict, runs: list) -> dict:
    values = [
        float(run["metric_value"])
        for run in runs
        if isinstance(run, dict) and _is_finite_number(run.get("metric_value"))
    ]

    ledger_seeds = sorted(
        {run.get("seed") for run in runs if isinstance(run, dict) and run.get("seed") is not None},
        key=_sort_key,
    )
    ledger_run_ids = sorted(
        {
            run.get("run_id")
            for run in runs
            if isinstance(run, dict) and run.get("run_id") is not None
        },
        key=_sort_key,
    )
    metric_names = {
        run.get("metric_name")
        for run in runs
        if isinstance(run, dict) and run.get("metric_name") is not None
    }
    metric_splits = {
        run.get("metric_split")
        for run in runs
        if isinstance(run, dict) and run.get("metric_split") is not None
    }

    computed_mean = sum(values) / len(values) if values else None
    computed_median = _median(values)
    computed_std = _population_std(values)
    computed_min = min(values) if values else None
    computed_max = max(values) if values else None

    ledger_run_record_count = len(runs)
    ledger_unique_seed_count = len(ledger_seeds)
    duplicate_seed_count = max(0, ledger_run_record_count - ledger_unique_seed_count)

    claimed_value = claim.get("claimed_value")
    aggregation_claimed = claim.get("claimed_aggregation", "mean")
    aggregation_computed = None

    if values and _is_finite_number(claimed_value):
        if computed_mean is not None and abs(claimed_value - computed_mean) < TOLERANCE:
            aggregation_computed = "mean"
        elif computed_max is not None and abs(claimed_value - computed_max) < TOLERANCE:
            aggregation_computed = "best"
        elif computed_min is not None and abs(claimed_value - computed_min) < TOLERANCE:
            aggregation_computed = "worst"
        elif any(abs(claimed_value - value) < TOLERANCE for value in values):
            aggregation_computed = "single_seed"
        else:
            aggregation_computed = "none"

    return {
        "claimed_value": claimed_value,
        "computed_mean": computed_mean,
        "best_seed_value": computed_max,
        "worst_seed_value": computed_min,
        "reported_seed_count": claim.get("claimed_seed_count"),
        "ledger_seed_count": ledger_unique_seed_count,
        "ledger_seeds": ledger_seeds,
        "metric_name_claimed": claim.get("metric"),
        "metric_name_ledger": _singleton_or_list(metric_names),
        "metric_split_claimed": claim.get("metric_split"),
        "metric_split_ledger": _singleton_or_list(metric_splits),
        "aggregation_claimed": aggregation_claimed,
        "aggregation_computed": aggregation_computed,
        "tolerance": TOLERANCE,
        "computed_std": computed_std,
        "computed_min": computed_min,
        "computed_max": computed_max,
        "computed_median": computed_median,
        "reported_run_count": claim.get("claimed_run_count"),
        "reported_trial_count": claim.get("claimed_trial_count"),
        "claimed_count_type": claim.get("claimed_count_type"),
        "ledger_unique_seed_count": ledger_unique_seed_count,
        "ledger_run_record_count": ledger_run_record_count,
        "duplicate_seed_count": duplicate_seed_count,
        "ledger_run_ids": ledger_run_ids,
        "claimed_low_value": claim.get("claimed_low_value"),
        "claimed_high_value": claim.get("claimed_high_value"),
        "uncertainty_value": claim.get("uncertainty_value"),
        "true_baseline_value": claim.get("true_baseline_value"),
    }


def _expected_value_text(claim: dict, evidence: dict) -> str:
    aggregation = claim.get("claimed_aggregation", "mean")
    claimed_value = claim.get("claimed_value")

    if aggregation in ("mean", "baseline_value"):
        return f"{evidence['computed_mean']:.4f}"
    if aggregation in ("best", "max"):
        return f"{evidence['best_seed_value']:.4f}"
    if aggregation in ("worst", "min"):
        return f"{evidence['worst_seed_value']:.4f}"
    if aggregation == "median":
        return f"{evidence['computed_median']:.4f}"
    if aggregation in ("std", "standard_deviation"):
        return f"{evidence['computed_std']:.4f}"
    if aggregation in ("range", "min_max_range"):
        return f"[{evidence['computed_min']:.4f}, {evidence['computed_max']:.4f}]"
    if aggregation == "mean_plus_minus_std":
        return f"{evidence['computed_mean']:.4f} +/- {evidence['computed_std']:.4f}"
    if aggregation == "single_seed":
        return f"{claimed_value:.4f} for seed {claim.get('claimed_seed_id')}"
    if claimed_value is not None:
        return f"{claimed_value:.4f}"
    return "the stated value(s)"


def verify(
    claim: dict,
    runs: list,
    checks: list[ValidationCheck] | None = None,
) -> dict:
    """Verify one structured metric claim against supplied run records."""

    active_checks = CHECKS if checks is None else checks
    safe_claim = claim if isinstance(claim, dict) else {}
    safe_runs = runs if isinstance(runs, list) else []

    validation_problems = _validate_inputs(claim, runs)
    evidence = compute_evidence(safe_claim, safe_runs)
    supporting_run_ids = [
        run.get("run_id") for run in safe_runs if isinstance(run, dict) and run.get("run_id")
    ]

    if validation_problems:
        return {
            "verdict": FT.VERDICT_INSUFFICIENT_EVIDENCE,
            "failure_type": FT.INSUFFICIENT_EVIDENCE,
            "supporting_run_ids": supporting_run_ids,
            "evidence": evidence,
            "rationale": "Input schema validation failed: " + "; ".join(validation_problems) + ".",
        }

    for check in active_checks:
        result = check(claim, runs)
        if result is None:
            continue

        failure_type = result["failure_type"]
        verdict = (
            FT.VERDICT_INSUFFICIENT_EVIDENCE
            if failure_type in FT.INSUFFICIENT_VERDICT_TYPES
            else FT.VERDICT_VIOLATION
        )
        return {
            "verdict": verdict,
            "failure_type": failure_type,
            "supporting_run_ids": supporting_run_ids,
            "evidence": evidence,
            "rationale": result["rationale"],
        }

    aggregation = claim.get("claimed_aggregation", "mean")
    expected_text = _expected_value_text(claim, evidence)
    rationale = (
        f"Claim's {aggregation} value ({expected_text}) matches the ledger across "
        f"{len(runs)} run record(s), with dataset, method, metric, and split consistent."
    )

    return {
        "verdict": FT.VERDICT_SUPPORTED,
        "failure_type": FT.SUPPORTED,
        "supporting_run_ids": supporting_run_ids,
        "evidence": evidence,
        "rationale": rationale,
    }
