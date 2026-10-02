"""Core TRACE-ML verification orchestration and evidence computation."""

from __future__ import annotations

import math
from collections.abc import Callable

from trace_ml.verification import failure_types as FT
from trace_ml.verification.numeric import finite_float, stable_midpoint
from trace_ml.verification.numeric import stable_mean as _mean
from trace_ml.verification.numeric import stable_population_std as _population_std
from trace_ml.verification.numeric import within_tolerance as _matches
from trace_ml.verification.seedset_checks import (
    COMPARISON_AGGREGATIONS,
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
        return float(ordered[midpoint])
    return stable_midpoint(ordered[midpoint - 1], ordered[midpoint])


def _is_finite_number(value) -> bool:
    return finite_float(value) is not None


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
        if field not in claim or value is None:
            problems.append(f"claim missing required field '{field}'")
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"claim field '{field}' must be a non-empty string")

    if "claimed_aggregation" not in claim:
        problems.append("claim missing required field 'claimed_aggregation'")

    aggregation = claim.get("claimed_aggregation", "mean")
    if not isinstance(aggregation, str) or not aggregation.strip():
        problems.append("claim field 'claimed_aggregation' must be a non-empty string")

    claimed_value = claim.get("claimed_value")
    if aggregation in ("range", "min_max_range"):
        for field in ("claimed_low_value", "claimed_high_value"):
            if field not in claim or claim.get(field) is None:
                problems.append(f"claim missing required numeric field '{field}'")
        if claimed_value is not None and not _is_finite_number(claimed_value):
            problems.append("claim field 'claimed_value' must be a finite number or null")
    else:
        if "claimed_value" not in claim:
            problems.append("claim missing required field 'claimed_value'")
        elif not _is_finite_number(claimed_value):
            problems.append("claim field 'claimed_value' must be a finite number")

    if aggregation == "mean_plus_minus_std" and claim.get("uncertainty_value") is None:
        problems.append("claim missing required numeric field 'uncertainty_value'")
    if aggregation == "single_seed" and claim.get("claimed_seed_id") is None:
        problems.append("claim missing required field 'claimed_seed_id'")

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

    for field in ("baseline", "claimed_baseline_method"):
        value = claim.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            problems.append(f"claim field '{field}' must be a non-empty string or null")

    claimed_seed_id = claim.get("claimed_seed_id")
    if claimed_seed_id is not None and (
        isinstance(claimed_seed_id, bool)
        or not isinstance(claimed_seed_id, (int, str))
        or (isinstance(claimed_seed_id, str) and not claimed_seed_id.strip())
    ):
        problems.append(
            "claim field 'claimed_seed_id' must be an integer, non-empty string, or null"
        )

    for field in ("candidate_run_ids", "baseline_run_ids"):
        run_ids = claim.get(field)
        if run_ids is None:
            continue
        valid_ids = (
            isinstance(run_ids, list)
            and bool(run_ids)
            and all(isinstance(run_id, str) and run_id.strip() for run_id in run_ids)
            and len(run_ids) == len(set(run_ids))
        )
        if not valid_ids:
            problems.append(
                f"claim field '{field}' must be a non-empty list of unique non-empty strings"
            )

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
            if field not in run:
                continue
            value = run[field]
            if not isinstance(value, str) or not value.strip():
                problems.append(f"run[{index}] field '{field}' must be a non-empty string")

        run_id = run.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            if run_id in seen_run_ids:
                problems.append(f"duplicate run_id '{run_id}'")
            seen_run_ids.add(run_id)

        if "seed" in run:
            seed = run["seed"]
            if (
                isinstance(seed, bool)
                or not isinstance(seed, (int, str))
                or (isinstance(seed, str) and not seed.strip())
            ):
                problems.append(f"run[{index}] field 'seed' must be an integer or non-empty string")
            else:
                seed_types.add(type(seed))

        if "status" in run:
            status = run["status"]
            if not isinstance(status, str) or status not in ("completed", "failed", "running"):
                problems.append(
                    f"run[{index}] field 'status' must be completed, failed, or running"
                )

        if "metric_value" in run and not _is_finite_number(run["metric_value"]):
            problems.append(f"run[{index}] field 'metric_value' must be a finite number")

    if len(seed_types) > 1:
        problems.append("run seed identifiers must use one consistent type")

    return problems


def _sort_key(value):
    return (type(value).__name__, repr(value))


def _reference_error(rationale: str) -> dict:
    return {"failure_type": FT.INSUFFICIENT_EVIDENCE, "rationale": rationale}


def _resolve_run_sets(claim: dict, runs: list) -> tuple[list, list, dict | None]:
    """Resolve the exact method and baseline records referenced by a claim.

    For ordinary claims, candidate_run_ids is an optional manifest for the supplied
    run list. For comparison claims, explicit method and baseline manifests are
    required so a caller cannot self-assert the baseline value.
    """

    ledger_ids = {run["run_id"] for run in runs}
    candidate_ids = claim.get("candidate_run_ids")
    baseline_ids = claim.get("baseline_run_ids")
    aggregation = claim.get("claimed_aggregation", "mean")

    if aggregation in COMPARISON_AGGREGATIONS:
        if not candidate_ids:
            return (
                [],
                [],
                _reference_error(
                    "Comparison claims require candidate_run_ids for the method evidence."
                ),
            )
        if not baseline_ids:
            return (
                [],
                [],
                _reference_error(
                    "Comparison claims require baseline_run_ids for independently supplied baseline evidence."
                ),
            )
        if not claim.get("baseline"):
            return (
                [],
                [],
                _reference_error("Comparison claims require a non-empty baseline method name."),
            )
        overlap = set(candidate_ids) & set(baseline_ids)
        if overlap:
            return (
                [],
                [],
                _reference_error(
                    f"Method and baseline run manifests overlap at run ID(s) {sorted(overlap)}."
                ),
            )
        referenced_ids = set(candidate_ids) | set(baseline_ids)
    else:
        if baseline_ids is not None:
            return (
                [],
                [],
                _reference_error("baseline_run_ids is only valid for comparison aggregations."),
            )
        referenced_ids = set(candidate_ids) if candidate_ids is not None else ledger_ids

    missing = referenced_ids - ledger_ids
    extra = ledger_ids - referenced_ids
    if missing:
        return (
            [],
            [],
            _reference_error(
                f"Referenced run ID(s) {sorted(missing)} are missing from the supplied ledger."
            ),
        )
    if extra:
        return (
            [],
            [],
            _reference_error(
                f"Supplied ledger contains unreferenced run ID(s) {sorted(extra)}; "
                "the evidence manifest and supplied records must match exactly."
            ),
        )

    if aggregation in COMPARISON_AGGREGATIONS:
        candidate_set = set(candidate_ids)
        baseline_set = set(baseline_ids)
        method_runs = sorted(
            (run for run in runs if run["run_id"] in candidate_set),
            key=lambda run: run["run_id"],
        )
        baseline_runs = sorted(
            (run for run in runs if run["run_id"] in baseline_set),
            key=lambda run: run["run_id"],
        )
        return method_runs, baseline_runs, None

    return sorted(runs, key=lambda run: run["run_id"]), [], None


def _check_baseline_evidence(claim: dict, baseline_runs: list) -> dict | None:
    if not baseline_runs:
        return _reference_error("No baseline run evidence was supplied for the comparison claim.")

    incomplete = [run["run_id"] for run in baseline_runs if run["status"] != "completed"]
    if incomplete:
        return _reference_error(f"Baseline run(s) {sorted(incomplete)} are not marked 'completed'.")

    expected_baseline = claim["baseline"]
    datasets = {run["dataset"] for run in baseline_runs}
    methods = {run["method"] for run in baseline_runs}
    metrics = {run["metric_name"] for run in baseline_runs}
    splits = {run["metric_split"] for run in baseline_runs}

    if datasets != {claim["dataset"]}:
        return {
            "failure_type": FT.DATASET_MISMATCH,
            "rationale": (
                f"Baseline ledger dataset(s) {sorted(datasets)} do not match "
                f"claim dataset '{claim['dataset']}'."
            ),
        }
    if methods != {expected_baseline}:
        return {
            "failure_type": FT.STALE_BASELINE,
            "rationale": (
                f"Baseline ledger method(s) {sorted(methods)} do not match "
                f"configured baseline '{expected_baseline}'."
            ),
        }
    if metrics != {claim["metric"]}:
        return {
            "failure_type": FT.METRIC_MISMATCH,
            "rationale": (
                f"Baseline ledger metric(s) {sorted(metrics)} do not match "
                f"claim metric '{claim['metric']}'."
            ),
        }
    if splits != {claim["metric_split"]}:
        return {
            "failure_type": FT.SPLIT_MISMATCH,
            "rationale": (
                f"Baseline ledger split(s) {sorted(splits)} do not match "
                f"claim split '{claim['metric_split']}'."
            ),
        }

    baseline_mean = _mean([float(run["metric_value"]) for run in baseline_runs])
    stated_value = claim.get("true_baseline_value")
    if stated_value is not None and not _matches(stated_value, baseline_mean, TOLERANCE):
        return {
            "failure_type": FT.STALE_BASELINE,
            "rationale": (
                f"Claim states baseline value {stated_value:.4f}, but the referenced "
                f"baseline runs have mean {baseline_mean:.4f}."
            ),
        }
    return None


def compute_evidence(claim: dict, runs: list, baseline_runs: list | None = None) -> dict:
    baseline_runs = baseline_runs or []
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

    computed_mean = _mean(values)
    computed_median = _median(values)
    computed_std = _population_std(values)
    computed_min = min(values) if values else None
    computed_max = max(values) if values else None

    baseline_values = [
        float(run["metric_value"])
        for run in baseline_runs
        if isinstance(run, dict) and _is_finite_number(run.get("metric_value"))
    ]
    baseline_run_ids = sorted(
        {
            run.get("run_id")
            for run in baseline_runs
            if isinstance(run, dict) and run.get("run_id") is not None
        },
        key=_sort_key,
    )
    computed_baseline_mean = _mean(baseline_values)

    ledger_run_record_count = len(runs)
    ledger_unique_seed_count = len(ledger_seeds)
    duplicate_seed_count = max(0, ledger_run_record_count - ledger_unique_seed_count)

    claimed_value = claim.get("claimed_value")
    aggregation_claimed = claim.get("claimed_aggregation", "mean")
    aggregation_computed = None
    computed_comparison_value = None
    if computed_mean is not None and computed_baseline_mean is not None:
        if aggregation_claimed == "relative_improvement":
            if computed_baseline_mean != 0:
                computed_comparison_value = computed_mean / computed_baseline_mean - 1.0
        elif aggregation_claimed in {"delta", "absolute_improvement"}:
            computed_comparison_value = computed_mean - computed_baseline_mean
        if computed_comparison_value is not None and not math.isfinite(computed_comparison_value):
            computed_comparison_value = None

    if (
        aggregation_claimed in COMPARISON_AGGREGATIONS
        and _is_finite_number(claimed_value)
        and computed_comparison_value is not None
        and math.isfinite(computed_comparison_value)
        and _matches(claimed_value, computed_comparison_value, TOLERANCE)
    ):
        aggregation_computed = aggregation_claimed
    elif values and _is_finite_number(claimed_value):
        if computed_mean is not None and _matches(claimed_value, computed_mean, TOLERANCE):
            aggregation_computed = "mean"
        elif computed_max is not None and _matches(claimed_value, computed_max, TOLERANCE):
            aggregation_computed = "best"
        elif computed_min is not None and _matches(claimed_value, computed_min, TOLERANCE):
            aggregation_computed = "worst"
        elif any(_matches(claimed_value, value, TOLERANCE) for value in values):
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
        "baseline_method_claimed": claim.get("baseline"),
        "baseline_run_ids": baseline_run_ids,
        "baseline_run_record_count": len(baseline_runs),
        "computed_baseline_mean": computed_baseline_mean,
        "computed_comparison_value": computed_comparison_value,
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
    if aggregation in COMPARISON_AGGREGATIONS:
        computed = evidence.get("computed_comparison_value")
        if computed is not None:
            return f"{computed:.4f}"
    if claimed_value is not None:
        return f"{claimed_value:.4f}"
    return "the stated value(s)"


def _result_from_failure(
    failure: dict,
    supporting_run_ids: list[str],
    evidence: dict,
) -> dict:
    failure_type = failure["failure_type"]
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
        "rationale": failure["rationale"],
    }


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
    supporting_run_ids = sorted(
        {
            run.get("run_id")
            for run in safe_runs
            if isinstance(run, dict) and isinstance(run.get("run_id"), str) and run.get("run_id")
        }
    )
    evidence = compute_evidence(safe_claim, safe_runs)

    if validation_problems:
        return {
            "verdict": FT.VERDICT_INSUFFICIENT_EVIDENCE,
            "failure_type": FT.INSUFFICIENT_EVIDENCE,
            "supporting_run_ids": supporting_run_ids,
            "evidence": evidence,
            "rationale": "Input schema validation failed: " + "; ".join(validation_problems) + ".",
        }

    method_runs, baseline_runs, reference_failure = _resolve_run_sets(claim, runs)
    if reference_failure is not None:
        return _result_from_failure(reference_failure, supporting_run_ids, evidence)

    supporting_run_ids = sorted(
        [run["run_id"] for run in method_runs] + [run["run_id"] for run in baseline_runs]
    )
    evidence = compute_evidence(claim, method_runs, baseline_runs)

    effective_claim = dict(claim)
    if claim.get("claimed_aggregation") in COMPARISON_AGGREGATIONS:
        baseline_failure = _check_baseline_evidence(claim, baseline_runs)
        if baseline_failure is not None:
            return _result_from_failure(baseline_failure, supporting_run_ids, evidence)

        computed_baseline_mean = evidence["computed_baseline_mean"]
        if computed_baseline_mean is None:
            return _result_from_failure(
                _reference_error("Baseline evidence does not contain a finite metric value."),
                supporting_run_ids,
                evidence,
            )
        effective_claim["true_baseline_value"] = computed_baseline_mean

    for check in active_checks:
        result = check(effective_claim, method_runs)
        if result is None:
            continue
        return _result_from_failure(result, supporting_run_ids, evidence)

    aggregation = claim.get("claimed_aggregation", "mean")
    expected_text = _expected_value_text(effective_claim, evidence)
    if baseline_runs:
        record_text = (
            f"{len(method_runs)} method run record(s) and "
            f"{len(baseline_runs)} baseline run record(s)"
        )
    else:
        record_text = f"{len(method_runs)} run record(s)"

    rationale = (
        f"Claim's {aggregation} value ({expected_text}) matches {record_text}, "
        "with dataset, method, metric, split, and referenced evidence consistent."
    )

    return {
        "verdict": FT.VERDICT_SUPPORTED,
        "failure_type": FT.SUPPORTED,
        "supporting_run_ids": supporting_run_ids,
        "evidence": evidence,
        "rationale": rationale,
    }
