"""Deterministic semantic checks for TRACE-ML claim/run pairs."""

from __future__ import annotations

import math

from trace_ml.verification import failure_types as FT

TOLERANCE = 5e-5

COMPARISON_AGGREGATIONS = frozenset({"delta", "absolute_improvement", "relative_improvement"})


def _failure(failure_type: str, rationale: str) -> dict:
    return {"failure_type": failure_type, "rationale": rationale}


def _mean(values):
    if not values:
        return None
    count = len(values)
    return math.fsum(value / count for value in values)


def _median(values):
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
    mean = _mean(values)
    result = math.hypot(*(value - mean for value in values)) / math.sqrt(len(values))
    return result if math.isfinite(result) else None


def _values_by_seed(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(run["seed"], set()).add(run["metric_value"])
    return grouped


def _conflicting_seed_values(runs):
    return [seed for seed, values in _values_by_seed(runs).items() if len(values) > 1]


def _repeated_seeds(runs):
    counts = {}
    for run in runs:
        seed = run["seed"]
        counts[seed] = counts.get(seed, 0) + 1
    return sorted(
        (seed for seed, count in counts.items() if count > 1),
        key=lambda value: (type(value).__name__, repr(value)),
    )


def check_runs_exist(claim, runs):
    del claim
    if not runs:
        return _failure(
            FT.INSUFFICIENT_EVIDENCE,
            "No candidate runs were found in the run ledger for this claim.",
        )
    return None


def check_runs_completed(claim, runs):
    del claim
    incomplete = [run["run_id"] for run in runs if run.get("status") != "completed"]
    if incomplete:
        return _failure(
            FT.INSUFFICIENT_EVIDENCE,
            f"Run(s) {incomplete} are not marked 'completed'; cannot verify claim.",
        )
    return None


def check_consistent_dataset_method_metric(claim, runs):
    datasets = {run["dataset"] for run in runs}
    methods = {run["method"] for run in runs}
    metrics = {run["metric_name"] for run in runs}
    splits = {run["metric_split"] for run in runs}

    if claim["dataset"] not in datasets or len(datasets) > 1:
        return _failure(
            FT.DATASET_MISMATCH,
            f"Claim dataset '{claim['dataset']}' does not match "
            f"run ledger dataset(s) {sorted(datasets)}.",
        )
    if claim["method"] not in methods or len(methods) > 1:
        return _failure(
            FT.METHOD_MISMATCH,
            f"Claim method '{claim['method']}' does not match "
            f"run ledger method(s) {sorted(methods)}.",
        )
    if claim["metric"] not in metrics or len(metrics) > 1:
        return _failure(
            FT.METRIC_MISMATCH,
            f"Claim metric '{claim['metric']}' does not match "
            f"run ledger metric(s) {sorted(metrics)}.",
        )
    if claim["metric_split"] not in splits or len(splits) > 1:
        return _failure(
            FT.SPLIT_MISMATCH,
            f"Claim metric_split '{claim['metric_split']}' does not match "
            f"run ledger split(s) {sorted(splits)}.",
        )
    return None


def check_seed_count(claim, runs):
    count_type = claim.get("claimed_count_type")
    seeds_in_ledger = sorted({run["seed"] for run in runs})
    unique_seed_count = len(seeds_in_ledger)
    run_record_count = len(runs)
    duplicate_seed_count = max(0, run_record_count - unique_seed_count)

    if count_type is None:
        claimed_count = claim.get("claimed_seed_count")
        if claimed_count is None:
            return None

        repeated = _repeated_seeds(runs)
        if repeated:
            conflicts = _conflicting_seed_values(runs)
            detail = " and contain conflicting metric values" if conflicts else ""
            return _failure(
                FT.AMBIGUOUS_SEED_COUNT,
                "Claim uses seed-level semantics, but repeated run records exist for "
                f"seed(s) {repeated}{detail}. Deduplicate the ledger or declare "
                "run-level count semantics.",
            )
        if unique_seed_count < claimed_count:
            return _failure(
                FT.SEED_OMISSION,
                f"Claim states {claimed_count} seeds, but only {unique_seed_count} "
                f"seeds {seeds_in_ledger} are present in the ledger.",
            )
        if unique_seed_count > claimed_count:
            return _failure(
                FT.RETROSPECTIVE_SEED_FILTERING,
                f"Run ledger contains {unique_seed_count} seeds {seeds_in_ledger}, "
                f"but claim only counts {claimed_count}; some seeds may have been "
                "dropped after the fact.",
            )
        return None

    if count_type == "seeds":
        claimed_count = claim.get("claimed_seed_count")
        if claimed_count is None:
            return None
        if claimed_count > unique_seed_count:
            return _failure(
                FT.SEED_OMISSION,
                f"Claim states {claimed_count} unique seeds, but only "
                f"{unique_seed_count} unique seeds {seeds_in_ledger} are present.",
            )
        if claimed_count < unique_seed_count:
            return _failure(
                FT.RETROSPECTIVE_SEED_FILTERING,
                f"Run ledger contains {unique_seed_count} unique seeds "
                f"{seeds_in_ledger}, but claim only counts {claimed_count}; "
                "some seeds may have been dropped after the fact.",
            )

        repeated = _repeated_seeds(runs)
        if repeated:
            conflicts = _conflicting_seed_values(runs)
            detail = " and contain conflicting metric values" if conflicts else ""
            return _failure(
                FT.AMBIGUOUS_SEED_COUNT,
                f"Repeated run records for seed(s) {repeated}{detail}. TRACE-ML "
                "cannot infer a unique per-seed aggregate from retry/duplicate records.",
            )
        return None

    if count_type == "runs":
        claimed_count = claim.get("claimed_run_count")
        if claimed_count is None:
            return None
        if claimed_count == run_record_count:
            return None
        if claimed_count > run_record_count:
            return _failure(
                FT.SEED_OMISSION,
                f"Claim states {claimed_count} runs, but only {run_record_count} "
                "run records exist in the ledger for this claim.",
            )
        return _failure(
            FT.RETROSPECTIVE_SEED_FILTERING,
            f"Ledger contains {run_record_count} run records, but claim only counts "
            f"{claimed_count} runs; some records may have been dropped after the fact.",
        )

    if count_type == "trials":
        claimed_count = claim.get("claimed_trial_count")
        if claimed_count is None:
            return None
        if duplicate_seed_count > 0:
            return _failure(
                FT.AMBIGUOUS_SEED_COUNT,
                f"Claim states {claimed_count} trials. The ledger has "
                f"{run_record_count} run records but only {unique_seed_count} unique "
                f"seeds ({duplicate_seed_count} duplicate seed record(s)), so TRACE-ML "
                "cannot infer whether 'trials' means runs or unique seeds.",
            )
        if claimed_count == run_record_count:
            return None
        if claimed_count > run_record_count:
            return _failure(
                FT.SEED_OMISSION,
                f"Claim states {claimed_count} trials, but only {run_record_count} "
                "are present in the ledger.",
            )
        return _failure(
            FT.RETROSPECTIVE_SEED_FILTERING,
            f"Ledger contains {run_record_count} trials, but claim only counts "
            f"{claimed_count}; some may have been dropped after the fact.",
        )

    return _failure(
        FT.UNSUPPORTED_CLAIM_TYPE,
        f"Unrecognized claimed_count_type '{count_type}'.",
    )


def check_baseline_method_identity(claim, runs):
    del runs
    if claim.get("claimed_aggregation") not in COMPARISON_AGGREGATIONS:
        return None

    claimed_baseline = claim.get("claimed_baseline_method")
    configured_baseline = claim.get("baseline")
    if not claimed_baseline or not configured_baseline:
        return None

    if claimed_baseline != configured_baseline:
        return _failure(
            FT.STALE_BASELINE,
            f"Claim cites baseline method '{claimed_baseline}', but the configured "
            f"baseline for this claim is '{configured_baseline}'.",
        )
    return None


def check_aggregation_correctness(claim, runs):
    values = [run["metric_value"] for run in runs]
    true_mean = _mean(values)
    best_value = max(values)
    worst_value = min(values)

    claimed_value = claim.get("claimed_value")
    claimed_aggregation = claim.get("claimed_aggregation", "mean")

    if claimed_value is None:
        matches_mean = matches_best = matches_worst = False
    else:
        matches_mean = abs(claimed_value - true_mean) < TOLERANCE
        matches_best = abs(claimed_value - best_value) < TOLERANCE
        matches_worst = abs(claimed_value - worst_value) < TOLERANCE

    if claimed_aggregation == "mean":
        if matches_mean:
            return None
        if matches_best and not matches_mean:
            return _failure(
                FT.SEED_CHERRY_PICK,
                f"Claim reports {claimed_value:.4f} as the {len(values)}-seed mean, "
                f"but it is the best seed's value. The true mean is {true_mean:.4f}.",
            )
        if matches_worst:
            return _failure(
                FT.AGGREGATION_MISMATCH,
                f"Claim reports {claimed_value:.4f} as the mean, but it matches the "
                f"worst seed rather than the true mean {true_mean:.4f}.",
            )
        if any(abs(claimed_value - value) < TOLERANCE for value in values):
            return _failure(
                FT.AGGREGATION_MISMATCH,
                f"Claim reports {claimed_value:.4f} as the mean, but it matches a "
                f"single seed rather than the true mean {true_mean:.4f}.",
            )
        return _failure(
            FT.FABRICATED_METRIC,
            f"Claimed value {claimed_value:.4f} matches neither the true mean "
            f"({true_mean:.4f}), best ({best_value:.4f}), worst ({worst_value:.4f}), "
            "nor any individual run.",
        )

    if claimed_aggregation in ("best", "max"):
        if matches_best:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            f"Claim reports {claimed_value:.4f} as {claimed_aggregation}, but the "
            f"actual best value is {best_value:.4f}.",
        )

    if claimed_aggregation in ("worst", "min"):
        if matches_worst:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            f"Claim reports {claimed_value:.4f} as {claimed_aggregation}, but the "
            f"actual worst value is {worst_value:.4f}.",
        )

    if claimed_aggregation == "median":
        true_median = _median(values)
        if abs(claimed_value - true_median) < TOLERANCE:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            f"Claim reports {claimed_value:.4f} as the median, but the actual median "
            f"across {len(values)} runs is {true_median:.4f}.",
        )

    if claimed_aggregation in ("std", "standard_deviation"):
        true_std = _population_std(values)
        if true_std is None:
            return _failure(
                FT.INSUFFICIENT_EVIDENCE,
                "Population standard deviation is not representable as a finite float.",
            )
        if abs(claimed_value - true_std) < TOLERANCE:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            f"Claim reports {claimed_value:.4f} as the population standard deviation, "
            f"but the actual value is {true_std:.4f}.",
        )

    if claimed_aggregation in ("range", "min_max_range"):
        low = claim.get("claimed_low_value")
        high = claim.get("claimed_high_value")
        if low is None or high is None:
            return _failure(
                FT.UNSUPPORTED_CLAIM_TYPE,
                "Range claims require both claimed_low_value and claimed_high_value.",
            )

        low_ok = abs(low - worst_value) < TOLERANCE
        high_ok = abs(high - best_value) < TOLERANCE
        if low_ok and high_ok:
            return None

        problems = []
        if not low_ok:
            problems.append(f"lower bound {low:.4f} (actual min {worst_value:.4f})")
        if not high_ok:
            problems.append(f"upper bound {high:.4f} (actual max {best_value:.4f})")
        return _failure(
            FT.AGGREGATION_MISMATCH,
            "Range claim bound(s) incorrect: " + "; ".join(problems) + ".",
        )

    if claimed_aggregation == "mean_plus_minus_std":
        true_std = _population_std(values)
        if true_std is None:
            return _failure(
                FT.INSUFFICIENT_EVIDENCE,
                "Population standard deviation is not representable as a finite float.",
            )
        uncertainty = claim.get("uncertainty_value")
        if uncertainty is None:
            return _failure(
                FT.UNSUPPORTED_CLAIM_TYPE,
                "mean_plus_minus_std requires uncertainty_value.",
            )

        problems = []
        if not matches_mean:
            problems.append(f"mean component {claimed_value:.4f} (actual mean {true_mean:.4f})")
        if abs(uncertainty - true_std) >= TOLERANCE:
            problems.append(
                f"std component {uncertainty:.4f} (actual population std {true_std:.4f})"
            )
        if not problems:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            "mean +/- std claim component(s) incorrect: " + "; ".join(problems) + ".",
        )

    if claimed_aggregation == "single_seed":
        seed_id = claim.get("claimed_seed_id")
        if seed_id is None:
            return _failure(
                FT.INSUFFICIENT_EVIDENCE,
                "single_seed requires claimed_seed_id.",
            )

        matching_values = [run["metric_value"] for run in runs if run["seed"] == seed_id]
        if not matching_values:
            return _failure(
                FT.INSUFFICIENT_EVIDENCE,
                f"No run for seed {seed_id} exists among the candidate runs.",
            )
        unique_values = set(matching_values)
        if len(unique_values) > 1:
            return _failure(
                FT.AMBIGUOUS_SEED_COUNT,
                f"Seed {seed_id} has conflicting metric values {sorted(unique_values)} "
                "across multiple run records.",
            )
        if abs(claimed_value - matching_values[0]) < TOLERANCE:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            f"Claim reports {claimed_value:.4f} for seed {seed_id}, but the ledger "
            f"value(s) are {matching_values}.",
        )

    if claimed_aggregation == "baseline_value":
        if matches_mean:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            f"Claim reports {claimed_value:.4f} as the baseline value, but the actual "
            f"baseline mean is {true_mean:.4f}.",
        )

    if claimed_aggregation in COMPARISON_AGGREGATIONS:
        true_baseline = claim.get("true_baseline_value")
        if true_baseline is None:
            return _failure(
                FT.UNSUPPORTED_CLAIM_TYPE,
                f"{claimed_aggregation} requires true_baseline_value.",
            )

        if claimed_aggregation == "relative_improvement":
            if true_baseline == 0:
                return _failure(
                    FT.UNSUPPORTED_CLAIM_TYPE,
                    "Cannot compute relative improvement against a zero baseline.",
                )
            true_value = true_mean / true_baseline - 1.0
        else:
            true_value = true_mean - true_baseline

        if not math.isfinite(true_value):
            return _failure(
                FT.INSUFFICIENT_EVIDENCE,
                f"Computed {claimed_aggregation} is not finite for the supplied evidence.",
            )
        if abs(claimed_value - true_value) < TOLERANCE:
            return None
        return _failure(
            FT.AGGREGATION_MISMATCH,
            f"Claim reports {claimed_value:.4f} as {claimed_aggregation}, but the "
            f"actual value is {true_value:.4f} (method mean {true_mean:.4f}, "
            f"baseline {true_baseline:.4f}).",
        )

    if claimed_aggregation in ("not_applicable", "unsupported_claim_type"):
        return _failure(
            FT.UNSUPPORTED_CLAIM_TYPE,
            f"Claim is explicitly tagged '{claimed_aggregation}'.",
        )

    return _failure(
        FT.UNSUPPORTED_CLAIM_TYPE,
        f"Unrecognized claimed_aggregation '{claimed_aggregation}'.",
    )
