"""Stable machine-readable failure labels returned by TRACE-ML."""

SUPPORTED = "supported"
FABRICATED_METRIC = "fabricated_metric"
METRIC_MISMATCH = "metric_mismatch"
SPLIT_MISMATCH = "split_mismatch"
STALE_BASELINE = "stale_baseline"
SEED_OMISSION = "seed_omission"
SEED_CHERRY_PICK = "seed_cherry_pick"
RETROSPECTIVE_SEED_FILTERING = "retrospective_seed_filtering"
AGGREGATION_MISMATCH = "aggregation_mismatch"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
AMBIGUOUS_SEED_COUNT = "ambiguous_seed_count"
UNSUPPORTED_CLAIM_TYPE = "unsupported_claim_type"

ALL_FAILURE_TYPES = [
    SUPPORTED,
    FABRICATED_METRIC,
    METRIC_MISMATCH,
    SPLIT_MISMATCH,
    STALE_BASELINE,
    SEED_OMISSION,
    SEED_CHERRY_PICK,
    RETROSPECTIVE_SEED_FILTERING,
    AGGREGATION_MISMATCH,
    INSUFFICIENT_EVIDENCE,
    AMBIGUOUS_SEED_COUNT,
    UNSUPPORTED_CLAIM_TYPE,
]

VERDICT_SUPPORTED = "supported"
VERDICT_VIOLATION = "violation"
VERDICT_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# These outcomes mean the verifier cannot make a reliable judgment from the
# supplied records. They are deliberately distinct from confirmed violations.
INSUFFICIENT_VERDICT_TYPES = {
    INSUFFICIENT_EVIDENCE,
    AMBIGUOUS_SEED_COUNT,
    UNSUPPORTED_CLAIM_TYPE,
}
