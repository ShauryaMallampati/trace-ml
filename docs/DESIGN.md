# Design

TRACE-ML is organized as a deterministic sequence of checks:

1. evidence existence;
2. completion status;
3. dataset, method, metric, and split identity;
4. declared seed/run/trial counts;
5. exact run-reference manifests;
6. comparison-baseline identity and baseline-run evidence;
7. aggregation semantics.

Checks run in a fixed order and short-circuit on the first actionable problem. This keeps failures stable and makes CI output easier to interpret.

## Input validation

The verifier rejects malformed or ambiguous records before applying semantic checks. Examples include duplicate run IDs, empty or mixed-type seed identifiers, non-finite metric values, invalid count fields, mismatched run-reference manifests, and unsupported aggregation roles.

## Numerical comparisons

For ordinary ML-scale metrics, the comparison threshold is `5e-5`: half of one unit in the fourth decimal place. At very large magnitudes, the verifier also allows a small two-ULP representation margin so values are not rejected solely because IEEE-754 cannot represent the same mathematical result bit-for-bit. Inputs that cannot be represented as finite Python floats return `insufficient_evidence`. Means and population standard deviations use scale-normalized arithmetic, and input records are canonicalized by run ID before verification so output does not depend on ledger order.

## Extending the verifier

New aggregation behavior belongs in `trace_ml/verification/seedset_checks.py`. Comparison aggregations must derive baseline values from referenced baseline runs, not from an unverified scalar. Add positive, negative, malformed-input, numerical-boundary, and reference-integrity tests with every new rule.

The public API should remain deterministic: fixed claim + fixed run records should always produce the same result.
