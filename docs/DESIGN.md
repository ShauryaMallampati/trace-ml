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

The default absolute tolerance is strictly below `5e-5`: half of one unit in the fourth decimal place. A caller with different precision requirements should wrap the verifier with a domain-specific policy instead of silently changing the global rule. Means use numerically stable summation, and input records are canonicalized by run ID before verification so output does not depend on ledger order.

## Extending the verifier

New aggregation behavior belongs in `trace_ml/verification/seedset_checks.py`. Comparison aggregations must derive baseline values from referenced baseline runs, not from an unverified scalar. Add positive, negative, malformed-input, numerical-boundary, and reference-integrity tests with every new rule.

The public API should remain deterministic: fixed claim + fixed run records should always produce the same result.
