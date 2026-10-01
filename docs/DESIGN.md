# Design

TRACE-ML is organized as a deterministic sequence of checks:

1. evidence existence;
2. completion status;
3. dataset, method, metric, and split identity;
4. declared seed/run/trial counts;
5. comparison-baseline identity;
6. aggregation semantics.

Checks run in a fixed order and short-circuit on the first actionable problem. This keeps failures stable and makes CI output easier to interpret.

## Input validation

The verifier rejects malformed or ambiguous records before applying semantic checks. Examples include duplicate run IDs, mixed seed identifier types, non-finite metric values, invalid count fields, and unsupported aggregation roles.

## Numerical comparisons

The default absolute tolerance is strictly below `5e-4`, matching values reported to four decimal places. A caller with different precision requirements should wrap the verifier with a domain-specific policy instead of silently changing the global rule.

## Extending the verifier

New aggregation behavior belongs in `trace_ml/verification/seedset_checks.py`. Add positive, negative, malformed-input, and boundary tests with every new rule.

The public API should remain deterministic: fixed claim + fixed run records should always produce the same result.
