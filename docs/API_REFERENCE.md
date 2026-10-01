# API reference

## `verify(claim, runs)`

Pure verification function in `trace_ml.verification.verify_claim`.

**Inputs**

- `claim`: mapping that follows `schemas/claim-record.schema.json`.
- `runs`: list of mappings that follow `schemas/run-record.schema.json`.

**Output**

A mapping with:

- `verdict`: `supported`, `violation`, or `insufficient_evidence`;
- `failure_type`: stable failure identifier;
- `supporting_run_ids`: run identifiers used by the decision;
- `evidence`: recomputed values and matched metadata;
- `rationale`: human-readable explanation.

The verifier is deterministic for fixed inputs. It performs no network access and no disk I/O.

## CLI

```text
trace-ml verify --claim CLAIM.json --runs RUNS.json [--output RESULT.json] [--compact]
trace-ml --version
```

## Supported aggregation roles

Mean, maximum/best, minimum/worst, median, population standard deviation, range, mean plus or minus standard deviation, named single-seed values, baseline means, and selected absolute or relative comparison claims.

## Non-judgment conditions

Malformed records, incomplete evidence, non-finite values, conflicting repeated-seed records, ambiguous count semantics, and unsupported claim types return `insufficient_evidence` instead of a guessed violation.
