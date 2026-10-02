# Changelog

## 1.1.1 - 2026-10-01

- Reject missing or null required claim/run fields before semantic verification, so malformed API inputs fail closed as `insufficient_evidence` instead of reaching aggregation or formatting code.
- Align runtime validation with the published JSON contracts for required aggregation values, range bounds, single-seed IDs, and mean-plus-minus-standard-deviation uncertainty.
- Exercise the bundled mean and comparison examples through the real CLI in the test suite.
- Use a normal package install in user quickstarts while keeping editable installs in the development workflow.

## 1.1.0 - 2026-10-01

- Enforce exact `candidate_run_ids` manifests instead of accepting silently missing or extra evidence.
- Require independently supplied `baseline_run_ids` for delta and improvement claims; baseline means are recomputed from those records.
- Tighten four-decimal absolute tolerance from `5e-4` to `5e-5`.
- Canonicalize evidence order and use scale-normalized mean/standard-deviation calculations that preserve subnormal values and avoid avoidable overflow.
- Add a small ULP-aware representation margin on top of the `5e-5` decimal tolerance and fail closed on numbers that cannot be represented as finite floats.
- Reject empty seed identifiers, duplicate reference IDs, and boolean single-seed identifiers.
- Add `--require-supported` for CI-friendly exit codes and expose `verify` at the top-level Python package.
- Distinguish dataset and method identity mismatches from split and baseline failures in the machine-readable taxonomy.

## 1.0.0 - 2026-07-13

- Added the installable `trace_ml` package and `trace-ml` CLI.
- Added JSON schemas, runnable examples, and developer documentation.
- Added deterministic checks for run identity, completion, counts, baselines, and aggregation semantics.
- Added malformed-input and edge-case handling with explicit `insufficient_evidence` outcomes.
- Added cross-platform CI for Python 3.10–3.13.
