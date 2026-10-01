# TRACE-ML

[![CI](https://github.com/ShauryaMallampati/trace-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/ShauryaMallampati/trace-ml/actions/workflows/ci.yml)

**Deterministic validation for machine-learning experiment claims.**

TRACE-ML is a developer tool that checks whether a structured ML metric claim is supported by the run records it references. It provides a command-line interface and a pure Python API for experiment pipelines, evaluation dashboards, release checks, and code review.

## What it checks

Given one claim and a set of run records, TRACE-ML verifies:

- the evidence exists and every referenced run completed;
- dataset, method, metric, and split identity match;
- stated seed, run, or trial counts agree with the ledger;
- comparison claims use the intended baseline;
- mean, best, worst, median, population standard deviation, range, mean ± std, single-seed, delta, and relative-improvement values are computed correctly.

Malformed or ambiguous inputs return `insufficient_evidence` instead of being forced into a pass/fail answer.

## Quick start

Requires **Python 3.10+**.

```bash
git clone https://github.com/ShauryaMallampati/trace-ml.git
cd trace-ml
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
trace-ml verify --claim examples/claim.json --runs examples/runs.json
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The bundled example contains two completed runs with accuracies 0.80 and 0.84 and a claim that their mean is 0.82. TRACE-ML recomputes that evidence and returns a `supported` verdict.

## Python API

```python
from trace_ml.verification.verify_claim import verify

result = verify(claim, runs)
print(result["verdict"])
print(result["evidence"])
```

The verification core is deterministic for fixed inputs, has no third-party runtime dependencies, and performs no network access or hidden model calls.

## Engineering decisions

- **Three-way outcomes:** `supported`, `violation`, and `insufficient_evidence` keep missing or ambiguous evidence separate from a contradiction.
- **Explicit schemas:** claim and run contracts are documented as JSON Schema rather than inferred from loosely structured input.
- **Stable failure taxonomy:** callers can branch on machine-readable failure types while still receiving a detailed rationale.
- **Fail-safe validation:** duplicate IDs, mixed seed types, non-finite values, incomplete runs, and unsupported aggregation semantics are handled explicitly.
- **One verification path:** the CLI calls the same `verify(claim, runs)` function exposed by the Python API.

## Development

```bash
python -m pip install -e ".[test]"
python -m ruff check src tests
python -m pytest
coverage run -m pytest
coverage report
python -m build
```

CI runs on Linux, macOS, and Windows across Python 3.10–3.13 and enforces at least **85% branch-aware coverage** of the verification package.

## Repository layout

```text
src/trace_ml/cli.py            command-line interface
src/trace_ml/verification/     deterministic verification rules
schemas/                       JSON contracts for claims and run records
examples/                      minimal runnable input pair
tests/                         regression, schema, malformed-input, and edge-case tests
docs/                          API, design, integration, and validation notes
```

Start with the [quickstart](docs/QUICKSTART.md), then see the [API reference](docs/API_REFERENCE.md), [design notes](docs/DESIGN.md), and [integration guide](docs/INTEGRATION.md).

## Scope

TRACE-ML validates a claim against the records you supply. It does **not** authenticate the source ledger, discover which runs should be associated with a claim, parse arbitrary prose, or determine whether a metric is appropriate for a particular ML task. Those responsibilities belong to the surrounding experiment system.

## License

[MIT](LICENSE).
