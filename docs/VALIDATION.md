# Testing and validation

The repository uses three layers of software verification:

- **Unit and regression tests** for aggregations, evidence manifests, independently evidenced baselines, count semantics, malformed input, numerical boundaries, schemas, and edge cases.
- **Branch-aware coverage** focused on `trace_ml.verification`, with an 85% minimum enforced in CI.
- **Cross-platform CI** on Linux, macOS, and Windows for Python 3.10–3.13.

Every CI job runs:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest
```

The Linux/Python 3.13 job additionally runs:

```bash
coverage run -m pytest
coverage report
python -m build
python -m venv /tmp/trace-ml-wheel
/tmp/trace-ml-wheel/bin/python -m pip install dist/*.whl
```

A separate read-only Secret Scan workflow checks the full Git history with Gitleaks. Regression tests also exercise input-order invariance, large finite metric values, exact candidate/baseline reference matching, and four-decimal tolerance boundaries. These checks establish software behavior for the implemented rules. They do not prove that an upstream run ledger is truthful or that a metric is appropriate for a particular ML task.
