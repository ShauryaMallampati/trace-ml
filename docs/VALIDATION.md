# Testing and validation

The repository uses three layers of software verification:

- **Unit and regression tests** for aggregations, comparison claims, count semantics, malformed input, schemas, and edge cases.
- **Branch-aware coverage** focused on `trace_ml.verification`, with an 85% minimum enforced in CI.
- **Cross-platform CI** on Linux, macOS, and Windows for Python 3.10–3.13.

Every CI job runs:

```bash
python -m ruff check src tests
python -m pytest
```

The Linux/Python 3.13 job additionally runs:

```bash
coverage run -m pytest
coverage report
python -m build
```

These checks establish software behavior for the implemented rules. They do not prove that an upstream run ledger is truthful or that a metric is appropriate for a particular ML task.
