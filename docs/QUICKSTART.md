# Quickstart

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1`.

The verification core has no third-party runtime dependency.

## Verify one claim

```bash
trace-ml verify --claim examples/claim.json --runs examples/runs.json
```

The result contains a three-way verdict, a stable failure type, supporting run IDs, recomputed evidence, and a rationale.

## Use the Python API

```python
from trace_ml import verify

result = verify(claim, runs)
if result["verdict"] != "supported":
    print(result["failure_type"], result["rationale"])
```

For a CI gate, add `--require-supported` to the CLI command. Supported claims exit `0`, violations exit `1`, and insufficient evidence exits `2`.

## Run the developer checks

```bash
python -m pip install -e ".[test]"
python -m ruff check src tests
python -m pytest
coverage run -m pytest
coverage report
python -m build
```
