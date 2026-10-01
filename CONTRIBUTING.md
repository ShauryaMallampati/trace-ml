# Contributing

Keep changes focused on the verifier, schemas, CLI, or developer documentation.

Before opening a pull request:

```bash
python -m pip install -e ".[test]"
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest
python -m build
```

New verification rules should include:

- a clear input contract;
- at least one supported case;
- at least one violation case;
- malformed or ambiguous-input behavior;
- regression tests for boundary conditions.

Do not weaken an existing rule or change fixture evidence merely to make a test pass. If behavior changes intentionally, document the reason and update the relevant regression tests.
