.PHONY: install format test check coverage build demo clean

install:
	python -m pip install -e ".[test]"

format:
	python -m ruff format src tests
	python -m ruff check src tests --fix

test:
	python -m pytest

check:
	python -m ruff check src tests
	python -m pytest

coverage:
	coverage run -m pytest
	coverage report

build:
	python -m build

demo:
	trace-ml verify --claim examples/claim.json --runs examples/runs.json

clean:
	rm -rf build dist .pytest_cache .coverage htmlcov *.egg-info src/*.egg-info
