import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError, validate

ROOT = Path(__file__).resolve().parents[1]
CLAIM_SCHEMA = json.loads((ROOT / "schemas" / "claim-record.schema.json").read_text())
RUN_SCHEMA = json.loads((ROOT / "schemas" / "run-record.schema.json").read_text())


def test_schemas_are_valid_draft_2020_12():
    Draft202012Validator.check_schema(CLAIM_SCHEMA)
    Draft202012Validator.check_schema(RUN_SCHEMA)


def test_example_claim_and_runs_validate():
    claim = json.loads((ROOT / "examples" / "claim.json").read_text())
    runs = json.loads((ROOT / "examples" / "runs.json").read_text())

    validate(instance=claim, schema=CLAIM_SCHEMA)
    for run in runs:
        validate(instance=run, schema=RUN_SCHEMA)


def test_schema_accepts_range_claim_without_scalar_claimed_value():
    claim = {
        "dataset": "demo",
        "method": "classifier-a",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "range",
        "claimed_low_value": 0.80,
        "claimed_high_value": 0.84,
    }
    validate(instance=claim, schema=CLAIM_SCHEMA)


def test_schema_accepts_string_seed_identifiers():
    run = {
        "run_id": "run-alpha",
        "dataset": "demo",
        "method": "classifier-a",
        "seed": "seed-alpha",
        "metric_name": "accuracy",
        "metric_split": "test",
        "metric_value": 0.82,
        "status": "completed",
    }
    validate(instance=run, schema=RUN_SCHEMA)


def test_schema_requires_count_field_for_explicit_count_type():
    claim = {
        "dataset": "demo",
        "method": "classifier-a",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "mean",
        "claimed_value": 0.82,
        "claimed_count_type": "runs",
    }
    with pytest.raises(ValidationError):
        validate(instance=claim, schema=CLAIM_SCHEMA)
