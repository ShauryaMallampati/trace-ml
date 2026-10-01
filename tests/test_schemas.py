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


@pytest.mark.parametrize(
    ("claim_name", "runs_name"),
    [
        ("claim.json", "runs.json"),
        ("comparison-claim.json", "comparison-runs.json"),
    ],
)
def test_example_claim_and_runs_validate(claim_name, runs_name):
    claim = json.loads((ROOT / "examples" / claim_name).read_text())
    runs = json.loads((ROOT / "examples" / runs_name).read_text())

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


def test_schema_requires_evidence_manifests_for_comparison_claim():
    claim = {
        "dataset": "demo",
        "method": "classifier-a",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "delta",
        "claimed_value": 0.02,
        "baseline": "classifier-b",
    }
    with pytest.raises(ValidationError):
        validate(instance=claim, schema=CLAIM_SCHEMA)


def test_schema_accepts_evidence_backed_comparison_claim():
    claim = {
        "dataset": "demo",
        "method": "classifier-a",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "delta",
        "claimed_value": 0.02,
        "baseline": "classifier-b",
        "candidate_run_ids": ["method-0"],
        "baseline_run_ids": ["baseline-0"],
    }
    validate(instance=claim, schema=CLAIM_SCHEMA)


def test_schema_rejects_empty_candidate_manifest():
    claim = {
        "dataset": "demo",
        "method": "classifier-a",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "mean",
        "claimed_value": 0.82,
        "candidate_run_ids": [],
    }
    with pytest.raises(ValidationError):
        validate(instance=claim, schema=CLAIM_SCHEMA)


def test_schema_rejects_empty_string_seed():
    run = {
        "run_id": "run-alpha",
        "dataset": "demo",
        "method": "classifier-a",
        "seed": "",
        "metric_name": "accuracy",
        "metric_split": "test",
        "metric_value": 0.82,
        "status": "completed",
    }
    with pytest.raises(ValidationError):
        validate(instance=run, schema=RUN_SCHEMA)


def test_schema_rejects_whitespace_only_identifiers():
    claim = {
        "dataset": "   ",
        "method": "classifier-a",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_aggregation": "mean",
        "claimed_value": 0.82,
    }
    with pytest.raises(ValidationError):
        validate(instance=claim, schema=CLAIM_SCHEMA)

    run = {
        "run_id": "   ",
        "dataset": "demo",
        "method": "classifier-a",
        "seed": 0,
        "metric_name": "accuracy",
        "metric_split": "test",
        "metric_value": 0.82,
        "status": "completed",
    }
    with pytest.raises(ValidationError):
        validate(instance=run, schema=RUN_SCHEMA)
