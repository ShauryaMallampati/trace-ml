import json
import subprocess
import sys

import pytest


def _write_inputs(tmp_path):
    claim = {
        "dataset": "d",
        "method": "m",
        "metric": "accuracy",
        "metric_split": "test",
        "claimed_seed_count": 1,
        "claimed_aggregation": "mean",
        "claimed_value": 0.8,
    }
    runs = [
        {
            "run_id": "r0",
            "dataset": "d",
            "method": "m",
            "seed": 0,
            "metric_name": "accuracy",
            "metric_split": "test",
            "metric_value": 0.8,
            "status": "completed",
        }
    ]
    claim_path = tmp_path / "claim.json"
    runs_path = tmp_path / "runs.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    runs_path.write_text(json.dumps(runs), encoding="utf-8")
    return claim_path, runs_path


def test_cli_verifies_json_files(tmp_path):
    claim_path, runs_path = _write_inputs(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trace_ml.cli",
            "verify",
            "--claim",
            str(claim_path),
            "--runs",
            str(runs_path),
            "--compact",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["verdict"] == "supported"


def test_cli_writes_output_file(tmp_path):
    claim_path, runs_path = _write_inputs(tmp_path)
    output_path = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "trace_ml.cli",
            "verify",
            "--claim",
            str(claim_path),
            "--runs",
            str(runs_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(output_path.read_text())["verdict"] == "supported"


@pytest.mark.parametrize("invalid_json", ["{not-json", '{"claimed_value": NaN}', '{"x": Infinity}'])
def test_cli_rejects_malformed_or_nonstandard_json(tmp_path, invalid_json):
    claim_path = tmp_path / "claim.json"
    runs_path = tmp_path / "runs.json"
    claim_path.write_text(invalid_json, encoding="utf-8")
    runs_path.write_text("[]", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trace_ml.cli",
            "verify",
            "--claim",
            str(claim_path),
            "--runs",
            str(runs_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Could not read valid JSON" in completed.stderr


def test_cli_require_supported_uses_machine_readable_exit_codes(tmp_path):
    claim_path, runs_path = _write_inputs(tmp_path)

    supported = subprocess.run(
        [
            sys.executable,
            "-m",
            "trace_ml.cli",
            "verify",
            "--claim",
            str(claim_path),
            "--runs",
            str(runs_path),
            "--require-supported",
        ],
        capture_output=True,
        text=True,
    )
    assert supported.returncode == 0

    claim = json.loads(claim_path.read_text())
    claim["claimed_value"] = 0.7
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    violation = subprocess.run(
        [
            sys.executable,
            "-m",
            "trace_ml.cli",
            "verify",
            "--claim",
            str(claim_path),
            "--runs",
            str(runs_path),
            "--require-supported",
        ],
        capture_output=True,
        text=True,
    )
    assert violation.returncode == 1

    runs_path.write_text("[]", encoding="utf-8")
    insufficient = subprocess.run(
        [
            sys.executable,
            "-m",
            "trace_ml.cli",
            "verify",
            "--claim",
            str(claim_path),
            "--runs",
            str(runs_path),
            "--require-supported",
        ],
        capture_output=True,
        text=True,
    )
    assert insufficient.returncode == 2


def test_cli_reports_version():
    completed = subprocess.run(
        [sys.executable, "-m", "trace_ml.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "TRACE-ML 1.1.0"
