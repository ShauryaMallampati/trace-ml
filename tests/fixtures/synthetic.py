"""Deterministic synthetic fixtures for verifier tests."""

from __future__ import annotations


def make_run(
    run_id: str,
    seed: int | str,
    metric_value: float,
    dataset: str = "synthetic_dataset",
    method: str = "synthetic_model",
    metric_name: str = "accuracy",
    metric_split: str = "test",
    status: str = "completed",
    experiment_group_id: str = "expgrp_synthetic_0001",
    baseline: str | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "experiment_group_id": experiment_group_id,
        "task_id": "synthetic_task",
        "dataset": dataset,
        "split_id": "synthetic_split_0",
        "split_hash": "deadbeef00000000",
        "method": method,
        "baseline": baseline,
        "seed": seed,
        "metric_name": metric_name,
        "metric_split": metric_split,
        "metric_value": metric_value,
        "command": f"synthetic: seed={seed}",
        "config_hash": "cafebabe00000000",
        "status": status,
        "timestamp": 0.0,
        "log_path": None,
    }


def make_runs(values: list[float], **kwargs) -> list[dict]:
    group_id = kwargs.pop("experiment_group_id", "expgrp_synthetic_0001")
    return [
        make_run(
            run_id=f"run_{group_id}_seed{seed}",
            seed=seed,
            metric_value=value,
            experiment_group_id=group_id,
            **kwargs,
        )
        for seed, value in enumerate(values)
    ]


def make_claim(
    claim_id: str = "claim_synthetic_0001",
    dataset: str = "synthetic_dataset",
    method: str = "synthetic_model",
    metric: str = "accuracy",
    metric_split: str = "test",
    claimed_value: float | None = 0.0,
    claimed_seed_count: int | None = 5,
    claimed_aggregation: str = "mean",
    candidate_run_ids: list[str] | None = None,
    baseline: str | None = None,
    **extra,
) -> dict:
    claim = {
        "claim_id": claim_id,
        "dataset": dataset,
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "claimed_value": claimed_value,
        "claimed_seed_count": claimed_seed_count,
        "claimed_aggregation": claimed_aggregation,
        "metric_split": metric_split,
    }
    if candidate_run_ids is not None:
        claim["candidate_run_ids"] = candidate_run_ids
    claim.update(extra)
    return claim


def run_ids(runs: list[dict]) -> list[str]:
    return [run["run_id"] for run in runs]
