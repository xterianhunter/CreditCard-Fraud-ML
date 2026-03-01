"""Lightweight MLflow helpers with safe no-op defaults."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any


def parse_tags(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    tags: dict[str, str] = {}
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid MLflow tag '{part}'. Expected key=value.")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("MLflow tag key cannot be empty.")
        tags[key] = value
    return tags


def _require_mlflow() -> Any:
    try:
        import mlflow  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive only
        raise RuntimeError("MLflow is not installed. Add it to requirements.txt.") from exc
    return mlflow


def start_mlflow_run(
    *,
    enabled: bool,
    experiment_name: str | None,
    tracking_uri: str | None,
    run_name: str | None,
    tags: dict[str, str] | None,
) -> AbstractContextManager[Any]:
    if not enabled:
        return nullcontext()

    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags or None)


def log_params(enabled: bool, params: dict[str, Any]) -> None:
    if not enabled:
        return
    mlflow = _require_mlflow()
    mlflow.log_params(params)


def log_metrics(enabled: bool, metrics: dict[str, float]) -> None:
    if not enabled:
        return
    mlflow = _require_mlflow()
    mlflow.log_metrics(metrics)


def log_artifact(enabled: bool, path: str, artifact_path: str | None = None) -> None:
    if not enabled:
        return
    mlflow = _require_mlflow()
    mlflow.log_artifact(path, artifact_path=artifact_path)
