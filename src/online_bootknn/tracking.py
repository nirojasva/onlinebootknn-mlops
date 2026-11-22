"""Declarative configuration loader and MLflow-compatible experiment tracker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def load_experiment_config(config_path: str | Path) -> Dict[str, Any]:
    """Load experiment configuration from a JSON or YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return dict(yaml.safe_load(f))

    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


class ExperimentTracker:
    """Tracks experiment parameters, metrics, and artifacts locally (and in MLflow if installed).

    In industry MLOps, every benchmark run should record:
    1. Exact hyperparameters (`params.json`)
    2. Computed evaluation metrics (`metrics.jsonl`)
    3. Timestamp and dataset metadata
    """

    def __init__(
        self,
        experiment_name: str = "spectral_anomaly_benchmark",
        tracking_dir: str | Path = "logs/experiments",
    ) -> None:
        self.experiment_name = experiment_name
        self.base_dir = Path(tracking_dir)
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.base_dir / f"{experiment_name}_{run_ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[Dict[str, Any]] = []

        if HAS_MLFLOW:
            mlflow.set_experiment(experiment_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Save experiment configuration and hyperparameters to disk (and MLflow)."""
        params_file = self.run_dir / "params.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

        if HAS_MLFLOW and mlflow.active_run() is not None:
            flat_params = {k: str(v)[:250] for k, v in params.items()}
            mlflow.log_params(flat_params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a dictionary of evaluation metrics (e.g., AUC_PR, AUC_ROC, latency)."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(context or {}),
            **metrics,
        }
        self.metrics_history.append(entry)

        metrics_file = self.run_dir / "metrics.jsonl"
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        if HAS_MLFLOW and mlflow.active_run() is not None:
            numeric_only = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            mlflow.log_metrics(numeric_only)
