"""Run comparative online anomaly detection benchmark via config file or CLI flags."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.stream import NumpyStream
from sklearn.model_selection import ParameterGrid

from online_bootknn.benchmark import (
    DEFAULT_PARAM_GRID,
    get_model_with_params,
    step_online_detector,
    summarize_group_metrics,
)
from online_bootknn.tracking import ExperimentTracker, load_experiment_config

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "benchmark_default.json"
DATA_PATH = ROOT_DIR / "datasets" / "raw"
OUTPUT_DIR = ROOT_DIR / "datasets" / "processed"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments to override config defaults without editing code."""
    parser = argparse.ArgumentParser(
        description="Run comparative online spectral anomaly detection benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON/YAML configuration file.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Override number of runs per scenario/window.",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Override list of window sizes (e.g. --window-sizes 60 120 240).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Filter specific methods to run (e.g. --methods OnlineBootKNN ExactStorm).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Filter specific scenario prefixes (e.g. --scenarios A1 DA1).",
    )
    return parser.parse_args(argv)


def run_comparative_benchmark(argv: Optional[List[str]] = None) -> None:
    """Execute the comparative benchmark using declarative config + CLI overrides."""
    args = parse_args(argv)
    cfg: dict[str, Any] = (
        load_experiment_config(args.config) if args.config.exists() else {}
    )

    n_runs: int = args.n_runs if args.n_runs is not None else cfg.get("n_runs", 5)
    window_sizes: List[int] = (
        args.window_sizes
        if args.window_sizes is not None
        else cfg.get("window_sizes", [60, 120, 240])
    )
    selected_methods: Optional[List[str]] = (
        args.methods if args.methods is not None else cfg.get("methods")
    )
    selected_scenarios: Optional[List[str]] = (
        args.scenarios if args.scenarios is not None else cfg.get("scenarios")
    )
    cols_smin: int = cfg.get("cols_pos_smin", 1)
    cols_smax: int = cfg.get("cols_pos_smax", 2049)

    tracker = ExperimentTracker(
        experiment_name=cfg.get("experiment_name", "spectral_online_anomaly_detection"),
        tracking_dir=ROOT_DIR / "logs" / "experiments",
    )
    tracker.log_params(
        {
            "n_runs": n_runs,
            "window_sizes": window_sizes,
            "methods": selected_methods or list(DEFAULT_PARAM_GRID.keys()),
            "scenarios": selected_scenarios or "ALL",
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spectra_files = sorted(
        f for f in DATA_PATH.iterdir() if f.suffix == ".csv" and not f.name.startswith("0_")
    )

    active_grid = {
        m: grid
        for m, grid in DEFAULT_PARAM_GRID.items()
        if selected_methods is None or m in selected_methods
    }

    for file_name in spectra_files:
        scenario_prefix = file_name.name.split("_")[0]
        if selected_scenarios and scenario_prefix not in selected_scenarios:
            continue

        result = pd.read_csv(
            DATA_PATH / file_name,
            sep=",",
            low_memory=False,
            dtype={"CURRENTTIMESTAMP": str},
        )
        cols = result.columns[cols_smin:cols_smax]

        stream = NumpyStream(
            result[cols].values,
            result["ANOMALY?"].values,
            dataset_name="PV",
            feature_names=cols,
        )

        for window_size in window_sizes:
            for iteration in range(n_runs):
                records = []
                schema = stream.get_schema()

                for model_name, param_grid in active_grid.items():
                    for params in ParameterGrid(param_grid):
                        np.random.seed(iteration)
                        learner = get_model_with_params(
                            model_name, params, window_size=window_size, schema=schema
                        )
                        stream.restart()
                        row = 0
                        evaluator = AnomalyDetectionEvaluator(schema)

                        while stream.has_more_instances():
                            instance = stream.next_instance()
                            (
                                raw_score,
                                cleaned_score,
                                error_score,
                                training_time,
                                scoring_time,
                            ) = step_online_detector(learner, instance)

                            evaluator.update(instance.y_index, cleaned_score)
                            records.append(
                                {
                                    "iteration": iteration,
                                    "timestamp": result.iloc[row, 0],
                                    "method": model_name,
                                    "param": params,
                                    "score": raw_score,
                                    "cleaned_score": cleaned_score,
                                    "training_time": training_time,
                                    "scoring_time": scoring_time,
                                    "ground_truth": instance.y_index,
                                    "prog_auc": evaluator.auc(),
                                    "error_type_score": error_score,
                                }
                            )
                            row += 1

                        # Log run summary metrics to ExperimentTracker
                        df_method = pd.DataFrame(records[-row:])
                        mwp_str = f"{model_name}_{window_size}_{params}"
                        summary = summarize_group_metrics(
                            df_method, file_name.name, iteration, model_name, mwp_str
                        )
                        if summary:
                            tracker.log_metrics(
                                {
                                    "AUC_ROC": summary["AUC_ROC"],
                                    "AUC_PR": summary["AUC_PR"],
                                    "Max_F1": summary["Max_F1"],
                                    "mean_scoring_time": summary["mean_scoring_time"],
                                },
                                context={
                                    "scenario": scenario_prefix,
                                    "method": model_name,
                                    "window_size": window_size,
                                    "iteration": iteration,
                                },
                            )

                df_to_save = pd.DataFrame(records)
                out_file = (
                    OUTPUT_DIR
                    / f"{scenario_prefix}_results_iter_{iteration}_pv_ds_ws_{window_size}.xlsx"
                )
                df_to_save.to_excel(out_file, index=False)
                logger.info("Saved benchmark results to %s", out_file)


if __name__ == "__main__":
    run_comparative_benchmark()
