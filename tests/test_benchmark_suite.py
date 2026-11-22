"""Automated unit tests for OnlineBootKNN, preprocessing, and comparative benchmark pipeline.

Compatible with both `pytest` (`pytest tests/`) and standard library `unittest`
(`python -m unittest discover -s tests -v`).
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.stream import NumpyStream
from sklearn.model_selection import ParameterGrid

# Allow importing scripts/gen_summaries_AD_PV_online.py in integration tests
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.append(str(ROOT_DIR / "scripts"))

from gen_summaries_AD_PV_online import process_file

from online_bootknn import (
    DEFAULT_PARAM_GRID,
    ExperimentTracker,
    OnlineBootKNN,
    calculate_roc_pr_auc,
    clean_score,
    featurewise_distance,
    get_model_with_params,
    load_experiment_config,
    split_summary_methods,
    step_online_detector,
    summarize_group_metrics,
    transform_array,
)
from gen_comparative_AD_PV_online import parse_args


class TestOnlineBootKNNDetector(unittest.TestCase):
    """Tests for the OnlineBootKNN streaming anomaly detector."""

    def test_warmup_and_anomaly_detection(self) -> None:
        rng = np.random.RandomState(42)
        detector = OnlineBootKNN(
            window_size=20,
            chunk_size=15,
            ensemble_size=10,
            transf="ZNORM",
            alpha=0.05,
            random_seed=42,
            n_jobs=1,
        )

        # 1. Warm-up phase: feed 20 normal sinusoidal spectra
        base_wave = np.sin(np.linspace(0, 2 * np.pi, 64))
        for _ in range(20):
            normal_spectrum = base_wave + rng.normal(0, 0.02, size=64)
            score = detector.score_array(normal_spectrum)
            detector.train_array(normal_spectrum)
            self.assertEqual(score, 0.0)

        # Detector should now be initialized
        self.assertFalse(detector.init)

        # 2. Streaming normal spectra to establish baseline distance distribution
        for _ in range(30):
            normal_spectrum = base_wave + rng.normal(0, 0.02, size=64)
            z_score = detector.score_array(normal_spectrum)
            detector.train_array(normal_spectrum)

        # 3. Inject a clear spectral anomaly (inverted shape + localized spike)
        anomalous_spectrum = -base_wave
        anomalous_spectrum[10:20] += 15.0
        anomaly_z = detector.score_array(anomalous_spectrum)

        self.assertTrue(detector.last_value_is_anomaly)
        self.assertGreater(anomaly_z, detector.z_critical_one_tail)
        self.assertGreaterEqual(detector.n_anomalies, 1)
        self.assertIsNotNone(detector.normal_reference)
        self.assertIsNotNone(detector.abnormal_reference)

class TestComparativeBenchmarkAndSummaries(unittest.TestCase):
    """Tests for the multi-model benchmark runner and summary aggregator."""

    def test_model_factory_and_unified_step(self) -> None:
        # Verify factory builds OnlineBootKNN and PySAD models with unified window_size
        obknn = get_model_with_params(
            "OnlineBootKNN",
            {"chunk_size": 10, "ensemble_size": 5, "n_jobs": 1},
            window_size=10,
        )
        self.assertIsInstance(obknn, OnlineBootKNN)

        exact_storm = get_model_with_params(
            "ExactStorm",
            {"max_radius": 0.1},
            window_size=10,
        )

        # Create a mock instance with .x attribute to test step_online_detector
        class DummyInstance:
            def __init__(self, x: np.ndarray) -> None:
                self.x = x

        inst = DummyInstance(np.ones(16, dtype=float))
        _, cleaned, _, t_time, s_time = step_online_detector(exact_storm, inst)
        self.assertIsInstance(cleaned, float)
        self.assertGreaterEqual(t_time, 0.0)
        self.assertGreaterEqual(s_time, 0.0)

    def test_summarize_group_metrics_and_roc_pr_auc(self) -> None:
        # Simulate a benchmark output group from gen_comparative_AD_PV_online.py
        df_group = pd.DataFrame(
            {
                "ground_truth": [0, 0, 0, 0, 1, 1],
                "cleaned_score": [0.1, 0.2, 0.15, 0.3, 2.5, 3.1],
                "score": [0.1, 0.2, 0.15, 0.3, 2.5, 3.1],
                "training_time": [0.01, 0.01, 0.02, 0.01, 0.01, 0.02],
                "scoring_time": [0.005, 0.005, 0.006, 0.005, 0.005, 0.006],
            }
        )

        summary = summarize_group_metrics(
            group=df_group,
            file_name="A1_results_iter_0_pv_ds_ws_120.xlsx",
            iteration=0,
            method="OnlineBootKNN",
            mwp="OnlineBootKNN_120_{'chunk_size': 240}",
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["scenario"], "A1")
        self.assertAlmostEqual(summary["AUC_ROC"], 1.0, places=5)
        self.assertAlmostEqual(summary["AUC_PR"], 1.0, places=5)
        self.assertEqual(summary["count_anomalies"], 2)
        self.assertEqual(summary["count_normal"], 4)

    def test_split_summary_methods_parser(self) -> None:
        m, w, p = split_summary_methods("OnlineBootKNN_240_{'alpha': 0.05}")
        self.assertEqual(m, "OnlineBootKNN")
        self.assertEqual(w, "240")
        self.assertEqual(p, "{'alpha': 0.05}")
        self.assertIn("OnlineBootKNN", DEFAULT_PARAM_GRID)

    def test_config_cli_and_experiment_tracker(self) -> None:
        # 1. Verify default JSON config loads cleanly
        cfg_path = ROOT_DIR / "configs" / "benchmark_default.json"
        cfg = load_experiment_config(cfg_path)
        self.assertEqual(cfg["n_runs"], 5)
        self.assertIn("OnlineBootKNN", cfg["methods"])

        # 2. Verify CLI argument parser overrides
        parsed = parse_args(["--n-runs", "3", "--methods", "OnlineBootKNN", "ExactStorm", "--scenarios", "A1", "DA1"])
        self.assertEqual(parsed.n_runs, 3)
        self.assertEqual(parsed.methods, ["OnlineBootKNN", "ExactStorm"])
        self.assertEqual(parsed.scenarios, ["A1", "DA1"])

        # 3. Verify ExperimentTracker logs params.json and metrics.jsonl
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = ExperimentTracker(experiment_name="unit_test_exp", tracking_dir=tmp_dir)
            tracker.log_params({"n_runs": 3, "window_size": 120})
            tracker.log_metrics({"AUC_PR": 0.975, "AUC_ROC": 0.991}, context={"scenario": "A1"})

            self.assertTrue((tracker.run_dir / "params.json").exists())
            self.assertTrue((tracker.run_dir / "metrics.jsonl").exists())
            self.assertEqual(len(tracker.metrics_history), 1)
            self.assertAlmostEqual(tracker.metrics_history[0]["AUC_PR"], 0.975)

    def test_end_to_end_10_instances_2048_wavelengths_3_runs_and_summarize(self) -> None:
        """End-to-end pipeline test:
        1. Create 10 dummy spectral instances with 2048 wavelengths.
        2. Run every method in DEFAULT_PARAM_GRID 3 times.
        3. Save benchmark Excel files and generate the summary table.
        """
        n_instances = 10
        n_wavelengths = 2048
        n_runs = 3
        window_size = 5

        rng = np.random.RandomState(42)
        wavelength_cols = [f"WL_{j}" for j in range(n_wavelengths)]
        base_spectrum = np.sin(np.linspace(0, 8 * np.pi, n_wavelengths))
        spectra_matrix = base_spectrum + rng.normal(0, 0.05, size=(n_instances, n_wavelengths))
        # Inject realistic spectral shape anomalies at rows 7 and 8
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0])
        spectra_matrix[7] = -base_spectrum + rng.normal(0, 0.05, size=n_wavelengths)
        spectra_matrix[8] = -base_spectrum + rng.normal(0, 0.05, size=n_wavelengths)

        dummy_df = pd.DataFrame(spectra_matrix, columns=wavelength_cols)
        dummy_df.insert(0, "CURRENTTIMESTAMP", [f"2026-11-15T12:00:0{k}" for k in range(n_instances)])
        dummy_df["ANOMALY?"] = labels

        cols = dummy_df.columns[1 : 1 + n_wavelengths]
        self.assertEqual(len(cols), 2048)

        stream = NumpyStream(
            dummy_df[cols].values,
            dummy_df["ANOMALY?"].values,
            dataset_name="DummyPV",
            feature_names=cols,
        )
        schema = stream.get_schema()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generated_files = []

            # Step 1: Run each method 3 times (like gen_comparative_AD_PV_online.py)
            for iteration in range(n_runs):
                records = []
                for model_name, param_grid in DEFAULT_PARAM_GRID.items():
                    for params in ParameterGrid(param_grid):
                        run_params = dict(params)
                        # Scale chunk/ensemble size to fit the 10-instance test window
                        if model_name == "OnlineBootKNN":
                            run_params["chunk_size"] = window_size
                            run_params["ensemble_size"] = 5
                            run_params["n_jobs"] = 1

                        learner = get_model_with_params(
                            model_name, run_params, window_size=window_size, schema=schema
                        )
                        stream.restart()
                        evaluator = AnomalyDetectionEvaluator(schema)
                        row = 0

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
                                    "timestamp": dummy_df.iloc[row, 0],
                                    "method": model_name,
                                    "param": run_params,
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

                out_file = tmp_path / f"A1_results_iter_{iteration}_pv_ds_ws_{window_size}.xlsx"
                pd.DataFrame(records).to_excel(out_file, index=False)
                generated_files.append(out_file)

            # Step 2: Summarize the 3 runs across all 9 methods (like gen_summaries_AD_PV_online.py)
            all_summaries = []
            for excel_file in generated_files:
                all_summaries.extend(process_file(excel_file, expected_rows=n_instances))

            summary_df = pd.DataFrame(all_summaries)
            summary_out = tmp_path / "summary_results_test.xlsx"
            summary_df.to_excel(summary_out, index=False)

            # Verify 9 methods * 3 runs = 27 summary rows
            expected_summary_rows = len(DEFAULT_PARAM_GRID) * n_runs
            self.assertEqual(len(summary_df), expected_summary_rows)
            self.assertTrue((summary_df["count_cleaned_score"] == n_instances).all())
            self.assertTrue((summary_df["count_anomalies"] == 2).all())
            self.assertTrue((summary_df["count_normal"] == 8).all())
            self.assertTrue(summary_out.exists())

            # Display aggregated summary across the 3 runs per method
            agg_table = (
                summary_df.groupby("method_window_and_param")[
                    ["AUC_ROC", "AUC_PR", "Max_F1", "mean_training_time", "mean_scoring_time"]
                ]
                .mean()
                .round(4)
            )
            print("\n=== End-to-End Dummy Benchmark Summary (10x2048, 3 Runs) ===")
            print(agg_table.to_string())


if __name__ == "__main__":
    unittest.main()
