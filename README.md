# OnlineBootKNN — Production MLOps & Real-Time Spectral Anomaly Detection

Production-ready Python package, automated test suite, declarative benchmarking pipeline, and containerized **FastAPI** inference service for **OnlineBootKNN** (AAAI 2026 & NeurIPS 2026).

---

## Architecture & MLOps Features

1. **Modular Python Package (`src/online_bootknn/`, `pyproject.toml`):**
   - `detector.py`: Core `OnlineBootKNN` streaming estimator supporting both `CapyMOA` instances and pure NumPy arrays (`score_array`, `train_array`).
   - `preprocessing.py`: Spectral transformations (`ZNORM`, `FOD`, `DIL`, `FT`, `SQRT`) and score sanitization (`clean_score`).
   - `metrics.py`: Threshold-independent (`ROC-AUC`, `PR-AUC`, `Max-F1`) and range-based (`VUS`) streaming evaluation metrics.
   - `benchmark.py`: Multi-model factory and unified test-then-train step across 9 state-of-the-art streaming anomaly detectors.
   - `tracking.py`: Declarative JSON/YAML config loader and MLflow-compatible `ExperimentTracker`.
2. **Automated Testing (`tests/`):**
   - `tests/test_benchmark_suite.py`: Unit tests and 10×2048 3-run end-to-end integration test.
   - `tests/test_paper_reproducibility.py`: 5-run scientific regression tests verifying exact `AUC-PR` (`Mean ± Std`) from AAAI 2026 (`A1`) and NeurIPS 2026 (`DA1`).
   - `tests/test_api.py`: Real-time API validation, latency (<10 ms), and live drift monitoring tests.
3. **Real-Time Inference & Drift Monitoring API (`api/main.py`):**
   - `GET /health`: Service readiness and sliding-window warm-up status.
   - `POST /score`: Real-time 2,048-wavelength anomaly detection and online Poisson bootstrap model update.
   - `GET /monitoring/stats`: Live streaming Z-score, rolling mean/std distance, anomaly rate, and latency observability.
4. **Containerization & CI/CD (`Dockerfile`, `.github/workflows/ci.yml`):**
   - Multi-layer `Dockerfile` with built-in `/health` `HEALTHCHECK`.
   - GitHub Actions workflow running automated tests on every push and Pull Request.

---

## Quickstart (Using `env_spectra`)

```bash
# 1. Run unit & end-to-end integration tests
/home/nicolas/spectral-benchmark/env_spectra/bin/python -m unittest tests/test_benchmark_suite.py -v

# 2. Run real-time API & monitoring tests
/home/nicolas/spectral-benchmark/env_spectra/bin/python -m unittest tests/test_api.py -v

# 3. Run terminal HTTP API demo (no browser required, works over SSH)
/home/nicolas/spectral-benchmark/env_spectra/bin/python api/main.py --demo

# 4. Run 5-run paper reproducibility verification
/home/nicolas/spectral-benchmark/env_spectra/bin/python -m unittest tests.test_paper_reproducibility.TestPaperReproducibility.test_a1_all_7_methods_match_published_table_1 -v
```
