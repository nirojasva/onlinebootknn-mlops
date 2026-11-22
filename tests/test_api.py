"""Unit tests for the Real-Time Spectral Anomaly Detection REST API (Topic 4 & Topic 6)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from api.main import SpectralAnomalyService, SpectrumRequest


class TestRealTimeAnomalyAPI(unittest.TestCase):
    """Tests for Topic 4 (REST API serving) and Topic 6 (Live monitoring & drift stats)."""

    def test_realtime_api_scoring_validation_and_monitoring(self) -> None:
        api_service = SpectralAnomalyService(
            window_size=10,
            chunk_size=10,
            ensemble_size=5,
            alpha=0.05,
            random_seed=42,
        )

        # 1. Health check before warm-up
        health_init = api_service.health()
        self.assertEqual(health_init["status"], "healthy")
        self.assertFalse(health_init["is_warmed_up"])

        # 2. Input validation must reject empty or NaN spectra
        with self.assertRaises(ValueError):
            api_service.score_spectrum(
                SpectrumRequest(wavelengths=[1.0, float("nan"), 3.0])
            )

        # 3. Stream 20 normal 2048-channel spectra (10 warm-up + 10 baseline)
        rng = np.random.RandomState(42)
        base_wave = np.sin(np.linspace(0, 4 * np.pi, 2048))
        for k in range(20):
            spectrum = (base_wave + rng.normal(0, 0.02, size=2048)).tolist()
            resp = api_service.score_spectrum(
                SpectrumRequest(
                    wavelengths=spectrum, timestamp=f"2026-11-22T12:00:{k:02d}"
                )
            )
            self.assertLess(resp.scoring_time_ms, 500.0)

        self.assertTrue(api_service.health()["is_warmed_up"])

        # 4. Send 1 anomalous 2048-channel spectrum and verify real-time detection
        anomaly_vec = (-base_wave + rng.normal(0, 0.02, size=2048)).tolist()
        anom_resp = api_service.score_spectrum(
            SpectrumRequest(
                wavelengths=anomaly_vec, timestamp="2026-11-22T12:00:21"
            )
        )
        self.assertTrue(anom_resp.is_anomaly)
        self.assertGreater(anom_resp.z_score, anom_resp.z_threshold)

        # 5. Check live monitoring endpoint stats
        mon = api_service.monitoring_stats()
        self.assertEqual(mon["total_requests"], 21)
        self.assertGreaterEqual(mon["n_anomalies"], 1)
        self.assertGreater(mon["anomaly_rate"], 0.0)
        print("\n=== Live API Monitoring Stats (/monitoring/stats) ===")
        for key, val in mon.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    unittest.main()
