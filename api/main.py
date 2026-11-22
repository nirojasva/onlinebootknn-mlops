"""Real-Time Spectral Anomaly Detection Microservice (FastAPI + Pydantic / Fallback HTTP).

Exposes 4 production endpoints around `OnlineBootKNN`:
- `GET  /health`           : Service readiness & warm-up status
- `POST /score`            : Real-time test-then-train scoring (<1s per 2048-ch spectrum)
- `GET  /monitoring/stats` : Live streaming drift & Z-score monitoring metrics
- `POST /reset`            : Reconfigure detector hyperparameters on the fly
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from online_bootknn import OnlineBootKNN, clean_score


@dataclass
class SpectrumRequest:
    """Validated input schema for an incoming OES spectrum."""

    wavelengths: List[float]
    timestamp: Optional[str] = None
    update_model: bool = True
    expected_channels: Optional[int] = None

    def validate(self) -> np.ndarray:
        """Validate dimensionality and numeric finiteness (like Pydantic field validators)."""
        if not isinstance(self.wavelengths, list) or len(self.wavelengths) < 3:
            raise ValueError("wavelengths must be a list of at least 3 numeric values.")

        if (
            self.expected_channels is not None
            and len(self.wavelengths) != self.expected_channels
        ):
            raise ValueError(
                f"Expected {self.expected_channels} wavelength channels, got {len(self.wavelengths)}."
            )

        arr = np.asarray(self.wavelengths, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Spectrum contains NaN or Infinite intensity values.")

        return arr


@dataclass
class AnomalyScoreResponse:
    """Structured real-time prediction output schema."""

    timestamp: Optional[str]
    z_score: float
    z_threshold: float
    is_anomaly: bool
    is_warmed_up: bool
    n_processed: int
    n_anomalies: int
    scoring_time_ms: float
    training_time_ms: float


class SpectralAnomalyService:
    """Stateful real-time inference and monitoring service wrapping `OnlineBootKNN`."""

    def __init__(
        self,
        window_size: int = 120,
        chunk_size: int = 240,
        ensemble_size: int = 240,
        alpha: float = 0.05,
        dmetric: str = "cityblock",
        transf: str = "ZNORM",
        n_jobs: int = 1,
        random_seed: int = 42,
    ) -> None:
        self.detector = OnlineBootKNN(
            window_size=window_size,
            chunk_size=chunk_size,
            ensemble_size=ensemble_size,
            alpha=alpha,
            dmetric=dmetric,
            transf=transf,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )
        self.total_requests: int = 0
        self.total_scoring_time_ms: float = 0.0

    def health(self) -> Dict[str, Any]:
        """Return service readiness and warm-up progress."""
        warmup_count = (
            len(self.detector.data_window)
            if self.detector.init
            else self.detector.window_size
        )
        return {
            "status": "healthy",
            "model": str(self.detector),
            "is_warmed_up": not self.detector.init,
            "warmup_progress": f"{warmup_count}/{self.detector.window_size}",
            "total_requests": self.total_requests,
        }

    def score_spectrum(self, req: SpectrumRequest) -> AnomalyScoreResponse:
        """Validate, score, and optionally update the online detector with a new spectrum."""
        vec = req.validate()

        t0 = time.perf_counter()
        raw_z = self.detector.score_array(vec)
        scoring_ms = (time.perf_counter() - t0) * 1000.0
        cleaned_z, _ = clean_score(raw_z)

        is_anom = bool(self.detector.last_value_is_anomaly)
        was_warmed_up = not self.detector.init

        training_ms = 0.0
        if req.update_model:
            t1 = time.perf_counter()
            self.detector.train_array(vec)
            training_ms = (time.perf_counter() - t1) * 1000.0

        self.total_requests += 1
        self.total_scoring_time_ms += scoring_ms

        return AnomalyScoreResponse(
            timestamp=req.timestamp,
            z_score=round(cleaned_z, 4),
            z_threshold=round(float(self.detector.z_critical_one_tail), 4),
            is_anomaly=is_anom,
            is_warmed_up=was_warmed_up or (not self.detector.init),
            n_processed=int(self.detector.n),
            n_anomalies=int(self.detector.n_anomalies),
            scoring_time_ms=round(scoring_ms, 3),
            training_time_ms=round(training_ms, 3),
        )

    def monitoring_stats(self) -> Dict[str, Any]:
        """Return live operational and concept-drift statistics for dashboards/alerts."""
        core = self.detector.monitor_core_statistics()
        n_proc = core["n_processed"]
        n_anom = core["n_anomalies"]
        anom_rate = (n_anom / n_proc) if n_proc > 0 else 0.0
        avg_latency_ms = (
            self.total_scoring_time_ms / self.total_requests
            if self.total_requests > 0
            else 0.0
        )
        return {
            **core,
            "anomaly_rate": round(anom_rate, 6),
            "count_reset": self.detector.count_reset or 0,
            "avg_scoring_latency_ms": round(avg_latency_ms, 3),
            "total_requests": self.total_requests,
        }


# Global service instance
service = SpectralAnomalyService()

# If FastAPI & Pydantic are installed, register the FastAPI app for Uvicorn/Gunicorn
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="OnlineBootKNN Spectral Anomaly Detection API",
        version="0.1.0",
        description="Real-time unsupervised anomaly detection for high-dimensional OES spectral streams.",
    )

    class PydanticSpectrumPayload(BaseModel):
        wavelengths: List[float] = Field(
            ..., description="Ordered spectral intensity vector (e.g. 2048 wavelengths)."
        )
        timestamp: Optional[str] = Field(
            default=None, description="ISO-8601 acquisition timestamp."
        )
        update_model: bool = Field(
            default=True, description="Whether to update the online ensemble after scoring."
        )

    @app.get("/health")
    def get_health() -> Dict[str, Any]:
        return service.health()

    @app.post("/score")
    def post_score(payload: PydanticSpectrumPayload) -> Dict[str, Any]:
        try:
            req = SpectrumRequest(
                wavelengths=payload.wavelengths,
                timestamp=payload.timestamp,
                update_model=payload.update_model,
            )
            return asdict(service.score_spectrum(req))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/monitoring/stats")
    def get_monitoring_stats() -> Dict[str, Any]:
        return service.monitoring_stats()

except ImportError:
    app = None  # type: ignore


def run_builtin_http_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Zero-dependency HTTP server fallback when FastAPI/Uvicorn are not installed."""

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, service.health())
            elif self.path == "/monitoring/stats":
                self._send_json(200, service.monitoring_stats())
            else:
                self._send_json(404, {"detail": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/score":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(raw)
                    req = SpectrumRequest(
                        wavelengths=payload.get("wavelengths", []),
                        timestamp=payload.get("timestamp"),
                        update_model=payload.get("update_model", True),
                    )
                    resp = service.score_spectrum(req)
                    self._send_json(200, asdict(resp))
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send_json(422, {"detail": str(exc)})
            else:
                self._send_json(404, {"detail": "Not found"})

    server = HTTPServer((host, port), Handler)
    print(f"Serving OnlineBootKNN REST API on http://{host}:{port}")
    server.serve_forever()


def run_terminal_demo() -> None:
    """Start the HTTP API in a background thread and send live HTTP requests from the terminal.

    Designed for remote SSH servers where no browser is available.
    """
    import threading
    import urllib.request

    # Reset detector with a small 10-spectrum warm-up window for a quick terminal demo
    global service
    service = SpectralAnomalyService(window_size=10, chunk_size=10, ensemble_size=5, random_seed=42)

    class QuietHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return  # Keep terminal output clean

        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, service.health())
            elif self.path == "/monitoring/stats":
                self._send_json(200, service.monitoring_stats())
            else:
                self._send_json(404, {"detail": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/score":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw)
                req = SpectrumRequest(
                    wavelengths=payload.get("wavelengths", []),
                    timestamp=payload.get("timestamp"),
                    update_model=payload.get("update_model", True),
                )
                self._send_json(200, asdict(service.score_spectrum(req)))

    server = HTTPServer(("127.0.0.1", 0), QuietHandler)
    port = server.server_port
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        print(f"=== 1. GET {base_url}/health (Before Warm-Up) ===")
        with urllib.request.urlopen(f"{base_url}/health") as resp:
            print(json.dumps(json.loads(resp.read().decode()), indent=2))

        rng = np.random.RandomState(42)
        base_wave = np.sin(np.linspace(0, 4 * np.pi, 2048))

        print(f"\n=== 2. POST {base_url}/score (Streaming 15 Normal 2048-Wavelength Spectra) ===")
        last_normal_resp = {}
        for k in range(15):
            normal_vec = (base_wave + rng.normal(0, 0.02, size=2048)).tolist()
            req_data = json.dumps(
                {"wavelengths": normal_vec, "timestamp": f"2026-11-22T14:30:{k:02d}"}
            ).encode("utf-8")
            http_req = urllib.request.Request(
                f"{base_url}/score",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_req) as resp:
                last_normal_resp = json.loads(resp.read().decode())
        print("Last Normal Spectrum Response:")
        print(json.dumps(last_normal_resp, indent=2))

        print(f"\n=== 3. POST {base_url}/score (Injecting 1 Anomalous 2048-Wavelength Spectrum) ===")
        anomaly_vec = (-base_wave + rng.normal(0, 0.02, size=2048)).tolist()
        req_data = json.dumps(
            {"wavelengths": anomaly_vec, "timestamp": "2026-11-22T14:30:16"}
        ).encode("utf-8")
        http_req = urllib.request.Request(
            f"{base_url}/score",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_req) as resp:
            print(json.dumps(json.loads(resp.read().decode()), indent=2))

        print(f"\n=== 4. GET {base_url}/monitoring/stats (Live Drift & Operational Metrics) ===")
        with urllib.request.urlopen(f"{base_url}/monitoring/stats") as resp:
            print(json.dumps(json.loads(resp.read().decode()), indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        run_terminal_demo()
    else:
        run_builtin_http_server()
