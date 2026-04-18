"""
aria.production.metrics_exporter
==================================
Prometheus metrics bridge for the ARIA production system.

Tracks
------
- aria_inference_latency_ms    : Histogram per model (pointnet, gnn, nav, manip)
- aria_task_success_rate       : Gauge — rolling success rate over last N tasks
- aria_sensor_drop_count_total : Counter — number of dropped sensor frames
- aria_model_drift_score       : Gauge — KL divergence of action distribution vs baseline
- aria_fps                     : Gauge — estimated frames-per-second
- aria_system_health           : Gauge — 1.0 = healthy, 0.0 = degraded

Usage (standalone HTTP server)
-------------------------------
    exporter = ARIAMetricsExporter(port=9090)
    exporter.start()

    # In inference loop:
    with exporter.measure_latency("pointnet"):
        out = backbone(cloud)

    exporter.record_task_result(success=True)
    exporter.record_sensor_drop()

Usage (integration with ROS2 node)
------------------------------------
    # Import and use ARIAMetricsExporter in your ROS2 node's __init__.
    # The HTTP server runs in a background thread.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from typing import Generator, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics will be logged only")


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

_LATENCY_BUCKETS = (
    1.0, 2.0, 5.0, 10.0, 20.0, 33.0, 50.0, 100.0, 200.0, 500.0
)

if _PROM_AVAILABLE:
    _inference_latency = Histogram(
        "aria_inference_latency_ms",
        "Inference latency per model in milliseconds",
        labelnames=["model"],
        buckets=_LATENCY_BUCKETS,
    )
    _task_success_rate   = Gauge("aria_task_success_rate",   "Rolling task success rate (0–1)")
    _sensor_drop_total   = Counter("aria_sensor_drop_count_total", "Total dropped sensor frames",
                                   labelnames=["sensor"])
    _model_drift_score   = Gauge("aria_model_drift_score",   "Action distribution drift score")
    _fps_gauge           = Gauge("aria_fps",                 "Estimated inference fps")
    _system_health_gauge = Gauge("aria_system_health",       "Overall system health (1=OK)")


# ---------------------------------------------------------------------------
# Exporter class
# ---------------------------------------------------------------------------

class ARIAMetricsExporter:
    """
    Manages Prometheus metric collection and exposes an HTTP scrape endpoint.

    Parameters
    ----------
    port           : int    — port for the HTTP metrics server
    rolling_window : int    — number of recent tasks used for success-rate calc
    fps_window     : int    — number of recent inference timestamps for FPS calc
    """

    def __init__(
        self,
        port: int = 9090,
        rolling_window: int = 100,
        fps_window: int = 50,
    ) -> None:
        self._port = port
        self._rolling_window = rolling_window
        self._task_results: deque[bool] = deque(maxlen=rolling_window)
        self._frame_timestamps: deque[float] = deque(maxlen=fps_window)
        self._is_running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Prometheus HTTP server in a background thread."""
        if not _PROM_AVAILABLE:
            logger.warning("Prometheus client not available — HTTP server not started")
            return
        if not self._is_running:
            start_http_server(self._port)
            self._is_running = True
            logger.info("Prometheus metrics server listening on :%d/metrics", self._port)

    # ------------------------------------------------------------------
    # Latency measurement
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def measure_latency(self, model: str) -> Generator[None, None, None]:
        """
        Context manager to record inference latency for a named model.

        Usage:
            with exporter.measure_latency("pointnet"):
                output = model(input)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if _PROM_AVAILABLE:
                _inference_latency.labels(model=model).observe(elapsed_ms)
            logger.debug("[latency] %s: %.2f ms", model, elapsed_ms)

            # Update FPS estimate
            self._frame_timestamps.append(time.perf_counter())
            if len(self._frame_timestamps) >= 2:
                span = self._frame_timestamps[-1] - self._frame_timestamps[0]
                fps  = (len(self._frame_timestamps) - 1) / (span + 1e-8)
                if _PROM_AVAILABLE:
                    _fps_gauge.set(fps)

    # ------------------------------------------------------------------
    # Task outcome
    # ------------------------------------------------------------------

    def record_task_result(self, success: bool) -> None:
        """Record a completed task result (True = success, False = failure)."""
        with self._lock:
            self._task_results.append(success)
            rate = sum(self._task_results) / max(len(self._task_results), 1)
        if _PROM_AVAILABLE:
            _task_success_rate.set(rate)
        logger.info("Task result: %s | rolling_success_rate=%.2f",
                    "OK" if success else "FAIL", rate)

    @property
    def rolling_success_rate(self) -> float:
        with self._lock:
            return sum(self._task_results) / max(len(self._task_results), 1)

    # ------------------------------------------------------------------
    # Sensor health
    # ------------------------------------------------------------------

    def record_sensor_drop(self, sensor: str = "rgbd") -> None:
        """Increment the dropped-frame counter for a named sensor."""
        if _PROM_AVAILABLE:
            _sensor_drop_total.labels(sensor=sensor).inc()
        logger.warning("Sensor drop detected: %s", sensor)

    # ------------------------------------------------------------------
    # Model drift
    # ------------------------------------------------------------------

    def record_drift_score(self, kl_divergence: float) -> None:
        """
        Record the KL divergence between current and baseline action distribution.
        A high score (> 0.1) may indicate distribution shift / model drift.
        """
        if _PROM_AVAILABLE:
            _model_drift_score.set(kl_divergence)
        if kl_divergence > 0.1:
            logger.warning("Model drift detected: KL=%.4f", kl_divergence)

    # ------------------------------------------------------------------
    # System health
    # ------------------------------------------------------------------

    def set_health(self, healthy: bool) -> None:
        """Set overall system health gauge (1 = healthy, 0 = degraded)."""
        val = 1.0 if healthy else 0.0
        if _PROM_AVAILABLE:
            _system_health_gauge.set(val)
        if not healthy:
            logger.error("ARIA system health: DEGRADED")

    # ------------------------------------------------------------------
    # Summary (for logging without Prometheus)
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"ARIAMetrics | "
            f"success_rate={self.rolling_success_rate:.2%} | "
            f"tasks={len(self._task_results)} | "
            f"fps_window={len(self._frame_timestamps)}"
        )
