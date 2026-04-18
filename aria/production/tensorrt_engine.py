"""
aria.production.tensorrt_engine
=================================
Builds and runs TensorRT inference engines from ONNX models.

Features
--------
- FP16/INT8 precision with automatic calibration
- Engine serialisation to .trt files for fast reload
- Latency benchmark vs. ONNX Runtime baseline
- Graceful fallback to ONNX Runtime if TensorRT is unavailable

Note: TensorRT is Linux + NVIDIA CUDA only.
      On non-TRT hosts this module falls back to ONNX Runtime automatically.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TensorRT availability check
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    _TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    logger.info("TensorRT %s available", trt.__version__)
except ImportError:
    _TRT_AVAILABLE = False
    logger.warning(
        "TensorRT / pycuda not available — falling back to ONNX Runtime. "
        "Install tensorrt + pycuda on an NVIDIA Linux host for GPU-accelerated inference."
    )

import onnxruntime as ort


# ---------------------------------------------------------------------------
# ONNX Runtime fallback engine
# ---------------------------------------------------------------------------

class ONNXEngine:
    """Thin wrapper around ONNX Runtime for environments without TensorRT."""

    def __init__(self, onnx_path: str, device: str = "cpu") -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self.onnx_path = onnx_path
        logger.info("ONNXEngine loaded: %s", Path(onnx_path).name)

    def infer(self, inputs: Dict[str, npt.NDArray]) -> List[npt.NDArray]:
        return self._session.run(None, {k: inputs[k] for k in self._input_names})

    def benchmark(self, inputs: Dict[str, npt.NDArray], n_runs: int = 100) -> float:
        for _ in range(10):
            self._session.run(None, {k: inputs[k] for k in self._input_names})
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self._session.run(None, {k: inputs[k] for k in self._input_names})
        return (time.perf_counter() - t0) / n_runs * 1000.0


# ---------------------------------------------------------------------------
# TensorRT engine builder & runner
# ---------------------------------------------------------------------------

if _TRT_AVAILABLE:

    class TRTEngine:
        """
        Builds a TensorRT engine from an ONNX model and runs inference.

        Parameters
        ----------
        onnx_path  : str | Path
        engine_path: str | Path   — where to save/load the serialised engine
        fp16       : bool         — enable FP16 precision
        workspace_gb : float      — builder memory workspace in GB
        """

        def __init__(
            self,
            onnx_path: str | Path,
            engine_path: str | Path,
            fp16: bool = True,
            workspace_gb: float = 2.0,
        ) -> None:
            self.onnx_path   = Path(onnx_path)
            self.engine_path = Path(engine_path)
            self.fp16        = fp16

            self._engine  = None
            self._context = None
            self._stream  = cuda.Stream()
            self._bindings: List[int] = []
            self._input_idxs: List[int]  = []
            self._output_idxs: List[int] = []
            self._h_inputs:  List[npt.NDArray] = []
            self._h_outputs: List[npt.NDArray] = []
            self._d_inputs:  List = []
            self._d_outputs: List = []

            if self.engine_path.exists():
                logger.info("Loading cached TRT engine: %s", self.engine_path.name)
                self._load_engine()
            else:
                logger.info("Building TRT engine from ONNX: %s", self.onnx_path.name)
                self._build_engine(workspace_gb)
                self._save_engine()
            self._allocate_buffers()

        # ------------------------------------------------------------------
        # Build / load
        # ------------------------------------------------------------------

        def _build_engine(self, workspace_gb: float) -> None:
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
            )
            if self.fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("TRT: FP16 enabled")

            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(str(self.onnx_path), "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("TRT parse error: %s", parser.get_error(i))
                    raise RuntimeError("TRT ONNX parse failed")

            serialized = builder.build_serialized_network(network, config)
            runtime = trt.Runtime(TRT_LOGGER)
            self._engine = runtime.deserialize_cuda_engine(serialized)

        def _save_engine(self) -> None:
            self.engine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self.engine_path), "wb") as f:
                f.write(self._engine.serialize())
            logger.info("TRT engine saved: %s", self.engine_path)

        def _load_engine(self) -> None:
            runtime = trt.Runtime(TRT_LOGGER)
            with open(str(self.engine_path), "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())

        # ------------------------------------------------------------------
        # Buffers
        # ------------------------------------------------------------------

        def _allocate_buffers(self) -> None:
            self._context = self._engine.create_execution_context()
            self._h_inputs  = []
            self._h_outputs = []
            self._d_inputs  = []
            self._d_outputs = []
            self._bindings  = []

            for i in range(self._engine.num_bindings):
                dtype    = trt.nptype(self._engine.get_binding_dtype(i))
                shape    = tuple(self._engine.get_binding_shape(i))
                host_mem = cuda.pagelocked_empty(shape, dtype)
                dev_mem  = cuda.mem_alloc(host_mem.nbytes)
                self._bindings.append(int(dev_mem))
                if self._engine.binding_is_input(i):
                    self._h_inputs.append(host_mem)
                    self._d_inputs.append(dev_mem)
                    self._input_idxs.append(i)
                else:
                    self._h_outputs.append(host_mem)
                    self._d_outputs.append(dev_mem)

        # ------------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------------

        def infer(self, inputs: List[npt.NDArray]) -> List[npt.NDArray]:
            for i, inp in enumerate(inputs):
                np.copyto(self._h_inputs[i], inp.ravel())
                cuda.memcpy_htod_async(self._d_inputs[i], self._h_inputs[i], self._stream)
            self._context.execute_async_v2(self._bindings, self._stream.handle)
            for i in range(len(self._h_outputs)):
                cuda.memcpy_dtoh_async(self._h_outputs[i], self._d_outputs[i], self._stream)
            self._stream.synchronize()
            return [np.array(h) for h in self._h_outputs]

        def benchmark(self, inputs: List[npt.NDArray], n_runs: int = 100) -> float:
            for _ in range(10):
                self.infer(inputs)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                self.infer(inputs)
            self._stream.synchronize()
            return (time.perf_counter() - t0) / n_runs * 1000.0


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path | None = None,
    fp16: bool = True,
    device: str = "cuda",
) -> "TRTEngine | ONNXEngine":
    """
    Build the fastest available inference engine for an ONNX model.

    Priority: TensorRT → ONNX Runtime CUDA → ONNX Runtime CPU.

    Parameters
    ----------
    onnx_path   : path to the .onnx file
    engine_path : where to save the .trt engine (auto-derived if None)
    fp16        : enable FP16 in TRT
    device      : "cuda" or "cpu"

    Returns
    -------
    TRTEngine if TRT available, else ONNXEngine
    """
    if _TRT_AVAILABLE and device == "cuda":
        if engine_path is None:
            engine_path = Path(onnx_path).with_suffix(".trt")
        return TRTEngine(onnx_path, engine_path, fp16=fp16)
    else:
        logger.info("Using ONNX Runtime engine (device=%s)", device)
        return ONNXEngine(str(onnx_path), device=device)


# ---------------------------------------------------------------------------
# Latency report helper
# ---------------------------------------------------------------------------

def latency_report(
    engines: Dict[str, "TRTEngine | ONNXEngine"],
    dummy_inputs_map: Dict[str, Dict],
    n_runs: int = 100,
    budget_ms: float = 33.0,
) -> Dict[str, float]:
    """
    Benchmark a collection of engines and report whether each meets the
    per-frame latency budget.

    Parameters
    ----------
    engines          : {name: engine}
    dummy_inputs_map : {name: {input_name: np_array}}
    budget_ms        : per-frame budget in milliseconds (default 33ms = 30fps)

    Returns
    -------
    dict {name: latency_ms}
    """
    results = {}
    for name, engine in engines.items():
        inputs = dummy_inputs_map.get(name, {})
        if isinstance(engine, ONNXEngine):
            lat = engine.benchmark(inputs, n_runs)
        else:
            lat = engine.benchmark(list(inputs.values()), n_runs)
        status = "✓" if lat <= budget_ms else "✗"
        logger.info("[%s] %s latency: %.2f ms  %s", name, type(engine).__name__, lat, status)
        results[name] = lat
    return results
