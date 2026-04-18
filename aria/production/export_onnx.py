"""
aria.production.export_onnx
=============================
Exports trained ARIA PyTorch models to ONNX format for production inference.

Exports:
1. PointNet++ backbone      → aria_pointnet.onnx
2. GNN feature extractor    → aria_gnn_extractor.onnx
3. Navigation policy        → aria_nav_policy.onnx
4. Manipulation policy      → aria_manip_policy.onnx

Each export validates that the ONNX output matches PyTorch within tolerance.

Run
---
    python -m aria.production.export_onnx \\
        --nav-model   checkpoints/nav/best_model/best_model.zip \\
        --manip-model checkpoints/manip/best_model/best_model.zip \\
        --output-dir  exports/onnx/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from aria.perception.pointnet_backbone import PointNetBackbone

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Generic export helper
# ---------------------------------------------------------------------------

def export_model(
    model: nn.Module,
    dummy_inputs: tuple | torch.Tensor,
    output_path: str | Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
    opset: int = 17,
    atol: float = 1e-4,
) -> None:
    """
    Export a PyTorch model to ONNX and validate numerical correctness.

    Parameters
    ----------
    model         : nn.Module (already in eval mode)
    dummy_inputs  : single tensor or tuple of tensors
    output_path   : path for the .onnx file
    input_names   : ONNX input node names
    output_names  : ONNX output node names
    dynamic_axes  : dict describing dynamic dimensions
    opset         : ONNX opset version
    atol          : absolute tolerance for output diff validation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # 1. ONNX IR check
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX IR check passed: %s", output_path.name)

    # 2. Numerical validation via ONNX Runtime
    _validate_onnx(model, dummy_inputs, str(output_path), atol=atol)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Exported: %s (%.2f MB)", output_path.name, size_mb)


def _validate_onnx(
    torch_model: nn.Module,
    dummy_inputs: tuple | torch.Tensor,
    onnx_path: str,
    atol: float = 1e-4,
) -> None:
    """Run PyTorch and ONNX Runtime side-by-side and compare outputs."""
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs = (dummy_inputs,)

    with torch.no_grad():
        torch_outs = torch_model(*dummy_inputs)
    if isinstance(torch_outs, torch.Tensor):
        torch_outs = (torch_outs,)
    torch_outs_np = [o.cpu().numpy() for o in torch_outs]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {
        sess.get_inputs()[i].name: dummy_inputs[i].cpu().numpy()
        for i in range(len(dummy_inputs))
    }
    ort_outs = sess.run(None, feed)

    for i, (t_out, o_out) in enumerate(zip(torch_outs_np, ort_outs)):
        max_diff = float(np.abs(t_out - o_out).max())
        if max_diff > atol:
            raise ValueError(
                f"ONNX validation FAILED: output[{i}] max_diff={max_diff:.6f} > atol={atol}"
            )
        logger.info("  output[%d] validated: max_diff=%.2e (atol=%.2e) ✓", i, max_diff, atol)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx(onnx_path: str, inputs: dict[str, np.ndarray], n_runs: int = 100) -> float:
    """
    Benchmark ONNX Runtime inference latency.

    Returns
    -------
    float — mean latency in milliseconds
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Warm-up
    for _ in range(10):
        sess.run(None, inputs)
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, inputs)
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000.0
    logger.info("ONNX Runtime latency [%s]: %.2f ms/inference (n=%d)",
                Path(onnx_path).name, elapsed_ms, n_runs)
    return elapsed_ms


# ---------------------------------------------------------------------------
# PointNet++ export
# ---------------------------------------------------------------------------

def export_pointnet(output_dir: Path, n_points: int = 2048, in_channels: int = 3) -> str:
    logger.info("Exporting PointNet++ backbone…")
    backbone = PointNetBackbone(in_channels=in_channels, global_feat_dim=512)
    dummy = torch.randn(1, n_points, 3 + in_channels)
    out_path = output_dir / "aria_pointnet.onnx"

    # Wrap to return only global_feat for simpler ONNX graph
    class _BackboneWrapper(nn.Module):
        def __init__(self, b): super().__init__(); self.b = b
        def forward(self, x):  _, g = self.b(x); return g

    export_model(
        _BackboneWrapper(backbone), dummy, out_path,
        input_names=["cloud"],
        output_names=["global_feat"],
        dynamic_axes={"cloud": {0: "batch_size", 1: "num_points"}},
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# SB3 policy export
# ---------------------------------------------------------------------------

def export_sb3_policy(
    model_zip: str,
    output_path: Path,
    env_type: str = "nav",
) -> str:
    """
    Export the actor network from a trained SB3 model to ONNX.

    SB3 actor takes a flat feature vector (after the extractor) and outputs
    action mean.  We export the full actor (extractor + policy net) by
    tracing through a dummy forward pass.
    """
    from stable_baselines3 import PPO, SAC
    logger.info("Loading SB3 model from: %s", model_zip)

    Algo = PPO if env_type == "nav" else SAC

    # Load without env to avoid spinning up physics
    model = Algo.load(model_zip, device="cpu")
    policy = model.policy.cpu().eval()

    # Determine observation shape from policy
    obs_space = model.observation_space

    # Build a flat dummy obs dict → convert to tensors
    dummy_obs: Dict[str, torch.Tensor] = {}
    for key, space in obs_space.spaces.items():
        dummy_obs[key] = torch.zeros((1, *space.shape), dtype=torch.float32)

    class _PolicyWrapper(nn.Module):
        def __init__(self, p): super().__init__(); self.p = p
        def forward(self, **kwargs):
            features = self.p.extract_features(kwargs, self.p.pi_features_extractor)
            latent   = self.p.mlp_extractor.forward_actor(features)
            return self.p.action_net(latent)

    wrapper = _PolicyWrapper(policy)

    with torch.no_grad():
        out_test = wrapper(**dummy_obs)
    action_dim = out_test.shape[-1]
    logger.info("Policy action_dim=%d", action_dim)

    # ONNX export with named inputs
    input_names = list(dummy_obs.keys())
    output_names = ["action_mean"]
    inputs_tuple = tuple(dummy_obs.values())

    export_model(
        wrapper, inputs_tuple, output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={k: {0: "batch"} for k in input_names},
        atol=1e-3,   # SB3 BN layers may introduce slight differences
    )
    return str(output_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA ONNX Export Pipeline")
    parser.add_argument("--nav-model",   type=str, default=None)
    parser.add_argument("--manip-model", type=str, default=None)
    parser.add_argument("--output-dir",  type=str, default="exports/onnx/")
    parser.add_argument("--n-points",    type=int, default=2048)
    parser.add_argument("--in-channels", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. PointNet++
    pn_path = export_pointnet(out, n_points=args.n_points, in_channels=args.in_channels)
    benchmark_onnx(pn_path, {"cloud": np.random.randn(1, args.n_points, 3 + args.in_channels).astype(np.float32)})

    # 2. Navigation policy
    if args.nav_model:
        nav_path = out / "aria_nav_policy.onnx"
        export_sb3_policy(args.nav_model, nav_path, env_type="nav")

    # 3. Manipulation policy
    if args.manip_model:
        manip_path = out / "aria_manip_policy.onnx"
        export_sb3_policy(args.manip_model, manip_path, env_type="manip")

    logger.info("All exports complete. Files in: %s", out.resolve())


if __name__ == "__main__":
    main()
