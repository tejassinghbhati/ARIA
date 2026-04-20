# ARIA — Autonomous Robot Intelligence Architecture

> **Embodied AI system** — perceive, reason, navigate, and manipulate in 3D indoor environments from natural language instructions.

---

## Overview

(I am currently working on this, will make sure to have it perfected soon)
ARIA is a production-grade embodied AI pipeline that enables a robot to:

1. **Understand** a natural language command (e.g. _"fetch the red mug from the shelf"_)
2. **Perceive** its 3D environment via RGB-D, LiDAR, and IMU sensor fusion
3. **Reason** about objects and spatial relationships using a live semantic scene graph
4. **Navigate** to a target object using a PPO-trained policy
5. **Manipulate** objects with a SAC-trained Franka Panda pick-and-place policy
6. **Deploy** at 30fps+ on physical hardware via ONNX/TensorRT, ROS2, and Docker

---

## Architecture

```
NL Command
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Language Node (Phi-3-mini LLM → Sub-goal planner)         │
│  SubGoals: [navigate_to "shelf"] → [pick_up "red mug"] …   │
└──────────────────────────┬──────────────────────────────────┘
                           │ grounded node IDs
┌──────────────────────────▼──────────────────────────────────┐
│  Perception Pipeline (10 Hz)                                │
│  RGB-D + LiDAR + IMU → SensorFusion → PointNet++ →         │
│  OccupancyMap + SceneGraph (NetworkX DiGraph)               │
└──────────────────────────┬──────────────────────────────────┘
                           │ scene graph tensors
┌──────────────────────────▼──────────────────────────────────┐
│  RL Policy (20 Hz)                                          │
│  GNNSceneGraphExtractor (GAT) → PPO Nav | SAC Manip         │
│  → /cmd_vel  |  → /franka/joint_commands                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
         Physical Robot (Franka Emika Panda)
```

---

## Repository Structure

```
ARIA/
├── aria/
│   ├── perception/         Phase 1 — sensor fusion, PointNet++, scene graph, NLP
│   ├── rl/                 Phase 2 & 3 — Gym envs, GNN extractor, PPO/SAC training
│   ├── sim2real/           Domain randomization, calibration
│   └── production/         ONNX export, TensorRT engine, Prometheus metrics
├── ros2_ws/                ROS2 Lifecycle nodes
├── docker/                 Dockerfiles + docker-compose
├── monitoring/             Prometheus + Grafana dashboard
├── configs/                YAML configuration files
└── tests/                  Unit + integration tests
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

---

## Phase-by-Phase Execution

### Phase 1 — 3D Perception

```python
from aria.perception import SensorFusion, OccupancyMap, SceneGraph, NLPGrounder

fusion = SensorFusion()
frame  = fusion.process(rgbd_frame, lidar_scan)

omap = OccupancyMap(resolution_m=0.05)
omap.update(frame.points_world)

graph = SceneGraph()
# ... add detected nodes ...

grounder = NLPGrounder()
plan = grounder.plan("fetch the red mug from the shelf", graph)
```

### Phase 2 — Navigation Training

```bash
python -m aria.rl.train_nav --config configs/nav_training.yaml
tensorboard --logdir runs/nav/
```

### Phase 3 — Manipulation Training

```bash
python -m aria.rl.train_manip --config configs/manip_training.yaml
```

### Phase 4 — Production Export & Monitoring

```bash
python -m aria.production.export_onnx \
    --nav-model   checkpoints/nav/best_model/best_model.zip \
    --manip-model checkpoints/manip/best_model/best_model.zip \
    --output-dir  exports/onnx/

cd docker && docker compose up
# Grafana: http://localhost:3000  |  Prometheus: http://localhost:9090
```

### ROS2 Deployment

```bash
cd ros2_ws && colcon build --packages-select aria_ros
source install/setup.bash
ros2 launch aria_ros aria_full.launch.py
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| 3D Vision | PyTorch, Open3D, PointNet++ (native) |
| Scene Graph | NetworkX, CLIP (ViT-B/32) |
| NLP Planner | Phi-3-mini (HuggingFace Transformers) |
| Simulation | PyBullet + Gymnasium |
| RL | Stable-Baselines3 (PPO nav, SAC manip) |
| GNN | Dense GAT (custom, SB3-compatible) |
| Sim-to-Real | Domain randomization (texture/lighting/physics) |
| Robot | Franka Emika Panda (7-DOF, PyBullet URDF) |
| Production | ONNX + TensorRT (FP16), ROS2 Humble |
| Containerization | Docker + NVIDIA Container Toolkit |
| Monitoring | Prometheus + Grafana |

---

## License

MIT
