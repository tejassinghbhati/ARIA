# ARIA: Autonomous Robot Intelligence Architecture

**An Embodied AI framework for perception, reasoning, navigation, and manipulation in 3D environments.**

---

## Abstract

The Autonomous Robot Intelligence Architecture (ARIA) is an advanced, production-grade embodied AI framework designed to bridge the gap between high-level natural language reasoning and low-level robotic control. ARIA integrates multimodal perception (RGB-D, LiDAR, IMU) with large language model (LLM) based sub-goal planning, culminating in deep reinforcement learning (DRL) policies for robust navigation and dexterous manipulation. The system is engineered for high-frequency deployment on physical hardware (Franka Emika Panda) utilizing optimized ONNX/TensorRT runtimes within a ROS2 ecosystem.

## System Architecture

ARIA operates via a cohesive pipeline that translates natural language into physical robotic action:

1. **Semantic Planning Node**: Utilizes a Phi-3-mini LLM for zero-shot task decomposition, grounding natural language instructions into actionable sub-goals (e.g., `[navigate_to "shelf"] → [pick_up "red mug"]`).
2. **Multimodal Perception Pipeline (10 Hz)**: Fuses RGB-D, LiDAR, and IMU data to construct a dense 3D representation. PointNet++ features are extracted to maintain a dynamic, live semantic scene graph represented as a NetworkX directed graph.
3. **Reinforcement Learning Control (20 Hz)**: A Graph Neural Network (GNN) based feature extractor processes the scene graph to condition Proximal Policy Optimization (PPO) for mobile navigation and Soft Actor-Critic (SAC) for continuous manipulation control.
4. **Hardware Deployment**: Action vectors are mapped to continuous control commands (`/cmd_vel`, `/franka/joint_commands`) on the physical robot.

## Repository Organization

```text
ARIA/
├── aria/
│   ├── perception/         # Sensor fusion, PointNet++ feature extraction, Scene Graph generation
│   ├── rl/                 # Gymnasium environments, GNN extractors, PPO/SAC training loops
│   ├── sim2real/           # Domain randomization and physics calibration modules
│   └── production/         # ONNX/TensorRT export pipelines and Prometheus metrics
├── ros2_ws/                # ROS2 Lifecycle nodes for hardware integration
├── docker/                 # Containerization and orchestration (Docker Compose)
├── monitoring/             # System telemetry via Prometheus and Grafana
├── configs/                # Centralized YAML experiment configurations
└── tests/                  # Unit and integration test suites
```

## Installation & Setup

### Environment Configuration

We recommend using a dedicated virtual environment. Ensure your system meets the hardware requirements (NVIDIA GPU recommended for accelerated inference).

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Test Suite Execution

Validate the installation by running the comprehensive test suite:

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## Execution Modules

The framework is partitioned into modular execution phases to facilitate independent research and development on sub-components.

### 1. 3D Perception & Grounding

The perception module processes raw sensor streams to construct a semantically rich representation of the environment.

```python
from aria.perception import SensorFusion, OccupancyMap, SceneGraph, NLPGrounder

# Initialize perception primitives
fusion = SensorFusion()
frame = fusion.process(rgbd_frame, lidar_scan)

# Update spatial representation
omap = OccupancyMap(resolution_m=0.05)
omap.update(frame.points_world)

# Construct semantic relationships
graph = SceneGraph()
# ... integrate detected nodes ...

# Ground language to scene graph
grounder = NLPGrounder()
plan = grounder.plan("fetch the red mug from the shelf", graph)
```

### 2. Navigation Policy Training

Train the mobile navigation policy using Proximal Policy Optimization (PPO).

```bash
python -m aria.rl.train_nav --config configs/nav_training.yaml
tensorboard --logdir runs/nav/
```

### 3. Manipulation Policy Training

Train the robotic arm manipulation policy using Soft Actor-Critic (SAC).

```bash
python -m aria.rl.train_manip --config configs/manip_training.yaml
```

### 4. Production Deployment & Telemetry

Export trained models to optimized runtimes and instantiate the telemetry stack.

```bash
# Export models to ONNX
python -m aria.production.export_onnx \
    --nav-model checkpoints/nav/best_model/best_model.zip \
    --manip-model checkpoints/manip/best_model/best_model.zip \
    --output-dir exports/onnx/

# Launch monitoring infrastructure
cd docker && docker compose up -d
# Grafana: http://localhost:3000 | Prometheus: http://localhost:9090
```

### 5. ROS2 Hardware Integration

Deploy the complete software stack to physical hardware using ROS2 Lifecycle nodes.

```bash
cd ros2_ws
colcon build --packages-select aria_ros
source install/setup.bash
ros2 launch aria_ros aria_full.launch.py
```

## Technology Stack

| Subsystem | Core Technologies |
| :--- | :--- |
| **3D Computer Vision** | PyTorch, Open3D, PointNet++ |
| **Semantic Representation**| NetworkX, CLIP (ViT-B/32) |
| **Language Planning** | HuggingFace Transformers (Phi-3-mini) |
| **Simulation Environment** | PyBullet, Gymnasium |
| **Reinforcement Learning** | Stable-Baselines3 (PPO, SAC), Custom Dense GAT |
| **Sim-to-Real Transfer** | Physics, Texture, and Lighting Domain Randomization |
| **Hardware Platform** | Franka Emika Panda (7-DOF) |
| **Deployment & Inference** | ONNX, TensorRT (FP16), ROS2 Humble |
| **DevOps & Telemetry** | Docker, NVIDIA Container Toolkit, Prometheus, Grafana |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
