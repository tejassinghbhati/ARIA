# ARIA: Autonomous Robot Intelligence Architecture

**An Embodied AI framework for perception, reasoning, navigation, and manipulation in 3D environments.**

---

## Abstract

The Autonomous Robot Intelligence Architecture (ARIA) is an advanced, production-grade embodied AI framework designed to bridge the gap between high-level natural language reasoning and low-level robotic control. ARIA integrates multimodal perception (RGB-D, LiDAR, IMU) with large language model (LLM) based sub-goal planning, culminating in deep reinforcement learning (DRL) policies for robust navigation and dexterous manipulation. The system is engineered for high-frequency deployment on physical hardware (Franka Emika Panda) utilizing optimized ONNX/TensorRT runtimes within a ROS2 ecosystem.

---

## System Architecture

ARIA operates via a cohesive pipeline that translates natural language into physical robotic action:

1. **Semantic Planning Node**: Utilizes a Phi-3-mini LLM for zero-shot task decomposition, grounding natural language instructions into actionable sub-goals (e.g., `[navigate_to "shelf"] → [pick_up "red mug"]`).
2. **Multimodal Perception Pipeline (10 Hz)**: Fuses RGB-D, LiDAR, and IMU data to construct a dense 3D representation. PointNet++ features are extracted to maintain a dynamic, live semantic scene graph represented as a NetworkX directed graph.
3. **Reinforcement Learning Control (20 Hz)**: A Graph Neural Network (GNN) based feature extractor processes the scene graph to condition Proximal Policy Optimization (PPO) for mobile navigation and Soft Actor-Critic (SAC) for continuous manipulation control.
4. **Hardware Deployment**: Action vectors are mapped to continuous control commands (`/cmd_vel`, `/franka/joint_commands`) on the physical robot.

---

## Repository Organization

```
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

---

## Installation & Setup (Windows)

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11.x | 3.12+ not supported (open3d limitation) |
| Microsoft C++ Build Tools | 14.0+ | Required for pybullet compilation |
| Git | Any | For CLIP installation from source |
| Docker Desktop | Any | Optional — for monitoring stack and ROS2 |

#### Install Python 3.11

Download from [python.org](https://www.python.org/downloads/release/python-3119/). During install, check **"Add Python to PATH"**.

Verify:
```powershell
py -3.11 --version
# Python 3.11.9
```

#### Install Microsoft C++ Build Tools

Required to compile `pybullet` from source. Download from:
`https://visualstudio.microsoft.com/visual-cpp-build-tools/`

Run the installer and select **"Desktop development with C++"**. After install, **restart PowerShell**.

### Environment Setup

```powershell
# Create a Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate it (run this every time you open a new terminal)
.venv\Scripts\activate

# Verify you are on 3.11
python --version
# Python 3.11.9

# Install all dependencies (allow extra time for large downloads)
pip install --timeout 300 -r requirements.txt

# Install ARIA as an editable package
pip install -e .
```

> **Note:** The `requirements.txt` install downloads ~500 MB of packages including PyTorch. If it times out, re-run the same command — pip will resume from cached packages.

### PyTorch-Geometric (optional, for GNN training)

PyG sparse dependencies require a separate install matched to your CUDA version. Run **after** the main requirements install:

```powershell
# CPU only
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# CUDA 12.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

---

## Running the Tests

Always use `python -m pytest` (not bare `pytest`) to ensure the venv Python is used, not any system-installed pytest.

```powershell
# Activate venv first
.venv\Scripts\activate

# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# All tests with coverage
python -m pytest --cov=aria --cov-report=term-missing
```

---

## Execution Modules

### 1. Perception & Language Grounding (Python API)

Use the perception primitives directly in Python — no additional setup required beyond the pip install.

```python
from aria.perception import SensorFusion, OccupancyMap, SceneGraph, NLPGrounder

# Initialize sensor fusion
fusion = SensorFusion()
frame = fusion.process(rgbd_frame, lidar_scan)

# Build occupancy map
omap = OccupancyMap(resolution_m=0.05)
omap.update(frame.points_world)

# Build and query scene graph
graph = SceneGraph()
# ... add detected nodes ...

# Ground a natural language command to a task plan
grounder = NLPGrounder()
plan = grounder.plan("fetch the red mug from the shelf", graph)
```

### 2. Navigation Policy Training (PPO)

```powershell
python -m aria.rl.train_nav --config configs/nav_training.yaml
```

Monitor training in TensorBoard:

```powershell
tensorboard --logdir runs/nav/
# Open http://localhost:6006 in your browser
```

### 3. Manipulation Policy Training (SAC)

```powershell
python -m aria.rl.train_manip --config configs/manip_training.yaml
```

### 4. ONNX Export for Deployment

After training, export models to ONNX for optimized inference:

```powershell
python -m aria.production.export_onnx `
    --nav-model checkpoints/nav/best_model/best_model.zip `
    --manip-model checkpoints/manip/best_model/best_model.zip `
    --output-dir exports/onnx/
```

### 5. Monitoring Stack (Prometheus + Grafana)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) to be installed and running.

```powershell
cd docker
docker compose up -d
```

| Service | URL |
|---|---|
| Grafana dashboards | http://localhost:3000 |
| Prometheus metrics | http://localhost:9090 |

To run RL training inside Docker (GPU required):

```powershell
cd docker
docker compose --profile training up --build
```

To stop:

```powershell
docker compose down
```

### 6. ROS2 Hardware Integration

ROS2 is only needed when deploying to a physical Franka robot. It is **not required** for simulation or training.

#### Option A — Native Windows ROS2 Install

Install ROS2 Humble for Windows following the official guide:
`https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html`

After install, open a **new** PowerShell window and run:

```powershell
# Source ROS2 environment
call C:\opt\ros\humble\x64\setup.bat

# Build the ARIA ROS2 package
cd E:\ARIA\ros2_ws
colcon build --packages-select aria_ros

# Source the built workspace
call install\setup.bat

# Launch the full ARIA stack
ros2 launch aria_ros aria_full.launch.py
```

#### Option B — Docker (recommended)

The Docker Compose stack runs the full ROS2 pipeline in Linux containers, avoiding the native Windows ROS2 install. Requires Docker Desktop with the NVIDIA Container Toolkit for GPU support.

```powershell
cd E:\ARIA\docker
docker compose up --build
```

---

## Technology Stack

| Subsystem | Core Technologies |
|---|---|
| **3D Computer Vision** | PyTorch, Open3D, PointNet++ |
| **Semantic Representation** | NetworkX, CLIP (ViT-B/32) |
| **Language Planning** | HuggingFace Transformers (Phi-3-mini) |
| **Simulation Environment** | PyBullet, Gymnasium |
| **Reinforcement Learning** | Stable-Baselines3 (PPO, SAC), Custom Dense GAT |
| **Sim-to-Real Transfer** | Physics, Texture, and Lighting Domain Randomization |
| **Hardware Platform** | Franka Emika Panda (7-DOF) |
| **Deployment & Inference** | ONNX, TensorRT (FP16), ROS2 Humble |
| **DevOps & Telemetry** | Docker, NVIDIA Container Toolkit, Prometheus, Grafana |

---

## Troubleshooting

**`open3d` install fails**
open3d only supports Python 3.10–3.11. Make sure your venv was created with `py -3.11 -m venv .venv`, not the system Python.

**`pybullet` build fails with MSVC error**
Install Microsoft C++ Build Tools (see Prerequisites above), then restart PowerShell before retrying.

**`pytest` uses the wrong Python**
Always run `python -m pytest` instead of bare `pytest`. The bare command may resolve to a system or conda Python rather than the venv.

**`pip install` times out on large packages**
Re-run with a higher timeout: `pip install --timeout 300 -r requirements.txt`. Cached packages are skipped so it resumes from where it left off.

**`docker compose up` — no configuration file found**
Run from inside the `docker/` directory, or specify the path explicitly: `docker compose -f docker\docker-compose.yml up`

**`colcon` / `ros2` not recognized**
These require a ROS2 installation. If you only need simulation and training, ROS2 is not required — skip to sections 1–4 above.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.