"""
aria.rl.train_manip
====================
Phase 3 training script — SAC manipulation agent (Franka Panda pick-and-place).

Run
---
    python -m aria.rl.train_manip --config configs/manip_training.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from aria.rl.envs.manip_env import ManipEnv
from aria.rl.extractors.gnn_extractor import GNNSceneGraphExtractor
from aria.sim2real.domain_randomizer import DomainRandomizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, n_envs: int = 1, seed: int = 7, use_dr: bool = True):
    """Create vectorised ManipEnv with optional domain randomization."""
    dr_cfg = cfg.get("domain_randomization", {})
    randomizer = DomainRandomizer(dr_cfg) if (use_dr and dr_cfg.get("enabled", True)) else None

    def _make():
        return ManipEnv(cfg=cfg, domain_randomizer=randomizer)

    vec_env = make_vec_env(
        _make, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv
    )
    return VecMonitor(vec_env)


def build_model(env, cfg: dict) -> SAC:
    sac_cfg = cfg.get("sac", {})
    pol_cfg = cfg.get("policy", {})
    gnn_cfg = pol_cfg.get("gnn", {})

    policy_kwargs = dict(
        features_extractor_class=GNNSceneGraphExtractor,
        features_extractor_kwargs=dict(
            features_dim=pol_cfg.get("mlp_features_dim", 512),
            gnn_hidden=gnn_cfg.get("hidden_channels", 128),
            gnn_layers=gnn_cfg.get("num_layers", 3),
            gnn_heads=gnn_cfg.get("heads", 4),
            gnn_output_dim=gnn_cfg.get("output_dim", 256),
            gnn_dropout=gnn_cfg.get("dropout", 0.1),
        ),
        net_arch=[256, 256],
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
        buffer_size=sac_cfg.get("buffer_size", 500_000),
        learning_starts=sac_cfg.get("learning_starts", 10_000),
        batch_size=sac_cfg.get("batch_size", 512),
        tau=float(sac_cfg.get("tau", 0.005)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        train_freq=sac_cfg.get("train_freq", 1),
        gradient_steps=sac_cfg.get("gradient_steps", 1),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        target_update_interval=sac_cfg.get("target_update_interval", 1),
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg.get("logging", {}).get("tensorboard_log", "runs/manip/"),
        verbose=cfg.get("logging", {}).get("verbose", 1),
        seed=cfg.get("environment", {}).get("seed", 7),
    )
    return model


def train(cfg: dict) -> None:
    ck_cfg = cfg.get("checkpointing", {})
    env_cfg = cfg.get("environment", {})
    seed = env_cfg.get("seed", 7)
    n_envs = env_cfg.get("num_envs", 4)

    logger.info("Building %d ManipEnv instances…", n_envs)
    train_env = build_env(cfg, n_envs=n_envs, seed=seed, use_dr=True)
    eval_env  = build_env(cfg, n_envs=1, seed=seed + 200, use_dr=False)

    logger.info("Building SAC model with GNN extractor…")
    model = build_model(train_env, cfg)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ck_cfg.get("best_model_path", "checkpoints/manip/best_model"),
        log_path=ck_cfg.get("checkpoint_dir", "checkpoints/manip/"),
        eval_freq=max(ck_cfg.get("eval_freq", 20_000) // n_envs, 1),
        n_eval_episodes=ck_cfg.get("eval_episodes", 20),
        deterministic=True,
        render=False,
    )
    ck_cb = CheckpointCallback(
        save_freq=max(ck_cfg.get("save_freq", 25_000) // n_envs, 1),
        save_path=ck_cfg.get("checkpoint_dir", "checkpoints/manip/"),
        name_prefix="aria_manip",
    )

    total_timesteps = cfg.get("sac", {}).get("total_timesteps", 1_500_000)
    logger.info("Starting SAC training for %d timesteps…", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([eval_cb, ck_cb]),
        reset_num_timesteps=True,
    )

    final_path = Path(ck_cfg.get("checkpoint_dir", "checkpoints/manip/")) / "aria_manip_final"
    model.save(str(final_path))
    logger.info("Training complete. Model saved to: %s", final_path)
    train_env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA — SAC Manipulation Training")
    parser.add_argument("--config", type=str, default="configs/manip_training.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    if args.resume:
        env_cfg = cfg.get("environment", {})
        train_env = build_env(cfg, n_envs=env_cfg.get("num_envs", 4), seed=env_cfg.get("seed", 7))
        model = SAC.load(args.resume, env=train_env)
        model.learn(total_timesteps=cfg.get("sac", {}).get("total_timesteps", 1_500_000),
                    reset_num_timesteps=False)
        train_env.close()
    else:
        train(cfg)


if __name__ == "__main__":
    main()
