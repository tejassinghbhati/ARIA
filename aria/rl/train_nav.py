"""
aria.rl.train_nav
==================
Phase 2 training script — PPO navigation agent.

Run
---
    python -m aria.rl.train_nav --config configs/nav_training.yaml

The script:
1. Loads config from YAML
2. Creates vectorised NavEnv with SubprocVecEnv
3. Wraps with GNNSceneGraphExtractor
4. Trains PPO to convergence (2M steps default)
5. Saves best checkpoint by eval success rate
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from aria.rl.envs.nav_env import NavEnv
from aria.rl.extractors.gnn_extractor import GNNSceneGraphExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, n_envs: int = 1, seed: int = 42):
    """Create a vectorised NavEnv."""
    def _make():
        env = NavEnv(cfg=cfg)
        return env

    vec_env = make_vec_env(
        _make,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )
    return VecMonitor(vec_env)


def build_model(env, cfg: dict) -> PPO:
    """Construct the PPO model with the GNN policy extractor."""
    obs_cfg  = cfg.get("observation", {})
    ppo_cfg  = cfg.get("ppo", {})
    pol_cfg  = cfg.get("policy", {})
    gnn_cfg  = pol_cfg.get("gnn", {})

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
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 512),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg.get("logging", {}).get("tensorboard_log", "runs/nav/"),
        verbose=cfg.get("logging", {}).get("verbose", 1),
        seed=cfg.get("environment", {}).get("seed", 42),
    )
    return model


def train(cfg: dict) -> None:
    ck_cfg   = cfg.get("checkpointing", {})
    env_cfg  = cfg.get("environment", {})
    n_envs   = env_cfg.get("num_envs", 8 if os.cpu_count() and os.cpu_count() > 4 else 4)
    seed     = env_cfg.get("seed", 42)

    logger.info("Building %d parallel NavEnv instances…", n_envs)
    train_env = build_env(cfg, n_envs=n_envs, seed=seed)
    eval_env  = build_env(cfg, n_envs=1, seed=seed + 100)

    logger.info("Building PPO model with GNN extractor…")
    model = build_model(train_env, cfg)

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ck_cfg.get("best_model_path", "checkpoints/nav/best_model"),
        log_path=ck_cfg.get("checkpoint_dir", "checkpoints/nav/"),
        eval_freq=max(ck_cfg.get("eval_freq", 25_000) // n_envs, 1),
        n_eval_episodes=ck_cfg.get("eval_episodes", 20),
        deterministic=True,
        render=False,
    )
    ck_cb = CheckpointCallback(
        save_freq=max(ck_cfg.get("save_freq", 50_000) // n_envs, 1),
        save_path=ck_cfg.get("checkpoint_dir", "checkpoints/nav/"),
        name_prefix="aria_nav",
    )
    callbacks = CallbackList([eval_cb, ck_cb])

    total_timesteps = cfg.get("ppo", {}).get("total_timesteps", 2_000_000)
    logger.info("Starting PPO training for %d timesteps…", total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=True)

    final_path = Path(ck_cfg.get("checkpoint_dir", "checkpoints/nav/")) / "aria_nav_final"
    model.save(str(final_path))
    logger.info("Training complete. Model saved to: %s", final_path)

    train_env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA — PPO Navigation Training")
    parser.add_argument(
        "--config", type=str,
        default="configs/nav_training.yaml",
        help="Path to nav_training.yaml",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint .zip to resume from",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override ppo.total_timesteps from config",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override environment.seed from config",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    # Apply CLI overrides
    if args.timesteps is not None:
        cfg.setdefault("ppo", {})["total_timesteps"] = args.timesteps
        logger.info("CLI override: total_timesteps=%d", args.timesteps)
    if args.seed is not None:
        cfg.setdefault("environment", {})["seed"] = args.seed
        logger.info("CLI override: seed=%d", args.seed)

    if args.resume:
        import stable_baselines3 as sb3
        logger.info("Resuming from checkpoint: %s", args.resume)
        env_cfg = cfg.get("environment", {})
        n_envs  = env_cfg.get("num_envs", 4)
        seed    = env_cfg.get("seed", 42)
        train_env = build_env(cfg, n_envs=n_envs, seed=seed)
        model = PPO.load(args.resume, env=train_env)
        ck_cfg = cfg.get("checkpointing", {})
        total_timesteps = cfg.get("ppo", {}).get("total_timesteps", 2_000_000)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        train_env.close()
    else:
        train(cfg)


if __name__ == "__main__":
    main()
