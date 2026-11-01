# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train or play the custom Unitree Go2 ball-chasing task.

Run with the Isaac Lab launcher from any checkout of this repository.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

TASK_ID = "Isaac-Go2-Ball-Chase-v0"

CUSTOM_DIR = Path(__file__).resolve().parent
if str(CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DIR))

# Start Isaac Sim FIRST.
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Go2 ball chase: upper RSL-RL PPO + frozen lower locomotion policy"
)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--mode", choices=("train", "play"), default="train")
parser.add_argument("--walking_policy", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Only after SimulationApp is alive do we import task/simulation modules.
import gymnasium as gym
import torch

import go2_ball_task  # noqa: F401  # registers the Gym environment

from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg


def resolve_checkpoint(log_root: Path, checkpoint: str | None, load_run: str | None) -> Path:
    if checkpoint is not None:
        direct = Path(checkpoint).expanduser()
        if direct.is_file():
            return direct.resolve()

    if load_run is None:
        raise FileNotFoundError(
            "Upper-policy checkpoint not found. "
            "Pass an existing --checkpoint path, or pass --load_run plus --checkpoint."
        )

    run_dir = log_root / load_run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if checkpoint:
        candidate = run_dir / checkpoint
        if candidate.is_file():
            return candidate.resolve()
        matches = sorted(run_dir.glob(checkpoint))
    else:
        matches = sorted(run_dir.glob("model_*.pt"))

    if not matches:
        raise FileNotFoundError(f"No checkpoint found under: {run_dir}")

    return matches[-1].resolve()


def main() -> None:
    if args_cli.task not in gym.registry:
        raise RuntimeError(
            f"Task '{args_cli.task}' was not registered. Run this custom "
            "training entry point with the Isaac Lab launcher: "
            "./isaaclab.sh -p /path/to/go2_ball_chase/train.py"
        )

    walking_policy = Path(args_cli.walking_policy).expanduser().resolve()
    if not walking_policy.is_file():
        raise FileNotFoundError(f"Walking policy does not exist: {walking_policy}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed

    env_cfg.actions.pre_trained_policy.policy_path = str(walking_policy)

    # During playback, keep the episode running indefinitely while the robot
    # remains upright. A fall still triggers the base-height termination and
    # resets both the robot and the ball.
    if args_cli.mode == "play":
        env_cfg.terminations.time_out = None

    agent_cfg = load_cfg_from_registry(
        args_cli.task,
        "rsl_rl_cfg_entry_point",
    )
    agent_cfg.seed = args_cli.seed

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # Remove or migrate compatibility fields according to the installed
    # RSL-RL version before serializing the configuration for the runner.
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        metadata.version("rsl-rl-lib"),
    )

    log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None,
    )
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None if args_cli.mode == "play" else str(run_dir),
        device=agent_cfg.device,
    )

    try:
        if args_cli.mode == "train":
            if args_cli.resume:
                checkpoint = resolve_checkpoint(
                    log_root,
                    args_cli.checkpoint,
                    args_cli.load_run,
                )
                print(f"[Ball] Resume upper PPO : {checkpoint}")
                runner.load(str(checkpoint))

            print(f"[Ball] task              : {args_cli.task}")
            print(f"[Ball] num envs          : {args_cli.num_envs}")
            print(f"[Ball] walking policy    : {walking_policy}")
            print(f"[Ball] upper PPO runner  : {runner.__class__.__name__}")
            print(f"[Ball] max iterations    : {agent_cfg.max_iterations}")
            print(f"[Ball] log directory     : {run_dir}")

            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )

        else:
            checkpoint = resolve_checkpoint(
                log_root,
                args_cli.checkpoint,
                args_cli.load_run,
            )
            print(f"[Ball] Load upper policy : {checkpoint}")
            print(f"[Ball] Walking policy    : {walking_policy}")

            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            obs = env.get_observations()

            with torch.inference_mode():
                while simulation_app.is_running():
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Report failures before Kit shutdown messages obscure the traceback.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
