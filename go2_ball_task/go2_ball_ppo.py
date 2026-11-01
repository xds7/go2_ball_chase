# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the upper Go2 ball-tracking policy."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2BallChasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Standard RSL-RL PPO config for the trainable upper policy."""

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    # Upper policy runs at 10 Hz.
    # 32 steps -> 3.2 seconds of rollout per environment.
    num_steps_per_env = 32

    max_iterations = 1000
    save_interval = 50

    experiment_name = "go2_ball_chase"

    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.6,
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,

        num_learning_epochs=5,
        num_mini_batches=4,

        learning_rate=3.0e-4,
        schedule="adaptive",

        gamma=0.99,
        lam=0.95,

        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # This clips the HIGH-LEVEL action only:
    # [vx, vy, wz].
    clip_actions = 1.0
