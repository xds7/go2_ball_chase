# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registration for the standalone Go2 ball-chasing task."""

import gymnasium as gym


gym.register(
    id="Isaac-Go2-Ball-Chase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__package__}.ball_task:Go2BallChaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__package__}.go2_ball_ppo:Go2BallChasePPORunnerCfg",
    },
)
