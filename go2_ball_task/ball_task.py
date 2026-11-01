# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone hierarchical Go2 ball-chasing environment.

The trainable policy is the upper tracking policy.
The lower locomotion policy is a frozen TorchScript ``policy.pt`` loaded by
``PreTrainedPolicyActionCfg``.

This environment inherits directly from ``ManagerBasedRLEnvCfg``.
It does NOT inherit ``UnitreeGo2FlatEnvCfg``.
"""

from __future__ import annotations

import math
import os

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


TARGET_DISTANCE = 0.70
TARGET_HEIGHT = 0.11

# Recommended: set once in the shell before launching training:
#   export GO2_WALKING_POLICY=/absolute/path/to/policy.pt
WALKING_POLICY_PATH = os.environ.get(
    "GO2_WALKING_POLICY",
    "/ABSOLUTE/PATH/TO/GO2/ROUGH/policy.pt",
)

# Reuse the interface of the rough-terrain policy supplied to ``train.py``.
# In particular, its observation has a 187-value height scan; using the flat
# config here produces 48 observations for a policy trained with 235.
_official_go2_rough_cfg = UnitreeGo2RoughEnvCfg()
LOW_LEVEL_ACTION_CFG = _official_go2_rough_cfg.actions.joint_pos
LOW_LEVEL_OBSERVATION_CFG = _official_go2_rough_cfg.observations.policy


# Custom observation / reward / reset functions
def ball_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return planar ball displacement in the robot base frame."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    displacement_w = ball.data.root_pos_w.torch - robot.data.root_pos_w.torch
    displacement_b = math_utils.quat_apply_inverse(robot.data.root_quat_w.torch, displacement_w)

    # Keep the position observation in a modest numerical range.
    return displacement_b[:, :2] / 3.0


def ball_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return planar robot-to-ball distance."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    return torch.linalg.vector_norm(
        ball.data.root_pos_w.torch[:, :2] - robot.data.root_pos_w.torch[:, :2],
        dim=-1,
    )


def distance_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward maintaining TARGET_DISTANCE from the ball."""
    error = (ball_distance(env) - TARGET_DISTANCE) / 0.45
    return torch.exp(-(error * error))


def facing_ball_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward keeping the target in front of the robot."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    displacement_w = ball.data.root_pos_w.torch - robot.data.root_pos_w.torch
    displacement_b = math_utils.quat_apply_inverse(robot.data.root_quat_w.torch, displacement_w)

    planar_norm = torch.linalg.vector_norm(
        displacement_b[:, :2],
        dim=-1,
    ).clamp_min(1.0e-6)

    return displacement_b[:, 0] / planar_norm


def relative_speed_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize excessive relative speed near the target."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    distance = ball_distance(env)
    near_target = torch.exp(
        -(((distance - TARGET_DISTANCE) / 0.50) ** 2)
    )

    relative_velocity = (
        robot.data.root_lin_vel_w.torch[:, :2]
        - ball.data.root_lin_vel_w.torch[:, :2]
    )

    return near_target * torch.sum(
        relative_velocity * relative_velocity,
        dim=-1,
    )


def too_close_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize entering the ball too closely."""
    return (ball_distance(env) < 0.40).float()


def reset_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_radius: float = 1.2,
    max_radius: float = 2.5,
    min_speed: float = 0.25,
    max_speed: float = 0.50,
) -> None:
    """Reset the ball with a random planar position and velocity."""
    ball = env.scene[asset_cfg.name]
    robot = env.scene["robot"]

    count = env_ids.numel()
    if count == 0:
        return

    angle = torch.empty(
        count,
        device=env.device,
    ).uniform_(-math.pi, math.pi)

    radius = torch.empty(
        count,
        device=env.device,
    ).uniform_(min_radius, max_radius)

    velocity_angle = torch.empty(
        count,
        device=env.device,
    ).uniform_(-math.pi, math.pi)

    speed = torch.empty(
        count,
        device=env.device,
    ).uniform_(min_speed, max_speed)

    root_state = ball.data.default_root_state.torch[env_ids].clone()

    root_state[:, 0] = (
        robot.data.root_pos_w.torch[env_ids, 0]
        + radius * torch.cos(angle)
    )
    root_state[:, 1] = (
        robot.data.root_pos_w.torch[env_ids, 1]
        + radius * torch.sin(angle)
    )
    root_state[:, 2] = (
        env.scene.env_origins[env_ids, 2]
        + TARGET_HEIGHT
    )

    root_state[:, 7] = speed * torch.cos(velocity_angle)
    root_state[:, 8] = speed * torch.sin(velocity_angle)
    root_state[:, 9:13] = 0.0

    ball.write_root_pose_to_sim(
        root_state[:, :7],
        env_ids=env_ids,
    )
    ball.write_root_velocity_to_sim(
        root_state[:, 7:13],
        env_ids=env_ids,
    )


def move_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    update_period: float = 0.1,
    boundary: float = 3.0,
) -> None:
    """Move the kinematic ball and reflect its velocity at arena boundaries."""
    ball = env.scene[asset_cfg.name]
    poses = ball.data.root_pose_w.torch[env_ids].clone()
    velocities = ball.data.root_vel_w.torch[env_ids].clone()
    origins = env.scene.env_origins[env_ids]

    next_xy = poses[:, :2] + velocities[:, :2] * update_period
    relative_xy = next_xy - origins[:, :2]
    hit_boundary = torch.abs(relative_xy) > boundary
    velocities[:, :2] = torch.where(hit_boundary, -velocities[:, :2], velocities[:, :2])

    poses[:, :2] += velocities[:, :2] * update_period
    poses[:, 2] = origins[:, 2] + TARGET_HEIGHT
    velocities[:, 2:] = 0.0

    ball.write_root_pose_to_sim(poses, env_ids=env_ids)
    ball.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# Scene
@configclass
class Go2BallChaseSceneCfg(InteractiveSceneCfg):
    """Scene for the standalone ball-chasing task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.2,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.25, 0.02),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, TARGET_HEIGHT),
        ),
    )

    # Required by the rough-terrain walking policy, even though this task uses
    # a plane. The observation layout must remain identical to training.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


# MDP config
@configclass
class ActionsCfg:
    """High-level action -> frozen low-level walking policy."""

    pre_trained_policy: nav_mdp.PreTrainedPolicyActionCfg = (
        nav_mdp.PreTrainedPolicyActionCfg(
            asset_name="robot",
            policy_path=WALKING_POLICY_PATH,

            # Existing Go2 walking policy is evaluated at 50 Hz:
            # 200 Hz physics / 4 = 50 Hz.
            low_level_decimation=4,

            # These two configs are the only pieces reused from the
            # official Go2 rough environment.
            low_level_actions=LOW_LEVEL_ACTION_CFG,
            low_level_observations=LOW_LEVEL_OBSERVATION_CFG,

            debug_vis=False,
        )
    )


@configclass
class ObservationsCfg:
    """Observations for the upper tracking policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        ball_position_b = ObsTerm(
            func=ball_position_b,
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )

        last_action = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Rewards for the upper tracking task only."""

    distance_tracking = RewTerm(
        func=distance_tracking_reward,
        weight=2.0,
    )

    facing_ball = RewTerm(
        func=facing_ball_reward,
        weight=0.5,
    )

    relative_speed = RewTerm(
        func=relative_speed_penalty,
        weight=-0.2,
    )

    too_close = RewTerm(
        func=too_close_penalty,
        weight=-2.0,
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    termination = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )


@configclass
class EventsCfg:
    """Task reset events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball = EventTerm(
        func=reset_ball,
        mode="reset",
        params={
            "min_radius": 1.2,
            "max_radius": 2.5,
            "min_speed": 0.25,
            "max_speed": 0.50,
        },
    )

    move_ball = EventTerm(
        func=move_ball,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "update_period": 0.1,
            "boundary": 3.0,
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.15,
        },
    )


# Environment
@configclass
class Go2BallChaseEnvCfg(ManagerBasedRLEnvCfg):
    """Independent hierarchical Go2 ball-chasing environment."""

    scene: Go2BallChaseSceneCfg = Go2BallChaseSceneCfg(
        num_envs=256,
        env_spacing=6.0,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # The upper action itself is the velocity command.
    # Therefore this task does not need CommandManager.
    commands = None
    curriculum = None

    def __post_init__(self):
        super().__post_init__()

        # Physics: 200 Hz.
        self.sim.dt = 0.005

        # Upper tracking policy: 10 Hz.
        # One upper action is held for 20 physics steps.
        self.decimation = 20

        # PreTrainedPolicyActionCfg.low_level_decimation=4 means that the
        # frozen walking policy executes at 50 Hz inside the same interval.
        self.episode_length_s = 20.0

        self.sim.render_interval = self.decimation
        self.scene.height_scanner.update_period = (
            self.actions.pre_trained_policy.low_level_decimation * self.sim.dt
        )
