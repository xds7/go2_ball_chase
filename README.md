# Go2 Ball Chase

Isaac Lab 上で Unitree Go2 に動くボールを追わせる強化学習タスクです。上位の PPO が速度指令を学習し、事前学習済みの歩行ポリシーが関節を制御します。ボールは衝突しない運動学的な目標です。

## 必要なもの

- Isaac Lab と、その環境に対応する Isaac Sim・RSL-RL
- Go2 の rough terrain 用 TorchScript 歩行ポリシー（`policy.pt`）

歩行ポリシーと学習済みの追跡モデルは、このリポジトリには含まれません。

## 学習

Isaac Lab のルートディレクトリで実行します。`/path/to/go2_ball_chase` と `policy.pt` のパスを実際の場所に置き換えてください。

```bash
./isaaclab.sh -p /path/to/go2_ball_chase/train.py \
  --mode train --headless --walking_policy /path/to/policy.pt \
  --num_envs 256 --max_iterations 1000
```

チェックポイントは `logs/rsl_rl/go2_ball_chase/` 以下に保存されます。

## 再生

```bash
./isaaclab.sh -p /path/to/go2_ball_chase/train.py \
  --mode play --num_envs 1 --walking_policy /path/to/policy.pt \
  --checkpoint /path/to/model_1000.pt
```

学習を再開する場合は、学習コマンドに `--resume --checkpoint /path/to/model_1000.pt` を追加します。

## 構成

- `train.py`: 学習・再生の起動
- `go2_ball_task/ball_task.py`: 環境、観測、報酬
- `go2_ball_task/go2_ball_ppo.py`: 上位 PPO の設定
