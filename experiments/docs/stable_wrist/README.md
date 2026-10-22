# 实验：稳定姿态 + 右手腕位置，按 reward_inference 加权合成 z

- 日期：2026-10-21
- 脚本：`experiments/stable_wrist.py`，查看器 `experiments/stable_wrist_view.py`（单测 `tests/test_stable_wrist.py`）
- 模型：`new_model_for_training_code_inference`（z_dim 256，z 在半径 16 的球面上）
- 数据：训练 replay buffer `checkpoint/buffers/train_reduced`（100 万帧）
- 原始结果：本目录 `metrics.json`、`per_rollout.csv`、`stratify_all.png`；`results.npz`、`stable_db.pkl`、`wrist_db.pkl`
  已 gitignore，只在本地

## 1. 问题

把"一个能稳定保持的姿态"和"一个右手腕在某处的帧"当作两帧 reward，按 `FBModel.reward_inference` 合成一个 z：
z = project(1·B(stable) + w·B(wrist))。策略从 default pose 出发，能否既摆出 stable 姿态，又把右手送到 wrist 帧的位置？
w 取 1 和 0.5。

## 2. 设置

- **stable 帧（100）**：buffer 中随机取 2 万帧，查静态稳定性（pose_hold score，见 `doc/motion_query.md`），保留 ≥ 85 的帧，
  再用 `spread` 按加权 qpos 距离（权重同 `multi_vocab.py`）挑 100 个两两不太相似的。稳定性中位数 88.0。
  **其中 41 个骨盆低于 0.4 m**（坐、跪、躺），稳定性筛选本身不限制直立。
- **wrist 帧（4000）**：新指标 `right_wrist_x/y/z` 表示右手腕（`right_wrist_yaw_link`）相对骨盆的位置，在 heading 坐标系下
  （只去掉 yaw）。在 5 cm 的三维网格上 `stratify`，范围取全 buffer 的 0.5%–99.5% 分位：x ∈ [-0.30, 0.45]，
  y ∈ [-0.50, 0.40]，z ∈ [-0.35, 0.50]。格内随机取帧，4125 个格子中 3911 个各取 1 帧。分布见 `stratify_all.png`。
- **去速度**：两组帧都用 `zero_velocity` 把 next_obs 中的速度清零，与 goal_inference 的做法相同。
- **配对与 z**：每个 wrist 帧随机配一个 stable 帧，每个 stable 帧平均用 40 次。
  z[w, i] = project_z(B_stable[pair[i]] + w·B_wrist[i])，共 4000 × 2 = 8000 个 z。
- **执行**：用 `pose_hold.isaac.IsaacBackend`，即不带动作任务的 Isaac Lab 场景，观测构造同 pose_hold。
  从 default pose 起步（根在原点，离地间隙 1 cm），确定性动作，跑 5 s（250 步）。2048 env 一批。
- **自检**：
  - 合成的 z 与直接调用 `reward_inference(两帧, reward=[1, w])` 的余弦 ≥ 0.99999994。
  - 查看器中 MuJoCo 摆出的仿真末帧手腕与 Isaac 记录的 `sim_wrist` 相差 ≤ 3 mm。

运行（两段分开跑，各自在内存看门狗下；Isaac 关闭时会结束整个进程，所以 `close` 放在保存之后）：

```bash
uv run python scripts/mem_guard.py --limit 90 --log mem_sel.log -- \
    python -m experiments.stable_wrist --model-folder new_model_for_training_code_inference --select-only
uv run python scripts/mem_guard.py --limit 90 --log mem_roll.log -- \
    python -m experiments.stable_wrist --model-folder new_model_for_training_code_inference --reuse-selection
uv run python -m experiments.stable_wrist_view          # ←/→ 翻页，↑/↓ 换权重，E 排序/原序，F 只看摔倒
```

耗时与内存（RTX 5070，32 GB 内存）：

| 阶段 | 耗时 | 系统内存峰值 |
|---|---|---|
| 选帧（全量扫描 174 s，补算稳定性 1.99 万个姿态） | 约 6 min | 36% |
| rollout（8000 次） | 71 s | 23% |

## 3. 指标定义

| 指标 | 含义 |
|---|---|
| `wrist_err` | 末帧右手腕（相对骨盆、heading 坐标系）与目标的距离（m）。目标取 wrist 帧 next_obs 中的手腕列，即 B 实际看到的位置 |
| `wrist_err_tail` | 最后 1 s 的 `wrist_err` 均值 |
| `baseline_err` | 所配 stable 帧自身的手腕与目标的距离。即"完全不管手"时的误差，作为参照 |
| `pose_err` | 末帧与 stable 帧的关节 RMSE（rad），不含右臂 7 个关节 |
| `height_err` | 末帧骨盆高度减去 stable 帧骨盆高度（m） |
| `fell` | `height_err` < -0.25 m。不用绝对高度，因为 stable 帧本身可能是躺姿 |

## 4. 结果

| | w = 1 | w = 0.5 |
|---|---|---|
| wrist_err p10 / p50 / p90 (m) | 0.100 / **0.294** / 0.625 | 0.211 / **0.442** / 0.728 |
| < 5 cm / < 10 cm | 2.2% / 10.1% | 0.3% / 1.3% |
| baseline_err p50 (m) | 0.477 | 0.477 |
| 比 stable 帧自身更接近目标的比例 | 85.4% | 71.7% |
| baseline − wrist_err 的中位数 (m) | 0.111 | 0.024 |
| pose_err p50 (rad) | 0.315 | 0.160 |
| 摔倒率 | 7.1% | 0.1% |

`wrist_err_tail` 与末帧几乎相同，5 s 末已稳定，不是还没走到。

**按目标位置（w = 1，各轴按分位数分 5 档，表中为各档误差中位数）**：

| 轴 | 档 1 | 档 2 | 档 3 | 档 4 | 档 5 |
|---|---|---|---|---|---|
| x（前后，从 -0.61 起） | 0.40 | 0.34 | 0.26 | 0.23 | 0.26 |
| y（左右，从 -0.61 起，右为负） | 0.29 | 0.22 | 0.24 | 0.32 | 0.46 |
| z（上下，从 -0.46 起） | 0.35 | 0.28 | 0.29 | 0.28 | 0.28 |

按 stable 帧分组：配直立 stable 帧（骨盆 > 0.6 m）与配低姿 stable 帧时，w = 1 的误差中位数几乎相同（0.295 vs 0.292 m）。

## 5. 推断

1. **一帧 wrist 帧的 B 只能把手往目标方向拉一部分。** w = 1 时误差中位数从 48 cm（不管手）降到 29 cm，85% 的配对有改善，
   但只有 10% 进入 10 cm 以内。z 是两个 B 相加后再投影的结果，策略得到的是"两者之间"的某个行为，不是两个约束同时满足。
2. **w 是在姿态与手之间的权衡。** w 从 0.5 升到 1：
   - 手更准：误差中位数 44 → 29 cm；
   - 姿态更偏离 stable 帧：pose_err 0.16 → 0.32 rad；
   - 摔倒率 0.1% → 7.1%。

   w = 0.5 时手几乎不动，中位改善仅 2.4 cm，z 基本由 stable 帧主导。
3. **身后和身体左侧的目标最难。** x < -0.14 m（手在骨盆后方）或 y > 0.19 m（右手越过身体中线到左侧）时，
   误差中位数为 40–46 cm；身前、身体右侧约 22–26 cm。高度方向差别不大。这与 buffer 中这些位置帧少
   （见 `stratify_all.png` 的候选数曲线）一致，但本实验不能区分"数据少"和"与 stable 姿态冲突"两种原因。

## 6. 常见误读

- **"next-qpos 和 next_obs 是同一帧"**：在本 buffer 中两者并不完全一致。由 next-qpos 用 MuJoCo FK 算的手腕位置，
  与 privileged_state 中的手腕列相比：中位差 1.7 cm，p95 9 cm，最大 93 cm（躺姿居多）。
  因此 stratify 用的是前者（近似均匀），误差目标用的是后者（与 B 一致）。查看器里 wrist 机器人的姿态来自 next-qpos，
  可能与绿色目标球有几厘米的偏差。
- **把 `fell` 理解为摔倒到地上**：它的含义是"比 stable 姿态低 25 cm 以上"。stable 帧本身有 41% 是低姿。
- **躺姿下的 heading 坐标系**：骨盆接近竖直时 yaw 定义不稳定，此时手腕的 x/y 分量会随小角度变化而跳动。
  看低姿配对的个例时要留意这一点。
