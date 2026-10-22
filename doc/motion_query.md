# motion_query

motion_query（`humanoidverse/query/motion_query.py`）是一个惰性的链式查询库。它从 replay buffer 或 LAFAN 动作中取帧，算出每帧的指标，再用 DuckDB SQL 筛选、加列、排序、采样，最后输出一张表或一个 `MotionDatabase`。

```python
db = (
    Source(model_folder).scan()                # 惰性：collect / to_database 之前什么都不跑
    .where("base_height >= 0.6 AND base_speed BETWEEN 0.5 AND 1.0")
    .with_columns(cmd_speed="greatest(abs(base_vx), abs(base_vy))")
    .case("bucket", {"slow": "cmd_speed < 0.5", "fast": "cmd_speed >= 0.5"})
    .order_by("base_speed DESC")
    .to_database()
)
db.save(model_folder / "motion_query" / "motion_db.pkl")
```

它只是一个库，具体的查询写成 `scripts/queries/` 下的脚本。结果 pickle 可以交给 `motion_viewer` 查看，也可以交给 `motion_latent` 推断 z。

## 数据源

`Source` 决定帧从哪里来、怎么扫描，它的字段就是查询脚本的命令行参数（由 tyro 生成）。默认读 replay buffer；设置了 `lafan` 就改读 LAFAN 动作。

| 数据源 | 类 | 帧的来源 | `frame` 的含义 |
| --- | --- | --- | --- |
| replay buffer | `H5Frames` | `checkpoint/buffers/train_reduced`（没有则用 `train`）下的 `buffer.hdf5`，按片读取，不整个载入 | buffer 中的下标；轨迹 buffer (T, E) 展平为 t * E + e |
| LAFAN | `LafanFrames` | `lafan_dir` 下名字匹配 `lafan` 通配符的 .pkl，用 Isaac 环境的 motion library 按环境 dt 重采样，同时算出 backward 观测 | 拼接后的序号；另有 `motion`、`motion_time` |

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model_folder` | （必填） | 模型目录，包含 checkpoint 与 buffer |
| `buffer` | 自动 | 指定 buffer 目录 |
| `lafan` | None | 逗号分隔的通配符，如 `"walk*,run*"` |
| `lafan_dir` | `humanoidverse/data/lafan_29dof` | LAFAN pkl 目录 |
| `chunk` | 100000 | 每次读取的帧数 |
| `max_frames` | None | 只扫描前 N 帧 |
| `stride` | 1 | 每隔 stride 帧取一帧 |
| `max_workers` | CPU 核数 | 计算指标的进程数 |
| `device` | cuda | LAFAN 环境所用设备 |

注意：LAFAN 帧是按环境 dt 重采样的，和 pose_hold 按原始 fps 取的帧并不一一对应。

## 表结构

每帧是表 `frames` 的一行，列分三类：`frame`（帧在数据源中的下标）、`METRICS` 中的全部指标，以及 `qpos_0` .. `qpos_35`。指标在扫描时由 MuJoCo 从 qpos/qvel 算出。条件判断用 float64，保留的行存为 float32。

| 类别 | 列 | 单位 / 约定 |
| --- | --- | --- |
| 骨盆位置与速度 | `base_height`, `base_speed`, `base_vx`, `base_vy`, `base_vz`, `move_angle`, `com_speed`, `yaw_rate` | m、m/s（朝向坐标系）、deg、rad/s |
| 姿态 | `base_roll`, `base_pitch`, `torso_roll`, `torso_pitch`, `base_tilt`, `torso_tilt` | deg；tilt 为偏离竖直方向的无符号角，躺下时约 90 |
| 手臂关节 | `left/right_elbow_flexion`, `left/right_arm_abduction` | deg；flexion 为 0 表示伸直，abduction 外展为正 |
| 部位高度 | `torso_height`, `left/right_hand_height`, `left/right_knee_height`, `left/right_foot_height` | m（世界 z） |
| 接触与相对高度 | `foot_clearance`, `left/right_wrist_above_shoulder` | m；foot_clearance 是较低一侧脚踝离脚掌平放时高度的距离 |
| 右手腕相对骨盆 | `right_wrist_x`, `right_wrist_y`, `right_wrist_z` | m；`right_wrist_yaw_link` 相对骨盆，heading 坐标系（只去 yaw，x 前 y 左 z 上），同 privileged_state 的 local_body_pos |
| 姿态原始值 | `qpos_0` .. `qpos_35` | 根位置 0-2、根四元数 wxyz 3-6、关节角 7-35 |
| 静态稳定性 | `fidelity`, `stability` 等（仅在调用 `with_stability` / `stable` 之后才有） | 0-100 分 |

要新增指标，在 `envs/g1_env_helper/queries/metrics.py` 的 `METRICS` 中注册一个 `Metric` 实例。

## 查询操作

`Query` 的每个方法都返回一个新的 `Query`，调用 `collect()` 或 `to_database()` 时才真正执行。“逐行”指该操作只看单行，可以在扫描时分块执行（见「执行模型」）。

| 方法 | 作用 | 逐行 | 示例 |
| --- | --- | --- | --- |
| `where(cond, **params)` | 保留满足 SQL 条件的行，多个 where 之间是 AND；`$name` 绑定参数 | 是 | `where("base_speed >= $v", v=0.5)` |
| `with_columns(**cols)` | 新增或替换列：字符串为 SQL 表达式，函数为 Python 列函数 | SQL 是，函数否 | `with_columns(cmd_speed="greatest(abs(base_vx), abs(base_vy))")` |
| `case(name, {label: cond}, default)` | 新增标签列，取第一个匹配的条件 | 是 | `case("grp", {"stand": "base_speed < 0.05"}, default="move")` |
| `order_by(*keys)` | 排序 | 否 | `order_by("base_speed DESC", "frame")` |
| `limit(n)` | 取前 n 行 | 否 | `limit(1000)` |
| `sql(stmt, per_chunk, **bindings)` | 一条完整的 DuckDB 语句（也可以传 .sql 文件的 Path），支持 CTE、窗口函数、JOIN；DataFrame 参数绑定为表，其他参数绑定为 `$name`；结果必须保留 `frame` 和 `qpos_*` | 仅 `per_chunk=True` 时 | `sql("SELECT f.* FROM frames f JOIN keep USING (frame)", keep=df)` |
| `with_stability(columns, cache)` | 新增静态稳定性列（pose_hold score） | 否 | `with_stability({"stability": "score", "base": "base_score"})` |
| `stable(min_score, cache)` | 等于 `with_stability` + `where("stability >= min_score")` | 否 | `stable(90)` |
| `apply(fn)` | 任意 DataFrame → DataFrame 的步骤 | 否 | `apply(lambda df: df.drop_duplicates("frame"))` |
| `sample(n)` | 进入采样链，见「采样」 | 否 | `sample(4000).partition(...)` |

查询有两种结束方式：

- `collect()` 返回 pandas 表。已收集的表可以用 `Query(df)` 继续查询。
- `to_database()` 返回 `MotionDatabase`，并从数据源取回各帧的 qpos、qvel、action、next_obs。数值列变为 metrics；文本列（如 bucket）存为编码，对应的取值放在 `labels`。

## 静态稳定性

BFM 策略从静止状态保持一个姿态 5 秒，得到两个相互独立的 pose_hold 分数（0-100）：

- **复现分数** `fidelity_score`：和原姿态比，基座朝向、关节、终点位移差多少。它和 `data/pose_hold/lafan_29dof/ranking.csv` 中的 `score` 是同一个量。
- **基座稳定分数** `stability_score`：不看原姿态，只看整个仿真过程中骨盆高度、XY 位置、yaw、倾斜（重力方向）相对各自中位数变化了多少。数值基本不变就是稳定。一个姿态如果被稳稳保持在另一个形状上，它稳定但复现差；摔倒则两者都差。

实现在 `humanoidverse/pose_hold/`（`scores.py` 计算分数，`cache.py` 管缓存）。

```python
q.stable(90)                                         # 只保留 stability_score >= 90 的行
q.stable(90, min_fidelity=85)                        # 同时要求 fidelity_score >= 85
q.with_stability()                                   # 加 fidelity 和 stability 两列
q.with_stability({"stability": "stability_score", "fall": "height_score", "dist": "distance_score"})
q.stable(90, cache=StabilityCache(num_envs=1024))    # 换模型 / batch 大小 / 评估参数
```

可取的缓存列：
- 复现：`score`（= `fidelity_score`）, `base_score`, `joint_score`, `pose_score`, `distance_score`, `cost`, `valid`, `base_error_mean_deg`, `base_error_max_deg`, `joint_rmse_mean`, `joint_error_max`, `end_displacement`, `max_displacement`, `tail_base_speed`, `height_shift`
- 稳定：`stability_score`, `height_score`, `drift_score`, `yaw_score`, `tilt_score`, `height_dev_max`, `drift_dev_max`, `yaw_dev_max_deg`, `tilt_dev_max_deg`
- 实际保持住的姿态（整个仿真过程的中位数）：`hold_height`（骨盆高度，m），`torso_tilt_deg`（torso_link 的 z 轴和竖直向上的夹角，度）

**缓存。** 分数存在 `.cache/pose_hold.sqlite`。第一次使用时会自动导入 pose_hold 的 ranking.csv（约 26.5 万个姿态，耗时约 9 秒）；ranking.csv 更新后会重新导入。

- 缓存键是去掉朝向的根四元数加 29 个关节角，按 1e-4 rad 量化。根位置和朝向不影响结果，因为 pose_hold 会把根放回原点，并按离地间隙重设高度。
- 不同评估器的分数分开存放，用模型文件的 sha256、duration、clearance 和 ScoreConfig 来区分。
- 评估器身份不包含 pose_hold 的代码和物理文件，所以改了打分代码后要手动删掉缓存文件。

**补算。** 到达这一步的行里，没有缓存的姿态会先去重，再交给一个 Isaac Sim 子进程，按 `num_envs`（默认 3072）批量仿真。每批跑完就写入缓存，中断后已完成的进度不会丢。启动开销约 25 秒，之后的耗时取决于批数。用子进程的原因是：LAFAN 数据源已经在查询进程里启动了一个 Isaac app，而一个进程只能有一个。

```bash
uv run python -m humanoidverse.pose_hold.cache import [--ranking PATH] [--force]
uv run python -m humanoidverse.pose_hold.cache stats
```

## 采样

`sample(n)` 把查询接到 `motion_sampling` 的采样链上，从到达这一步的行里挑出 n 行。各阶段由外到内嵌套：外层分配名额，内层在自己的名额内挑行。如果链不以 `spread` 结尾，最后一级取表中靠前的若干行。

| 阶段 | 作用 | 关键参数 |
| --- | --- | --- |
| `partition(by, share, spacing_floor)` | 把 n 按某一列的取值分配（按 share 比例或均分） | `spacing_floor`：某组太密、在间距下限下凑不够名额时少取，剩余名额按比例分给其他组 |
| `stratify(cells, by, plot)` | 在若干指标构成的网格上均匀取样，候选不足的格子全部取走 | `grid(base_vx=0.15, range=(-1.05, 1.05))`；`by` 按外层 partition 的取值分别设网格；`plot` 保存分布图 |
| `prefer(by, keep)` | 只保留某一列最低的 keep 比例，但不少于名额 | 常与 `SymmetryError()` 列搭配 |
| `spread(weights, min_dist)` | 按加权 qpos 距离保持两两间隔（相似度稀疏化），必须是最后一级 | `weights` 的格式同 `bench/motion_similarity.py` |

```python
query.sample(4000)
    .partition("grp", {"stand": 0.05, "move": 0.6, "posture": 0.35}, spacing_floor=0.5)
    .stratify({"move": grid(base_vx=0.15, base_vy=0.15, range=(-1.05, 1.05))}, by="grp")
    .prefer("symmetry_error", keep=0.5)
    .spread(SIMILARITY)
    .to_database()
```

采样链也可以脱离查询单独使用：`sample().partition(...)(df, n)` 返回被选中行的下标。

## 执行模型

链头连续的逐行操作在扫描时逐块执行。从第一个非逐行操作开始，后面的步骤都在收集后的整张表上执行。

扫描阶段占用的内存只取决于命中的行数。静态稳定性只对到达它的行查缓存，未命中的姿态才交给 Isaac 子进程。

## 优化建议

最重要的一条：把便宜的逐行条件连续地写在链的最前面，把昂贵的步骤（Python 列函数、静态稳定性、采样）放在后面。

1. **不要打断前导逐行操作。** 只有链头连续的 `where` / `case` / SQL `with_columns` / `sql(per_chunk=True)` 会在扫描时分块执行。一旦出现 `order_by`、Python 列函数或 `with_stability`，后面的 where 就要在整张已收集的表上跑，内存里得先放下所有行（每行带 36 列 qpos）。
2. **先用 `max_frames` / `stride` 调条件。** 扫描的主要开销是逐帧用 MuJoCo 算指标。条件调好之前，用 `--max-frames 200000` 或 `--stride 10` 快速看命中数，最后再全量跑。
3. **脚本开头保留 `OMP_NUM_THREADS=1`。** 指标由 `max_workers` 个进程并行计算，每个进程再开多线程会互相抢占 CPU。
4. **静态稳定性放在筛选之后、采样之前。** 按 pose_hold 的实测速度，未缓存的姿态约 170 个/秒（3072 并行，26.5 万帧用时 1538 秒），一百万个约需 1.6 小时。放在 `sample` 之后，最终行数会少于 n。运行时会先打印 `N poses, M distinct, K not cached`；K 太大时可以中断、加上条件再跑，已算完的批次不会丢。
5. **缓存命中率取决于数据源。** 导入的分数来自 LAFAN 原始 fps 的帧。`--lafan` 数据源按环境 dt 重采样，大部分帧是插值出来的姿态；replay buffer 的帧首次查询时也都不在缓存里。两者第一次都需要补算，之后重复查询会全部命中。
6. **分摊启动开销。** 每次补算都要启动一次 Isaac Sim（约 25 秒）。一条链里只调用一次 `with_stability`，需要多个分项时用 `columns` 一次取齐。显存不够时用 `StabilityCache(num_envs=1024)` 降低并行数。
7. **Python 列函数放在后面。** `SymmetryError()` 这类列函数要对到达的每一行都算一遍，先用 SQL 把行数降下来。
8. **先 `prefer` 再 `spread`。** `spread` 要在候选池上反复二分距离阈值，池子越小越快；`prefer` 和 `stratify` 会先把池子切小。
9. **调采样参数时不要重新扫描。** 把筛选结果 `collect()` 后存成 parquet，再用 `Query(df).sample(...)` 反复调参。但表里不带帧数据，最终输出 MotionDatabase 时仍要从 Source 再走一遍，或者用 `sql(..., keep=df)` JOIN 回去。
10. **参数用 `$name` 绑定，不要用 f-string 拼 SQL。** 例如 `where("base_speed >= $v", v=speed)`，可以避免引号和精度问题。
11. **大查询放在内存看门狗下跑。** 扫描进程池与采样进程池（`spread` / `stratify`）的每个 worker 约占 0.8 GB，默认 worker 数等于 CPU 核数；`MOTION_WORKERS` 可以控制采样进程池的大小。用 `uv run python scripts/mem_guard.py --limit 90 -- python ...` 运行，系统内存超过 90% 时会结束整个进程组。

## 现有查询脚本与示例

现有脚本都在 `scripts/queries/`，运行方式为 `uv run python scripts/queries/<name>.py --model-folder /path/to/model [--lafan "walk*"] [--max-frames N]`。

| 脚本 | 查询内容 | 用到的操作 |
| --- | --- | --- |
| `crouch_still.py` | 低姿、几乎静止、躯干不过度前倾 | where |
| `walk_forward.py` | 直立、中速前进，按速度从快到慢排序 | where、order_by |
| `walk_backward.py` | 后退（move_angle 在 ±180 度的 30 度以内） | where |
| `multi_vocab.py` | holosoma bfm_walk_token 的 token 词表：直立行走帧加低姿静态帧 | where、case、SymmetryError、partition / stratify / prefer / spread |
| `walk_sequence_vocab.py` | holosoma bfm_walk_sequence 的变长 token（来自 motion_sequence 数据库） | 单独使用 `sample()` 采样链 |

示例：选出能稳定保持的低姿，按骨盆高度和躯干倾角均匀取 500 帧。

```python
db = (
    source.scan()
    .where("base_height < 0.55 AND base_speed <= 0.1 AND foot_clearance <= 0.05")  # 扫描时分块执行
    .stable(85)                                   # 只对剩下的行查缓存 / 补算
    .sample(500)
    .stratify({"base_height": Cell(0.05, 0.1, 0.55), "torso_tilt": Cell(15, 0, 120)})
    .spread(SIMILARITY)
    .to_database()
)
```
