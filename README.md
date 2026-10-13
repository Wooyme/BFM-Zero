<h1 align="center"> BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning </h1>

<div align="center">

[[arXiv]](https://arxiv.org/abs/2511.04131)
[[Paper]](https://lecar-lab.github.io/BFM-Zero/resources/paper.pdf)
[[Website]](https://lecar-lab.github.io/BFM-Zero/)

<!-- [[Arxiv]](https://lecar-lab.github.io/SoFTA/) -->
<!-- [[Video]](https://www.youtube.com/) -->

<img src="static/images/ip.png" style="height:50px;" />
<img src="static/images/meta.png" style="height:50px;" />
</div>

## Code

Code will be released in stages:

- [x] **Pretrained checkpoints + sim-to-sim / sim-to-real deployment**  
  → [`deploy`](https://github.com/LeCAR-Lab/BFM-Zero/tree/deploy) branch

- [x] **Minimal inference code + tutorial**  
  → [`minimal_inference`](https://github.com/LeCAR-Lab/BFM-Zero/tree/minimal_inference) branch

- [x] **Full training and evaluation pipelines**

- [x] **A distributed training in MJLab (6 hours training) + Teleoperation**      
  → [`UFO`](https://github.com/Roboparty/UFO) implementation

- [ ] **Minimal training code (RTX 4090 support)**

# BFM-Zero Training

Humanoidverse training for BFM-Zero with Isaac Sim or MuJoCo.

## Requirements

- Python 3.10
- CUDA-capable GPU
- Isaac Sim (Linux) or MuJoCo for simulation

## Installation

### 1. Clone and fetch large files (Git LFS)

This repo uses [Git LFS](https://git-lfs.github.com/) for motion data and model. After cloning, install LFS and pull the large files:

```bash
git clone https://github.com/LeCAR-Lab/BFM-Zero.git
cd BFM-Zero
git lfs install
git lfs pull
```
Note: If the repository exceeded its LFS budget,  you can access the data here: https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/main/data

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip: `pip install uv`

### 3. Install dependencies

From this directory (BFM-Zero):

```bash
uv sync
```

## Data

- **Motion data**: Included via Git LFS in `humanoidverse/data/` after `git lfs pull`. `lafan_29dof.pkl` is for evaluation; `lafan_29dof_10s-clipped.pkl` is for training.
- If you are unsure about the data format, please check the discussion in
  [Issue #12](https://github.com/LeCAR-Lab/BFM-Zero/issues/12).

### Convert GMR G1 motions

Convert the trusted GMR `.pkl` files in a directory into a separate BFM-Zero dataset:

```bash
uv run python scripts/convert_gmr_to_bfm.py ../GMR/bvh/gmr_g1 \
    -o humanoidverse/data/gmr_g1_10s-clipped.pkl
```

The converter runs on CPU using NumPy, SciPy, Joblib, and MuJoCo; it does not start Isaac Sim or import GMR.
It maps the 29 joints by name from the source MJCF to the BFM G1 model, resamples to 30 FPS
(linear position/joint interpolation and quaternion SLERP), and writes non-overlapping 300-frame clips.
Trailing frames and motions shorter than 300 resampled frames are discarded and reported.
Each entry is named `<source_stem>_clip<N>` and contains the same seven fields as
`lafan_29dof_10s-clipped.pkl`, with float32 arrays and integer `fps=30`.
**`smpl_joints` is an all-zero `[300, 24, 3]` placeholder**, since GMR robot motions do not contain
the original human SMPL joints; the current BFM-Zero robot motion loader does not use this field.

By default, `--height-mode sequence` shifts each complete resampled motion vertically so its
lowest robot collision geometry point reaches `--ground-offset` (default: 0 meters).
This uses the target G1 meshes, spheres, and cylinders over all frames, before splitting clips;
all clips of one motion share the same offset. XY, joint motion, and relative height changes are preserved.
Use `--height-mode none` to preserve the original root trajectory. This is format conversion and
uniform height alignment, not IK, contact correction, or terrain reconstruction.
GMR and BFM G1 models have some different fixed body offsets, so Cartesian body trajectories may differ.

Defaults for `--source-mjcf` and `--target-mjcf` point to the sibling GMR `g1_mocap_29dof.xml`
and this repository's `g1_29dof.xml`, respectively; override them if the models live elsewhere.
The source MJCF must match the model used to export `dof_pos`; the converter expects GMR's XYZW quaternion convention.
Existing output files require `--overwrite`. Output must be outside the source directory.
The result is validated and reloaded before publication, and existing LAFAN files are not modified.

For training, set `TrainConfig.env.lafan_tail_path` (the `HumanoidVerseIsaacConfig` field in
`humanoidverse/train.py`) to `humanoidverse/data/gmr_g1_10s-clipped.pkl`.
For tracking inference, pass this path as `--data_path`.

### Render motion data on G1

Render a clipped BFM-Zero dataset directly to MP4 without a trained policy or Isaac Sim:

```bash
# Show motion names and zero-based indices.
uv run python scripts/render_bfm_motion.py humanoidverse/data/gmr_g1_10s-clipped.pkl --list

# Export one clip (defaults to index 0 if no selector is provided).
uv run python scripts/render_bfm_motion.py humanoidverse/data/gmr_g1_10s-clipped.pkl \
    --motion FDCGB_0_clip0 -o videos/FDCGB_0_clip0.mp4

# Export every clip as a separate video.
uv run python scripts/render_bfm_motion.py humanoidverse/data/gmr_g1_10s-clipped.pkl \
    --all --output-dir videos/gmr_g1
```

The script uses the BFM G1 model and ground plane, preserves the stored root trajectory and joint angles,
and converts XYZW quaternions to MuJoCo's WXYZ convention. **This is kinematic reference playback,
not a policy rollout or a test of physical feasibility.** Both LAFAN and converted GMR libraries are supported;
`smpl_joints` is not used. Video FPS comes from the motion, so 300 frames at 30 FPS produces a 10-second video.

Default resolution is 1280×720, with a camera following the pelvis. Options include `--index 0`,
`--camera fixed`, `--distance 3`, `--azimuth 135`, `--elevation -20`, `--width 640 --height 360`,
and `--max-frames 30` for a short preview. Without an output argument, videos go to
`videos/<input_stem>/<motion_name>.mp4`. Existing videos require `--overwrite`.
Rendering uses MuJoCo, MediaPy, and FFmpeg. The default OpenGL backend is `egl` for headless GPU rendering
(or the existing `MUJOCO_GL` environment variable); `--backend osmesa` supports software rendering when
OSMesa is installed, and `--backend glfw` uses a display. `--list` does not initialize OpenGL.


## Training

### Launch

```bash
uv run python -m humanoidverse.train
```
Note: change `buffer_device` to "cuda:0" if you have larger vRAM.

### Main parameters

Training is driven by `humanoidverse.train` (see `train_bfm_zero()` in `train.py`). Key options:

| Area | Parameters |
|------|------------|
| **Scale** | `num_env_steps`, `online_parallel_envs`, `buffer_size`, `checkpoint_every_steps` |
| **Paths** | `work_dir`, env’s `lafan_tail_path` (expert motion data) |
| **Run** | `seed`, `use_wandb`, `wandb_pname` / `wandb_gname` / `wandb_ename` |
| **Policy / optim** | `update_agent_every`, `num_agent_updates`, `num_seed_steps`; agent config (e.g. `batch_size`, `lr_actor`, `lr_critic`, `discount`) |
| **Robot / env** | Overridden via `hydra_overrides` (e.g. `robot=...`, `robot.control.action_scale=...`, `env.config.lie_down_init=...`) |

Override from code by passing a custom `TrainConfig`, or extend the CLI to accept Hydra/tyro overrides.

Tips: After 50-100 M steps training, eval/emd should lower than 0.75.


<div align="center">
<img src="static/images/training_curve.png" style="height:300px;" />
</div>

---

## Inference

After training, three scripts handle inference and export:

| Script | Purpose |
|--------|---------|
| **`humanoidverse.tracking_inference`** | Motion tracking → extract latent \(z\), export ONNX |
| **`humanoidverse.goal_inference`** | Goal-reaching → compute \(z\) for different goals |
| **`humanoidverse.reward_inference`** | Reward-based tasks → compute \(z\) and evaluate performance |

### Example outputs (videos)

Videos from `BFM-Zero/model` after running each inference script with `--save_mp4`:

**1. Tracking inference** — expert (left) vs policy (right):

<img src="https://github.com/LeCAR-Lab/BFM-Zero/blob/main/model/tracking_inference/tracking.gif" controls width="400">


**2. Goal inference** — goal-reaching rollout:

<img src="https://github.com/LeCAR-Lab/BFM-Zero/blob/main/model/goal_inference/goal.gif" controls width="400">

**3. Reward inference** — example task (e.g. move-ego):

<img src="https://github.com/LeCAR-Lab/BFM-Zero/blob/main/model/reward_inference/move-ego-low0.6-0-0.7.gif" controls width="400"></img>

---

All scripts use **tyro** for the CLI. General usage:

```bash
uv run python -m humanoidverse.tracking_inference --help
uv run python -m humanoidverse.goal_inference --help
uv run python -m humanoidverse.reward_inference --help
```

**Common arguments:**

- `--model_folder`: Path to the trained model directory (must contain `checkpoint/` and `config.json`).
- `--data_path` (optional): Override the default LaFan data path.
- `--simulator`: `isaacsim` (default) or `mujoco`. **Use `--simulator mujoco` to run without Isaac Lab** (MuJoCo only; output is directly usable for sim2sim visualization).
- `--headless` (default: `True`): Run without GUI; use `--no-headless` to show the viewer.
- `--save_mp4`: Save rendered videos.

**Output:** All inference scripts export the policy to ONNX (`{model_name}.onnx`) in their respective output subdirectories under `exported/`.

---

### Tracking inference

Runs motion tracking, exports ONNX, and optionally saves a comparison video (expert vs policy).

```bash
uv run python -m humanoidverse.tracking_inference \
    --model_folder /path/to/model \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --no-headless \
    --save_mp4
```

- `--model_folder` should point to the **outer** model directory (the one that contains `checkpoint/`).
- You can set motion IDs and visualization length inside the script if exposed.

**Outputs** (under `model_folder/tracking_inference/`):

- `zs_{MOTION_ID}.pkl`: Latent \(z\) for each motion.
- `tracking.mp4`: Expert vs policy comparison (when `--save_mp4` is set).

---

### Goal inference

Computes \(z\) for predefined goals and optionally renders goal-reaching videos.

```bash
uv run python -m humanoidverse.goal_inference \
    --model_folder /path/to/model \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --save_mp4
```

- Iterates over predefined goals and computes the corresponding \(z\).
- Requires `goal_frames_lafan29dof.json` (the script searches for it in several locations).

**Outputs** (under `model_folder/goal_inference/`):

- `goal_reaching.pkl`: Dictionary `{goal_name -> z}`.
- `videos/*.mp4`: Per-goal videos (if `--save_mp4 True`).

---

### Reward inference

Runs reward-based task inference: computes \(z\) and optionally runs rollouts for evaluation.

```bash
uv run python -m humanoidverse.reward_inference \
    --model_folder /path/to/model \
    --save_mp4 
```

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--num_samples` | Number of samples in the buffer per inference run. |
| `--n_inferences` | Number of inference latents per reward task. |
| `--episode_length` | Steps per rollout. |
| `--skip_rollouts` | If `True`, only compute \(z\); do not run visualization rollouts. |

**Outputs** (under `model_folder/reward_inference/`):

- `reward_locomotion.pkl`: Dictionary `{task_name -> z}`.
- `videos/*.mp4`: Per-task videos (when `--save_mp4` is set).

---

### Motion query

Each query is a script in `scripts/queries/` that selects frames of the replay buffer (or of LAFAN motions with `--lafan "walk*,run*"`). It does this with one chain of SQL-like operations over per-frame motion metrics (`METRICS` in `humanoidverse/envs/g1_env_helper/queries/metrics.py`). The buffer is streamed from its hdf5; no model and no simulator are needed.

```bash
uv run python scripts/queries/walk_forward.py --model-folder /path/to/model [--max-frames N] [--stride K]
```

```python
db = (
    source.scan()                                         # lazy until collect() / to_database()
    .where("base_height >= 0.6")
    .where("base_speed BETWEEN 0.5 AND 1.0 AND move_angle BETWEEN -30 AND 30")
    .with_columns(cmd_speed="greatest(abs(base_vx), abs(base_vy))")    # SQL expression or Python column function
    .case("tier", {"slow": "cmd_speed < 0.7"}, default="fast")         # label of the first matching condition
    .order_by("base_speed DESC")
    .to_database()
)
```

When the SQL gets complex, write it as one raw statement over the table `frames`, inline or from a `.sql` file. Other keyword arguments bind `$name` parameters, and DataFrames bind extra tables. The raw statement can sit anywhere in the chain:

```python
db = (
    source.scan()
    .sql(Path("my_query.sql"), per_chunk=True, lim=1.05)   # per_chunk: row-local, so it runs while scanning
    .sample(4000).spread()
    .to_database()
)
```

Without `per_chunk`, a statement may use `ORDER BY`, `LIMIT`, aggregates, windows or joins, and runs once on all the rows that reach it.

Conditions and expressions are DuckDB SQL over the columns `frame` (index in the source), every metric, and `qpos_0 .. qpos_35`. The leading `where` / `case` / SQL `with_columns` steps run while scanning, chunk by chunk. Everything after the first other step (`order_by`, `limit`, `sql`, Python column functions, `sample`) runs on the rows that pass them. `humanoidverse/query/motion_query.py` (`Source`, `Query`) and `humanoidverse/query/motion_sampling.py` are libraries; copy a script to write a new query. To add a metric, register a `Metric` object (a frozen dataclass with `__call__(model, data) -> float`) in `METRICS`.

**Outputs**: the script decides; the simple ones write `model_folder/motion_query/motion_db.pkl` (or `--output`). This is a `MotionDatabase` loadable with `MotionDatabase.load` and searchable with `sort` / `top_k` / `filter(min_<metric>=..., max_<metric>=...)` / `where` (SQL condition or callable) / `nearest`. Text columns (e.g. `bucket`) are stored as codes, with their values in `db.labels`.

### Motion viewer

Browses a `MotionDatabase` interactively: queries at a prompt, frames shown in the MuJoCo viewer with the selected metrics overlaid. Needs a display.

```bash
uv run python -m humanoidverse.motion_viewer --db /path/to/model/motion_query/motion_db.pkl
```

```text
> where base_speed > 1 AND abs(move_angle) < 30    # SQL condition, same as motion_latent --where
> sort base_speed                                   # descending; "sort base_speed asc" for ascending
> head 10                                           # table from the cursor; "head 10 all" for every metric
>                                                   # empty line: next frame (also n / p / <index>)
> play 20                                           # step through the view at 20 fps, then stop
> back                                              # undo the last query; reset for the whole database
> save /tmp/walk_fast.pkl                           # current view, usable as motion_latent --db
```

`help` lists the rest (`filter`, `nearest`, `limit`, `describe`, `info`, `columns`, `center`).

### Motion latent

Turns frames of a `MotionDatabase` (from motion query) into latent z. Frames are selected with a SQL condition (DuckDB) over the metric columns and `source_index`: comparisons, `AND` / `OR` / `NOT`, arithmetic, `abs`, `BETWEEN`, `IN (..)`.

```bash
uv run python -m humanoidverse.motion_latent \
    --model_folder /path/to/model \
    --where "base_speed > 0.5 AND abs(move_angle) < 30 AND torso_pitch < 15" \
    --name walk_forward
```

By default all selected frames give one z (backward embedding averaged over them, i.e. reward inference with reward 1 on the selection); `--per_frame` gives one z per frame (goal inference). `--sort_by <metric> [--ascending] --limit N` keeps the first N frames after sorting. `--db` defaults to `model_folder/motion_query/motion_db.pkl`.

**Outputs:** `model_folder/motion_latent/latent_z.pkl` (or `--output`): `{name: z}` with z of shape `(1, z_dim)` or `(N, z_dim)`. An existing file is updated, so several selections can be collected under different `--name`s (default: the `--where` expression).

### Motion vocab

`scripts/queries/walk_vocab.py` selects the fixed token vocabulary of holosoma `bfm_walk_token` from the replay buffer:

1. Its query keeps upright frames inside the command range: torso tilt ≤ 30°, a foot within 0.05 m of the ground, wrists below the shoulders, and pelvis height in range.
2. A `case` splits them into walking-command buckets: stand, low speed, 5 translation tiers of `max(|vx|, |vy|)`, and 3 turning tiers of `|yaw_rate|`.
3. `SymmetryError` adds each frame's left-right symmetry error.
4. The chain continues with `.sample(total)`, which picks frames with velocity uniformity > symmetry > pose similarity:

```python
.sample(total)
.partition("grp", {"stand": 0.05, "translate": 0.85, "turn": 0.1}, spacing_floor=0.5)
.stratify(translate=grid(base_vx=0.15, base_vy=0.15, limit=1.05), turn=grid(yaw_rate=0.1, limit=1.05))
.prefer("symmetry_error", keep=0.5)
.spread(SIMILARITY)
```

- `partition` splits the total over the command modes. With `spacing_floor`, a mode too dense to fill its quota with frames far enough apart is capped, and its leftover goes to the others.
- `stratify` spreads frames evenly over velocity grid cells; a cell short of candidates gives all it has. Its keywords name values of the enclosing `partition`, so stand passes through unchanged.
- `prefer` keeps the most symmetric half, never fewer than the quota.
- `spread` keeps frames pairwise apart in a weighted qpos distance (yaw-free base quaternion, base height and joints). Frames are visited in random order and kept when no kept frame is within a threshold, and the threshold is bisected until the quota is filled.

```bash
uv run python scripts/queries/walk_vocab.py --model-folder /path/to/model --latent
```

**Outputs** (under `model_folder/motion_vocab/`, or `--output`):

- `motion_db.pkl`: the selected frames. The `bucket` metric gives the bucket's code, and `db.labels["bucket"]` gives the names. Browse it with `motion_viewer` (`where bucket = 3`).
- `latent_z.pkl` (with `--latent`): `{bucket name: z}`, one z per frame from goal inference. Bucket names sort in token order, so it can be passed directly to holosoma `scripts/build_bfm_vocab.py`.

Velocities follow the walking command: `base_vx` / `base_vy` are in m/s in the heading frame, and `yaw_rate` is in rad/s.

---



## License

BFM-Zero is licensed under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Citation

If you find this project useful in your research, please consider citing:

<!--
```bibtex
@article{li2025bfmzero,
  title   = {BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning},
  author  = {Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
  journal = {arXiv preprint arXiv:2505.06776},
  year    = {2025}
}
```
Wrong arXiv id here!!
--> 

```bibtex
@misc{li2025bfmzeropromptablebehavioralfoundation,
      title={BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning}, 
      author={Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
      year={2025},
      eprint={2511.04131},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.04131}, 
}
```

## Contact

If you have any problems, please contact [liyitang475@gmail.com](mailto:liyitang475@gmail.com).
