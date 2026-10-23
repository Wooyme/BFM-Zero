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

### Rank initial poses using BFM holding

```bash
# Evaluate every original LAFAN frame using Isaac Sim (3,072 parallel environments).
uv run python -m humanoidverse.pose_hold.eval

# A small inspection run with an explicit GPU batch size.
uv run python -m humanoidverse.pose_hold.eval --num-envs 128 \
    --stride 100 --max-frames 128 -o data/pose_hold/preview

# Resume interrupted work with the same inputs, model, code and scoring config.
uv run python -m humanoidverse.pose_hold.eval --resume
```

Every original frame is evaluated, including airborne, low and lying poses. Defaults use
`new_model_for_training_code_inference`, the trusted pickles in
`humanoidverse/data/lafan_29dof`, and a 5-second closed-loop rollout. `--model-folder`, `--data-path`,
`--motion NAME` (repeatable), `--stride`, and `--max-frames` select the model and inputs. Original
frame indices and FPS are retained; targets are not resampled to the policy frequency.

Initialization preserves the base orientation and 29 joint angles, centers XY, and shifts Z until
Isaac Sim's collision geometry is 1 cm above the fixed floor (`--clearance 0.01`). All
velocities and past actions are zero. History repeats the stationary initial observation, and
kinematics are refreshed without advancing physics. The initial observation, including its aligned
root height, gives one fixed `goal_inference` latent. The deterministic actor controls the robot
throughout the rollout using the saved training configuration's scaling, hard-waist gains, limits
and armature. Noise, randomization, motion-reference tasks and automatic resets are absent.

The policy loader keeps only actor, backward and observation-normalizer weights. Isaac Sim runs
headless GPU physics with `--num-envs 3072` by default, the configuration validated on the current
RTX 5070. Set `--num-envs` explicitly to fit another GPU. The backend and parallel count remain fixed
throughout a ranking run. `--device cuda:0` selects the GPU.

At 50 Hz, each rollout compares its full base quaternion (including yaw, invariant to quaternion
sign) and 29 joints to the fixed initial target. For `A(x) = 0.7 mean(x²) + 0.3 P95(x²)`, the default
cost is `0.4 A(base_angle / 15°) + 0.4 A(joint_RMSE / 0.2 rad) + 0.2 (end_distance / 0.25 m)²`.
The score is `100 exp(-cost)`; ranking uses cost to avoid underflow ties. Scores include initial
landing and all subsequent motion, without an unscored settling period. All weights and scales are
CLI options; `--base-scale` takes radians. Independent base/joint/pose/distance ranks are also kept.
Maximum displacement, path length and final-second speeds expose continuous or returning movement.

This `score` (also `fidelity_score`) measures how faithfully the pose is **reproduced**. A separate
`stability_score` measures how stable the **base** is, independent of the target: how much pelvis height, XY
position, yaw and tilt (the gravity direction in the base frame, against its mean) change from their own medians
over the rollout. Its cost is `0.25 A(Δheight / 0.1 m) + 0.25 A(ΔXY / 0.25 m) + 0.25 A(Δyaw / 15°) + 0.25 A(Δtilt / 15°)`,
with per-term `height/drift/yaw/tilt_score` and the maxima `height_dev_max`, `drift_dev_max`, `yaw_dev_max_deg`,
`tilt_dev_max_deg`. A pose held steadily in another posture is stable but not faithful; a fall is neither.
What is actually held is described by medians over the rollout: `hold_height` (pelvis height, m) and
`torso_tilt_deg` (angle of the torso_link z axis to world up).
Code is in the `humanoidverse/pose_hold/` package (`scores.py` holds the scores).

Output in `data/pose_hold/lafan_29dof/`:

- `ranking.csv`: every initial frame, source name/index, mean/P95/max errors, displacements, scores
  and ranks. `valid` means a numerically completed rollout, not a stability certificate; finite falls
  still contribute their full error. Numerical failures have score zero and rank after valid samples.
- `motion_db.pkl`: ranked **initial frames**, zero initial `qvel`, original unshifted `qpos_original`,
  metrics and source labels, directly browsable with the existing viewer.
- `parts/batch_*.npz`: completed batches of scalar summaries, initial poses and per-frame latents;
  used for resume. No trajectories or per-time-step score arrays are saved.
- `manifest.json`, `summary.json`: provenance, completion statistics and measured throughput.
  Model/configuration/physics/code fingerprints prevent incompatible resumes.

```bash
uv run python -m humanoidverse.motion_viewer --db data/pose_hold/lafan_29dof/motion_db.pkl
```

For example, `columns score base_error_mean_deg joint_rmse_mean end_displacement` changes the overlay;
`where end_displacement < 0.1 AND joint_rmse_p95 < 0.15` narrows the ranked view. Input motions are
unchanged. Existing results require `--resume` or explicit `--overwrite`.

### Extract and browse self-stable motion sequences

Build sequence candidates from the original LAFAN motions and their pose-hold ranking, without
loading a checkpoint or starting Isaac Sim:

```bash
uv run python -m humanoidverse.motion_sequence.build
```

Defaults are `--data-path humanoidverse/data/lafan_29dof`,
`--ranking data/pose_hold/lafan_29dof/ranking.csv`, `--threshold 83`, and
`--output-dir data/motion_sequence/lafan_29dof`. Existing outputs require `--overwrite`.
Every source frame must have exactly one finite score. Duplicate/missing scores, invalid frame
indices, unknown sources/motions, and source hashes differing from the ranking's sibling
`manifest.json` cause extraction to fail.

A frame is stable exactly when **`score > threshold`** (83 itself is unstable at the default).
Frames are restored to temporal order within each `(source_file, motion_name)` before extraction:

- `stable`: each maximal continuous stable run, including isolated stable frames.
- `transition`: the last stable frame before an unstable run, all unstable frames in that run,
  and the first subsequent stable frame. Unstable prefixes/tails without both stable endpoints
  do not form transitions.

Endpoints can be shared. For `S S U U S S`, the result is stable `[0,1]`, transition `[1,4]`,
and stable `[4,5]`. Segments never cross source motions, and there is no length limit or score smoothing.

The output directory contains:

- `sequence_db.pkl`: a versioned Joblib index containing source references/hashes, chronological
  per-frame score snapshots, and sequence records. It does not duplicate the original motions.
- `sequences.csv`: ID, kind, relative source filename, motion name, inclusive start/end frame,
  FPS, endpoint/minimum/mean scores, frame count, and duration in seconds.
- `summary.json`: provenance, sequence counts/length statistics and unclosed unstable frame counts.

IDs are zero-based and deterministic in source filename, motion name, then start-frame order;
stable segments precede transitions with the same start. Frame count is `end-start+1`;
duration is `(end-start)/fps`, so a single-frame segment has zero elapsed duration.
The public Python API is `SequenceDatabase.load(path)`, `.sequences`, and
`.get_frames(sequence_id)`, returning `(qpos, scores)` with both endpoints included and WXYZ root
quaternions. Only load trusted pickle files. Keep the original motion files available; the viewer
loads them on demand and checks their hashes against the index.

```bash
uv run python -m humanoidverse.motion_sequence.viewer
# Or select another index:
uv run python -m humanoidverse.motion_sequence.viewer --db /path/to/sequence_db.pkl
```

The MuJoCo window shows original reference poses with sequence/source coordinates, score and
stability status. Playback starts paused, follows the root with the camera, and uses the source FPS.
Centering subtracts only the segment's **starting** XY, preserving translation within the segment;
use `--no-center` to keep the original world coordinates. This is kinematic pose playback.

| Command | Behavior |
| --- | --- |
| `head [n]`, `info` | List the next n sequences (default 20), or inspect the selected sequence. |
| `seq <id>`, `next`, `prev` | Select a sequence and pause at its first frame; next/prev wrap. |
| `show <i>` or `<i>` | Jump to a local zero-based frame within the sequence. |
| `n [k]`, `p [k]`, empty line | Step forward/backward locally, wrapping within the sequence. |
| `play [speed]`, `stop` | Play at source FPS times speed (default 1), or pause. Singletons stay paused. |
| `loop on\|off` | Repeat the current sequence (default on), or pause at its last frame. |
| `center on\|off` | Toggle the translation of the segment's start to XY zero. |
| `quit`, Ctrl-D | Stop playback, close the window and exit. |


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

1. Its query keeps frames with a foot within 0.05 m of the ground and wrists below the shoulders, inside the command range. They are either upright (pelvis height 0.55–1.0 m, torso tilt ≤ 30°) or a static low posture: pelvis height < 0.55 m, torso tilt ≤ 100° (unsigned, about 90° lying prone or supine), and `|vx|`, `|vy|`, `|yaw_rate|` < 0.05.
2. A `case` splits them into buckets: stand, low speed, 5 translation tiers of `max(|vx|, |vy|)`, 3 turning tiers of `|yaw_rate|`, and `10_posture` (every frame below 0.55 m, from squat to kneel to lying).
3. `SymmetryError` adds each frame's left-right symmetry error, scored on its own: each pitch joint (the gait swing axis) may be axially symmetric (left = right) or centrally symmetric (left and right deviating oppositely from the default pose), each roll or yaw joint, waist included, must be axially symmetric.
4. The chain continues with `.sample(total)`, which picks frames with velocity (posture: height and tilt) uniformity > symmetry > pose similarity:

```python
.sample(total)
.partition("grp", {"stand": 0.05, "translate": 0.72, "turn": 0.08, "posture": 0.15}, spacing_floor=0.5)
.stratify(
    {
        "translate": grid(base_vx=0.15, base_vy=0.15, range=(-1.05, 1.05)),
        "turn": grid(yaw_rate=0.1, range=(-1.05, 1.05)),
        "posture": {"base_height": Cell(0.05, 0, 0.55), "torso_tilt": Cell(10, 0, 90)},
    },
    by="grp",
)
.prefer("symmetry_error", keep=0.5)
.spread(SIMILARITY)
```

- `partition` splits the total over the command modes. With `spacing_floor`, a mode too dense to fill its quota with frames far enough apart is capped, and its leftover goes to the others.
- `stratify` spreads frames evenly over velocity grid cells (posture: 0.05 m pelvis height × 10° torso tilt cells, tilt above 90° in the last); a cell short of candidates gives all it has. `by="grp"` keys its grids by the values of the enclosing `partition` on `grp`, so stand passes through unchanged; without `by`, one grid applies to all rows.
- `prefer` keeps the most symmetric half, never fewer than the quota.
- `spread` keeps frames pairwise apart in a weighted qpos distance (yaw-free base quaternion, base height and joints). Frames are visited in random order and kept when no kept frame is within a threshold, and the threshold is bisected until the quota is filled.

```bash
uv run python scripts/queries/walk_vocab.py --model-folder /path/to/model --latent
```

**Outputs** (under `model_folder/motion_vocab/`, or `--output`):

- `motion_db.pkl`: the selected frames. The `bucket` metric gives the bucket's code, and `db.labels["bucket"]` gives the names. Browse it with `motion_viewer` (`where bucket = 3`).
- `latent_z.pkl` (with `--latent`): `{bucket name: z}`, one z per frame from goal inference. Bucket names sort in token order, so it can be passed directly to holosoma `scripts/build_bfm_vocab.py`.

Velocities follow the walking command: `base_vx` / `base_vy` are in m/s in the heading frame, and `yaw_rate` is in rad/s.

### Locomotion and posture vocabulary

`vocab/loco_posture_vocab/loco_posture_vocab.py` builds a fixed 4096-token vocabulary: 2048 upright locomotion
commands and 2048 static posture goals. It retains `multi_vocab.py`'s velocity grid and similarity weights.
Sampling prefers a small `min(abs(left_wrist_height - base_height), abs(right_wrist_height - base_height))`
before similarity spreading, without a symmetry preference. Static standing belongs to the posture half.

```bash
uv run python -m vocab.loco_posture_vocab.loco_posture_vocab \
    --model-folder /path/to/model --lafan '*' --latent \
    --output vocab/loco_posture_vocab/runs/run_001
```

Query workspaces live under `vocab/<query-name>/`. The current query's cache, inspection sheets,
historical runs and exports are in [`vocab/loco_posture_vocab/`](vocab/loco_posture_vocab/README.md).
The business-specific query, sampling, visualization and CEM modules live in that workspace;
the old `scripts/queries/` entry points remain compatibility wrappers.
Its `latest/` link points to the latest validated run. The old model-directory workspace path remains
a compatibility symlink; the model and source motions have not moved.

For repeated full LAFAN runs, add `--scan-cache /path/to/cache`. The cache stores source frames,
query metrics and collision heights, and invalidates when inputs, relevant geometry or query code
change. Scanning runs in a subprocess so its Isaac instance exits before posture evaluation.

The four nonoverlapping pelvis-height bands are `[0, 0.25)`, `[0.25, 0.45)`, `[0.45, 0.65)` and
`[0.65, 1.0]` metres, for supine/prone, kneeling/half-kneeling, crouching/rising and standing.
Each query also constrains **wrist, foot and knee heights** and torso tilt. It permits asymmetric support,
one raised hand/foot and intermediate rising poses. The exact SQL clauses and their relaxed tolerances
are in `vocab/loco_posture_vocab/posture_vocab.py::pose_conditions`.
Both wrists must be no lower than the pelvis band's lower edge; the band's upper edge does not limit wrists.
This condition stays strict during query relaxation, CEM and double-kneeling compensation.
Each band initially receives 512 tokens; short bands can transfer quota to other postures,
while the overall 50/50 split stays fixed. All six
posture kinds must have at least one token. `--total` must therefore be even and at least 12.

Kneeling uses strict support geometry: both ankles are at most 10cm high and differ by at most 3cm.
For double kneeling, all four knee/ankle heights span at most 5cm and both knees are at most 8cm high.
For half kneeling, the knees differ by at least 8cm, the lower knee is 2–7cm above both ankles and at
most 11cm high, and the higher knee is no higher than the pelvis. These support conditions remain
strict under `--relax`; transitional deep squats are excluded. Genuine-pose CEM uses the same criteria.
The thresholds are configurable fields of `Postures` in `vocab/loco_posture_vocab/posture_vocab.py`.

Use `uv run python -m vocab.loco_posture_vocab.inspect_posture_queries --cache /path/to/lafan_scan_cache/table.pkl`
to filter the cached full scan, render two-view knee/ankle inspection sheets and export
`query_samples.pkl` / `query_candidates.pkl` for `motion_viewer`. Knee centres are marked red and
ankle centres blue. The inspection includes diverse motion clips and rejected deep-squat examples.

Selection first uses poses whose lowest MJCF collision point is within `--ground-tol 0.05` metres of
the floor. If band quotas or coverage are insufficient, it relaxes body tolerances by `--relax 0.05`.
If that is still insufficient, airborne poses may be shifted rigidly by `clearance - lowest_collision_z`;
all world heights, root qpos and the backward observation's root height are adjusted consistently.
Fallback rows are admitted only when they address an existing missing cell or short height band.
This fallback uses forward kinematics and preserves joint angles. Disable it with `--no-align-airborne`.
Penetrating poses are not rescued. Isaac evaluation subsequently uses the actual USD collision geometry.

With `--latent`, postures use fresh per-frame `goal_inference` with zero qvel and observation velocities.
The script evaluates 5-second holds, then runs CEM for low stability, height drift or posture mismatch,
retaining the original goal as an incumbent. Stability above 95 is a soft target; CEM aims for 96 and
uses a 3cm height tolerance. An optimized latent normally must retain its posture class.
Unresolved goals also try a stable neighbouring cell as a new CEM search centre. Half-kneeling first
searches from its own latents; on failure it may borrow a nearby-height double-kneeling latent.
Crouching latents cannot fill half-kneeling cells. Within the kneeling height band, scarce genuine
half-kneeling samples retain their quota and the remaining slots go to double kneeling.
This explicit compensation mode checks target height, the mandatory wrist lower bound and Stability, and records the seed kind,
actual held kind and `height_only` flag rather than requiring half-kneeling geometry.
Compensated targets receive another static hold from the reached pose. Their exported qpos and
observations are updated together, with all goal velocities zeroed.
Numeric failures are replaced with unused candidates, or stop export if no valid replacement exists.

Before final selection, CEM can also synthesize missing cells from nearby-cell goal latents. Cells are
posture kind × 5cm height; targets fill height gaps within observed support and seed missing kinds,
without requesting every Cartesian combination of permissive query bounds. Synthesized tokens are
accepted by their **actual reached pose and median rollout height**, with a second hold from that pose. Inclination remains
a broad pose-query condition and a diagnostic, without a target grid or CEM inclination objective. During synthesis, the 3cm
height target refers to the destination cell centre rather than the parent pose. `--max-fill 64` and
`--fill-attempts 2` bound synthesis; `--fill-seeds 8` tries diverse nearby parents for each missing cell
and retains at most one accepted token per task. Increase these for sparse sources. `--no-fill-cells` disables synthesis,
and `--no-cem` disables both optimization and synthesis. Selection without `--latent` does neither.

Outputs default to `vocab/loco_posture_vocab/`; use `--output` to select a run subdirectory:

- `motion_db.pkl`: source/target frames, zero-velocity posture observations and final latents/scores when requested.
- `latent_z.pkl` with `--latent`: `{bucket: (N, z_dim)}`, locomotion first, followed by the four posture bands.
- `selection.csv` and `plots/`: source frame, parent frame, candidate stage and sampling distributions.
- `summary.json`: filter counts, quotas, six-kind coverage, synthesis attempts, unfilled cells and final quality.
- `posture_rollouts.npz` with `--latent`: final rollouts and scores aligned to posture rows in `motion_db.pkl`.
- `initial_motion_db.pkl` and, with `--latent`, `initial_z.npy`: selected goals before CEM, saved before evaluation.

The source may be a replay buffer, `--lafan` motions or `--latents` rollout states. A stationary-only
source generally cannot provide the required locomotion half; use a source containing both groups.

To save the last state of each latent after five seconds from the model's configured default pose:

```bash
uv run python scripts/queries/rollout_vocab.py --model-folder /path/to/model --latents /path/to/latent_z.pkl
uv run python -m humanoidverse.motion_viewer --db /path/to/default_rollout_last.pkl
```

The rollout preserves the configured initial height, starts with zero velocity, and keeps actual final
velocities. The output sits beside `latent_z.pkl`; `source_index` is the token index, and `bucket` and
`token_in_bucket` preserve vocabulary grouping. Use `--duration` to change the rollout length.

### Latent residual VQ

`humanoidverse.latent_vq` tokenizes the goal latent of every LAFAN frame into two tokens, after the action tokenizer of VQ-BeT: an MLP encoder, a 2-layer residual VQ (the first code is coarse, the second quantizes what it left) and an MLP decoder whose output is projected onto the z sphere like `project_z`. Codebooks are EMA updated with k-means initialization and dead-code replacement; the loss is L1 reconstruction + 5 × commitment.

```bash
# z = project_z(B(next_obs)) per frame, velocities kept (velocity_multiplier 1); needs Isaac
uv run python -m humanoidverse.latent_vq.extract --model-folder /path/to/model
# 4 motions held out for validation; --all trains on every motion
uv run python -m humanoidverse.latent_vq.train --model-folder /path/to/model \
    --chunk 8 --codebook-size 256 --hidden-dim 1024 --layer-num 2
```

With `--chunk T`, one token pair encodes T consecutive z of one motion (the action chunks of VQ-BeT). Training takes a window at every frame; tokenizing takes back-to-back windows, the last of each motion ending on its last frame.

Training prints L1, cosine of z_hat to z (mean, 5th percentile, and from the first token alone), codes used and perplexity per layer, and unique token pairs, on the train and held-out motions. At the end it reports, on the held-out motions, the cosine of a flat k-means with `codebook_size²` centers (the same token budget, capped at 4096).

**Outputs** (under `model_folder/latent_vq/`): `goal_z.npz` (`z`, `qpos`, `qvel`, `motion`, `motion_time` of every frame); `<run>/rvq.pt` (`LatentRVQVAE.load`, then `encode(z)` of `(N, T, z_dim)` windows → `(N, 2)` tokens and `decode(tokens)` → windows), `<run>/tokens.npz` (`tokens` and first frame `starts` of each window, `chunk`, and per frame `z_hat`, `motion`, `motion_time`, `is_val`) and `<run>/metrics.json`.

### Experiments

One-off experiments live in `experiments/` (run from the repository root as `python -m experiments.<name>`), each with a write-up under `experiments/docs/<name>/`: setup, raw results, inferences, takeaways and common misreadings.

| Experiment | Question | Write-up |
|---|---|---|
| `tracking_window` | How the tracking window T = 1..8 changes z and what the actor does with it | [experiments/docs/tracking_window](experiments/docs/tracking_window/README.md) |
| `motion_to_z` | How differences between reference motion frames show up in the goal z, and how often a close z hides a far motion | [experiments/docs/motion_to_z](experiments/docs/motion_to_z/README.md) |
| `z_to_motion` | How a controlled change of z (cosine, direction) changes the actions and trajectories of the policy | [experiments/docs/z_to_motion](experiments/docs/z_to_motion/README.md) |
| `dof_noise_z` | How noise on N upper-body, lower-body or arbitrary joints of a reference frame changes its goal z | [experiments/docs/dof_noise_z](experiments/docs/dof_noise_z/README.md) |
| `rvq_capacity` | How closely residual VQ reproduces the goal z, at what token budget, on training and unseen motions | [experiments/docs/rvq_capacity](experiments/docs/rvq_capacity/README.md) |
| `rvq_rollout` | How well the policy still tracks motions when z is replaced by its residual VQ reconstruction | [experiments/docs/rvq_rollout](experiments/docs/rvq_rollout/README.md) |
| `hand_compose` | Whether walking plus a hand target can come from two additive latent tables, z = project(z_walk + λ z_hand), instead of one product vocab | [experiments/docs/hand_compose](experiments/docs/hand_compose/README.md) |
| `stable_wrist` | Whether z = project(B(stable pose) + w B(right wrist frame)), as reward_inference, makes the policy hold the pose and reach the wrist position from the default pose | [experiments/docs/stable_wrist](experiments/docs/stable_wrist/README.md) |

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
