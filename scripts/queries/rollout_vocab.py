"""Roll out each vocabulary latent from the configured default pose; export last states for motion_viewer."""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_latents(path):
    groups = joblib.load(path)
    if not isinstance(groups, dict) or not groups:
        raise ValueError('Expected a nonempty {bucket: (N, z_dim)} vocabulary')
    names = sorted(groups)
    arrays = [np.asarray(groups[name], dtype=np.float32) for name in names]
    if any(a.ndim != 2 or not len(a) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Every bucket must have finite, nonempty 2D latents')
    if len({a.shape[1] for a in arrays}) != 1:
        raise ValueError('Latent dimensions differ between buckets')
    codes = np.concatenate([np.full(len(a), i, dtype=np.int32) for i, a in enumerate(arrays)])
    within = np.concatenate([np.arange(len(a), dtype=np.int32) for a in arrays])
    return np.concatenate(arrays), names, codes, within


def default_qpos(config):
    robot = config['robot']
    initial = robot['init_state']
    # Training config root rotations are XYZW; exported qpos uses WXYZ.
    rotation = np.asarray(initial['rot'], dtype=np.float32)[[3, 0, 1, 2]]
    joints = [initial['default_joint_angles'][name] for name in robot['dof_names']]
    return np.asarray([*initial['pos'], *rotation, *joints], dtype=np.float32)


def main(
    model_folder: Path = Path('new_model_for_training_code_inference'),
    latents: Path = Path('new_model_for_training_code_inference/loco_posture_vocab/full_lafan_compensated/latent_z.pkl'),
    output: Path | None = None,
    duration: float = 5.0,
    num_envs: int = 2048,
    device: str = 'cuda:0',
    seed: int = 0,
):
    """Each latent starts independently from config default joints/root pose and zero velocity.

    The initial height is preserved exactly (no ground alignment). Actual final velocities are kept.
    Bucket names sort in the same token order as the vocabulary exporter. Output is a MotionDatabase
    readable with `python -m humanoidverse.motion_viewer --db <output>`.
    """
    import torch

    from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase, compute_metrics
    from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model
    from humanoidverse.envs.g1_env_helper.queries import METRICS
    from humanoidverse.pose_hold.eval import make_backend
    from humanoidverse.pose_hold.policy import GoalPolicy, Observations, load_training_config, sha256
    from humanoidverse.pose_hold.scores import ScoreConfig
    from scripts.queries.cem_rebuild import OBS_KEYS, rollout

    sys.stdout.reconfigure(line_buffering=True)
    if not math.isfinite(duration) or duration <= 0 or num_envs < 1:
        raise ValueError('Require positive duration and num_envs')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(device)
    z, names, codes, within = load_latents(latents)
    output = output or latents.with_name('default_rollout_last.pkl')
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_training_config(model_folder)
    initial = default_qpos(config)
    qpos = np.tile(initial, (len(z), 1))
    backend = make_backend(config, min(num_envs, len(z)), 0.01, device)
    try:
        policy, observations = GoalPolicy(model_folder, device), Observations(config, device)
        print(f'Rollout {len(z)} latents from default_pos for {duration}s, batch={backend.batch_size}', flush=True)
        with torch.inference_mode():
            result = rollout(backend, policy, observations, qpos,
                             torch.as_tensor(z, dtype=torch.float32, device=device), duration, ScoreConfig(),
                             reset_kwargs={'align_ground': False})
        if not np.allclose(result['initial_qpos'], qpos, atol=1e-5, rtol=0):
            raise RuntimeError('Simulator did not start from the configured default pose')
        if not np.allclose(result['initial_qvel'], 0, atol=1e-5, rtol=0):
            raise RuntimeError('Initial velocities were not zero')
        print('Rollouts complete; computing final-frame query metrics', flush=True)
        metrics = compute_metrics(env_model(), result['final_qpos'], result['final_qvel'], list(METRICS), max_workers=1)
        metrics.update({k: v for k, v in result.items() if v.ndim == 1})
        metrics.update(token_index=np.arange(len(z)), bucket=codes, token_in_bucket=within)
        recipe = dict(script='scripts/queries/rollout_vocab.py', latents=str(latents.resolve()),
                      latents_sha256=sha256(latents), model_folder=str(model_folder.resolve()),
                      duration=duration, steps=math.ceil(duration/backend.dt), dt=backend.dt,
                      initial='config.robot.init_state default joints/root; zero velocity',
                      initial_qpos=initial.tolist(), align_ground=False, seed=seed,
                      buckets={name: int((codes==i).sum()) for i,name in enumerate(names)})
        frames = dict(qpos=result['final_qpos'], qvel=result['final_qvel'], z=z,
                      next_obs={k: result[f'final_obs_{k}'] for k in OBS_KEYS},
                      initial_qpos=result['initial_qpos'], initial_qvel=result['initial_qvel'])
        db = MotionDatabase(metrics=metrics, source_index=np.arange(len(z)), frames=frames,
                            labels={'bucket': names}, recipe=json.dumps(recipe))
        db.save(output)
        summary = dict(**recipe, output=str(output.resolve()), frames=len(db),
                       numeric_valid=int(result['valid'].sum()),
                       finite_final_states=bool(np.isfinite(frames['qpos']).all() and np.isfinite(frames['qvel']).all()))
        output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(f'Saved {len(db)} final frames to {output}; {summary["numeric_valid"]}/{len(db)} numerically valid', flush=True)
    finally:
        backend.close()


if __name__ == '__main__':
    import tyro

    tyro.cli(main)
