"""Stable goal latents of every frame of the rec* LAFAN motions.

The goal latent of each frame (goal_inference with the velocities zeroed, as motion_latent) is held by the BFM
policy from rest in the frame's pose (pose_hold, Isaac Lab). Latents whose rollout is not stable enough
(stability_score < min_stability) or whose held pelvis height (median over the rollout) is more than height_tol
away from the reference height (pelvis of the pose placed on the ground by pose_hold's reset) are optimized
with CEM around the goal latent until both hold.

Saves latent_z.pkl ({motion name: (frames, z_dim) z}) and record_latent_db.pkl, a LatentDatabase (query/latent_db.py)
of the rollouts: one row per frame, in frame order, with its latent and the last state of its rollout (qpos, qvel
and the backward-map observation). Open it with humanoidverse.motion_viewer or query it with Source(latents=...).

Run: uv run python scripts/mem_guard.py --limit 90 -- python scripts/queries/test.py [--max-frames N] ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per metric worker process

import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.query.motion_query import Source, to_database, zero_velocity

OBS_KEYS = ("state", "last_action", "privileged_state")  # next_obs of the other frame sources
CEM_SETTINGS = ("min_stability", "height_tol", "margin", "population", "elite_frac", "iterations", "sigma0", "sigma_min",
                "height_penalty", "duration", "num_envs")


def rollout(backend, policy, observations, qpos: np.ndarray, z, duration: float, score_config) -> dict[str, np.ndarray]:
    """pose_hold.eval.evaluate_batch with given latents z (n, d): hold each pose of qpos (n, 36) from rest, in
    batches of the backend size. Scores plus ref_height (pelvis after the reset), final_qpos, final_qvel and
    the backward-map observation of the last state (final_obs_<key>)."""
    import torch

    from humanoidverse.pose_hold.scores import Scores

    parts = []
    for b in range(0, len(qpos), backend.batch_size):
        q, zb = qpos[b : b + backend.batch_size], z[b : b + backend.batch_size]
        state = backend.reset(q)
        obs, tensors = observations.build(state, np.zeros((len(q), 29)), reset=True)
        initial = tensors["qpos"].clone()
        scoring = Scores(initial, math.ceil(duration / backend.dt), backend.dt, score_config,
                         torso=observations.body_names.index("torso_link"))
        for step in range(scoring.steps):
            state, applied = backend.step(policy.act(obs, zb, mean=True))
            obs, tensors = observations.build(state, applied)
            scoring.add(step, tensors, state["valid"])
        result = scoring.finish()
        result["ref_height"] = initial[:, 2].cpu().numpy()
        result["final_qpos"] = tensors["qpos"].cpu().numpy()
        result["final_qvel"] = tensors["qvel"].cpu().numpy()
        result |= {f"final_obs_{k}": obs[k].cpu().numpy() for k in OBS_KEYS}
        parts.append(result)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["height_err"] = out["hold_height"] - out["ref_height"]
    return out


def fitness(result: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Stability minus a penalty per cm of held height beyond the tolerance (0 when invalid), and feasibility."""
    excess = np.maximum(0, np.abs(result["height_err"]) - cfg["height_tol"]) / 0.01
    fit = np.where(result["valid"], result["stability_score"] - cfg["height_penalty"] * excess, 0)
    feasible = result["valid"] & (result["stability_score"] >= cfg["min_stability"] + cfg["margin"]) & (excess == 0)
    return fit, feasible


def cem_worker(job: Path, out: Path) -> None:
    """Isaac subprocess: evaluate the goal latents of job, CEM the failing ones, re-evaluate all, save to out."""
    import torch

    from humanoidverse.pose_hold.eval import make_backend
    from humanoidverse.pose_hold.policy import GoalPolicy, Observations, load_training_config
    from humanoidverse.pose_hold.scores import ScoreConfig

    sys.stdout.reconfigure(line_buffering=True)  # the Isaac app exits without flushing
    torch.set_num_threads(1)
    torch.manual_seed(0)
    with np.load(job, allow_pickle=False) as data:
        qpos, z0 = data["qpos"], data["z0"]
        model_folder, device = Path(str(data["model_folder"])), str(data["device"])
        cfg = {k: data[k].item() for k in CEM_SETTINGS}
    device = device if ":" in device else f"{device}:0"
    config = load_training_config(model_folder)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    backend = make_backend(config, int(cfg["num_envs"]), 0.01, device)
    score_config = ScoreConfig()
    start = time.time()
    try:
        policy = GoalPolicy(model_folder, device)
        observations = Observations(config, device)
        with torch.inference_mode():
            z0_t = torch.as_tensor(z0, dtype=torch.float32, device=device)
            first = rollout(backend, policy, observations, qpos, z0_t, cfg["duration"], score_config)
            todo = np.flatnonzero(~first["valid"] | (first["stability_score"] < cfg["min_stability"])
                                  | (np.abs(first["height_err"]) > cfg["height_tol"]))
            print(f"Goal latents: {len(qpos)} frames, {len(todo)} below stability {cfg['min_stability']} or off height "
                  f"({time.time() - start:.0f}s)")

            z = z0_t.clone()
            n, P = len(todo), int(cfg["population"])
            n_elite = max(2, round(P * cfg["elite_frac"]))
            per_batch = max(1, int(cfg["num_envs"]) // P)
            mu = z0_t[todo].clone()
            sigma = torch.full_like(mu, cfg["sigma0"])
            best_z, best_key = mu.clone(), np.full(n, -np.inf)
            found = np.zeros(n, dtype=bool)
            iters = np.zeros(n, dtype=np.int32)
            for it in range(int(cfg["iterations"])):
                active = np.flatnonzero(~found)
                if not len(active):
                    break
                for b in range(0, len(active), per_batch):
                    rows = active[b : b + per_batch]
                    r = torch.as_tensor(rows, device=device)
                    cand = policy.project_z(mu[r, None] + sigma[r, None] * torch.randn(len(rows), P, mu.shape[1], device=device))
                    cand[:, 0] = policy.project_z(mu[r])
                    flat = cand.reshape(-1, cand.shape[-1])
                    result = rollout(backend, policy, observations, np.repeat(qpos[todo[rows]], P, axis=0), flat,
                                     cfg["duration"], score_config)
                    fit, feasible = (x.reshape(len(rows), P) for x in fitness(result, cfg))
                    key = fit + 1000 * feasible  # any feasible latent beats every infeasible one
                    top = np.argmax(key, axis=1)
                    better = key[np.arange(len(rows)), top] > best_key[rows]
                    best_key[rows[better]] = key[better, top[better]]
                    bt, tt = torch.as_tensor(better, device=device), torch.as_tensor(top, device=device)
                    best_z[r[bt]] = cand[bt, tt[bt]]
                    found[rows] |= feasible.any(axis=1)
                    iters[rows] = it + 1
                    elite = torch.as_tensor(np.argsort(-fit, axis=1)[:, :n_elite], device=device)
                    e = torch.gather(cand, 1, elite[..., None].expand(-1, -1, cand.shape[-1]))
                    mu[r] = 0.7 * e.mean(1) + 0.3 * mu[r]
                    sigma[r] = 0.7 * e.std(1).clamp(min=cfg["sigma_min"]) + 0.3 * sigma[r]
                print(f"  CEM iteration {it + 1}: {found.sum()}/{n} frames feasible ({time.time() - start:.0f}s)")
            z[torch.as_tensor(todo, device=device)] = best_z

            # the rollouts kept: one pass over all chosen latents (scores shift with the batch layout)
            final = rollout(backend, policy, observations, qpos, z, cfg["duration"], score_config)
            optimized = np.zeros(len(qpos), dtype=bool)
            optimized[todo] = True
            feasible = ~optimized
            feasible[todo] = found
            cem_iterations = np.zeros(len(qpos), dtype=np.int32)
            cem_iterations[todo] = iters
            z_cos = torch.nn.functional.cosine_similarity(z, z0_t, dim=-1).cpu().numpy()
            # how far the latent is from what the backward map infers of the state it leads to
            obs = {k: torch.as_tensor(final[f"final_obs_{k}"], device=device) for k in OBS_KEYS}
            z_back = torch.cat([policy.goal_inference({k: v[b : b + 4096] for k, v in obs.items()}).float()
                                for b in range(0, len(z), 4096)])
            z_backward_cos = torch.nn.functional.cosine_similarity(z, z_back, dim=-1).cpu().numpy()
            np.savez(out, z=z.cpu().numpy(), stability_initial=first["stability_score"], height_err_initial=first["height_err"],
                     optimized=optimized, feasible=feasible, cem_iterations=cem_iterations, z_cos=z_cos,
                     z_backward_cos=z_backward_cos,
                     **{k: final[k] for k in ("final_qpos", "final_qvel", "ref_height", "hold_height", "height_err",
                                              "score", "stability_score", "valid")},
                     **{f"final_obs_{k}": final[f"final_obs_{k}"] for k in OBS_KEYS})
            print(f"Saved the rollouts ({time.time() - start:.0f}s)")
    finally:
        backend.close()  # closing the Isaac app ends the process: everything is saved before


def main(
    source: Source = Source(model_folder=Path("new_model_for_training_code_inference"), lafan="rec*"),
    min_stability: float = 95.0,
    height_tol: float = 0.03,
    margin: float = 1.0,
    population: int = 32,
    elite_frac: float = 0.25,
    iterations: int = 10,
    sigma0: float = 0.3,
    sigma_min: float = 0.02,
    height_penalty: float = 5.0,
    duration: float = 5.0,
    num_envs: int = 2048,
    output: Path | None = None,
):
    """Args:
    min_stability: stability_score every latent should reach.
    height_tol: m between the held pelvis height and the reference pose's.
    margin: CEM aims at min_stability + margin, as the final pass scores shift a little with the batch layout.
    population / elite_frac / iterations: CEM samples per frame and iteration, share kept as elites, iteration cap.
    sigma0 / sigma_min: initial and smallest per-dimension std of the CEM samples before projecting onto the z sphere.
    height_penalty: stability points lost per cm of held height beyond height_tol.
    duration: s of each rollout (pose_hold's 5 s).
    num_envs: parallel Isaac Sim environments.
    output: output folder (default: model_folder/motion_query/rec_stable).
    """
    import torch

    from humanoidverse.motion_latent import infer_z, load_model
    from humanoidverse.query.latent_db import LatentDatabase

    output = output or Path(source.model_folder) / "motion_query" / "rec_stable"
    output.mkdir(parents=True, exist_ok=True)
    query = source.scan()
    df = query.collect()
    db = to_database(query.frames, df)
    zero_velocity(db)
    model = load_model(source.model_folder, source.device)
    z0 = infer_z(model, db.frames, per_frame=True, device=source.device)
    del model
    torch.cuda.empty_cache()
    print(f"{len(z0)} goal latents")

    settings = dict(min_stability=min_stability, height_tol=height_tol, margin=margin, population=population,
                    elite_frac=elite_frac, iterations=iterations, sigma0=sigma0, sigma_min=sigma_min,
                    height_penalty=height_penalty, duration=duration, num_envs=num_envs)
    with tempfile.TemporaryDirectory() as tmp:
        job, out = Path(tmp) / "job.npz", Path(tmp) / "result.npz"
        np.savez(job, qpos=db.frames["qpos"].astype(np.float32), z0=z0, model_folder=str(Path(source.model_folder).resolve()),
                 device=source.device, **settings)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "cem", str(job), str(out)], check=True)
        with np.load(out) as data:
            res = dict(data)

    motions = db.frames["motion"].astype(str)
    names = sorted(np.unique(motions))
    joblib.dump({name: res["z"][motions == name].astype(np.float32) for name in names}, output / "latent_z.pkl")
    rollouts = LatentDatabase(
        z=res["z"],
        qpos=res["final_qpos"],
        qvel=res["final_qvel"],
        next_obs={k: res[f"final_obs_{k}"] for k in OBS_KEYS},
        recipe=f"scripts/queries/test.py lafan={source.lafan} {settings}; states are the last of the rollouts",
    )
    rollouts.save(output / "record_latent_db.pkl")

    passed = (res["stability_score"] >= min_stability) & (np.abs(res["height_err"]) <= height_tol) & res["valid"]
    print(f"{len(passed)} frames: {res['optimized'].sum()} optimized ({res['feasible'][res['optimized']].sum()} feasible in CEM), "
          f"{passed.sum()} pass after the final pass, {(~passed).sum()} fail "
          f"({(res['stability_score'] < min_stability).sum()} unstable, {(np.abs(res['height_err']) > height_tol).sum()} off height)")
    print(f"z vs goal inference of the rollout's last state: median cosine {np.median(res['z_backward_cos']):.3f}")
    print(f"Saved {output / 'latent_z.pkl'} and {output / 'record_latent_db.pkl'}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "cem":
        cem_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        import tyro

        tyro.cli(main, config=(tyro.conf.OmitArgPrefixes,))
