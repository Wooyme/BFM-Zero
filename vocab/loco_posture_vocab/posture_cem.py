"""Isaac subprocess for pose-preserving CEM and neighbouring-cell synthesis.

Uses the same rollout, scores, sphere projection and CEM defaults as
scripts/queries/cem_rebuild.py, with original latents retained as incumbents.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import compute_metrics
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model
from vocab.loco_posture_vocab.posture_vocab import BUCKETS, HEIGHT_COLUMNS, Postures, annotate, cell_keys, neighbour_order


@dataclass(frozen=True)
class Optimization:
    min_stability: float = 95.0
    height_tol: float = 0.03
    margin: float = 1.0
    population: int = 32
    elite_frac: float = 0.25
    iterations: int = 10
    sigma0: float = 0.3
    sigma_min: float = 0.02
    height_penalty: float = 5.0
    duration: float = 5.0
    num_envs: int = 2048
    seed: int = 0
    fill_cells: bool = True
    max_fill: int = 64
    fill_attempts: int = 2
    fill_seeds: int = 8

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 <= self.min_stability < 100 or not 0 <= self.margin <= 100 - self.min_stability:
            raise ValueError("Require 0 <= min_stability < 100 and 0 <= margin <= 100-min_stability")
        if self.height_tol < 0 or self.height_penalty < 0 or self.duration <= 0:
            raise ValueError("Require nonnegative height tolerance/penalty and positive duration")
        if self.population < 2 or not 0 < self.elite_frac <= 1 or self.iterations < 0 or self.num_envs < 1:
            raise ValueError("Invalid CEM population, elite_frac, iterations or num_envs")
        if self.sigma_min <= 0 or self.sigma0 < self.sigma_min or self.max_fill < 0 or self.fill_attempts < 1 or self.fill_seeds < 1:
            raise ValueError("Invalid CEM sigma or fill budget")


def held_geometry(qpos, cfg):
    import pandas as pd

    names = list(HEIGHT_COLUMNS) + ["torso_tilt", "torso_pitch", "foot_clearance", "base_vx", "base_vy"]
    valid = np.isfinite(qpos).all(1) & (np.linalg.norm(qpos[:, 3:7], axis=1) > 1e-8)
    safe = qpos.copy()
    safe[~valid] = 0
    safe[~valid, 3] = 1
    metrics = compute_metrics(env_model(), safe, np.zeros((len(safe), safe.shape[1] - 1)), names, max_workers=1)
    return annotate(pd.DataFrame(metrics), cfg, cfg.relax), valid


def objective(result, geometry, target_bucket, target_kind, target_cell, cfg, opt, filling=False, height_only=None):
    """A matching posture always beats a nonmatching one, even below 95 points."""
    valid = result["valid"] & np.isfinite(result["stability_score"]) & np.isfinite(result["height_err"])
    kind = geometry.pose_kind.astype("string").fillna("unclassified").to_numpy(dtype=str) == target_kind
    if not filling:
        kind |= np.isin(target_kind, ["transition", "rising"])
    same = (geometry.bucket.astype("string").fillna("unclassified").to_numpy(dtype=str) == target_bucket) & kind
    height_only = np.zeros(len(geometry), dtype=bool) if height_only is None else np.asarray(height_only, dtype=bool)
    height_only = height_only & (target_kind == "half_kneel")
    final_height = geometry.base_height.to_numpy()
    held_height = np.asarray(result.get("hold_height", final_height))
    # Compensation relaxes support/torso geometry, while the wrist lower bound stays mandatory.
    lo, hi = cfg.height_edges[1:3]
    same = np.where(height_only, (final_height >= lo) & (final_height < hi) & (held_height >= lo) & (held_height < hi), same)
    band = np.array([list(BUCKETS).index(str(b)) for b in target_bucket])
    lower_wrist = np.asarray(cfg.height_edges)[band]
    wrist_min = geometry.w_min.to_numpy()
    wrist_ok = wrist_min >= lower_wrist
    same &= wrist_ok
    height_error = np.abs(result["height_err"])
    target_error = np.zeros(len(geometry))
    # Graded subtype errors let an absent kind be approached before a hard
    # query boundary is crossed (e.g. supine -> prone, or two knees -> one).
    pitch = geometry.torso_pitch.to_numpy()
    knee_min, knee_max = geometry.k_min.to_numpy(), geometry.k_max.to_numpy()
    ankle_min, ankle_max = geometry.f_min.to_numpy(), geometry.f_max.to_numpy()
    shape_error = np.where(target_kind == "prone", np.maximum(0, 20 - pitch) / 20, 0.0)
    shape_error += np.where(target_kind == "supine", np.maximum(0, pitch + 20) / 20, 0.0)
    ankle_error = np.maximum(0, ankle_max - ankle_min - cfg.ankle_level_tol)
    ankle_error += np.maximum(0, ankle_max - cfg.kneel_ankle_max)
    plane_error = np.maximum(0, np.maximum(knee_max, ankle_max) - np.minimum(knee_min, ankle_min) - cfg.kneel_plane_tol)
    plane_error += np.maximum(0, knee_max - cfg.kneel_knee_max)
    half_error = np.maximum(0, cfg.half_knee_gap - (knee_max - knee_min))
    half_error += np.maximum(0, cfg.half_low_knee_min - (knee_min - ankle_max))
    half_error += np.maximum(0, knee_min - ankle_min - cfg.half_low_knee_max)
    half_error += np.maximum(0, knee_min - cfg.half_low_knee_height_max)
    half_error += np.maximum(0, knee_max - geometry.base_height.to_numpy())
    shape_error += np.where(target_kind == "kneel", (ankle_error + plane_error) / 0.05, 0.0)
    shape_error += np.where(target_kind == "half_kneel", (ankle_error + half_error) / 0.05, 0.0)
    shape_error += np.where(target_kind == "crouch", np.maximum(0, 0.10 - knee_min) / 0.05, 0.0)
    shape_error[height_only] = 0
    shape_error += np.maximum(0, lower_wrist - wrist_min) / 0.05
    in_cell = np.ones(len(geometry), dtype=bool)
    if filling:
        achieved = cell_keys(geometry, cfg)
        in_cell = np.array([k == (str(t[0]), str(t[1]), int(t[2])) for k, t in zip(achieved, target_cell)])
        steady = geometry.assign(base_height=result.get("hold_height", geometry.base_height))
        steady_cells = cell_keys(steady, cfg)
        in_cell &= np.array([k == (str(t[0]), str(t[1]), int(t[2])) for k, t in zip(steady_cells, target_cell)])
        lower = target_cell[:, 2].astype(float) * cfg.height_cell
        upper = lower + cfg.height_cell
        height_cell = (final_height >= lower) & (final_height < upper) & (held_height >= lower) & (held_height < upper)
        in_cell = np.where(height_only, height_cell, in_cell)
        hc = (target_cell[:, 2].astype(float) + 0.5) * cfg.height_cell
        # A fill target is independent of the parent's reset height. Keeping the
        # parent's 3cm constraint here would prevent moving into a new height cell.
        height_error = np.abs(steady.base_height.to_numpy() - hc)
        target_error = height_error / cfg.height_cell
    excess = np.maximum(0, height_error - opt.height_tol) / 0.01
    fit = result["stability_score"] - opt.height_penalty * excess - 10 * (target_error + shape_error)
    fit = np.nan_to_num(fit, nan=-1e9, posinf=-1e9, neginf=-1e9)
    # Finite, shape-preserving rollouts are admissible even below the stability target.
    admissible = valid & same & in_cell
    feasible = admissible & (result["stability_score"] >= opt.min_stability + opt.margin) & (excess == 0)
    key = np.where(valid, fit + 10000 * admissible + 100000 * feasible, -1e9)
    return key, admissible, feasible


def should_optimize(result, matching, opt):
    return ~result["valid"] | ~matching | (result["stability_score"] <= opt.min_stability) | (np.abs(result["height_err"]) > opt.height_tol)


def worker(job: Path, output: Path):
    import json
    import sys
    import time

    import torch

    from humanoidverse.pose_hold.eval import make_backend
    from humanoidverse.pose_hold.policy import GoalPolicy, Observations, load_training_config
    from humanoidverse.pose_hold.scores import ScoreConfig
    from scripts.queries.cem_rebuild import rollout

    sys.stdout.reconfigure(line_buffering=True)
    torch.set_num_threads(1)
    with np.load(job, allow_pickle=False) as data:
        qpos, z0 = data["qpos"], data["z0"]
        buckets, kinds, cells = data["buckets"], data["kinds"], data["cells"]
        folder, device = Path(str(data["model_folder"])), str(data["device"])
        cfg = Postures(**json.loads(str(data["postures"])))
        opt = Optimization(**json.loads(str(data["optimization"])))
        filling, enabled = bool(data["filling"]), bool(data["cem"])
        height_only = data["height_only"].copy() if "height_only" in data else np.zeros(len(qpos), dtype=bool)
        seed_kind = data["seed_kinds"].copy() if "seed_kinds" in data else kinds.copy()
    opt.validate()
    torch.manual_seed(opt.seed)
    if torch.device(device).type == "cuda":
        device = device if ":" in device else f"{device}:0"
        torch.cuda.set_device(device)
    config = load_training_config(folder)
    backend = make_backend(config, min(opt.num_envs, len(qpos) * opt.population), cfg.clearance, device)
    try:
        policy, observations = GoalPolicy(folder, device), Observations(config, device)
        scores = ScoreConfig()

        def evaluate(q, z):
            result = rollout(backend, policy, observations, q, z, opt.duration, scores)
            geometry, valid = held_geometry(result["final_qpos"], cfg)
            result["valid"] &= valid
            return result, geometry

        with torch.inference_mode():
            started = time.monotonic()
            z = torch.as_tensor(z0, dtype=torch.float32, device=device)
            first, geom = evaluate(qpos, z)
            best_key, matching, found = objective(first, geom, buckets, kinds, cells, cfg, opt, filling, height_only)
            active_mask = ~found if filling else should_optimize(first, matching, opt)
            print(
                f"Initial {'cell seeds' if filling else 'posture goals'}: {len(z)}, "
                f"{int((first['stability_score'] > opt.min_stability).sum())} above {opt.min_stability}, "
                f"{int(matching.sum())} matching, {int(active_mask.sum())} need CEM "
                f"({time.monotonic() - started:.0f}s)"
            )
            attempted = np.zeros(len(z), dtype=bool)
            iterations = np.zeros(len(z), dtype=np.int32)
            seed_restarts = np.zeros(len(z), dtype=np.int32)
            stable_seeds = matching & first["valid"] & (first["stability_score"] > opt.min_stability)
            mu, best = z.clone(), z.clone()
            best_height = first["hold_height"].copy()
            sigma = torch.full_like(mu, opt.sigma0)
            p, elite_n = opt.population, max(2, round(opt.population * opt.elite_frac))
            per_batch = max(1, backend.batch_size // p)
            for iteration in range(opt.iterations if enabled else 0):
                active = np.flatnonzero(active_mask & ~found)
                if not len(active):
                    break
                if not filling and iteration == max(1, opt.iterations // 2):
                    # Borrow a stable neighbouring cell as another search centre;
                    # keep the incumbent and the original reset/height target.
                    for row in active:
                        related = (buckets == buckets[row]) & (kinds == kinds[row])
                        if kinds[row] == "half_kneel":
                            related |= (buckets == buckets[row]) & (kinds == "kneel")
                        seeds = np.flatnonzero((stable_seeds | found) & related)
                        seeds = seeds[seeds != row]
                        if len(seeds):
                            order = neighbour_order(best_height[seeds], buckets[seeds], kinds[seeds],
                                                    first["ref_height"][row], buckets[row], kinds[row], attempt=1)
                            parent = seeds[order[0]]
                            mu[row] = policy.project_z(best[parent] + opt.sigma_min * torch.randn_like(best[parent]))
                            sigma[row] = opt.sigma0
                            seed_restarts[row] += 1
                            if kinds[row] == "half_kneel" and kinds[parent] == "kneel":
                                height_only[row] = True
                                seed_kind[row] = kinds[parent]
                for start in range(0, len(active), per_batch):
                    rows = active[start : start + per_batch]
                    idx = torch.as_tensor(rows, device=device)
                    candidates = policy.project_z(mu[idx, None] + sigma[idx, None] * torch.randn(len(rows), p, z.shape[1], device=device))
                    candidates[:, 0] = best[idx]  # incumbent is evaluated in the same batch as challengers
                    if p > 2:
                        candidates[:, 1] = policy.project_z(mu[idx])
                    repeated = np.repeat(rows, p)
                    result, geometry = evaluate(qpos[repeated], candidates.reshape(-1, z.shape[1]))
                    key, admissible, feasible = objective(
                        result, geometry, buckets[repeated], kinds[repeated], cells[repeated], cfg, opt, filling, height_only[repeated]
                    )
                    key = key.reshape(len(rows), p)
                    search_key = key.copy()
                    if not filling:
                        incumbent = key[:, 0].copy()
                        key[~admissible.reshape(len(rows), p)] = -1e9
                        key[:, 0] = incumbent
                    top = key.argmax(1)
                    # Compare with this iteration's incumbent, not a score from a
                    # different PhysX batch layout. This prevents accepting regression.
                    changed = top != 0
                    for local in np.flatnonzero(changed):
                        best[rows[local]] = candidates[local, top[local]]
                    best_key[rows] = key[np.arange(len(rows)), top]
                    best_height[rows] = result["hold_height"].reshape(len(rows), p)[np.arange(len(rows)), top]
                    found[rows] = feasible.reshape(len(rows), p)[np.arange(len(rows)), top]
                    attempted[rows], iterations[rows] = True, iteration + 1
                    elite = torch.as_tensor(np.argsort(-search_key, axis=1)[:, :elite_n], device=device)
                    e = torch.gather(candidates, 1, elite[..., None].expand(-1, -1, z.shape[1]))
                    mu[idx] = 0.7 * e.mean(1) + 0.3 * mu[idx]
                    sigma[idx] = 0.7 * e.std(1).clamp(min=opt.sigma_min) + 0.3 * sigma[idx]
                    if len(active) > 4 * per_batch and (start // per_batch + 1) % 4 == 0:
                        print(
                            f"  CEM {iteration + 1}: {min(start + per_batch, len(active))}/{len(active)} goals processed "
                            f"({time.monotonic() - started:.0f}s)"
                        )
                print(f"{'Cell fill' if filling else 'Pose'} CEM {iteration + 1}: {int(found.sum())}/{len(z)} feasible")
            final, geometry = evaluate(qpos, best)
            # Recheck original and optimized latents together whenever a candidate
            # changed. An optimized pose must match the same geometric query.
            if not filling:
                changed = np.flatnonzero(torch.any(best != z, dim=1).cpu().numpy())
                if len(changed):
                    pair_q = np.repeat(qpos[changed], 2, axis=0)
                    pair_z = torch.stack((z[changed], best[changed]), dim=1).reshape(-1, z.shape[1])
                    result, g = evaluate(pair_q, pair_z)
                    repeat = np.repeat(changed, 2)
                    key, admissible, _ = objective(result, g, buckets[repeat], kinds[repeat], cells[repeat], cfg, opt, height_only=height_only[repeat])
                    keep_new = admissible.reshape(-1, 2)[:, 1] & (key.reshape(-1, 2)[:, 1] > key.reshape(-1, 2)[:, 0])
                    best[changed[~keep_new]] = z[changed[~keep_new]]
                    final, geometry = evaluate(qpos, best)
                    _, matching, _ = objective(final, geometry, buckets, kinds, cells, cfg, opt, height_only=height_only)
                    mismatch = ~matching & torch.any(best != z, dim=1).cpu().numpy()
                    if mismatch.any():
                        best[mismatch] = z[mismatch]
                        final, geometry = evaluate(qpos, best)
            else:
                # Generated tokens will reset from the reached pose, so their
                # reported scores must be measured from that pose, not the parent.
                reached = final["final_qpos"].copy()
                safe = np.isfinite(reached).all(1) & (np.linalg.norm(reached[:, 3:7], axis=1) > 1e-8)
                qpos[safe] = reached[safe]
                final, geometry = evaluate(qpos, best)
            _, matching, feasible = objective(final, geometry, buckets, kinds, cells, cfg, opt, filling, height_only)
            result = {
                k: final[k]
                for k in (
                    "stability_score",
                    "height_err",
                    "ref_height",
                    "hold_height",
                    "torso_tilt_deg",
                    "score",
                    "valid",
                    "final_qpos",
                    "final_qvel",
                )
            }
            result |= {k: v for k, v in final.items() if k.startswith("final_obs_")}
            result |= {
                "z": best.cpu().numpy(),
                "reference_qpos": qpos,
                "matching": matching,
                "feasible": feasible,
                "optimized": torch.any(best != z, dim=1).cpu().numpy(),
                "attempted": attempted,
                "cem_iterations": iterations,
                "seed_restarts": seed_restarts,
                "stability_initial": first["stability_score"],
                "height_err_initial": first["height_err"],
                "height_only": height_only,
                "seed_kind": seed_kind,
                "held_pose_kind": geometry.pose_kind.to_numpy(dtype=str),
                "held_bucket": geometry.bucket.to_numpy(dtype=str),
            }
            result |= {f"held_{k}": geometry[k].to_numpy() for k in (*HEIGHT_COLUMNS, "torso_tilt", "torso_pitch")}
            np.savez(output, **result)
            print(
                f"Final: {int(final['valid'].sum())}/{len(z)} valid, "
                f"{int((final['stability_score'] > opt.min_stability).sum())} above {opt.min_stability}, "
                f"{int(matching.sum())} matching postures/cells"
            )
    finally:
        backend.close()


if __name__ == "__main__":
    import sys

    worker(Path(sys.argv[1]), Path(sys.argv[2]))
