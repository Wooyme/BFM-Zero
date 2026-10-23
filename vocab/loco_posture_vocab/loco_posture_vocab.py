"""4096 tokens: 50% upright locomotion, 50% zero-velocity posture goals.

Run from the repository root:
    uv run python scripts/queries/loco_posture_vocab.py --model-folder MODEL --lafan '*' --latent

Posture queries combine pelvis, wrist, foot and knee heights (see
vocab/loco_posture_vocab/posture_vocab.py). First use grounded frames, then relax
body tolerances; only if coverage is short, translate airborne poses by their
lowest collision point. Finally CEM can fill holes from nearby-cell goal latents.
Coverage targets are holes inside the observed height support, not every
Cartesian combination of allowed bounds. --max-fill bounds synthesis work;
unfilled cells remain explicitly listed in summary.json. --no-cem skips all CEM.
Without --latent the script only selects frames and reports remaining holes.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tyro

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import compute_metrics, qpos_columns, qpos_of
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model, qpos_features
from humanoidverse.envs.g1_env_helper.queries import METRICS
from humanoidverse.query.motion_query import Source, zero_velocity
from vocab.loco_posture_vocab.posture_cem import Optimization
from vocab.loco_posture_vocab.posture_vocab import (
    BUCKETS,
    SIMILARITY,
    Postures,
    annotate,
    cell_keys,
    collision_heights,
    coverage_deficits,
    locomotion,
    materialize,
    neighbour_order,
    prepare_candidates,
    select_frames,
)


def save_summary(output, report):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=convert) + "\n")


def evaluate(source, qpos, z, buckets, kinds, cfg, opt, cem=True, cells=None, height_only=None, seed_kinds=None):
    """Keep Isaac initialization in a subprocess, also when Source itself uses Isaac."""
    filling = cells is not None
    if cells is None:
        cells = [(b, k, 0) for b, k in zip(buckets, kinds)]
    with tempfile.TemporaryDirectory(prefix="loco-posture-") as tmp:
        job, result = Path(tmp) / "job.npz", Path(tmp) / "result.npz"
        np.savez(
            job,
            qpos=np.asarray(qpos, np.float32),
            z0=np.asarray(z, np.float32),
            buckets=np.asarray(buckets, str),
            kinds=np.asarray(kinds, str),
            cells=np.asarray(cells, str),
            postures=json.dumps(asdict(cfg)),
            optimization=json.dumps(asdict(opt)),
            model_folder=str(source.model_folder.resolve()),
            device=source.device,
            filling=filling,
            cem=cem,
            height_only=np.zeros(len(qpos), dtype=bool) if height_only is None else np.asarray(height_only, dtype=bool),
            seed_kinds=np.asarray(kinds if seed_kinds is None else seed_kinds, str),
        )
        subprocess.run(
            [sys.executable, "-m", "vocab.loco_posture_vocab.posture_cem", str(job), str(result)],
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        with np.load(result, allow_pickle=False) as data:
            return dict(data)


def goal_latents(source, db, post_rows):
    """Always infer posture goals from zero-velocity next_obs, bypassing kept z/B."""
    import torch

    from humanoidverse.motion_latent import infer_z
    from humanoidverse.pose_hold.policy import GoalPolicy

    policy = GoalPolicy(source.model_folder, source.device)
    mask = np.zeros(len(db), dtype=bool)
    mask[post_rows] = True
    z = None
    for rows, posture in ((np.flatnonzero(~mask), False), (np.flatnonzero(mask), True)):
        if not len(rows):
            continue
        parts = []
        for start in range(0, len(rows), 1024):
            frames = db.select(rows[start : start + 1024]).frames
            parts.append(infer_z(policy, {"next_obs": frames["next_obs"]} if posture else frames, per_frame=True, device=source.device))
        inferred = np.concatenate(parts)
        if z is None:
            z = np.empty((len(db), inferred.shape[1]), dtype=np.float32)
        z[rows] = inferred
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if z is None or not np.isfinite(z).all():
        raise ValueError("Goal inference produced no latents or nonfinite latents")
    return z


class AugmentedFrames:
    """Original frames plus individually simulated, virtual source frames."""

    def __init__(self, frames):
        self.frames = frames
        self.first_generated = frames.num_frames
        self.generated = {}

    def gather_frames(self, idx):
        original = idx < self.first_generated
        raw = (
            self.frames.gather_frames(idx[original])
            if original.any()
            else self.frames.gather_frames(np.array([next(iter(self.generated.values()))["parent_frame"]]))
        )

        def gather(template, path=()):
            if isinstance(template, dict):
                return {k: gather(v, (*path, k)) for k, v in template.items()}
            out = np.zeros((len(idx), *template.shape[1:]), dtype=template.dtype)
            if original.any():
                out[original] = template
            for row in np.flatnonzero(~original):
                value = self.generated[int(idx[row])]
                for key in path:
                    value = value.get(key, {}) if isinstance(value, dict) else {}
                if not isinstance(value, dict):
                    out[row] = value
            return out

        return gather(raw)


def ordered_fill_tasks(tasks, post, limit):
    """Prioritize missing kinds and interleave bands, so one band cannot use the entire budget."""
    present = set(post.pose_kind)
    queues = {b: sorted([t for t in tasks if t[0] == b], key=lambda t: t[1] in present) for b in BUCKETS}
    out = []
    while any(queues.values()) and len(out) < limit:
        for bucket in BUCKETS:
            if queues[bucket] and len(out) < limit:
                out.append(queues[bucket].pop(0))
    return out


def fill_candidates(source, frames, post, targets, cfg, opt, total, report):
    tasks, _ = coverage_deficits(post, targets, cfg, total)
    tasks = ordered_fill_tasks(tasks, post, opt.max_fill)
    report["fill"] = {"requested": len(tasks), "budget": opt.max_fill, "attempts": [], "accepted": 0}
    if not tasks or post.empty:
        report["fill"]["reason"] = "no tasks or no posture goal available as a neighbouring seed"
        return post
    accepted, signatures = [], set()
    remaining = np.arange(len(tasks))
    for attempt in range(opt.fill_attempts):
        if not len(remaining):
            break
        eligible, parent_indices = [], []
        for i in remaining:
            bucket, kind, h = tasks[i]
            parents = fill_parent_indices(post, (bucket, kind, h), cfg, attempt, opt.fill_seeds)
            eligible.extend([i] * len(parents))
            parent_indices.extend(parents)
        if not eligible:
            break
        batch_tasks = [tasks[i] for i in eligible]
        parents = post.iloc[parent_indices].reset_index(drop=True)
        # Materialization also translates airborne seed observations consistently.
        seed_db = materialize(frames, parents)
        zero_velocity(seed_db)
        seed_db.frames["qvel"][:] = 0
        seeds = goal_latents(source, seed_db, np.arange(len(seed_db)))
        height_only = np.array([t[1] == "half_kneel" and k == "kneel"
                                for t, k in zip(batch_tasks, parents.pose_kind)])
        result = evaluate(
            source,
            seed_db.frames["qpos"],
            seeds,
            [t[0] for t in batch_tasks],
            [t[1] for t in batch_tasks],
            cfg,
            replace(opt, seed=opt.seed + attempt),
            cells=batch_tasks,
            height_only=height_only,
            seed_kinds=parents.pose_kind.to_numpy(),
        )
        accepted_tasks = set()
        for j in np.argsort(-result['stability_score'], kind='stable'):
            task_index = eligible[j]
            if task_index in accepted_tasks:
                continue
            key = np.round(result["z"][j], 6).tobytes()
            if (not result["valid"][j] or not result["matching"][j]
                    or result["stability_score"][j] <= opt.min_stability or key in signatures):
                continue
            qpos = result["final_qpos"][j : j + 1]
            metrics = compute_metrics(env_model(), qpos, np.zeros((1, qpos.shape[1] - 1)), list(METRICS), max_workers=1)
            row = parents.iloc[j].copy()
            for name, values in metrics.items():
                row[name] = values[0]
            for name, value in zip(qpos_columns(qpos.shape[1]), qpos[0]):
                row[name] = value
            frame = frames.first_generated + len(frames.generated)
            parent = int(parents.frame.iloc[j])
            row["frame"], row["parent_frame"] = frame, parent
            row["origin"], row["candidate_stage"], row["height_shift"] = f"cem:{frame}", "cem", 0.0
            row["collision_min_height"] = collision_heights(qpos)[0]
            classified = annotate(pd.DataFrame([row]), cfg, cfg.relax).iloc[0]
            # Only accept the actual reached pose, never assign a desired cell label.
            actual_key = cell_keys(pd.DataFrame([classified]), cfg)[0]
            if actual_key != tasks[task_index] and not (height_only[j] and actual_key[2] == tasks[task_index][2]):
                continue
            classified["actual_pose_kind"] = classified.pose_kind
            classified["compensation_seed_kind"] = parents.pose_kind.iloc[j] if height_only[j] else ""
            classified["height_only"] = bool(height_only[j])
            if height_only[j]:
                classified["bucket"], classified["pose_kind"] = tasks[task_index][:2]
            signatures.add(key)
            frames.generated[frame] = {
                "qpos": qpos[0],
                "qvel": result["final_qvel"][j],
                "z": result["z"][j],
                "next_obs": {k.removeprefix("final_obs_"): v[j] for k, v in result.items() if k.startswith("final_obs_")},
                "parent_frame": parent,
            }
            accepted.append(classified)
            accepted_tasks.add(task_index)
        retry = [i for i in remaining if i not in accepted_tasks]
        report["fill"]["attempts"].append({
            "attempt": attempt + 1, "tried": len(eligible), "tried_targets": len(set(eligible)), "remaining": len(retry),
            "parent_kinds": parents.pose_kind.tolist(), "parent_frames": parents.frame.tolist(),
            "stability": result['stability_score'].tolist(), "matching": result['matching'].tolist(),
            "held_height": result['hold_height'].tolist() if 'hold_height' in result else [],
            "held_wrists": np.stack([result['held_left_wrist_height'], result['held_right_wrist_height']], axis=1).tolist()
                if 'held_left_wrist_height' in result else [],
        })
        remaining = np.asarray(retry, dtype=int)
    report["fill"]["accepted"] = len(accepted)
    report["fill"]["failed_targets"] = [list(tasks[i]) for i in remaining]
    return pd.concat([post, pd.DataFrame(accepted)], ignore_index=True) if accepted else post


def fill_parent_indices(post, target, cfg, attempt, count):
    """Try diverse nearby seeds, prioritizing genuine half kneeling before double kneeling."""
    bucket, kind, h = target
    order = neighbour_order(post.base_height, post.bucket, post.pose_kind,
                            (h + .5) * cfg.height_cell, bucket, kind, attempt)
    preferred = 'kneel' if kind == 'half_kneel' and attempt > 0 else kind
    preferred_order = order[post.pose_kind.to_numpy()[order] == preferred]
    if len(preferred_order):
        order = preferred_order
    order = order[:max(256, count)]
    if len(order) <= count:
        return order.tolist()
    features = qpos_features(env_model(), qpos_of(post.iloc[order]), SIMILARITY)
    picked, distance = [0], np.full(len(order), np.inf)
    while len(picked) < count:
        distance = np.minimum(distance, ((features - features[picked[-1]]) ** 2).sum(axis=1))
        distance[picked] = -np.inf
        picked.append(int(distance.argmax()))
    return order[picked].tolist()


def zero_postures(db, rows):
    zero_velocity(db, rows)
    db.frames["qvel"][rows] = 0
    for name in ("base_vx", "base_vy", "base_vz", "base_speed", "com_speed", "yaw_rate", "cmd_speed"):
        if name in db.metrics:
            db.metrics[name][rows] = 0


def goal_modes(selected):
    """Preserve explicit compensation permissions through the final evaluation."""
    height_only = selected.get("height_only", pd.Series(False, index=selected.index)).fillna(False).to_numpy(dtype=bool)
    seeds = selected.pose_kind.to_numpy(dtype=str).copy()
    if "compensation_seed_kind" in selected:
        seeds = seeds.astype(object)
        seeds[height_only] = selected.compensation_seed_kind.to_numpy()[height_only]
    if np.any(height_only & ((seeds != "kneel") | (selected.pose_kind.to_numpy() != "half_kneel"))):
        raise ValueError("Half-kneeling compensation must use a kneeling seed")
    return dict(height_only=height_only, seed_kinds=np.asarray(seeds, str))


def compensate_half_kneel(source, db, selected, rows, result, cfg, opt, first_generated, report):
    """After own-pose search, validate related compensation from its reached goal pose."""
    kinds = selected.pose_kind.to_numpy()[rows]
    stable = result['valid'] & (result['stability_score'] > opt.min_stability)
    height_ok = np.abs(result['height_err']) <= opt.height_tol
    genuine = (result['held_pose_kind'] == 'half_kneel') & (result['held_bucket'] == '11_kneel')
    keep = stable & height_ok & genuine
    result['height_only'][keep & (kinds == 'half_kneel')] = False
    pending = np.flatnonzero((kinds == 'half_kneel') & ~keep)
    donors = np.flatnonzero((kinds == 'kneel') & (result['held_pose_kind'] == 'kneel')
                            & stable & height_ok & result['matching'])
    stats = dict(requested=len(pending), accepted=0, remaining=len(pending), attempts=[])
    report['half_kneel_compensation'] = stats
    if not len(pending) or not len(donors) or opt.max_fill <= 0:
        return result
    targets = {i: ('11_kneel', 'half_kneel', int(np.floor(selected.base_height.iloc[rows[i]] / cfg.height_cell + 1e-6)))
               for i in pending}
    allowed = set(sorted(set(targets.values()))[:opt.max_fill])
    pending = np.array([i for i in pending if targets[i] in allowed], dtype=int)
    unattempted = stats['requested'] - len(pending)
    rng = np.random.default_rng(opt.seed + 1000)
    for attempt in range(opt.fill_attempts):
        if not len(pending):
            break
        parents = []
        for i in pending:
            centre = (targets[i][2] + .5) * cfg.height_cell
            order = neighbour_order(result['hold_height'][donors], result['held_bucket'][donors],
                                    result['held_pose_kind'][donors], centre, '11_kneel', 'half_kneel', attempt=1)
            parents.append(donors[order[min(attempt, len(order)-1)]])
        parents = np.asarray(parents, dtype=int)
        seeds = result['z'][parents].copy()
        radius = np.linalg.norm(seeds, axis=1, keepdims=True)
        seeds += rng.normal(0, opt.sigma_min, seeds.shape).astype(seeds.dtype)
        seeds *= radius / np.maximum(np.linalg.norm(seeds, axis=1, keepdims=True), 1e-8)
        batch = evaluate(source, result['final_qpos'][parents], seeds, ['11_kneel'] * len(pending),
                         ['half_kneel'] * len(pending), cfg, replace(opt, seed=opt.seed+1000+attempt),
                         cells=[targets[i] for i in pending], height_only=np.ones(len(pending), dtype=bool),
                         seed_kinds=result['held_pose_kind'][parents])
        accepted = batch['valid'] & batch['matching'] & (batch['stability_score'] > opt.min_stability)
        stats['attempts'].append(dict(attempt=attempt+1, tried=len(pending), accepted=int(accepted.sum())))
        local = pending[accepted]
        positions = rows[local]
        if len(local):
            selected.loc[positions, 'requested_height'] = selected.base_height.iloc[positions].to_numpy()
            selected.loc[positions, 'requested_frame'] = selected.frame.iloc[positions].to_numpy()
            selected.loc[positions, 'parent_frame'] = selected.frame.iloc[rows[parents[accepted]]].to_numpy()
            virtual = np.arange(first_generated + stats['accepted'], first_generated + stats['accepted'] + len(local))
            selected.loc[positions, 'frame'] = virtual
            selected.loc[positions, 'origin'] = [f'cem_compensation:{i}' for i in virtual]
            selected.loc[positions, 'candidate_stage'] = 'cem_compensation'
            db.source_index[positions] = virtual
            db.frames['qpos'][positions] = batch['final_qpos'][accepted]
            for name in db.frames['next_obs']:
                if f'final_obs_{name}' in batch:
                    db.frames['next_obs'][name][positions] = batch[f'final_obs_{name}'][accepted]
            metrics = compute_metrics(env_model(), db.frames['qpos'][positions], np.zeros_like(db.frames['qvel'][positions]),
                                      list(METRICS), max_workers=1)
            geometry = annotate(pd.DataFrame(metrics), cfg, cfg.relax)
            for name in geometry:
                if name not in ('bucket', 'pose_kind') and pd.api.types.is_numeric_dtype(geometry[name]):
                    values = geometry[name].to_numpy(dtype=float)
                    selected.loc[positions, name] = values
                    if name in db.metrics:
                        db.metrics[name][positions] = values
            for name, values in zip(qpos_columns(db.frames['qpos'].shape[1]), db.frames['qpos'][positions].T):
                selected.loc[positions, name] = values
            selected.loc[positions, 'height_shift'] = 0.
            selected.loc[positions, 'collision_min_height'] = collision_heights(db.frames['qpos'][positions])
            if 'collision_min_height' in db.metrics:
                db.metrics['collision_min_height'][positions] = selected.collision_min_height.iloc[positions].to_numpy()
            zero_postures(db, positions)
            for name in result:
                if name in ('stability_initial', 'height_err_initial'):
                    continue
                if name in ('cem_iterations', 'seed_restarts'):
                    result[name][local] += batch[name][accepted] + (1 if name == 'seed_restarts' else 0)
                elif name == 'attempted':
                    result[name][local] |= batch[name][accepted]
                else:
                    result[name][local] = batch[name][accepted]
            result['optimized'][local] = True
            stats['accepted'] += len(local)
        pending = pending[~accepted]
    stats['remaining'] = len(pending) + unattempted
    return result


def deduplicate_posture_latents(source, db, selected, rows, z, result, cfg, opt, report):
    """Keep locomotion fixed; replace exact posture duplicates with validated nearby latents."""
    combined = z.copy()
    combined[rows] = result['z']
    _, inverse, counts = np.unique(combined, axis=0, return_inverse=True, return_counts=True)
    lookup = {int(pos): i for i, pos in enumerate(rows)}
    pending = []
    for group in np.flatnonzero(counts > 1):
        members = np.flatnonzero(inverse == group)
        walking = [p for p in members if int(p) not in lookup]
        keep = walking[0] if walking else members[0]
        pending.extend(lookup[int(p)] for p in members if p != keep and int(p) in lookup)
    stats = dict(before=int(len(counts)), attempted=len(pending), replaced=0)
    report['latent_duplicates'] = stats
    if not pending:
        return result
    repeat = np.repeat(np.asarray(pending, dtype=int), 12)
    rng = np.random.default_rng(opt.seed + 2000)
    seeds = result['z'][repeat].copy()
    radius = np.linalg.norm(seeds, axis=1, keepdims=True)
    seeds += rng.normal(0, opt.sigma_min, seeds.shape).astype(seeds.dtype)
    seeds *= radius / np.maximum(np.linalg.norm(seeds, axis=1, keepdims=True), 1e-8)
    positions = rows[repeat]
    batch = evaluate(source, db.frames['qpos'][positions], seeds, selected.bucket.to_numpy()[positions],
                     selected.pose_kind.to_numpy()[positions], cfg, opt, cem=False,
                     height_only=result['height_only'][repeat], seed_kinds=result['seed_kind'][repeat])
    valid = batch['valid'] & batch['matching'] & (batch['stability_score'] > opt.min_stability)
    valid &= np.abs(batch['height_err']) <= opt.height_tol
    for i, local in enumerate(pending):
        candidates = np.arange(12*i, 12*(i+1))
        candidates = candidates[valid[candidates]]
        if not len(candidates):
            continue
        choice = candidates[np.argmax(batch['stability_score'][candidates])]
        for name in result:
            if name in ('stability_initial', 'height_err_initial', 'cem_iterations', 'seed_restarts', 'attempted'):
                continue
            result[name][local] = batch[name][choice]
        result['optimized'][local] = True
        stats['replaced'] += 1
    combined[rows] = result['z']
    stats['after'] = int(len(np.unique(combined, axis=0)))
    return result


def main(
    source: Source,
    total: int = 4096,
    latent: bool = False,
    output: Path | None = None,
    scan_cache: Path | None = None,
    postures: Postures = Postures(),
    optimization: Optimization = Optimization(),
    cem: bool = True,
):
    """Select frames; --latent also infers, evaluates and optionally optimizes posture latents.

    CEM stability is a soft target. Numeric failures never enter an exported vocabulary.
    --height-edges sets the five nonoverlapping height boundaries. --max-fill and
    --fill-attempts bound neighbouring-cell synthesis; --no-align-airborne disables
    the collision-height fallback. Individual filters are documented in posture_vocab.py.
    """
    postures.validate()
    optimization.validate()
    if total < 12 or total % 2:
        raise ValueError("total must be even and at least 12 to cover all six postures")
    output = output or Path(__file__).resolve().parents[2] / "vocab" / "loco_posture_vocab"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "complete": False,
        "total": total,
        "latent": latent,
        "cem": cem,
        "postures": asdict(postures),
        "optimization": asdict(optimization),
    }
    try:
        if scan_cache is None:
            query = source.scan()
            frames_source, raw = query.frames, query.collect()
        else:
            from vocab.loco_posture_vocab.scan_cache import load_scan

            frames_source, raw = load_scan(source, scan_cache)
        if raw.empty:
            raise ValueError("The source contains no frames")
        table, post, targets, candidate_report = prepare_candidates(raw, postures, total // 2)
        report["candidates"] = candidate_report
        save_summary(output, report)
        frames = AugmentedFrames(frames_source)
        # Standalone airborne walking frames are excluded too; upright walking
        # otherwise retains multi_vocab's locomotion conditions.
        loco = locomotion(table.loc[table.collision_min_height.abs() <= postures.ground_tol]).copy()
        loco["cmd_speed"] = np.maximum(np.abs(loco.base_vx), np.abs(loco.base_vy))
        report["loco_candidates"] = len(loco)
        if len(loco) < total // 2:
            raise ValueError(f"Locomotion has {len(loco)} candidates for {total // 2}; scan more frames")
        if latent and cem and optimization.fill_cells:
            post = fill_candidates(source, frames, post, targets, postures, optimization, total // 2, report)
            save_summary(output, report)
        selected = select_frames(loco, post, total, postures, output / "plots")
        selected["parent_frame"] = selected.get("parent_frame", selected.frame).fillna(selected.frame)
        db = materialize(frames, selected)
        rows = np.flatnonzero(selected.grp.to_numpy() == "posture")
        zero_postures(db, rows)
        db.save(output / "initial_motion_db.pkl")
        if latent:
            z = goal_latents(source, db, rows)
            for i, frame in enumerate(selected.frame):
                if frame in frames.generated:
                    z[i] = frames.generated[frame]["z"]
            np.save(output / "initial_z.npy", z)
            result = evaluate(
                source,
                db.frames["qpos"][rows],
                z[rows],
                selected.bucket.to_numpy()[rows],
                selected.pose_kind.to_numpy()[rows],
                postures,
                optimization,
                cem=cem,
                **goal_modes(selected.iloc[rows]),
            )
            # Refill numeric failures from unused candidates of the same posture.
            used = set(selected.frame)
            for refill in range(3):
                bad = np.flatnonzero(~result["valid"] | ~np.isfinite(result["z"]).all(1))
                if not len(bad):
                    break
                for local in bad:
                    pos = rows[local]
                    spare = post.loc[(post.pose_kind == selected.pose_kind.iloc[pos]) & ~post.frame.isin(used)]
                    if spare.empty:
                        raise ValueError(f"No numeric-valid replacement for {selected.pose_kind.iloc[pos]}")
                    replacement = spare.iloc[0]
                    selected.loc[pos, replacement.index] = replacement
                    used.add(replacement.frame)
                db = materialize(frames, selected)
                zero_postures(db, rows)
                fresh = goal_latents(source, db.select(rows[bad]), np.arange(len(bad)))
                retry = evaluate(
                    source,
                    db.frames["qpos"][rows[bad]],
                    fresh,
                    selected.bucket.to_numpy()[rows[bad]],
                    selected.pose_kind.to_numpy()[rows[bad]],
                    postures,
                    replace(optimization, seed=optimization.seed + refill + 1),
                    cem=cem,
                    **goal_modes(selected.iloc[rows[bad]]),
                )
                for name in result:
                    result[name][bad] = retry[name]
            if not result["valid"].all() or not np.isfinite(result["z"]).all():
                raise ValueError("Numeric failures remain after replacing posture candidates")
            if cem and optimization.fill_cells:
                result = compensate_half_kneel(source, db, selected, rows, result, postures, optimization,
                                               max(frames.first_generated + len(frames.generated), int(selected.frame.max())+1), report)
            if cem:
                result = deduplicate_posture_latents(source, db, selected, rows, z, result, postures, optimization, report)
            z[rows] = result["z"]
            db.frames["z"] = z
            for name, values in result.items():
                if values.ndim == 1:
                    if values.dtype.kind in "biuf":
                        db.labels.pop(name, None)
                        db.metrics[name] = np.full(len(db), np.nan, dtype=np.float32)
                        db.metrics[name][rows] = values
                    else:
                        labels, codes = np.unique(values.astype(str), return_inverse=True)
                        db.labels[name] = labels.tolist()
                        db.metrics[name] = np.full(len(db), np.nan, dtype=np.float32)
                        db.metrics[name][rows] = codes
            selected.loc[rows, "height_only"] = result["height_only"]
            selected.loc[rows, "compensation_seed_kind"] = result["seed_kind"]
            selected.loc[rows, "actual_pose_kind"] = result["held_pose_kind"]
            stable = result["stability_score"] > optimization.min_stability
            height_ok = np.abs(result["height_err"]) <= optimization.height_tol
            report["quality"] = {
                "evaluated": len(rows),
                "stable": int(stable.sum()),
                "below_target": int((~stable).sum()),
                "stability_pass_rate": float(stable.mean()),
                "height_failures": int((~height_ok).sum()),
                "pose_mismatches": int((~result["matching"]).sum()),
                "optimized": int(result["optimized"].sum()),
                "neighbour_restarts": int(result["seed_restarts"].sum()),
                "height_only_compensation": int(result["height_only"].sum()),
                "held_kinds": pd.Series(result["held_pose_kind"]).value_counts().to_dict(),
                "stability_quantiles": np.quantile(result["stability_score"], [0, 0.25, 0.5, 0.75, 1]).tolist(),
            }
            grouped = {bucket: z[selected.bucket.to_numpy() == bucket] for bucket in sorted(selected.bucket.unique())}
            joblib.dump(grouped, output / "latent_z.pkl")
            np.savez_compressed(output / "posture_rollouts.npz", **result)
        else:
            # A source's optimized z/B is not the newly requested zero-velocity goal.
            db.frames.pop("z", None)
            db.frames.pop("B", None)
        db.recipe = json.dumps(
            {
                "script": "loco_posture_vocab.py",
                "source": asdict(source),
                "postures": asdict(postures),
                "optimization": asdict(optimization),
            },
            default=str,
        )
        db.save(output / "motion_db.pkl")
        chosen_post = selected.loc[selected.grp == "posture"]
        missing = sorted(set(targets) - set(cell_keys(chosen_post, postures)))
        report |= {
            "complete": True,
            "selected": len(db),
            "selected_buckets": selected.bucket.value_counts().to_dict(),
            "selected_kinds": selected.pose_kind.value_counts().to_dict(),
            "missing_cells": [list(k) for k in missing],
            "selected_stages": chosen_post.candidate_stage.value_counts().to_dict(),
        }
        selected.drop(columns=[c for c in selected if c.startswith("qpos_")]).to_csv(output / "selection.csv", index=False)
        save_summary(output, report)
        print(json.dumps({k: report[k] for k in ("selected", "selected_buckets", "selected_kinds", "selected_stages")}, ensure_ascii=False))
        print(f"Uncovered support cells: {len(missing)}; saved {output}")
        if latent:
            print(f"Posture quality: {report['quality']}")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_summary(output, report)
        raise


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.OmitArgPrefixes,))
