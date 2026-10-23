"""Inspect strict knee/ankle queries on cached LAFAN frames with two-view contact sheets."""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase, qpos_of
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model, qpos_features, remove_yaw
from vocab.loco_posture_vocab.posture_vocab import SIMILARITY, Postures, annotate


def representative_rows(table, count, model):
    table = table.reset_index(drop=True)
    if len(table) <= count:
        return table
    rng = np.random.default_rng(0)
    picked = []
    motion = table.origin.str.rsplit(':', n=1).str[0]
    for name in sorted(motion.unique()):
        group = np.flatnonzero(motion.to_numpy() == name)
        picked.append(int(group[len(group)//2]))
        if len(picked) >= count:
            break
    for i in rng.permutation(len(table)):
        if len(picked) >= min(count, len(motion.unique())+4):
            break
        if i not in picked:
            picked.append(int(i))
    features = qpos_features(model, qpos_of(table), SIMILARITY)
    distance = np.full(len(table), np.inf)
    for i in picked:
        distance = np.minimum(distance, ((features-features[i])**2).sum(1))
    while len(picked) < count:
        distance[picked] = -1
        i = int(distance.argmax())
        picked.append(i)
        distance = np.minimum(distance, ((features-features[i])**2).sum(1))
    return table.iloc[picked]


def save_database(table, path, cfg):
    q = qpos_of(table).astype(np.float32)
    metrics, labels = {}, {}
    for name in table:
        if name.startswith('qpos_') or name in ('frame',):
            continue
        if pd.api.types.is_numeric_dtype(table[name]):
            metrics[name] = table[name].to_numpy(dtype=float, na_value=np.nan)
        else:
            values = table[name].astype('string').fillna('unclassified').to_numpy(dtype=str)
            unique, codes = np.unique(values, return_inverse=True)
            labels[name], metrics[name] = unique.tolist(), codes
    metrics['is_kneel'] = (table.pose_kind == 'kneel').to_numpy(dtype=float)
    metrics['is_half_kneel'] = (table.pose_kind == 'half_kneel').to_numpy(dtype=float)
    MotionDatabase(metrics, table.frame.to_numpy(dtype=int),
                   frames=dict(qpos=q, qvel=np.zeros((len(q), q.shape[1]-1), np.float32)),
                   labels=labels, recipe=json.dumps(asdict(cfg))).save(path)


def sheet(model, renderer, table, title, path):
    width, height = renderer.width, renderer.height
    columns, card_height = 3, height+98
    rows = int(np.ceil(len(table)/columns))
    canvas = Image.new('RGB', (2*width*columns, card_height*rows+60), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    draw.text((12,8), title, fill='black', font=font)
    draw.text((12,30), 'Red: knees; blue: ankles; green: wrists. Heights in m. Two views; root yaw removed.', fill='black', font=font)
    data, camera, option = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    option.sitegroup[:] = 0
    camera.distance, camera.elevation = 1.55, -15
    camera.lookat[:] = [0,0,.40]
    body_names = ['left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link',
                  'left_wrist_roll_link','right_wrist_roll_link']
    for i, (_, row) in enumerate(table.iterrows()):
        q = qpos_of(pd.DataFrame([row]))[0]
        q[:2] = 0
        q[3:7] = remove_yaw(q[None,3:7])[0]
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model,data)
        x,y = (i%columns)*2*width, 60+(i//columns)*card_height
        for view, angle in enumerate((125,35)):
            camera.azimuth = angle
            renderer.update_scene(data,camera=camera,scene_option=option)
            picture = Image.fromarray(renderer.render())
            overlay = ImageDraw.Draw(picture)
            cameras = renderer.scene.camera
            position = (cameras[0].pos+cameras[1].pos)/2
            forward = (cameras[0].forward+cameras[1].forward)/2
            up = (cameras[0].up+cameras[1].up)/2
            right = np.cross(forward,up)
            focal = .5*height/np.tan(np.deg2rad(model.vis.global_.fovy)/2)
            for j,name in enumerate(body_names):
                delta = data.xpos[model.body(name).id]-position
                depth = delta@forward
                px,py = width/2+focal*(delta@right)/depth, height/2-focal*(delta@up)/depth
                color = (245,45,20) if j<2 else ((20,150,255) if j<4 else (20,190,65))
                overlay.ellipse((px-5,py-5,px+5,py+5),fill=color,outline='white',width=1)
            canvas.paste(picture,(x+view*width,y))
        draw.text((x+5,y+height+3),f'{i+1}. {row.origin}',fill='black',font=font)
        draw.text((x+5,y+height+25),f'H={row.base_height:.3f}  knees={row.left_knee_height:.3f}, {row.right_knee_height:.3f}',fill='black',font=font)
        draw.text((x+5,y+height+47),f'ankles={row.left_foot_height:.3f}, {row.right_foot_height:.3f}',fill='black',font=font)
        draw.text((x+5,y+height+69),f'wrists={row.left_wrist_height:.3f}, {row.right_wrist_height:.3f}',fill='black',font=font)
    canvas.save(path)


def main(
    cache: Path = Path('vocab/loco_posture_vocab/lafan_scan_cache/table.pkl'),
    output: Path = Path('vocab/loco_posture_vocab/inspection'),
    samples: int = 12,
):
    cfg = Postures()
    cfg.validate()
    if samples < 1:
        raise ValueError('samples must be positive')
    output.mkdir(parents=True,exist_ok=True)
    raw = joblib.load(cache)
    table = annotate(raw.loc[raw.collision_min_height.abs()<=cfg.ground_tol],cfg)
    old = ((table.base_height>=.25)&(table.base_height<.45)&(table.w_min<=table.base_height+.25)
           &(table.w_max<=table.base_height+.45)&(table.f_min<=.15)&(table.f_max<=.30)
           &(table.k_min<=.18)&(table.k_max<=table.base_height+.15)&(table.torso_tilt<=100))
    rejected = table.loc[old & table.bucket.isna() & (table.f_max<=cfg.kneel_ankle_max)
                         & (table.f_max-table.f_min<=cfg.ankle_level_tol) & (table.k_min-table.f_max>cfg.half_low_knee_max)]
    model = env_model()
    groups = {'kneel':table.loc[table.pose_kind=='kneel'], 'half_kneel':table.loc[table.pose_kind=='half_kneel'],
              'rejected_deep_squat':rejected}
    selected = []
    with mujoco.Renderer(model,height=240,width=260) as renderer:
        for kind, candidates in groups.items():
            if candidates.empty:
                continue
            chosen = representative_rows(candidates,samples,model).copy()
            chosen['sample_group'] = kind
            selected.append(chosen)
            sheet(model,renderer,chosen,kind,output/f'{kind}.png')
            print(f'{kind}: {len(candidates)} candidates; rendered {len(chosen)}',flush=True)
    sample_table = pd.concat(selected,ignore_index=True)
    sample_table.drop(columns=[c for c in sample_table if c.startswith('qpos_')]).to_csv(output/'samples.csv',index=False)
    save_database(sample_table,output/'query_samples.pkl',cfg)
    accepted = table.loc[table.pose_kind.isin(['kneel','half_kneel'])]
    save_database(accepted,output/'query_candidates.pkl',cfg)
    summary = dict(scanned=len(raw),grounded=len(table),postures=asdict(cfg),
                   candidates={k:len(v) for k,v in groups.items()},samples=sample_table.sample_group.value_counts().to_dict())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    import tyro
    tyro.cli(main)
