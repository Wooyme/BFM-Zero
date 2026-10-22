import argparse
from pathlib import Path
from typing import Mapping

from humanoidverse.query.motion_query import Source

import numpy as np
import pandas as pd


def plot_base_height(groups: Mapping[str, pd.Series], path: Path, title: str, bins: int = 60):
    """Histograms of base_height, one per group on shared bins, saved to path: group "all" filled gray
    behind, the others as outlines; dashed lines: p5/p50/p95 of the last group."""
    from matplotlib.figure import Figure  # no pyplot: no backend needed

    heights = np.concatenate([np.asarray(h) for h in groups.values()])
    edges = np.linspace(heights.min(), heights.max(), bins + 1)
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.subplots()
    if "all" in groups:
        ax.hist(groups["all"], bins=edges, color="0.8", label=f"all ({len(groups['all'])})")
    for name, h in groups.items():
        if name != "all":
            ax.hist(h, bins=edges, histtype="step", lw=1.5, label=f"{name} ({len(h)})")
    for q in np.percentile(list(groups.values())[-1], [5, 50, 95]):
        ax.axvline(q, color="k", ls="--", lw=0.8)
    ax.set_xlabel("base_height (m)")
    ax.set_ylabel("frames")
    ax.set_title(title)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def record_base_height(model_folder: Path):
    latents = Path("humanoidverse/data/miyari_29dof/record_1022.pkl")
    record = Source(model_folder=model_folder, latents=latents)

    all_frames = record.scan().with_stability().collect()
    result = all_frames[all_frames.stability >= 90]
    print(np.max(result.base_height))
    print(np.min(result.base_height))
    print(result.base_height)

    plot_base_height({"all": all_frames.base_height, "stability>=95": result.base_height},
                     Path("outputs/queries") / f"{latents.stem}_base_height.png",
                     title=f"{latents.stem}: base_height (dashed: p5/p50/p95 of stability>=95)")


def lafan_base_height(model_folder: Path, pattern: str = "rec*"):
    frames = Source(model_folder=model_folder, lafan=pattern).scan().collect()
    motion = frames.origin.str.rsplit(":", n=1).str[0]
    groups = {name: frames.base_height[motion == name] for name in sorted(motion.unique())}
    if len(groups) > 1:
        groups["all"] = frames.base_height
    for name, h in groups.items():
        print(f"{name}: {len(h)} frames, min {h.min():.3f}, max {h.max():.3f}, mean {h.mean():.3f}")

    plot_base_height(groups, Path("outputs/queries") / f"lafan_{pattern.replace('*', '')}_base_height.png",
                     title=f"lafan_29dof/{pattern}: base_height (dashed: p5/p50/p95 of {list(groups)[-1]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", choices=["record", "lafan"], default="record")
    parser.add_argument("--lafan", default="rec*", help="comma-separated globs of lafan_29dof motions")
    args = parser.parse_args()

    model_folder = Path("new_model_for_training_code_inference")
    if args.source == "record":
        record_base_height(model_folder)
    else:
        lafan_base_height(model_folder, args.lafan)
