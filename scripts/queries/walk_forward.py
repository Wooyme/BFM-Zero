"""Walking forward at a medium speed, standing upright, fastest first.

Run: uv run python scripts/queries/walk_forward.py --model-folder /path/to/model [--max-frames N] ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per metric worker process

from pathlib import Path

import rich
import tyro

from humanoidverse.query.motion_query import Source


def main(source: Source, output: Path | None = None, top_k: int = 20):
    """Args:
    output: MotionDatabase pickle (default: model_folder/motion_query/motion_db.pkl).
    top_k: rows to print.
    """
    db = (
        source.scan()
        .where("base_height >= 0.6")
        .where("base_speed BETWEEN 0.5 AND 1.0 AND move_angle BETWEEN -30 AND 30")
        .order_by("base_speed DESC")
        .to_database()
    )
    db.save(output or source.model_folder / "motion_query" / "motion_db.pkl")
    rich.print(db.to_dataframe().head(top_k).to_string())


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.OmitArgPrefixes,))
