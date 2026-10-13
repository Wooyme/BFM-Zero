"""Walking backward: move_angle within 30 deg of +-180.

Run: uv run python scripts/queries/walk_backward.py --model-folder /path/to/model [--max-frames N] ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per metric worker process

from pathlib import Path

import tyro

from humanoidverse.query.motion_query import Source


def main(source: Source, output: Path | None = None):
    """Args:
    output: MotionDatabase pickle (default: model_folder/motion_query/motion_db.pkl).
    """
    db = (
        source.scan()
        .where("base_height >= 0.6 AND base_speed >= 0.2 AND abs(move_angle) >= 150")
        .to_database()
    )
    db.save(output or source.model_folder / "motion_query" / "motion_db.pkl")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.OmitArgPrefixes,))
