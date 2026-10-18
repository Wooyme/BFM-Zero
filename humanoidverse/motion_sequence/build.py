"""Extract stable runs and stable-ended transitions from ranked original motion frames."""

import argparse
import csv
import json
import math
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from humanoidverse.motion_sequence.database import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    DEFAULT_RANKING,
    Sequence,
    SequenceDatabase,
    extract_spans,
    sha256,
)
from scripts.render_bfm_motion import load_motions, motion_qpos

OUTPUT_NAMES = ("sequence_db.pkl", "sequences.csv", "summary.json")


def source_path(value: str, data_path: Path) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else data_path / path).resolve())


def build_database(data_path: Path = DEFAULT_DATA, ranking: Path = DEFAULT_RANKING, threshold: float = 83.0) -> SequenceDatabase:
    """Validate complete score coverage, then segment each original motion independently."""
    data_path, ranking = Path(data_path).resolve(), Path(ranking).resolve()
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    files = sorted(data_path.glob("*.pkl"))
    if not files or not data_path.is_dir():
        raise ValueError(f"No motion pickle files at {data_path}")
    columns = ["source_file", "motion_name", "frame_index", "score"]
    ranking_hash = sha256(ranking)
    rows = pd.read_csv(ranking, usecols=columns, dtype={"source_file": str, "motion_name": str}, keep_default_na=False)
    if rows.empty or (rows[["source_file", "motion_name"]] == "").any().any():
        raise ValueError("Ranking must contain source_file, motion_name and all original frames")
    for column in ("frame_index", "score"):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if not np.isfinite(rows[column].to_numpy()).all():
            raise ValueError(f"Ranking contains nonfinite {column}")
    if (rows.frame_index < 0).any() or (rows.frame_index != np.floor(rows.frame_index)).any():
        raise ValueError("frame_index must contain nonnegative integers")
    rows["source_file"] = rows.source_file.map(lambda value: source_path(value, data_path))
    if rows.duplicated(["source_file", "motion_name", "frame_index"]).any():
        raise ValueError("Ranking contains duplicate source/motion/frame scores")
    unknown_files = set(rows.source_file) - {str(path) for path in files}
    if unknown_files:
        raise ValueError(f"Ranking references unknown source files: {sorted(unknown_files)}")
    groups = dict(tuple(rows.groupby(["source_file", "motion_name"], sort=False)))
    manifest_path = ranking.with_name("manifest.json")
    expected_hashes = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected_hashes = {source_path(item["path"], data_path): item["sha256"] for item in manifest["identity"]["inputs"]}
    sequences, score_arrays, source_hashes = [], {}, {}
    for path in files:
        relative = path.relative_to(data_path).as_posix()
        source_hashes[relative] = sha256(path)
        if expected_hashes is not None and expected_hashes.get(str(path)) != source_hashes[relative]:
            raise ValueError(f"Source hash does not match ranking manifest: {path}")
        for name, motion in sorted(load_motions(path).items()):
            qpos, fps = motion_qpos(motion)
            frame_count = len(qpos)
            group = groups.pop((str(path), name), None)
            if group is None:
                raise ValueError(f"Missing ranking scores for {relative}:{name}")
            group = group.sort_values("frame_index")
            if (group.frame_index >= frame_count).any():
                raise ValueError(f"Ranking frame_index out of bounds for {relative}:{name}")
            if not np.array_equal(group.frame_index.to_numpy(), np.arange(frame_count)):
                raise ValueError(f"Missing ranking scores for frames of {relative}:{name}")
            scores = group.score.to_numpy(dtype=np.float64, copy=True)
            score_arrays[relative, name] = scores
            for kind, start, end in extract_spans(scores, threshold):
                span = scores[start : end + 1]
                sequences.append(
                    Sequence(
                        sequence_id=len(sequences),
                        kind=kind,
                        source_file=relative,
                        motion_name=name,
                        start_frame=start,
                        end_frame=end,
                        fps=fps,
                        start_score=float(scores[start]),
                        end_score=float(scores[end]),
                        min_score=float(span.min()),
                        mean_score=float(span.mean()),
                    )
                )
    if groups:
        raise ValueError(f"Ranking references unknown motions: {sorted(groups)}")
    if sha256(ranking) != ranking_hash:
        raise ValueError("Ranking changed during extraction; retry with a completed ranking")
    return SequenceDatabase(str(data_path), str(ranking), ranking_hash, threshold, source_hashes, score_arrays, sequences)


def check_output(output_dir: Path, data_path: Path, overwrite: bool) -> None:
    if output_dir.resolve().is_relative_to(data_path.resolve()):
        raise ValueError("Output must be outside the input motion directory")
    existing = [str(output_dir / name) for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs already exist: {existing}; use --overwrite to replace them")


def write_outputs(db: SequenceDatabase, output_dir: Path, overwrite: bool = False) -> dict:
    """Stage all artifacts before replacing outputs; the source motion data is never copied."""
    output_dir = Path(output_dir)
    check_output(output_dir, Path(db.data_path), overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = db.summary()
    with tempfile.TemporaryDirectory(prefix=".build-", dir=output_dir) as directory:
        staging = Path(directory)
        db.save(staging / "sequence_db.pkl")
        with (staging / "sequences.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[item.name for item in fields(Sequence)] + ["num_frames", "duration"])
            writer.writeheader()
            writer.writerows(sequence.to_record() for sequence in db.sequences)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        for name in OUTPUT_NAMES:
            (staging / name).replace(output_dir / name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--threshold", type=float, default=83.0, help="Stable iff score > threshold (default: 83)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    check_output(args.output_dir, args.data_path, args.overwrite)
    db = build_database(args.data_path, args.ranking, args.threshold)
    summary = write_outputs(db, args.output_dir, args.overwrite)
    print(f"Saved {len(db)} sequences to {args.output_dir.resolve()}")
    for kind, values in summary["groups"].items():
        print(f"  {kind}: {values['count']} sequences, {values['singletons']} singletons, max {values['max_frames']} frames")
    print(f"Unclosed unstable frames: leading={summary['unclosed_leading_frames']}, trailing={summary['unclosed_trailing_frames']}")


if __name__ == "__main__":
    main()
