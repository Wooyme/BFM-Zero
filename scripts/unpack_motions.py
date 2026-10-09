"""Split a motion dictionary into one joblib .pkl file per motion.

Usage:
    python scripts/unpack_motions.py
    python scripts/unpack_motions.py path/to/motions.pkl -o path/to/output

Each output contains {motion_name: motion_data}, matching the motion library's
directory input format. Only load pickle files from trusted sources.
"""

import argparse
from pathlib import Path

import joblib


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "humanoidverse/data/lafan_29dof.pkl"


def unpack_motions(source: Path, output_dir: Path, overwrite: bool = False) -> int:
    motions = joblib.load(source)
    if not isinstance(motions, dict) or not motions:
        raise ValueError("Expected a non-empty dictionary of motions.")

    # Validate all names and destinations before writing any files. Dots are
    # excluded because the motion loader takes the name before the first dot.
    destinations = []
    for name, motion in motions.items():
        if not isinstance(name, str) or not name or any(c in name for c in '/\\.\0'):
            raise ValueError(f"Motion name cannot be used as a filename: {name!r}")
        if not isinstance(motion, dict):
            raise ValueError(f"Expected a dictionary for motion {name!r}.")
        destination = output_dir / f"{name}.pkl"
        if destination.resolve() == source.resolve():
            raise ValueError(f"Output would overwrite the source: {destination}")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError(f"Output is not a regular file: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {destination}; use --overwrite to replace it.")
        destinations.append((name, destination))

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, destination in destinations:
        with destination.open("wb" if overwrite else "xb") as stream:
            joblib.dump({name: motions[name]}, stream)
    return len(destinations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help="Input .pkl file (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path, help="Output directory (default: input path without its extension)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing motion files")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.input.with_suffix("")
    try:
        count = unpack_motions(args.input, output_dir, args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Extracted {count} motions to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
