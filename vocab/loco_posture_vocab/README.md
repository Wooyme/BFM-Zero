# Locomotion and posture vocab workspace

The query-specific implementation lives here:

- `posture_vocab.py`: posture queries, category rules, coverage and sampling.
- `posture_cem.py`: posture-specific objectives, CEM and compensation rules.
- `loco_posture_vocab.py`: vocabulary generation and export.
- `inspect_posture_queries.py`: query-specific visual inspection.
- `scan_cache.py`: source-scan cache integration for this query.

Run from the project root with `python -m vocab.loco_posture_vocab.loco_posture_vocab`.
The former `scripts/queries/` commands remain thin compatibility entry points.

Locomotion candidates come only from `walk*` and `run*` motions and prefer the
most symmetric half within each velocity cell (using `SymmetryError` with
`SIMILARITY` weights, never fewer than the quota). Posture sampling prefers
`wrist_pelvis_distance`.

This workspace was moved from `new_model_for_training_code_inference/loco_posture_vocab/`.
The original compatibility symlink has been removed. `workspace.json` records paths
relative to this directory; `migration_manifest.json` records the original file sizes and SHA-256 hashes.
Historical logs, manifests and code snapshots retain their original paths and contents.
Auxiliary scripts that inferred model location from directory depth now read `workspace.json`;
their original versions and both hashes are retained in `migration_original_scripts/` and the migration manifest.

- `latest/` → `full_lafan_walk_run_symmetry_20261023/`: latest validated 4096-token vocabulary.
- `latest/latent_z.pkl`: final vocabulary, 2048 locomotion and 2048 posture latents.
- `latest/motion_db.pkl`: selected goal frames and final latents.
- `latest/default_rollout_last.pkl`: all 4096 last frames after rollout from `default_pos`.
- `latest/audit.json`, `latest/quality.png`: quality and actual rollout coverage.
- `latest/visual_check/`: final posture inspection sheets and viewer-readable samples.
- `latest/visual_check/loco_goals.png`: selected walk/run locomotion samples.
- `latest/export_checks.json`: source, uniqueness, alignment and default-rollout checks.
- `full_lafan_wrist_floor/`: previous validated vocabulary, preserved for comparison.
- `query_support_wrist_floor/`: latest source-query visual inspection.
- `lafan_scan_cache/`: reusable full-data scan.
- Other `full_lafan_*` and `query_support_*` directories: previous runs and query iterations.

Run workspace tools from the project root, using a fresh run directory:

```bash
uv run python -m vocab.loco_posture_vocab.loco_posture_vocab \
    --model-folder new_model_for_training_code_inference --lafan '*' --latent \
    --scan-cache vocab/loco_posture_vocab/lafan_scan_cache \
    --output vocab/loco_posture_vocab/runs/run_001

uv run python -m vocab.loco_posture_vocab.inspect_posture_queries \
    --cache vocab/loco_posture_vocab/lafan_scan_cache/table.pkl \
    --output vocab/loco_posture_vocab/inspection

uv run python -m humanoidverse.motion_viewer \
    --db vocab/loco_posture_vocab/latest/default_rollout_last.pkl
```

The latest run has 4096 unique latents; all 2048 locomotion tokens come from `walk*`
and `run*` motions with symmetry preference. All 4096 final latents were rolled out
from the configured default pose for 5 seconds and are numerically valid.
Posture Stability exceeds 95 for 1896/2048 tokens; goal coverage is 28/28 cells and
strict posture-rollout coverage is 26/28 cells. The remaining cells (supine height
cell 0 and stand height cell 17) are recorded in `latest/audit.json`.
