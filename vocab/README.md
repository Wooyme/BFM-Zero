# Vocab query workspaces

Create one workspace per query at `vocab/<query-name>/`. Keep query configuration,
query-specific source code, inspection images, sampling results, caches, CEM outputs
and rollout results together, with separate subdirectories for successive runs.
Business rules, optimization objectives and dedicated tools belong to their workspace;
only reusable infrastructure and thin compatibility entry points remain in shared directories.
Models and source motions stay in their original locations.

Current workspace: [loco_posture_vocab](loco_posture_vocab/README.md).
