# Extension points and future work

## Scope
This note records the design considerations for physics simulation bench for capsule-preserving endoscopic procedures.

## Current project context
The repository currently implements MRI-grid signed-distance fields, capsule clearance, Pennes bioheat, laser deposition, six-axis wrench telemetry. The main code paths are `sim/sdf_capsule_tracker.py` and `sim/pennes_bioheat_solver.py`. The principal engineering concerns are voxel spacing, surface normals, thermal boundary conditions, Arrhenius damage, force calibration, time-step stability.

## Design questions
- What inputs, units, coordinate conventions, and preconditions does this part of the system require?
- Which state is authoritative, and which values are derived?
- How are invalid, stale, incomplete, or physically implausible inputs represented?
- Which assumptions are supported by source code or measured evidence, and which remain provisional?
- What observable evidence allows a reviewer to reproduce the behavior?

## Recommended design record
Document the selected approach, alternatives considered, rationale, expected failure modes, and any measurable trade-offs. Keep numerical constants linked to source or calibration evidence; do not copy unexplained values into a second source of truth.

## Verification
Pair this design note with the relevant scenarios under `docs/verification/scenarios/`. A design decision is not considered verified merely because the code compiles.

## Status
Working reference. Confirm implementation details against the source before using this note as a hardware, clinical, or production specification.
