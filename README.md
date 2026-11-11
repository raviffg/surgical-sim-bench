# surgical-sim-bench

Physics bench for capsule-preserving endoscopic procedures. Three coupled
solvers: anatomy as a signed distance field, tissue as a bioheat field, tool as
a six-axis wrench.

## Modules

- `sim/sdf_capsule_tracker.py` - Euclidean SDF on 0.6 mm isotropic T2 MRI grids, signed clearance `d(x)` and surface normals `grad d(x)`, sub-millimeter capsule violation tracking.
- `sim/pennes_bioheat_solver.py` - 3D finite-difference Pennes solver with Beer-Lambert laser source and Arrhenius damage integral `Omega(t)`, comparing Ho:YAG (2120 nm, 0.4 mm) against Thulium fiber (1940 nm, 0.15 mm).
- `sim/wrench_telemetry.py` - 6-axis wrench synthesiser, `M = r x F`, at 1 kHz with bias, noise and anti-alias filtering.

## Why these three

Clearance, heat and force are the only quantities that decide whether a capsule
survives a case. Mesh intersection is a rendering problem; a signed distance
crossing is a physical one.

## Run

```
python sim/pennes_bioheat_solver.py
python sim/wrench_telemetry.py
```

`SimpleITK` is optional and only used to read NIfTI volumes; the SDF, heat and
wrench paths run on `numpy` and `scipy` alone.

## Layout

- `sim/` - solvers
- `data/` - local volumes, ignored by git
- `out/` - generated fields and telemetry, ignored by git
