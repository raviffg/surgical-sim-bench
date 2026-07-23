"""Sub-millimeter SDF anatomy tracking for capsule-preserving procedures.

Builds a Euclidean signed distance field from a segmented T2 MRI label volume on
an isotropic 0.6 mm voxel grid, then tracks a surgical capsule as an implicit
surface. Every instrument sample becomes a signed clearance ``d(x)`` and a
surface normal ``n(x) = grad d(x)``, so a boundary violation is a scalar
crossing instead of a mesh-intersection guess.

Sign convention: ``d < 0`` inside the capsule, ``d > 0`` in surrounding tissue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

try:  # optional reader; the tracker itself stays dependency-light
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None

VOXEL_MM = 0.6
VIOLATION_MARGIN_MM = 0.3  # half a voxel: sub-millimeter, pre-contact


@dataclass(frozen=True)
class VoxelGrid:
    spacing_mm: float = VOXEL_MM
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Capsule:
    """Finite cylinder: axis from ``p0`` to ``p1`` with hemispherical caps."""

    p0_mm: np.ndarray
    p1_mm: np.ndarray
    radius_mm: float

    def surface_points(self, n_theta: int = 24, n_axial: int = 12) -> np.ndarray:
        axis = self.p1_mm - self.p0_mm
        length = float(np.linalg.norm(axis))
        unit = axis / length if length else np.array([0.0, 0.0, 1.0])
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, unit))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(unit, seed)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(unit, e1)
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        ring = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
        centres = self.p0_mm + np.linspace(0.0, 1.0, n_axial)[:, None] * axis
        return (centres[:, None, :] + self.radius_mm * ring[None, :, :]).reshape(-1, 3)


class SDFVolume:
    """Euclidean SDF on a regular grid with trilinear evaluation."""

    def __init__(self, values: np.ndarray, grid: VoxelGrid = VoxelGrid()) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.grid = grid
        grads = np.gradient(self.values, grid.spacing_mm, edge_order=1)
        self._grad = np.stack(grads[::-1], axis=-1)  # -> (x, y, z)

    @classmethod
    def from_mask(cls, mask: np.ndarray, grid: VoxelGrid = VoxelGrid()) -> "SDFVolume":
        mask = np.asarray(mask, dtype=bool)
        sampling = (grid.spacing_mm,) * 3
        outside = ndimage.distance_transform_edt(~mask, sampling=sampling)
        inside = ndimage.distance_transform_edt(mask, sampling=sampling)
        return cls(outside - inside, grid)

    @classmethod
    def from_nifti(cls, path: str) -> "SDFVolume":
        if sitk is None:
            raise RuntimeError("SimpleITK is required to read NIfTI volumes")
        image = sitk.ReadImage(path)
        spacing = image.GetSpacing()[::-1]
        if max(spacing) - min(spacing) > 1e-6:
            raise ValueError(f"anisotropic voxels not supported: {spacing}")
        return cls.from_mask(sitk.GetArrayFromImage(image) > 0, VoxelGrid(spacing[0]))

    def _trilinear(self, field: np.ndarray, points_mm: np.ndarray) -> np.ndarray:
        shape = np.array(field.shape)
        idx = np.clip((np.atleast_2d(points_mm) - self.grid.origin_mm)
                      / self.grid.spacing_mm, 0.0, shape - 1.0)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, shape - 1)
        frac = idx - lo
        out = np.zeros(len(idx))
        for corner in range(8):
            pick = [(hi, lo)[(corner >> axis) & 1][:, axis] for axis in range(3)]
            weight = np.prod([(frac[:, a] if (corner >> a) & 1 else 1.0 - frac[:, a])
                              for a in range(3)], axis=0)
            out += field[pick[0], pick[1], pick[2]] * weight
        return out

    def clearance(self, points_mm: np.ndarray) -> np.ndarray:
        """Signed distance d(x) in millimetres; negative inside the capsule."""
        return self._trilinear(self.values, points_mm)

    def normals(self, points_mm: np.ndarray) -> np.ndarray:
        """Unit surface normals n(x) = grad d(x) / |grad d(x)|."""
        grad = np.stack([self._trilinear(self._grad[..., c], points_mm)
                         for c in range(3)], axis=-1)
        norm = np.linalg.norm(grad, axis=-1, keepdims=True)
        return grad / np.where(norm < 1e-9, 1.0, norm)


@dataclass(frozen=True)
class ViolationFrame:
    min_clearance_mm: float
    penetration_mm: float
    violations: int
    worst_point_mm: tuple[float, float, float]
    normal: tuple[float, float, float]


class CapsuleTracker:
    """Tracks instrument samples against a capsule boundary and accumulates harm."""

    def __init__(self, sdf: SDFVolume, capsule: Capsule,
                 margin_mm: float = VIOLATION_MARGIN_MM) -> None:
        self.sdf, self.capsule = sdf, capsule
        self.margin_mm = margin_mm
        self._worst_penetration = 0.0
        self._violation_total = 0

    def registration_error_mm(self) -> float:
        """Mean |d| over the analytic capsule surface; should be sub-voxel."""
        return float(np.mean(np.abs(self.sdf.clearance(self.capsule.surface_points()))))

    def track(self, points_mm: Iterable[Iterable[float]]) -> ViolationFrame:
        pts = np.asarray(list(points_mm), dtype=np.float64)
        d = self.sdf.clearance(pts)
        normals = self.sdf.normals(pts)
        penetration = float(max(0.0, -np.min(d)))
        self._worst_penetration = max(self._worst_penetration, penetration)
        self._violation_total += int(np.count_nonzero(d < self.margin_mm))
        worst = int(np.argmin(d))
        return ViolationFrame(float(np.min(d)), penetration,
                              int(np.count_nonzero(d < self.margin_mm)),
                              tuple(float(c) for c in pts[worst]),
                              tuple(float(c) for c in normals[worst]))

    def summary(self) -> dict[str, float]:
        return {"registration_error_mm": self.registration_error_mm(),
                "worst_penetration_mm": self._worst_penetration,
                "violation_samples": float(self._violation_total),
                "voxel_mm": self.sdf.grid.spacing_mm}
