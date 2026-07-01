"""3D finite-difference Pennes bioheat solver with Arrhenius damage tracking.

    rho cp dT/dt = div(k grad T) + w_b rho_b c_b (T_a - T) + Q_laser

The laser source is a Beer-Lambert absorption profile with a Gaussian radial
footprint, so optical penetration depth is the only parameter separating Ho:YAG
(2120 nm, 0.4 mm) from Thulium fiber laser (1940 nm, 0.15 mm). Damage is
integrated with the Arrhenius model and thresholded at Omega = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_GAS = 8.314        # J/mol/K
ARRHENIUS_A = 3.1e98  # 1/s
ARRHENIUS_EA = 6.28e5  # J/mol


@dataclass(frozen=True)
class Tissue:
    k: float = 0.50            # W/m/K
    rho: float = 1050.0        # kg/m^3
    cp: float = 3600.0         # J/kg/K
    perfusion: float = 2000.0  # W/m^3/K, w_b rho_b c_b lumped
    t_arterial: float = 37.0   # deg C

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.cp)


@dataclass(frozen=True)
class LaserSource:
    name: str
    wavelength_nm: float
    penetration_mm: float
    power_w: float = 0.7
    fibre_radius_mm: float = 0.3

    @property
    def mu_a(self) -> float:
        return 1.0 / (self.penetration_mm * 1e-3)


HO_YAG = LaserSource("Ho:YAG", 2120.0, 0.4)
THULIUM_FIBER = LaserSource("Thulium fiber", 1940.0, 0.15)


class PennesSolver:
    """Explicit finite-difference solver on a uniform Cartesian grid."""

    def __init__(self, shape: tuple[int, int, int] = (41, 41, 61),
                 dx_mm: float = 0.2, tissue: Tissue = Tissue()) -> None:
        self.tissue, self.dx = tissue, dx_mm * 1e-3
        self.temperature = np.full(shape, tissue.t_arterial, dtype=np.float64)
        self.damage = np.zeros(shape, dtype=np.float64)

    @property
    def dt(self) -> float:
        """Explicit stability limit for the 7-point Laplacian plus perfusion."""
        return 0.35 * self.dx ** 2 / (6.0 * self.tissue.alpha)

    def _source(self, laser: LaserSource) -> np.ndarray:
        nx, ny, nz = self.temperature.shape
        x = (np.arange(nx) - nx // 2) * self.dx
        y = (np.arange(ny) - ny // 2) * self.dx
        z = np.arange(nz) * self.dx
        radius = laser.fibre_radius_mm * 1e-3
        radial = np.exp(-2.0 * (x[:, None] ** 2 + y[None, :] ** 2) / radius ** 2)
        fluence = laser.power_w / (np.pi * radius ** 2)
        return laser.mu_a * fluence * radial[:, :, None] * np.exp(-laser.mu_a * z)[None, None, :]

    @staticmethod
    def _arrhenius_rate(temperature: np.ndarray) -> np.ndarray:
        exponent = np.clip(-ARRHENIUS_EA / (R_GAS * (temperature + 273.15)), -700.0, 0.0)
        return ARRHENIUS_A * np.exp(exponent)

    def run(self, laser: LaserSource, duration_s: float = 1.5) -> "PennesSolver":
        t, dt, source = self.tissue, self.dt, self._source(laser)
        for _ in range(int(duration_s / dt)):
            # Insulated (Neumann) boundaries: edge padding, never periodic wrap.
            pad = np.pad(self.temperature, 1, mode="edge")
            lap = (pad[2:, 1:-1, 1:-1] + pad[:-2, 1:-1, 1:-1]
                   + pad[1:-1, 2:, 1:-1] + pad[1:-1, :-2, 1:-1]
                   + pad[1:-1, 1:-1, 2:] + pad[1:-1, 1:-1, :-2]
                   - 6.0 * self.temperature) / self.dx ** 2
            perfusion = t.perfusion * (t.t_arterial - self.temperature)
            self.temperature += (t.k * lap + perfusion + source) * dt / (t.rho * t.cp)
            self.damage += self._arrhenius_rate(self.temperature) * dt
        return self

    def coagulated(self) -> dict[str, float]:
        nx, ny, _ = self.damage.shape
        axis = self.damage[nx // 2, ny // 2, :]
        depths = np.nonzero(axis >= 1.0)[0]
        mask = self.damage >= 1.0
        return {"peak_temp_c": float(self.temperature.max()),
                "coagulation_depth_mm": float(depths.max() * self.dx * 1e3) if depths.size else 0.0,
                "lesion_volume_mm3": float(mask.sum()) * (self.dx * 1e3) ** 3,
                "damaged_voxels": float(mask.sum())}


def compare_wavelengths(duration_s: float = 1.5) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for laser in (HO_YAG, THULIUM_FIBER):
        results[laser.name] = {"wavelength_nm": laser.wavelength_nm,
                               "penetration_mm": laser.penetration_mm,
                               **PennesSolver().run(laser, duration_s).coagulated()}
    return results


if __name__ == "__main__":
    for name, row in compare_wavelengths().items():
        print(f"{name:14s} depth={row['coagulation_depth_mm']:.2f} mm  "
              f"peak={row['peak_temp_c']:.1f} C  "
              f"volume={row['lesion_volume_mm3']:.3f} mm3")
