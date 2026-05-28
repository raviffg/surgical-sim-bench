"""6-axis contact wrench synthesiser for endoscopic tool telemetry.

Converts a contact force ``F`` applied at lever arm ``r`` into a full wrench:

    M = r x F + M_applied

and streams it at 1 kHz with bias, sensor noise and a first-order anti-alias
filter, so downstream force-feedback and tissue-injury models see signals with
the bandwidth and drift of a real distal force/torque sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE_HZ = 1000.0
SENSOR_NOISE_N = 0.02
SENSOR_BIAS_N = np.array([0.05, -0.03, 0.08])
SENSOR_BIAS_NM = np.array([0.002, 0.001, -0.001])


@dataclass(frozen=True)
class Wrench:
    force_n: np.ndarray
    torque_nm: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.force_n, self.torque_nm])


def contact_wrench(force_n, contact_mm, applied_torque_nm=None) -> Wrench:
    force = np.asarray(force_n, dtype=float)
    torque = np.cross(np.asarray(contact_mm, dtype=float) * 1e-3, force)
    if applied_torque_nm is not None:
        torque = torque + np.asarray(applied_torque_nm, dtype=float)
    return Wrench(force, torque)


class WrenchStream:
    """Synthesises and filters a 6-axis telemetry channel."""

    def __init__(self, seed: int = 7, cutoff_hz: float = 120.0) -> None:
        self.rng = np.random.default_rng(seed)
        alpha = 2.0 * np.pi * cutoff_hz / SAMPLE_RATE_HZ
        self._k = alpha / (1.0 + alpha)
        self._state = np.zeros(6)

    def _filter(self, raw: np.ndarray) -> np.ndarray:
        self._state += self._k * (raw - self._state)
        return self._state.copy()

    def sample(self, wrench: Wrench) -> np.ndarray:
        raw = wrench.as_vector()
        raw[:3] += SENSOR_BIAS_N + self.rng.normal(0.0, SENSOR_NOISE_N, 3)
        raw[3:] += SENSOR_BIAS_NM + self.rng.normal(0.0, SENSOR_NOISE_N / 100.0, 3)
        return self._filter(raw)

    def _stream(self, duration_s: float, force_at) -> np.ndarray:
        steps = int(duration_s * SAMPLE_RATE_HZ)
        frames = np.zeros((steps, 6))
        for i in range(steps):
            force, contact = force_at(np.arange(steps)[i] / SAMPLE_RATE_HZ)
            frames[i] = self.sample(contact_wrench(force, contact))
        return frames

    def morcellation(self, duration_s: float = 1.0, blade_hz: float = 18.0,
                     peak_n: float = 1.8) -> np.ndarray:
        """Oscillating cutting load with a slow tissue-stiffness ramp."""

        def force_at(t: float):
            envelope = 0.35 + 0.65 * min(t / max(duration_s, 1e-6), 1.0)
            axial = peak_n * envelope * (0.6 + 0.4 * np.sin(2 * np.pi * blade_hz * t))
            lateral = 0.25 * peak_n * envelope * np.sin(2 * np.pi * blade_hz * t + 0.7)
            return np.array([lateral, 0.15 * axial, axial]), (0.0, 0.0, 22.0)

        return self._stream(duration_s, force_at)

    def enucleation(self, duration_s: float = 1.0, peak_n: float = 2.4) -> np.ndarray:
        """Traction ramp to capsule release, then elastic snap-back."""

        def force_at(t: float):
            frac = t / max(duration_s, 1e-6)
            pull = peak_n * (frac / 0.8 if frac <= 0.8 else np.exp(-(frac - 0.8) * 40.0))
            return np.array([0.0, 0.0, pull]), (0.0, 6.0, 18.0)

        return self._stream(duration_s, force_at)


if __name__ == "__main__":
    stream = WrenchStream()
    for label, frames in (("morcellation", stream.morcellation()),
                          ("enucleation", stream.enucleation())):
        print(f"{label:14s} frames={len(frames)}  "
              f"peak_force={float(np.abs(frames[:, :3]).max()):.2f} N")
