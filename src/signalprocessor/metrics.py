from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import G0
from .core import trapezoid_integrate
from .records import MotionRecord


@dataclass(frozen=True, slots=True)
class GroundMotionParameters:
    pga: float
    pgv: float
    pgd: float
    arias_intensity: float
    d5_75: float
    d5_95: float
    cav: float
    rms_acceleration: float
    bracketed_duration: float
    final_velocity: float
    final_displacement: float


def integrate_motion(acceleration_si: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    velocity = trapezoid_integrate(np.asarray(acceleration_si, dtype=np.float64), float(dt), 0.0)
    displacement = trapezoid_integrate(velocity, float(dt), 0.0)
    return velocity, displacement


def cumulative_arias(acceleration_si: np.ndarray, dt: float) -> np.ndarray:
    acc = np.asarray(acceleration_si, dtype=np.float64)
    acc2 = acc * acc
    integral = trapezoid_integrate(acc2, float(dt), 0.0)
    return (np.pi / (2.0 * G0)) * integral


def _time_at_fraction(time: np.ndarray, cumulative: np.ndarray, fraction: float) -> float:
    total = float(cumulative[-1])
    if total <= 0.0:
        return float(time[0])
    target = fraction * total
    idx = int(np.searchsorted(cumulative, target, side="left"))
    if idx <= 0:
        return float(time[0])
    if idx >= cumulative.size:
        return float(time[-1])
    x0 = cumulative[idx - 1]
    x1 = cumulative[idx]
    if x1 == x0:
        return float(time[idx])
    alpha = (target - x0) / (x1 - x0)
    return float(time[idx - 1] + alpha * (time[idx] - time[idx - 1]))


def significant_duration(time: np.ndarray, arias_curve: np.ndarray, start: float, end: float) -> float:
    return _time_at_fraction(time, arias_curve, end) - _time_at_fraction(time, arias_curve, start)


def bracketed_duration(acceleration_si: np.ndarray, dt: float, threshold_g: float = 0.05) -> float:
    mask = np.abs(acceleration_si) >= threshold_g * G0
    if not np.any(mask):
        return 0.0
    idx = np.flatnonzero(mask)
    return float((idx[-1] - idx[0]) * dt)


def compute_ground_motion_parameters(record: MotionRecord, *, threshold_g: float = 0.05) -> GroundMotionParameters:
    acc = record.acceleration_si()
    dt = record.dt
    vel, disp = integrate_motion(acc, dt)
    arias = cumulative_arias(acc, dt)
    abs_acc = np.abs(acc)
    cav = float(np.trapezoid(abs_acc, dx=dt))
    rms = float(np.sqrt(np.trapezoid(acc * acc, dx=dt) / max(record.duration, dt)))
    return GroundMotionParameters(
        pga=float(abs_acc.max()),
        pgv=float(np.abs(vel).max()),
        pgd=float(np.abs(disp).max()),
        arias_intensity=float(arias[-1]),
        d5_75=float(significant_duration(record.time, arias, 0.05, 0.75)),
        d5_95=float(significant_duration(record.time, arias, 0.05, 0.95)),
        cav=cav,
        rms_acceleration=rms,
        bracketed_duration=bracketed_duration(acc, dt, threshold_g=threshold_g),
        final_velocity=float(vel[-1]),
        final_displacement=float(disp[-1]),
    )
