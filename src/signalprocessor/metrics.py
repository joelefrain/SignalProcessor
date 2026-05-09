from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import G0
from .integration import integrate_motion
from .motion import Motion


def cumulative_arias(accel_mps2: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    squared = np.asarray(accel_mps2, dtype=np.float64) ** 2
    out = np.empty_like(squared)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * float(dt) * (squared[1:] + squared[:-1]))
    return np.pi / (2.0 * G0) * out


def significant_duration(
    time: NDArray[np.float64],
    arias: NDArray[np.float64],
    *,
    p1: float = 0.05,
    p2: float = 0.95,
) -> float:
    total = float(arias[-1])
    if total <= 0.0:
        return 0.0
    norm = arias / total
    t1 = float(np.interp(float(p1), norm, time))
    t2 = float(np.interp(float(p2), norm, time))
    return max(0.0, t2 - t1)


def threshold_durations(
    motion: Motion,
    *,
    threshold_g: float = 0.05,
) -> tuple[float, float]:
    above = np.abs(motion.accel) / G0 >= float(threshold_g)
    if not np.any(above):
        return 0.0, 0.0
    idx = np.flatnonzero(above)
    bracketed = float(motion.time[idx[-1]] - motion.time[idx[0]])
    uniform = float(np.sum(above) * motion.dt)
    return bracketed, uniform


def motion_metrics(motion: Motion) -> dict[str, float]:
    velocity, displacement = integrate_motion(motion)
    arias = cumulative_arias(motion.accel, motion.dt)
    cav_mps = float(np.trapz(np.abs(motion.accel), motion.time))
    cav_gs = cav_mps / G0
    bracketed, uniform = threshold_durations(motion)
    acc_g = motion.accel / G0
    return {
        "npts": float(motion.npts),
        "dt_s": float(motion.dt),
        "duration_s": float(motion.duration),
        "pga_g": float(np.max(np.abs(acc_g))),
        "pgv_cm_s": float(np.max(np.abs(velocity)) * 100.0),
        "pgd_cm": float(np.max(np.abs(displacement)) * 100.0),
        "velocity_final_cm_s": float(velocity[-1] * 100.0),
        "displacement_final_cm": float(displacement[-1] * 100.0),
        "arias_m_s": float(arias[-1]),
        "d5_95_s": significant_duration(motion.time, arias, p1=0.05, p2=0.95),
        "d5_75_s": significant_duration(motion.time, arias, p1=0.05, p2=0.75),
        "cav_m_s": cav_mps,
        "cav_g_s": cav_gs,
        "rms_g": float(np.sqrt(np.mean(acc_g * acc_g))),
        "bracketed_0p05g_s": bracketed,
        "uniform_0p05g_s": uniform,
    }
