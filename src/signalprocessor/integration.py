from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .motion import Motion


def cumulative_trapezoid_uniform(
    y: NDArray[np.float64], dt: float, initial: float = 0.0
) -> NDArray[np.float64]:
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    out[0] = float(initial)
    out[1:] = float(initial) + np.cumsum(0.5 * float(dt) * (y[1:] + y[:-1]))
    return out


def integrate_motion(
    motion: Motion,
    *,
    v0: float = 0.0,
    u0: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    velocity = cumulative_trapezoid_uniform(motion.accel, motion.dt, initial=v0)
    displacement = cumulative_trapezoid_uniform(velocity, motion.dt, initial=u0)
    return velocity, displacement
