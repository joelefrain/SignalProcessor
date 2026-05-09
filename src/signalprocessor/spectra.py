from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from .constants import G0
from .motion import Motion

try:  # pragma: no cover - exercised only when numba is installed.
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*_args, **_kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

    prange = range  # type: ignore


@njit(cache=False, parallel=True, fastmath=True)
def _newmark_spectrum_numba(
    ag: NDArray[np.float64],
    dt: float,
    periods: NDArray[np.float64],
    damping: float,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    n_periods = periods.size
    sd = np.zeros(n_periods, dtype=np.float64)
    sv = np.zeros(n_periods, dtype=np.float64)
    sa = np.zeros(n_periods, dtype=np.float64)
    saa = np.zeros(n_periods, dtype=np.float64)
    a0c = 1.0 / (beta * dt * dt)
    a1c = gamma / (beta * dt)
    a2c = 1.0 / (beta * dt)
    a3c = 1.0 / (2.0 * beta) - 1.0
    a4c = gamma / beta - 1.0
    a5c = dt * (gamma / (2.0 * beta) - 1.0)

    for ip in prange(n_periods):
        t = periods[ip]
        if t <= 0.0:
            max_abs = 0.0
            for i in range(ag.size):
                val = abs(ag[i])
                if val > max_abs:
                    max_abs = val
            sd[ip] = 0.0
            sv[ip] = 0.0
            sa[ip] = max_abs / G0
            saa[ip] = max_abs / G0
            continue

        omega = 2.0 * np.pi / t
        k = omega * omega
        c = 2.0 * damping * omega
        khat = k + a0c + a1c * c

        u = 0.0
        v = 0.0
        acc_rel = -ag[0]
        max_u = 0.0
        max_v = 0.0
        max_abs_acc = abs(acc_rel + ag[0])

        for i in range(1, ag.size):
            p_next = -ag[i]
            p_eff = (
                p_next
                + a0c * u
                + a2c * v
                + a3c * acc_rel
                + c * (a1c * u + a4c * v + a5c * acc_rel)
            )
            u_next = p_eff / khat
            acc_next = a0c * (u_next - u) - a2c * v - a3c * acc_rel
            v_next = v + dt * ((1.0 - gamma) * acc_rel + gamma * acc_next)

            au = abs(u_next)
            av = abs(v_next)
            aa = abs(acc_next + ag[i])
            if au > max_u:
                max_u = au
            if av > max_v:
                max_v = av
            if aa > max_abs_acc:
                max_abs_acc = aa

            u = u_next
            v = v_next
            acc_rel = acc_next

        sd[ip] = max_u
        sv[ip] = omega * max_u
        sa[ip] = (omega * omega * max_u) / G0
        saa[ip] = max_abs_acc / G0

    return sd, sv, sa, saa


def _newmark_spectrum_python(
    ag: NDArray[np.float64],
    dt: float,
    periods: NDArray[np.float64],
    damping: float,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    sd = np.zeros(periods.size, dtype=np.float64)
    sv = np.zeros(periods.size, dtype=np.float64)
    sa = np.zeros(periods.size, dtype=np.float64)
    saa = np.zeros(periods.size, dtype=np.float64)
    a0c = 1.0 / (beta * dt * dt)
    a1c = gamma / (beta * dt)
    a2c = 1.0 / (beta * dt)
    a3c = 1.0 / (2.0 * beta) - 1.0
    a4c = gamma / beta - 1.0
    a5c = dt * (gamma / (2.0 * beta) - 1.0)

    for ip, period in enumerate(periods):
        if period <= 0.0:
            pga = float(np.max(np.abs(ag)) / G0)
            sa[ip] = pga
            saa[ip] = pga
            continue

        omega = 2.0 * np.pi / float(period)
        k = omega * omega
        c = 2.0 * float(damping) * omega
        khat = k + a0c + a1c * c

        u = 0.0
        v = 0.0
        acc_rel = -float(ag[0])
        max_u = 0.0
        max_v = 0.0
        max_abs_acc = abs(acc_rel + float(ag[0]))

        for i in range(1, ag.size):
            p_eff = (
                -float(ag[i])
                + a0c * u
                + a2c * v
                + a3c * acc_rel
                + c * (a1c * u + a4c * v + a5c * acc_rel)
            )
            u_next = p_eff / khat
            acc_next = a0c * (u_next - u) - a2c * v - a3c * acc_rel
            v_next = v + dt * ((1.0 - gamma) * acc_rel + gamma * acc_next)

            max_u = max(max_u, abs(u_next))
            max_v = max(max_v, abs(v_next))
            max_abs_acc = max(max_abs_acc, abs(acc_next + float(ag[i])))
            u, v, acc_rel = u_next, v_next, acc_next

        sd[ip] = max_u
        sv[ip] = omega * max_u
        sa[ip] = (omega * omega * max_u) / G0
        saa[ip] = max_abs_acc / G0
    return sd, sv, sa, saa


def response_spectrum(
    motion: Motion,
    periods: NDArray[np.float64] | None = None,
    *,
    damping: float = 0.05,
    beta: float = 0.25,
    gamma: float = 0.5,
    use_numba: bool = True,
    num_periods: int = 200,
    t_min: float = 0.02,
    t_max: float = 5.0,
) -> dict[str, NDArray[np.float64]]:
    if periods is None:
        periods = logspace_periods(t_min=t_min, t_max=t_max, n=num_periods)
    periods = np.asarray(periods, dtype=np.float64)
    if periods.ndim != 1 or periods.size == 0:
        raise ValueError("periods must be a non-empty 1D array.")
    if np.any(periods < 0.0):
        raise ValueError("periods cannot be negative.")
    if use_numba and HAVE_NUMBA:
        sd, sv, sa, saa = _newmark_spectrum_numba(
            motion.accel, motion.dt, periods, float(damping), float(beta), float(gamma)
        )
    else:
        sd, sv, sa, saa = _newmark_spectrum_python(
            motion.accel, motion.dt, periods, float(damping), float(beta), float(gamma)
        )
    return {
        "period_s": periods,
        "sd_m": sd,
        "sv_m_s": sv,
        "sa_g": sa,
        "saa_g": saa,
    }


@lru_cache(maxsize=32)
def logspace_periods(
    t_min: float = 0.02, t_max: float = 5.0, n: int = 200
) -> NDArray[np.float64]:
    return np.logspace(np.log10(float(t_min)), np.log10(float(t_max)), int(n)).astype(
        np.float64
    )


def significant_duration(
    motion: Motion,
    energy_range: tuple[float, float] = (0.05, 0.95),
) -> float:
    """Calculate significant duration based on Arias intensity.

    Significant duration is the time interval during which 90% of the Arias intensity
    is generated, by default from 5% to 95% cumulative energy.

    Args:
        motion: Motion object with acceleration time series
        energy_range: Tuple (lower, upper) energy bounds (0-1). Default is (0.05, 0.95).

    Returns:
        Significant duration in seconds
    """
    accel = np.asarray(motion.accel, dtype=np.float64)
    dt = float(motion.dt)

    # Compute squared acceleration
    accel_sq = accel**2

    # Compute cumulative energy (proportional to Arias intensity)
    energy = np.cumsum(accel_sq)
    energy = energy / energy[-1]  # Normalize to [0, 1]

    # Find indices where energy reaches lower and upper bounds
    lower_bound, upper_bound = float(energy_range[0]), float(energy_range[1])

    idx_lower = np.searchsorted(energy, lower_bound, side="left")
    idx_upper = np.searchsorted(energy, upper_bound, side="right")

    # Ensure valid indices
    idx_lower = max(0, min(idx_lower, len(energy) - 1))
    idx_upper = max(0, min(idx_upper, len(energy) - 1))

    # Calculate duration in seconds
    duration = (idx_upper - idx_lower) * dt

    return max(0.0, duration)
