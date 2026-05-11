from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import EPS, G


def _as_periods(periods_s: np.ndarray | list[float]) -> np.ndarray:
    periods = np.asarray(periods_s, dtype=float)
    if periods.ndim != 1 or periods.size == 0:
        raise ValueError("periods_s must be a non-empty 1D array")
    if np.any(periods <= 0):
        raise ValueError("all periods must be positive")
    return periods


def oscillator_response(
    acceleration_mps2: np.ndarray,
    dt: float,
    periods_s: np.ndarray | list[float],
    damping: float = 0.05,
    *,
    keep_history: bool = False,
) -> dict[str, np.ndarray]:
    """Vectorized Newmark average-acceleration response for many periods."""

    acc = np.asarray(acceleration_mps2, dtype=float)
    periods = _as_periods(periods_s)
    omega = 2.0 * np.pi / periods
    c = 2.0 * damping * omega
    k = omega**2
    beta = 0.25
    gamma = 0.5
    den = 1.0 + c * gamma * dt + k * beta * dt * dt

    u = np.zeros(periods.size, dtype=float)
    v = np.zeros(periods.size, dtype=float)
    a_rel = -acc[0] - c * v - k * u
    max_u = np.abs(u)
    max_v = np.abs(v)
    max_abs_acc = np.abs(a_rel + acc[0])
    peak_index = np.zeros(periods.size, dtype=int)
    peak_sign = np.ones(periods.size, dtype=float)
    u_hist = np.empty((acc.size, periods.size), dtype=float) if keep_history else None
    if keep_history:
        u_hist[0, :] = u

    for i in range(acc.size - 1):
        u_pred = u + dt * v + dt * dt * (0.5 - beta) * a_rel
        v_pred = v + dt * (1.0 - gamma) * a_rel
        a_new = (-acc[i + 1] - c * v_pred - k * u_pred) / den
        u = u_pred + beta * dt * dt * a_new
        v = v_pred + gamma * dt * a_new
        a_rel = a_new

        abs_u = np.abs(u)
        improved = abs_u > max_u
        max_u = np.maximum(max_u, abs_u)
        max_v = np.maximum(max_v, np.abs(v))
        max_abs_acc = np.maximum(max_abs_acc, np.abs(a_rel + acc[i + 1]))
        peak_index[improved] = i + 1
        peak_sign[improved] = np.sign(u[improved])
        if keep_history:
            u_hist[i + 1, :] = u

    out = {
        "period_s": periods,
        "sd_m": max_u,
        "relative_velocity_mps": max_v,
        "psv_mps": omega * max_u,
        "psa_mps2": omega**2 * max_u,
        "psa_g": omega**2 * max_u / G,
        "sa_abs_mps2": max_abs_acc,
        "sa_abs_g": max_abs_acc / G,
        "peak_index": peak_index,
        "peak_sign": np.where(peak_sign == 0.0, 1.0, peak_sign),
    }
    if keep_history and u_hist is not None:
        out["u_history_m"] = u_hist
    return out


def response_spectrum(
    acceleration_mps2: np.ndarray,
    dt: float,
    periods_s: np.ndarray | list[float],
    damping: float = 0.05,
) -> pd.DataFrame:
    response = oscillator_response(acceleration_mps2, dt, periods_s, damping, keep_history=False)
    keys = ("period_s", "sd_m", "psv_mps", "psa_mps2", "psa_g", "sa_abs_mps2", "sa_abs_g")
    return pd.DataFrame({key: response[key] for key in keys})


def make_period_grid(
    min_s: float = 0.05,
    max_s: float = 5.0,
    num: int = 120,
    spacing: str = "log",
) -> np.ndarray:
    if spacing == "linear":
        return np.linspace(min_s, max_s, int(num))
    return np.geomspace(min_s, max_s, int(num))


def interpolate_spectrum_loglog(
    target_periods_s: np.ndarray,
    target_sa_g: np.ndarray,
    query_periods_s: np.ndarray,
) -> np.ndarray:
    periods = np.asarray(target_periods_s, dtype=float)
    values = np.asarray(target_sa_g, dtype=float)
    query = np.asarray(query_periods_s, dtype=float)
    if np.any(query < periods.min()) or np.any(query > periods.max()):
        clipped = np.clip(query, periods.min(), periods.max())
    else:
        clipped = query
    log_sa = np.interp(np.log(clipped), np.log(periods), np.log(np.maximum(values, EPS)))
    return np.exp(log_sa)

