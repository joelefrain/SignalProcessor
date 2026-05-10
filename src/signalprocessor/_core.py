from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover

    def njit(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


@njit(cache=True, fastmath=True)
def trapezoid_integrate(values: np.ndarray, dt: float, init: float = 0.0) -> np.ndarray:
    n = values.size
    out = np.empty(n, dtype=np.float64)
    out[0] = init
    half_dt = 0.5 * dt
    for i in range(1, n):
        out[i] = out[i - 1] + (values[i - 1] + values[i]) * half_dt
    return out


@njit(cache=True, fastmath=True)
def central_difference(values: np.ndarray, dt: float) -> np.ndarray:
    n = values.size
    out = np.empty(n, dtype=np.float64)
    if n == 1:
        out[0] = 0.0
        return out
    if n < 5:
        out[0] = (values[1] - values[0]) / dt
        for i in range(1, n - 1):
            out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)
        out[n - 1] = (values[n - 1] - values[n - 2]) / dt
        return out

    out[0] = (values[1] - values[0]) / dt
    out[1] = (values[2] - values[0]) / (2.0 * dt)
    for i in range(2, n - 2):
        out[i] = (
            values[i - 2] - 8.0 * values[i - 1] + 8.0 * values[i + 1] - values[i + 2]
        ) / (12.0 * dt)
    out[n - 2] = (values[n - 1] - values[n - 3]) / (2.0 * dt)
    out[n - 1] = (values[n - 1] - values[n - 2]) / dt
    return out


@njit(cache=True, fastmath=True)
def newmark_response_spectrum(
    acceleration: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = acceleration.size
    m = periods.size
    sd = np.zeros(m, dtype=np.float64)
    psv = np.zeros(m, dtype=np.float64)
    psa = np.zeros(m, dtype=np.float64)
    saa = np.zeros(m, dtype=np.float64)
    peak_index = np.zeros(m, dtype=np.int64)

    beta = 0.25
    gamma = 0.5
    pi2 = 2.0 * np.pi

    for j in range(m):
        period = periods[j]
        pga = 0.0
        for i in range(n):
            val = abs(acceleration[i])
            if val > pga:
                pga = val
        if period <= 1.0e-12:
            psa[j] = pga
            saa[j] = pga
            continue

        omega = pi2 / period
        k = omega * omega
        c = 2.0 * damping * omega

        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        k_eff = k + a0 + a1 * c

        u = 0.0
        v = 0.0
        rel_acc = -acceleration[0] - c * v - k * u
        max_u = abs(u)
        max_abs_acc = abs(rel_acc + acceleration[0])
        max_i = 0

        for i in range(n - 1):
            p_next = -acceleration[i + 1]
            p_eff = (
                p_next
                + a0 * u
                + a2 * v
                + a3 * rel_acc
                + c * (a1 * u + a4 * v + a5 * rel_acc)
            )
            u_next = p_eff / k_eff
            rel_acc_next = a0 * (u_next - u) - a2 * v - a3 * rel_acc
            v_next = v + dt * ((1.0 - gamma) * rel_acc + gamma * rel_acc_next)

            abs_u = abs(u_next)
            if abs_u > max_u:
                max_u = abs_u
                max_i = i + 1
            abs_acc = abs(rel_acc_next + acceleration[i + 1])
            if abs_acc > max_abs_acc:
                max_abs_acc = abs_acc

            u = u_next
            v = v_next
            rel_acc = rel_acc_next

        sd[j] = max_u
        psv[j] = omega * max_u
        psa[j] = k * max_u
        saa[j] = max_abs_acc
        peak_index[j] = max_i

    return sd, psv, psa, saa, peak_index
