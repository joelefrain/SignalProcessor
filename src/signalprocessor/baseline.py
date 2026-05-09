from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .integration import integrate_motion
from .motion import Motion


@dataclass(slots=True)
class Baselineouput:
    motion: Motion
    baseline: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    method: str
    info: dict[str, float | str | int]


def normalized_time(time: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    duration = float(time[-1] - time[0])
    if duration <= 0.0:
        raise ValueError("Duration must be positive.")
    return (time - time[0]) / duration, duration


def polynomial_design(tau: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    if order < 0:
        raise ValueError("Polynomial order must be >= 0.")
    return np.vander(tau, N=order + 1, increasing=True)


def _window_mask(time: NDArray[np.float64], windows: list[tuple[float, float]] | None) -> NDArray[np.bool_]:
    if not windows:
        return np.ones(time.size, dtype=bool)
    mask = np.zeros(time.size, dtype=bool)
    for start, end in windows:
        mask |= (time >= float(start)) & (time <= float(end))
    if not np.any(mask):
        raise ValueError("Baseline windows do not include any sample.")
    return mask


def fit_polynomial_baseline(
    motion: Motion,
    *,
    order: int = 1,
    windows: list[tuple[float, float]] | None = None,
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tau, _ = normalized_time(motion.time)
    design = polynomial_design(tau, order)
    mask = _window_mask(motion.time, windows)
    x = design[mask]
    y = motion.accel[mask]
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64)[mask])
        x = x * w[:, None]
        y = y * w
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return design @ coeffs, coeffs


def constrained_polynomial_baseline(
    motion: Motion,
    *,
    order: int = 1,
    v_final: float = 0.0,
    u_final: float = 0.0,
    v0: float = 0.0,
    u0: float = 0.0,
    windows: list[tuple[float, float]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tau, duration = normalized_time(motion.time)
    design = polynomial_design(tau, order)
    mask = _window_mask(motion.time, windows)

    a = motion.accel
    time_rel = motion.time - motion.time[0]
    rhs_v = np.trapz(a, time_rel) - (float(v_final) - float(v0))
    rhs_u = np.trapz((duration - time_rel) * a, time_rel) - (
        float(u_final) - float(u0) - float(v0) * duration
    )
    constraints = np.zeros((2, order + 1), dtype=np.float64)
    for k in range(order + 1):
        constraints[0, k] = duration / (k + 1)
        constraints[1, k] = duration * duration / ((k + 1) * (k + 2))
    d = np.array([rhs_v, rhs_u], dtype=np.float64)

    if order + 1 == 2:
        coeffs = np.linalg.solve(constraints, d)
        return design @ coeffs, coeffs

    x = design[mask]
    y = a[mask]
    ata = x.T @ x
    aty = x.T @ y
    kkt = np.block(
        [
            [ata, constraints.T],
            [constraints, np.zeros((constraints.shape[0], constraints.shape[0]))],
        ]
    )
    rhs = np.concatenate([aty, d])
    sol = np.linalg.solve(kkt, rhs)
    coeffs = sol[: order + 1]
    return design @ coeffs, coeffs


def correct_baseline(
    motion: Motion,
    *,
    method: str = "polynomial",
    order: int = 1,
    pre_event_end: float | None = None,
    windows: list[tuple[float, float]] | None = None,
    enforce_zero_end: bool = False,
    v_final: float = 0.0,
    u_final: float = 0.0,
) -> Baselineouput:
    method_norm = method.strip().lower()
    info: dict[str, float | str | int] = {"order": int(order)}

    if method_norm in {"none", "raw"}:
        baseline = np.zeros_like(motion.accel)
        coeffs = np.zeros(1, dtype=np.float64)
    elif method_norm in {"pre_event_mean", "pre-event-mean", "mean"}:
        if pre_event_end is None:
            n = max(10, int(0.05 * motion.npts))
            sample = motion.accel[:n]
            info["pre_event_samples"] = int(n)
        else:
            mask = motion.time <= float(pre_event_end)
            if not np.any(mask):
                raise ValueError("pre_event_end does not include any sample.")
            sample = motion.accel[mask]
            info["pre_event_end"] = float(pre_event_end)
        coeff = float(np.mean(sample))
        baseline = np.full_like(motion.accel, coeff)
        coeffs = np.array([coeff], dtype=np.float64)
    elif method_norm in {"final_velocity", "velocity_final"}:
        coeff = float(np.trapz(motion.accel, motion.time - motion.time[0]) / motion.duration)
        baseline = np.full_like(motion.accel, coeff)
        coeffs = np.array([coeff], dtype=np.float64)
    elif method_norm in {"polynomial", "poly", "least_squares"}:
        if enforce_zero_end:
            baseline, coeffs = constrained_polynomial_baseline(
                motion,
                order=order,
                windows=windows,
                v_final=v_final,
                u_final=u_final,
            )
            info["constraint"] = "vT,uT"
        else:
            baseline, coeffs = fit_polynomial_baseline(motion, order=order, windows=windows)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    corrected = motion.with_accel(
        motion.accel - baseline,
        name=f"{motion.name}_baseline",
        meta={"baseline_method": method_norm, "baseline_order": int(order)},
    )
    velocity, displacement = integrate_motion(corrected)
    info.update(
        {
            "velocity_final_mps": float(velocity[-1]),
            "displacement_final_m": float(displacement[-1]),
            "baseline_mean_mps2": float(np.mean(baseline)),
        }
    )
    return Baselineouput(corrected, baseline, coeffs, method_norm, info)
