from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal

from .core import central_difference
from .metrics import GroundMotionParameters, compute_ground_motion_parameters, integrate_motion
from .records import MotionRecord


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    remove_mean: bool = True
    pre_event_seconds: float | None = None
    baseline_order: int = 1
    constrain_final_velocity: bool = True
    constrain_final_displacement: bool = False
    target_final_velocity: float = 0.0
    target_final_displacement: float = 0.0
    despike: bool = True
    spike_sigma: float = 8.0
    taper_fraction: float = 0.02
    highpass_hz: float | None = 0.05
    lowpass_hz: float | None = None
    filter_order: int = 4
    zero_phase: bool = True


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    record: MotionRecord
    velocity: np.ndarray
    displacement: np.ndarray
    baseline: np.ndarray
    config: CorrectionConfig
    metrics: GroundMotionParameters
    diagnostics: dict[str, Any] = field(default_factory=dict)


def remove_mean(acc: np.ndarray, dt: float, seconds: float | None = None) -> tuple[np.ndarray, float]:
    n = acc.size if seconds is None else max(1, min(acc.size, int(round(seconds / dt))))
    offset = float(np.mean(acc[:n]))
    return acc - offset, offset


def polynomial_baseline(
    acceleration: np.ndarray,
    dt: float,
    order: int,
    *,
    constrain_velocity: bool = False,
    constrain_displacement: bool = False,
    target_final_velocity: float = 0.0,
    target_final_displacement: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    acc = np.asarray(acceleration, dtype=np.float64)
    if order < 0:
        return np.zeros_like(acc), np.zeros(0, dtype=np.float64)

    n = acc.size
    tau = np.linspace(0.0, 1.0, n, dtype=np.float64)
    design = np.vander(tau, N=order + 1, increasing=True)
    lhs = design.T @ design
    rhs = design.T @ acc

    constraints: list[np.ndarray] = []
    targets: list[float] = []
    duration = (n - 1) * dt
    time = np.arange(n, dtype=np.float64) * dt

    if constrain_velocity:
        raw_v = float(np.trapezoid(acc, dx=dt))
        constraints.append(np.asarray([duration / (k + 1) for k in range(order + 1)], dtype=np.float64))
        targets.append(raw_v - target_final_velocity)
    if constrain_displacement:
        raw_u = float(np.trapezoid((duration - time) * acc, dx=dt))
        constraints.append(
            np.asarray([duration * duration / ((k + 1) * (k + 2)) for k in range(order + 1)], dtype=np.float64)
        )
        targets.append(raw_u - target_final_displacement)

    if constraints and len(constraints) <= order + 1:
        cmat = np.vstack(constraints)
        zeros = np.zeros((cmat.shape[0], cmat.shape[0]), dtype=np.float64)
        kkt = np.block([[lhs, cmat.T], [cmat, zeros]])
        krhs = np.concatenate([rhs, np.asarray(targets, dtype=np.float64)])
        sol = np.linalg.lstsq(kkt, krhs, rcond=None)[0]
        coeffs = sol[: order + 1]
    else:
        coeffs = np.linalg.lstsq(design, acc, rcond=None)[0]

    baseline = design @ coeffs
    return baseline, coeffs


def despike_array(acceleration: np.ndarray, dt: float, *, sigma: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    acc = np.asarray(acceleration, dtype=np.float64).copy()
    deriv = central_difference(acc, float(dt))
    med = float(np.median(deriv))
    mad = float(np.median(np.abs(deriv - med)))
    scale = 1.4826 * mad if mad > 0.0 else float(np.std(deriv))
    if scale <= 0.0:
        return acc, np.zeros(0, dtype=np.int64)
    spike_idx = np.flatnonzero(np.abs(deriv - med) > sigma * scale)
    spike_idx = spike_idx[(spike_idx > 0) & (spike_idx < acc.size - 1)]
    if spike_idx.size == 0:
        return acc, spike_idx
    mask = np.ones(acc.size, dtype=bool)
    mask[spike_idx] = False
    acc[spike_idx] = np.interp(spike_idx, np.flatnonzero(mask), acc[mask])
    return acc, spike_idx


def cosine_taper(acceleration: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return acceleration.copy()
    fraction = float(min(fraction, 0.5))
    window = signal.windows.tukey(acceleration.size, alpha=2.0 * fraction)
    return acceleration * window


def butterworth_filter(
    acceleration: np.ndarray,
    dt: float,
    *,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    order: int,
    zero_phase: bool = True,
) -> np.ndarray:
    fs = 1.0 / dt
    nyq = 0.5 * fs
    hp = highpass_hz if highpass_hz and highpass_hz > 0.0 else None
    lp = lowpass_hz if lowpass_hz and lowpass_hz > 0.0 else None
    if lp is not None:
        lp = min(lp, 0.98 * nyq)
    if hp is not None:
        hp = min(hp, 0.95 * nyq)
    if hp is None and lp is None:
        return acceleration.copy()
    if hp is not None and lp is not None and hp >= lp:
        raise ValueError("highpass_hz must be lower than lowpass_hz")
    if hp is not None and lp is not None:
        wn: float | list[float] = [hp, lp]
        btype = "bandpass"
    elif hp is not None:
        wn = hp
        btype = "highpass"
    else:
        wn = lp
        btype = "lowpass"
    sos = signal.butter(order, wn, btype=btype, fs=fs, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, acceleration)
    return signal.sosfilt(sos, acceleration)


def correct_record(record: MotionRecord, config: CorrectionConfig | None = None) -> CorrectionResult:
    cfg = config or CorrectionConfig()
    acc = record.acceleration_si().astype(np.float64, copy=True)
    dt = record.dt
    diagnostics: dict[str, Any] = {}

    if cfg.despike:
        acc, spikes = despike_array(acc, dt, sigma=cfg.spike_sigma)
        diagnostics["spike_count"] = int(spikes.size)
        diagnostics["spike_indices"] = spikes

    if cfg.remove_mean:
        acc, offset = remove_mean(acc, dt, cfg.pre_event_seconds)
        diagnostics["mean_removed_mps2"] = offset

    constrain_disp = cfg.constrain_final_displacement and cfg.baseline_order >= 1
    baseline, coeffs = polynomial_baseline(
        acc,
        dt,
        cfg.baseline_order,
        constrain_velocity=cfg.constrain_final_velocity,
        constrain_displacement=constrain_disp,
        target_final_velocity=cfg.target_final_velocity,
        target_final_displacement=cfg.target_final_displacement,
    )
    acc = acc - baseline
    diagnostics["baseline_coefficients"] = coeffs

    if cfg.taper_fraction > 0.0:
        acc = cosine_taper(acc, cfg.taper_fraction)

    acc = butterworth_filter(
        acc,
        dt,
        highpass_hz=cfg.highpass_hz,
        lowpass_hz=cfg.lowpass_hz,
        order=cfg.filter_order,
        zero_phase=cfg.zero_phase,
    )

    velocity, displacement = integrate_motion(acc, dt)
    corrected = record.with_acceleration(acc, units="m/s^2", metadata={"processing": "signalprocessor"})
    metrics = compute_ground_motion_parameters(corrected)
    diagnostics["final_velocity_mps"] = float(velocity[-1])
    diagnostics["final_displacement_m"] = float(displacement[-1])
    return CorrectionResult(
        record=corrected,
        velocity=velocity,
        displacement=displacement,
        baseline=baseline,
        config=cfg,
        metrics=metrics,
        diagnostics=diagnostics,
    )
