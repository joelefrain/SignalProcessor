from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import EPS, G


def cumulative_trapezoid(y: np.ndarray, dt: float, initial: float = 0.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)
    out[0] = initial
    if y.size > 1:
        increments = 0.5 * (y[1:] + y[:-1]) * dt
        out[1:] = initial + np.cumsum(increments)
    return out


def integrate_motion(
    acceleration_mps2: np.ndarray,
    dt: float,
    initial_velocity: float = 0.0,
    initial_displacement: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = cumulative_trapezoid(acceleration_mps2, dt, initial_velocity)
    displacement = cumulative_trapezoid(velocity, dt, initial_displacement)
    return velocity, displacement


def arias_intensity(acceleration_mps2: np.ndarray, dt: float) -> np.ndarray:
    squared = np.asarray(acceleration_mps2, dtype=float) ** 2
    return (np.pi / (2.0 * G)) * cumulative_trapezoid(squared, dt, 0.0)


def arias_percentile_times(
    acceleration_mps2: np.ndarray,
    dt: float,
    percentiles: Iterable[float] = (5, 75, 95),
) -> dict[float, float]:
    ia = arias_intensity(acceleration_mps2, dt)
    total = float(ia[-1])
    time = np.arange(ia.size, dtype=float) * dt
    if total <= EPS:
        return {float(p): float(time[0]) for p in percentiles}
    fraction = ia / total
    result: dict[float, float] = {}
    for p in percentiles:
        target = float(p) / 100.0
        result[float(p)] = float(np.interp(target, fraction, time))
    return result


def significant_durations(acceleration_mps2: np.ndarray, dt: float) -> dict[str, float]:
    p = arias_percentile_times(acceleration_mps2, dt, (5, 20, 75, 80, 95))
    return {
        "D_5_75": p[75.0] - p[5.0],
        "D_5_95": p[95.0] - p[5.0],
        "D_20_80": p[80.0] - p[20.0],
    }


def cumulative_absolute_velocity(acceleration_mps2: np.ndarray, dt: float) -> float:
    return float(np.trapezoid(np.abs(acceleration_mps2), dx=dt))


def peak_ground_values(acceleration_mps2: np.ndarray, dt: float) -> dict[str, float]:
    velocity, displacement = integrate_motion(acceleration_mps2, dt)
    return {
        "PGA_mps2": float(np.max(np.abs(acceleration_mps2))),
        "PGA_g": float(np.max(np.abs(acceleration_mps2)) / G),
        "PGV_mps": float(np.max(np.abs(velocity))),
        "PGD_m": float(np.max(np.abs(displacement))),
        "final_velocity_mps": float(velocity[-1]),
        "final_displacement_m": float(displacement[-1]),
    }


def post_event_slope(y: np.ndarray, dt: float, start_fraction: float = 0.8) -> float:
    y = np.asarray(y, dtype=float)
    i0 = max(0, min(y.size - 2, int(start_fraction * y.size)))
    x = np.arange(y.size - i0, dtype=float) * dt
    if x.size < 2:
        return 0.0
    slope = np.polyfit(x, y[i0:], 1)[0]
    return float(slope)


def fourier_amplitude_spectrum(
    acceleration_mps2: np.ndarray,
    dt: float,
    *,
    window: str | None = "hann",
    smooth_bins: int = 0,
) -> pd.DataFrame:
    acc = np.asarray(acceleration_mps2, dtype=float)
    demeaned = acc - np.mean(acc)
    if window == "hann" and acc.size > 4:
        weights = np.hanning(acc.size)
        demeaned = demeaned * weights
        scale = np.sum(weights) / acc.size
    else:
        scale = 1.0
    freq = np.fft.rfftfreq(acc.size, dt)
    fas = np.abs(np.fft.rfft(demeaned)) * dt / max(scale, EPS)
    if smooth_bins and smooth_bins > 1:
        kernel = np.ones(int(smooth_bins), dtype=float) / float(smooth_bins)
        fas = np.convolve(fas, kernel, mode="same")
    return pd.DataFrame({"frequency_hz": freq, "fas_mps": fas})


def motion_summary(acceleration_mps2: np.ndarray, dt: float) -> dict[str, float]:
    velocity, displacement = integrate_motion(acceleration_mps2, dt)
    peaks = peak_ground_values(acceleration_mps2, dt)
    durations = significant_durations(acceleration_mps2, dt)
    summary = {
        **peaks,
        "arias_intensity_mps": float(arias_intensity(acceleration_mps2, dt)[-1]),
        "CAV_mps": cumulative_absolute_velocity(acceleration_mps2, dt),
        "post_event_velocity_slope_mps2": post_event_slope(velocity, dt),
        "post_event_displacement_slope_mps": post_event_slope(displacement, dt),
    }
    summary.update(durations)
    return summary
