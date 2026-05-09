from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .baseline import correct_baseline
from .filtering import butterworth_filter
from .motion import Motion
from .spectra import response_spectrum


@dataclass(slots=True)
class ScaleResult:
    motion: Motion
    factor: float
    periods: NDArray[np.float64]
    record_sa_g: NDArray[np.float64]
    target_sa_g: NDArray[np.float64]
    scaled_sa_g: NDArray[np.float64]
    method: str


def interpolate_spectrum(
    source_periods: NDArray[np.float64],
    source_sa_g: NDArray[np.float64],
    target_periods: NDArray[np.float64],
    *,
    loglog: bool = True,
) -> NDArray[np.float64]:
    source_periods = np.asarray(source_periods, dtype=np.float64)
    source_sa_g = np.asarray(source_sa_g, dtype=np.float64)
    target_periods = np.asarray(target_periods, dtype=np.float64)
    if loglog:
        eps = np.finfo(float).tiny
        x = np.log(np.maximum(source_periods, eps))
        y = np.log(np.maximum(source_sa_g, eps))
        return np.exp(np.interp(np.log(np.maximum(target_periods, eps)), x, y))
    return np.interp(target_periods, source_periods, source_sa_g)


def _period_mask(periods: NDArray[np.float64], period_range: tuple[float, float] | None) -> NDArray[np.bool_]:
    if period_range is None:
        return np.ones(periods.size, dtype=bool)
    lo, hi = period_range
    mask = (periods >= float(lo)) & (periods <= float(hi))
    if not np.any(mask):
        raise ValueError("period_range does not include any target period.")
    return mask


def scale_factor(
    record_sa_g: NDArray[np.float64],
    target_sa_g: NDArray[np.float64],
    *,
    periods: NDArray[np.float64],
    method: str = "log_least_squares",
    period_range: tuple[float, float] | None = None,
    single_period: float | None = None,
    weights: NDArray[np.float64] | None = None,
) -> float:
    method_norm = method.strip().lower()
    record = np.asarray(record_sa_g, dtype=np.float64)
    target = np.asarray(target_sa_g, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)

    if single_period is not None or method_norm in {"single", "period"}:
        if single_period is None:
            raise ValueError("single_period is required for single-period scaling.")
        rec = float(interpolate_spectrum(periods, record, np.array([single_period]))[0])
        tar = float(interpolate_spectrum(periods, target, np.array([single_period]))[0])
        return tar / max(rec, np.finfo(float).tiny)

    mask = _period_mask(periods, period_range)
    rec = np.maximum(record[mask], np.finfo(float).tiny)
    tar = np.maximum(target[mask], np.finfo(float).tiny)
    if weights is None:
        w = np.ones(rec.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[mask]

    if method_norm in {"linear", "least_squares", "ls"}:
        return float(np.sum(w * rec * tar) / np.sum(w * rec * rec))
    if method_norm in {"log", "log_least_squares", "log_ls"}:
        return float(np.exp(np.sum(w * np.log(tar / rec)) / np.sum(w)))
    raise ValueError(f"Unsupported scaling method: {method}")


def scale_motion_to_target(
    motion: Motion,
    target_periods: NDArray[np.float64],
    target_sa_g: NDArray[np.float64],
    *,
    damping: float = 0.05,
    method: str = "log_least_squares",
    period_range: tuple[float, float] | None = None,
    single_period: float | None = None,
    factor_bounds: tuple[float, float] | None = (0.1, 10.0),
) -> ScaleResult:
    spectrum = response_spectrum(motion, target_periods, damping=damping)
    rec_sa = spectrum["sa_g"]
    factor = scale_factor(
        rec_sa,
        target_sa_g,
        periods=target_periods,
        method=method,
        period_range=period_range,
        single_period=single_period,
    )
    if factor_bounds is not None:
        factor = float(np.clip(factor, factor_bounds[0], factor_bounds[1]))
    scaled = motion.scaled(factor, name=f"{motion.name}_scaled")
    return ScaleResult(
        motion=scaled,
        factor=factor,
        periods=np.asarray(target_periods, dtype=np.float64),
        record_sa_g=rec_sa,
        target_sa_g=np.asarray(target_sa_g, dtype=np.float64),
        scaled_sa_g=rec_sa * factor,
        method=method,
    )


def scale_suite_to_target(
    motions: Iterable[Motion],
    target_periods: NDArray[np.float64],
    target_sa_g: NDArray[np.float64],
    *,
    damping: float = 0.05,
    method: str = "log_least_squares",
    period_range: tuple[float, float] | None = None,
    factor_bounds: tuple[float, float] | None = (0.2, 5.0),
) -> tuple[list[ScaleResult], NDArray[np.float64]]:
    ouput: list[ScaleResult] = []
    scaled_spectra = []
    for motion in motions:
        result = scale_motion_to_target(
            motion,
            target_periods,
            target_sa_g,
            damping=damping,
            method=method,
            period_range=period_range,
            factor_bounds=factor_bounds,
        )
        ouput.append(result)
        scaled_spectra.append(np.maximum(result.scaled_sa_g, np.finfo(float).tiny))
    matrix = np.vstack(scaled_spectra)
    suite_geo_mean = np.exp(np.mean(np.log(matrix), axis=0))
    return ouput, suite_geo_mean


def _moving_average(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    width = max(1, int(width))
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def frequency_domain_spectral_match(
    motion: Motion,
    target_periods: NDArray[np.float64],
    target_sa_g: NDArray[np.float64],
    *,
    damping: float = 0.05,
    iterations: int = 3,
    max_factor_per_iteration: float = 1.8,
    smoothing_width: int = 7,
    highpass_hz: float | None = 0.05,
    lowpass_hz: float | None = None,
) -> ScaleResult:
    """Fast approximate spectral matching by shaping Fourier amplitudes.

    This is useful for exploration and preconditioning. Response-spectrum
    compatibility still needs the same engineering checks as any matching method.
    """

    current = motion
    target_periods = np.asarray(target_periods, dtype=np.float64)
    target_sa_g = np.asarray(target_sa_g, dtype=np.float64)
    last_rec_sa = np.zeros_like(target_sa_g)
    total_factor = 1.0

    for _ in range(int(iterations)):
        spec = response_spectrum(current, target_periods, damping=damping)
        last_rec_sa = np.maximum(spec["sa_g"], np.finfo(float).tiny)
        ratios = np.clip(
            target_sa_g / last_rec_sa,
            1.0 / float(max_factor_per_iteration),
            float(max_factor_per_iteration),
        )
        ratios = _moving_average(ratios, smoothing_width)

        freq = np.fft.rfftfreq(current.npts, d=current.dt)
        fft = np.fft.rfft(current.accel)
        amp_factor = np.ones_like(freq)
        valid = freq > 0.0
        period_at_freq = np.empty_like(freq[valid])
        period_at_freq[:] = 1.0 / freq[valid]
        amp_factor[valid] = interpolate_spectrum(
            target_periods,
            ratios,
            period_at_freq,
            loglog=True,
        )
        amp_factor[0] = 1.0
        shaped = np.fft.irfft(fft * amp_factor, n=current.npts)
        current = current.with_accel(shaped - np.mean(shaped), name=f"{motion.name}_matched")
        if highpass_hz or lowpass_hz:
            current = butterworth_filter(
                current,
                highpass_hz=highpass_hz,
                lowpass_hz=lowpass_hz,
                order=4,
                zero_phase=True,
                taper_fraction=0.01,
                pad_seconds=3.0,
            ).motion

    final_spec = response_spectrum(current, target_periods, damping=damping)
    rms_ratio = np.exp(np.mean(np.log(np.maximum(target_sa_g, 1e-12) / np.maximum(final_spec["sa_g"], 1e-12))))
    total_factor *= float(rms_ratio)
    current = current.scaled(total_factor, name=f"{motion.name}_matched")
    final_sa = final_spec["sa_g"] * total_factor
    return ScaleResult(
        motion=current,
        factor=total_factor,
        periods=target_periods,
        record_sa_g=last_rec_sa,
        target_sa_g=target_sa_g,
        scaled_sa_g=final_sa,
        method="frequency_domain_spectral_match",
    )
