from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .records import MotionRecord, Spectrum
from .spectra import response_spectrum


@dataclass(frozen=True, slots=True)
class ScalingResult:
    record: MotionRecord
    factor: float
    original_spectrum: Spectrum
    scaled_spectrum: Spectrum
    target_spectrum: Spectrum
    max_abs_error: float
    rms_log_error: float


def _period_mask(periods: np.ndarray, t_min: float | None, t_max: float | None) -> np.ndarray:
    mask = np.ones(periods.size, dtype=bool)
    if t_min is not None:
        mask &= periods >= t_min
    if t_max is not None:
        mask &= periods <= t_max
    return mask


def spectral_misfit(spectrum: Spectrum, target: Spectrum, *, t_min: float | None = None, t_max: float | None = None) -> dict[str, float]:
    target_on_periods = target.as_units(spectrum.units).interpolate(spectrum.periods)
    mask = _period_mask(spectrum.periods, t_min, t_max)
    ratio = np.maximum(spectrum.sa[mask], np.finfo(float).tiny) / np.maximum(target_on_periods[mask], np.finfo(float).tiny)
    log_error = np.log(ratio)
    rel_error = ratio - 1.0
    return {
        "max_abs_error": float(np.max(np.abs(rel_error))) if rel_error.size else 0.0,
        "rms_log_error": float(np.sqrt(np.mean(log_error * log_error))) if log_error.size else 0.0,
        "mean_ratio": float(np.exp(np.mean(log_error))) if log_error.size else 1.0,
    }


def linear_scale_factor(
    spectrum: Spectrum,
    target: Spectrum,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    weights=None,
) -> float:
    target_same = target.as_units(spectrum.units)
    target_sa = target_same.interpolate(spectrum.periods)
    mask = _period_mask(spectrum.periods, t_min, t_max)
    mask &= (spectrum.sa > 0.0) & (target_sa > 0.0)
    if not np.any(mask):
        raise ValueError("No positive spectral ordinates inside scaling range")
    if weights is None:
        w = np.ones(np.count_nonzero(mask), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[mask]
    log_factor = np.sum(w * (np.log(target_sa[mask]) - np.log(spectrum.sa[mask]))) / np.sum(w)
    return float(np.exp(log_factor))


def linear_scale(
    record: MotionRecord,
    target: Spectrum,
    *,
    periods=None,
    damping: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
) -> ScalingResult:
    damping = target.damping if damping is None else damping
    spectrum_periods = target.periods if periods is None else periods
    original = response_spectrum(record, spectrum_periods, damping=damping, output_units=target.units)
    factor = linear_scale_factor(original, target, t_min=t_min, t_max=t_max)
    scaled = record.with_acceleration(record.acceleration * factor, units=record.units, metadata={"scale_factor": factor})
    scaled_spec = response_spectrum(scaled, original.periods, damping=damping, output_units=target.units)
    metrics = spectral_misfit(scaled_spec, target, t_min=t_min, t_max=t_max)
    return ScalingResult(
        record=scaled,
        factor=factor,
        original_spectrum=original,
        scaled_spectrum=scaled_spec,
        target_spectrum=target,
        max_abs_error=metrics["max_abs_error"],
        rms_log_error=metrics["rms_log_error"],
    )


def suite_mean_spectrum(records: list[MotionRecord], periods, *, damping: float = 0.05, units: str = "g") -> Spectrum:
    specs = [response_spectrum(record, periods, damping=damping, output_units=units).sa for record in records]
    return Spectrum(periods=np.asarray(periods, dtype=np.float64), sa=np.mean(np.vstack(specs), axis=0), units=units, damping=damping)
