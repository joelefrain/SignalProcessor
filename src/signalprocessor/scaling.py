from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .records import MotionRecord, Spectrum
from .spectra import response_spectrum

ScaleMethod = Literal["log", "logarithmic", "linear", "least_squares", "ls"]


@dataclass(frozen=True, slots=True)
class ScalingResult:
    record: MotionRecord
    factor: float
    original_spectrum: Spectrum
    scaled_spectrum: Spectrum
    target_spectrum: Spectrum
    max_abs_error: float
    rms_log_error: float


def _period_mask(
    periods: np.ndarray, t_min: float | None, t_max: float | None
) -> np.ndarray:
    mask = np.ones(periods.size, dtype=bool)
    if t_min is not None:
        mask &= periods >= t_min
    if t_max is not None:
        mask &= periods <= t_max
    return mask


def spectral_misfit(
    spectrum: Spectrum,
    target: Spectrum,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
) -> dict[str, float]:
    target_on_periods = target.as_units(spectrum.units).interpolate(spectrum.periods)
    mask = _period_mask(spectrum.periods, t_min, t_max)
    ratio = np.maximum(spectrum.sa[mask], np.finfo(float).tiny) / np.maximum(
        target_on_periods[mask], np.finfo(float).tiny
    )
    log_error = np.log(ratio)
    rel_error = ratio - 1.0
    return {
        "max_abs_error": float(np.max(np.abs(rel_error))) if rel_error.size else 0.0,
        "rms_log_error": float(np.sqrt(np.mean(log_error * log_error)))
        if log_error.size
        else 0.0,
        "mean_ratio": float(np.exp(np.mean(log_error))) if log_error.size else 1.0,
    }


def _period_weights(weights, periods: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected_count = int(np.count_nonzero(mask))
    if weights is None:
        return np.ones(selected_count, dtype=np.float64)

    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("weights must be a one-dimensional array")
    if arr.size == periods.size:
        selected = arr[mask]
    elif arr.size == selected_count:
        selected = arr
    else:
        raise ValueError(
            "weights must have the same length as spectrum periods or the selected scaling range"
        )
    if not np.all(np.isfinite(selected)):
        raise ValueError("weights must be finite")
    if np.any(selected < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(selected > 0.0):
        raise ValueError("at least one selected weight must be positive")
    return selected


def _normalize_scale_method(method: str) -> str:
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "log": "log",
        "logarithmic": "log",
        "logaritmico": "log",
        "logarítmico": "log",
        "linear": "linear",
        "least_squares": "linear",
        "ls": "linear",
        "minimos_cuadrados": "linear",
        "mínimos_cuadrados": "linear",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("method must be 'log' or 'linear'") from exc


def linear_scale_factor(
    spectrum: Spectrum,
    target: Spectrum,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    weights=None,
    method: ScaleMethod = "log",
) -> float:
    target_same = target.as_units(spectrum.units)
    target_sa = target_same.interpolate(spectrum.periods)
    mask = _period_mask(spectrum.periods, t_min, t_max)
    mask &= (spectrum.sa > 0.0) & (target_sa > 0.0)
    if not np.any(mask):
        raise ValueError("No positive spectral ordinates inside scaling range")
    w = _period_weights(weights, spectrum.periods, mask)
    selected_record = spectrum.sa[mask]
    selected_target = target_sa[mask]
    normalized_method = _normalize_scale_method(method)
    if normalized_method == "linear":
        numerator = np.sum(w * selected_record * selected_target)
        denominator = np.sum(w * selected_record * selected_record)
        if denominator <= 0.0:
            raise ValueError("Cannot compute a linear scale factor from zero spectra")
        return float(numerator / denominator)

    log_factor = np.sum(
        w * (np.log(selected_target) - np.log(selected_record))
    ) / np.sum(w)
    return float(np.exp(log_factor))


def linear_scale(
    record: MotionRecord,
    target: Spectrum,
    *,
    periods=None,
    damping: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    weights=None,
    method: ScaleMethod = "log",
) -> ScalingResult:
    damping = target.damping if damping is None else damping
    spectrum_periods = target.periods if periods is None else periods
    original = response_spectrum(
        record, spectrum_periods, damping=damping, output_units=target.units
    )
    factor = linear_scale_factor(
        original, target, t_min=t_min, t_max=t_max, weights=weights, method=method
    )
    scaled = record.with_acceleration(
        record.acceleration * factor,
        units=record.units,
        metadata={
            "scale_factor": factor,
            "scale_method": _normalize_scale_method(method),
        },
    )
    scaled_spec = response_spectrum(
        scaled, original.periods, damping=damping, output_units=target.units
    )
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


def suite_mean_spectrum(
    records: list[MotionRecord], periods, *, damping: float = 0.05, units: str = "g"
) -> Spectrum:
    specs = [
        response_spectrum(record, periods, damping=damping, output_units=units).sa
        for record in records
    ]
    return Spectrum(
        periods=np.asarray(periods, dtype=np.float64),
        sa=np.mean(np.vstack(specs), axis=0),
        units=units,
        damping=damping,
    )
