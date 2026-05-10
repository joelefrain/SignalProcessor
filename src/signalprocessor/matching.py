from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import G0
from .processing import CorrectionConfig, correct_record
from .records import MotionRecord, Spectrum
from .scaling import linear_scale, spectral_misfit
from .spectra import response_spectrum


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    method: str = "frequency"
    max_iterations: int = 15
    tolerance_log: float = np.log(1.05)
    relaxation: float = 0.35
    t_min: float | None = None
    t_max: float | None = None
    max_periods_per_iteration: int = 10
    wavelet_cycles: float = 3.0
    frequency_clip: tuple[float, float] = (0.2, 5.0)
    preserve_baseline: bool = True
    initial_linear_scale: bool = True


@dataclass(frozen=True, slots=True)
class MatchingResult:
    record: MotionRecord
    target: Spectrum
    spectrum: Spectrum
    iterations: int
    converged: bool
    history: list[dict[str, float]] = field(default_factory=list)


def gaussian_cosine_wavelet(
    time: np.ndarray, center: float, period: float, *, cycles: float = 3.0
) -> np.ndarray:
    width = max(period * cycles / 2.0, period)
    x = time - center
    env = np.exp(-0.5 * (x / width) ** 2)
    wave = env * np.cos(2.0 * np.pi * x / period)
    wave -= np.mean(wave)
    peak = np.max(np.abs(wave))
    if peak > 0.0:
        wave = wave / peak
    return wave


def _baseline_safe(record: MotionRecord) -> MotionRecord:
    cfg_corr = CorrectionConfig(
        remove_mean=True,
        pre_event_seconds=None,
        baseline_order=1,
        constrain_final_velocity=True,
        constrain_final_displacement=False,
        despike=False,
        taper_fraction=0.0,
        highpass_hz=None,
        lowpass_hz=None,
    )
    return correct_record(record, cfg_corr).record


def _frequency_domain_update(
    record: MotionRecord,
    spectrum: Spectrum,
    target: Spectrum,
    *,
    relaxation: float,
    clip: tuple[float, float],
    mask: np.ndarray,
) -> MotionRecord:
    target_sa = target.as_units(spectrum.units).interpolate(spectrum.periods)
    ratio = target_sa / np.maximum(spectrum.sa, np.finfo(float).tiny)
    ratio = np.where(mask, ratio, 1.0)
    ratio = np.clip(ratio, clip[0], clip[1])

    control_freqs = 1.0 / spectrum.periods
    order = np.argsort(control_freqs)
    control_freqs = control_freqs[order]
    log_ratio = np.log(ratio[order])

    acc = record.acceleration_si()
    freqs = np.fft.rfftfreq(record.npts, record.dt)
    fft = np.fft.rfft(acc)
    multiplier = np.ones_like(freqs)
    mask = (freqs >= control_freqs[0]) & (freqs <= control_freqs[-1]) & (freqs > 0.0)
    if np.any(mask):
        multiplier[mask] = np.exp(
            np.interp(np.log(freqs[mask]), np.log(control_freqs), log_ratio)
        )
    adjusted = np.fft.irfft(fft * np.power(multiplier, relaxation), n=record.npts)
    return record.with_acceleration(adjusted, units="m/s^2")


def _wavelet_update(
    record: MotionRecord,
    spectrum: Spectrum,
    target: Spectrum,
    peak_times: np.ndarray,
    cfg: MatchingConfig,
) -> MotionRecord:
    target_sa = target.as_units(spectrum.units).interpolate(spectrum.periods)
    log_error = np.log(
        np.maximum(target_sa, np.finfo(float).tiny)
        / np.maximum(spectrum.sa, np.finfo(float).tiny)
    )
    mask = _matching_mask(spectrum.periods, cfg.t_min, cfg.t_max)
    ranked = np.argsort(np.where(mask, np.abs(log_error), -np.inf))[::-1]
    selected = ranked[: cfg.max_periods_per_iteration]
    adjustment_si = np.zeros(record.npts, dtype=np.float64)
    for idx in selected:
        period = float(target.periods[idx])
        wave = gaussian_cosine_wavelet(
            record.time,
            center=float(peak_times[idx]),
            period=period,
            cycles=cfg.wavelet_cycles,
        )
        desired_delta_si = (
            cfg.relaxation
            * log_error[idx]
            * target_sa[idx]
            * (G0 if spectrum.units == "g" else 1.0)
        )
        adjustment_si += desired_delta_si * wave
    return record.with_acceleration(
        record.acceleration_si() + adjustment_si, units="m/s^2"
    )


def _matching_mask(
    periods: np.ndarray, t_min: float | None, t_max: float | None
) -> np.ndarray:
    mask = np.ones(periods.size, dtype=bool)
    if t_min is not None:
        mask &= periods >= t_min
    if t_max is not None:
        mask &= periods <= t_max
    return mask


def match_spectrum(
    record: MotionRecord, target: Spectrum, config: MatchingConfig | None = None
) -> MatchingResult:
    cfg = config or MatchingConfig()
    method = cfg.method.lower()
    if method not in {"frequency", "wavelet", "hybrid"}:
        raise ValueError(
            "MatchingConfig.method must be 'frequency', 'wavelet', or 'hybrid'"
        )
    current = record
    if cfg.initial_linear_scale:
        current = linear_scale(
            record, target, periods=target.periods, t_min=cfg.t_min, t_max=cfg.t_max
        ).record

    history: list[dict[str, float]] = []
    converged = False
    spec = response_spectrum(
        current, target.periods, damping=target.damping, output_units=target.units
    )

    for iteration in range(cfg.max_iterations):
        spec, peak_times = response_spectrum(
            current,
            target.periods,
            damping=target.damping,
            output_units=target.units,
            return_peak_times=True,
        )
        target_sa = target.as_units(spec.units).interpolate(spec.periods)
        log_error = np.log(
            np.maximum(target_sa, np.finfo(float).tiny)
            / np.maximum(spec.sa, np.finfo(float).tiny)
        )
        mask = _matching_mask(spec.periods, cfg.t_min, cfg.t_max)
        active_error = log_error[mask]
        metrics = spectral_misfit(spec, target, t_min=cfg.t_min, t_max=cfg.t_max)
        history.append(
            {
                "iteration": float(iteration),
                "max_abs_log_error": float(np.max(np.abs(active_error))),
                "rms_log_error": metrics["rms_log_error"],
            }
        )
        if np.max(np.abs(active_error)) <= cfg.tolerance_log:
            converged = True
            break

        if method in {"frequency", "hybrid"}:
            current = _frequency_domain_update(
                current,
                spec,
                target,
                relaxation=cfg.relaxation,
                clip=cfg.frequency_clip,
                mask=mask,
            )
        if method in {"wavelet", "hybrid"}:
            current = _wavelet_update(current, spec, target, peak_times, cfg)
        current = current.with_acceleration(
            current.acceleration_si(),
            units="m/s^2",
            metadata={"matching_iteration": iteration + 1, "matching_method": method},
        )
        if cfg.preserve_baseline:
            current = _baseline_safe(current)

    spec = response_spectrum(
        current, target.periods, damping=target.damping, output_units=target.units
    )
    return MatchingResult(
        record=current,
        target=target,
        spectrum=spec,
        iterations=len(history),
        converged=converged,
        history=history,
    )
