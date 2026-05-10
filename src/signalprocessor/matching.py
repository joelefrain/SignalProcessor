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
    wavelet_regularization: float = 0.05
    wavelet_max_adjustment_fraction: float = 0.35
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


def _relative_displacement_histories(
    acceleration: np.ndarray, dt: float, periods: np.ndarray, damping: float
) -> np.ndarray:
    acc = np.asarray(acceleration, dtype=np.float64)
    per = np.asarray(periods, dtype=np.float64)
    histories = np.zeros((per.size, acc.size), dtype=np.float64)
    beta = 0.25
    gamma = 0.5
    for j, period in enumerate(per):
        if period <= 1.0e-12:
            continue
        omega = 2.0 * np.pi / float(period)
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
        rel_acc = -acc[0] - c * v - k * u
        histories[j, 0] = u
        for i in range(acc.size - 1):
            p_next = -acc[i + 1]
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
            histories[j, i + 1] = u_next
            u = u_next
            v = v_next
            rel_acc = rel_acc_next
    return histories


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
    active = np.flatnonzero(mask)
    if active.size == 0:
        return record

    order = np.argsort(np.abs(log_error[active]))[::-1]
    selected = active[order[: cfg.max_periods_per_iteration]]
    selected_periods = spectrum.periods[selected]
    acc = record.acceleration_si()
    max_acc = max(float(np.max(np.abs(acc))), np.finfo(float).eps)
    peak_indices = np.searchsorted(record.time, peak_times[selected], side="left")
    peak_indices = np.clip(peak_indices, 0, record.npts - 1)
    current_histories = _relative_displacement_histories(
        acc, record.dt, selected_periods, target.damping
    )
    current_peak_disp = current_histories[np.arange(selected.size), peak_indices]
    signs = np.where(current_peak_disp < 0.0, -1.0, 1.0)
    target_sa_si = target.as_units("m/s^2").interpolate(spectrum.periods)[selected]
    omega = 2.0 * np.pi / selected_periods
    target_sd = target_sa_si / (omega * omega)
    desired = signs * target_sd - current_peak_disp

    waves = [
        gaussian_cosine_wavelet(
            record.time,
            center=float(peak_times[idx]),
            period=float(spectrum.periods[idx]),
            cycles=cfg.wavelet_cycles,
        )
        for idx in selected
    ]
    sensitivity = np.empty((selected.size, selected.size), dtype=np.float64)
    for col, wave in enumerate(waves):
        wave_histories = _relative_displacement_histories(
            wave, record.dt, selected_periods, target.damping
        )
        sensitivity[:, col] = wave_histories[np.arange(selected.size), peak_indices]

    if not np.all(np.isfinite(sensitivity)):
        return record

    singular = np.linalg.svd(sensitivity, compute_uv=False)
    lambda_reg = max(float(singular[0]) * cfg.wavelet_regularization, 1.0e-12)
    lhs = sensitivity.T @ sensitivity + (lambda_reg**2) * np.eye(selected.size)
    rhs = sensitivity.T @ desired
    amplitudes = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    max_amplitude = max(cfg.wavelet_max_adjustment_fraction * max_acc, 1.0e-4 * G0)
    amplitudes = np.clip(amplitudes, -max_amplitude, max_amplitude)

    adjustment_si = np.zeros(record.npts, dtype=np.float64)
    for amplitude, wave in zip(amplitudes, waves, strict=True):
        adjustment_si += float(amplitude) * wave
    adjustment_si *= cfg.relaxation

    current_rms = float(np.sqrt(np.mean(log_error[mask] * log_error[mask])))
    best_record = record
    best_rms = current_rms
    for step in (1.0, 0.5, 0.25, 0.125, 0.0625):
        trial = record.with_acceleration(acc + step * adjustment_si, units="m/s^2")
        trial_spectrum = response_spectrum(
            trial,
            spectrum.periods,
            damping=target.damping,
            output_units=spectrum.units,
        )
        trial_target_sa = target.as_units(trial_spectrum.units).interpolate(
            trial_spectrum.periods
        )
        trial_error = np.log(
            np.maximum(trial_target_sa, np.finfo(float).tiny)
            / np.maximum(trial_spectrum.sa, np.finfo(float).tiny)
        )
        trial_rms = float(np.sqrt(np.mean(trial_error[mask] * trial_error[mask])))
        if trial_rms < best_rms:
            best_record = trial
            best_rms = trial_rms
    return best_record


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
            if method == "hybrid":
                spec, peak_times = response_spectrum(
                    current,
                    target.periods,
                    damping=target.damping,
                    output_units=target.units,
                    return_peak_times=True,
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
