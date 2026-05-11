from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import EPS, G
from .io import save_dataframe_csv, save_motion_csv
from .metrics import motion_summary
from .preprocessing import constrained_polynomial_baseline
from .spectra import interpolate_spectrum_loglog, oscillator_response, response_spectrum
from .types import GroundMotion


@dataclass(slots=True)
class ScalingResult:
    method: str
    input_motion: GroundMotion
    output_motion: GroundMotion
    periods_s: np.ndarray
    target_sa_g: np.ndarray
    initial_spectrum: pd.DataFrame
    final_spectrum: pd.DataFrame
    metrics: dict[str, float]
    details: dict[str, Any]

    def write_outputs(self, output_dir: str | Path, *, suffix: str | None = None) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        tag = suffix or f"_{self.method}"
        stem = self.input_motion.record_id
        paths = {
            "motion": save_motion_csv(self.output_motion, root / f"{stem}{tag}_acc.csv", acceleration_unit="g"),
            "spectrum": save_dataframe_csv(self.final_spectrum, root / f"{stem}{tag}_spectrum.csv"),
        }
        report = pd.DataFrame([{**{"method": self.method, "record_id": stem}, **self.metrics, **self.details}])
        paths["report"] = save_dataframe_csv(report, root / f"{stem}{tag}_report.csv")
        return paths


@dataclass(slots=True)
class ScalingComparison:
    results: dict[str, ScalingResult]
    summary: pd.DataFrame

    def write_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths = {"summary": save_dataframe_csv(self.summary, root / "scaling_comparison_summary.csv")}
        for result in self.results.values():
            paths.update({f"{result.method}_{k}": v for k, v in result.write_outputs(root).items()})
        return paths


def _period_mask(periods: np.ndarray, period_range_s: tuple[float, float] | None) -> np.ndarray:
    if period_range_s is None:
        return np.ones(periods.size, dtype=bool)
    lo, hi = period_range_s
    return (periods >= lo) & (periods <= hi)


def _control_periods(
    target_periods_s: np.ndarray,
    target_sa_g: np.ndarray,
    period_range_s: tuple[float, float] | None,
    max_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = _period_mask(target_periods_s, period_range_s)
    periods = np.asarray(target_periods_s, dtype=float)[mask]
    target = np.asarray(target_sa_g, dtype=float)[mask]
    if periods.size > max_periods:
        idx = np.unique(np.round(np.linspace(0, periods.size - 1, max_periods)).astype(int))
        periods = periods[idx]
        target = target[idx]
    return periods, target


def spectral_fit_metrics(current_sa_g: np.ndarray, target_sa_g: np.ndarray) -> dict[str, float]:
    current = np.maximum(np.asarray(current_sa_g, dtype=float), EPS)
    target = np.maximum(np.asarray(target_sa_g, dtype=float), EPS)
    rel = current / target - 1.0
    log_error = np.log(current / target)
    return {
        "max_abs_relative_error": float(np.max(np.abs(rel))),
        "mean_abs_relative_error": float(np.mean(np.abs(rel))),
        "rms_log_error": float(np.sqrt(np.mean(log_error**2))),
        "mean_log_bias": float(np.mean(log_error)),
    }


def _result_metrics(
    original: GroundMotion,
    output: GroundMotion,
    final_sa_g: np.ndarray,
    target_sa_g: np.ndarray,
    elapsed_s: float,
) -> dict[str, float]:
    before = motion_summary(original.acceleration_mps2, original.dt)
    after = motion_summary(output.acceleration_mps2, output.dt)
    fit = spectral_fit_metrics(final_sa_g, target_sa_g)
    return {
        **fit,
        "runtime_s": float(elapsed_s),
        "PGA_ratio": after["PGA_g"] / max(before["PGA_g"], EPS),
        "PGV_ratio": after["PGV_mps"] / max(before["PGV_mps"], EPS),
        "PGD_ratio": after["PGD_m"] / max(before["PGD_m"], EPS),
        "arias_ratio": after["arias_intensity_mps"] / max(before["arias_intensity_mps"], EPS),
        "D_5_95_ratio": after["D_5_95"] / max(before["D_5_95"], EPS),
        "CAV_ratio": after["CAV_mps"] / max(before["CAV_mps"], EPS),
    }


def linear_scale_factor(
    motion: GroundMotion,
    target_periods_s: np.ndarray,
    target_sa_g: np.ndarray,
    *,
    damping: float = 0.05,
    period_range_s: tuple[float, float] | None = None,
) -> tuple[float, pd.DataFrame, np.ndarray]:
    periods = np.asarray(target_periods_s, dtype=float)
    target = np.asarray(target_sa_g, dtype=float)
    spectrum = response_spectrum(motion.acceleration_mps2, motion.dt, periods, damping)
    mask = _period_mask(periods, period_range_s)
    current = np.maximum(spectrum["psa_g"].to_numpy(), EPS)
    ln_scale = np.mean(np.log(np.maximum(target[mask], EPS)) - np.log(current[mask]))
    return float(np.exp(ln_scale)), spectrum, current


def scale_linear(
    motion: GroundMotion,
    target_periods_s: np.ndarray,
    target_sa_g: np.ndarray,
    *,
    damping: float = 0.05,
    period_range_s: tuple[float, float] | None = None,
    scale_factor: float | None = None,
) -> ScalingResult:
    start = perf_counter()
    sf, initial_spectrum, _ = linear_scale_factor(
        motion,
        target_periods_s,
        target_sa_g,
        damping=damping,
        period_range_s=period_range_s,
    )
    if scale_factor is not None:
        sf = float(scale_factor)
    scaled = motion.copy(acceleration_mps2=motion.acceleration_mps2 * sf, record_id=f"{motion.record_id}_linear")
    final_spectrum = response_spectrum(scaled.acceleration_mps2, scaled.dt, target_periods_s, damping)
    elapsed = perf_counter() - start
    metrics = _result_metrics(motion, scaled, final_spectrum["psa_g"].to_numpy(), target_sa_g, elapsed)
    return ScalingResult(
        method="linear",
        input_motion=motion,
        output_motion=scaled,
        periods_s=np.asarray(target_periods_s, dtype=float),
        target_sa_g=np.asarray(target_sa_g, dtype=float),
        initial_spectrum=initial_spectrum,
        final_spectrum=final_spectrum,
        metrics=metrics,
        details={"scale_factor": sf},
    )


def scale_to_pga(motion: GroundMotion, target_pga_g: float) -> ScalingResult:
    pga = np.max(np.abs(motion.acc_g))
    sf = float(target_pga_g) / max(float(pga), EPS)
    periods = np.geomspace(0.05, 5.0, 80)
    target = np.full_like(periods, np.nan, dtype=float)
    initial = response_spectrum(motion.acceleration_mps2, motion.dt, periods)
    output = motion.copy(acceleration_mps2=motion.acceleration_mps2 * sf, record_id=f"{motion.record_id}_pga")
    final = response_spectrum(output.acceleration_mps2, output.dt, periods)
    return ScalingResult(
        method="pga",
        input_motion=motion,
        output_motion=output,
        periods_s=periods,
        target_sa_g=target,
        initial_spectrum=initial,
        final_spectrum=final,
        metrics={"runtime_s": 0.0, "scale_factor": sf},
        details={"scale_factor": sf, "target_pga_g": target_pga_g},
    )


def _smooth_log_gain(gain: np.ndarray, bins: int = 7) -> np.ndarray:
    if bins <= 1 or gain.size < 3:
        return gain
    bins = int(min(bins, gain.size if gain.size % 2 == 1 else gain.size - 1))
    kernel = np.ones(bins, dtype=float) / bins
    return np.exp(np.convolve(np.log(np.maximum(gain, EPS)), kernel, mode="same"))


def frequency_domain_match(
    motion: GroundMotion,
    target_periods_s: np.ndarray,
    target_sa_g: np.ndarray,
    *,
    damping: float = 0.05,
    period_range_s: tuple[float, float] | None = None,
    gain_limits: tuple[float, float] = (0.35, 2.8),
    blend_exponent: float = 0.65,
) -> ScalingResult:
    start = perf_counter()
    linear = scale_linear(
        motion,
        target_periods_s,
        target_sa_g,
        damping=damping,
        period_range_s=period_range_s,
    )
    acc = linear.output_motion.acceleration_mps2
    current = linear.final_spectrum["psa_g"].to_numpy()
    target = np.asarray(target_sa_g, dtype=float)
    periods = np.asarray(target_periods_s, dtype=float)
    ratio = np.clip(target / np.maximum(current, EPS), gain_limits[0], gain_limits[1])
    freq_control = 1.0 / periods[::-1]
    ratio_control = ratio[::-1]
    freq = np.fft.rfftfreq(acc.size, motion.dt)
    band = freq > 0
    if period_range_s is not None:
        tmin, tmax = period_range_s
        band &= (freq >= 1.0 / tmax) & (freq <= 1.0 / tmin)
    interp_gain = np.ones_like(freq)
    interp_gain[band] = np.interp(
        np.log(freq[band]),
        np.log(freq_control),
        ratio_control,
        left=1.0,
        right=1.0,
    )
    interp_gain = _smooth_log_gain(np.clip(interp_gain, gain_limits[0], gain_limits[1]))
    interp_gain = interp_gain**blend_exponent
    spectrum = np.fft.rfft(acc - np.mean(acc))
    matched_acc = np.fft.irfft(spectrum * interp_gain, n=acc.size)
    matched_acc -= np.mean(matched_acc[: max(3, int(0.05 * matched_acc.size))])
    output = motion.copy(acceleration_mps2=matched_acc, record_id=f"{motion.record_id}_frequency")
    final_spectrum = response_spectrum(output.acceleration_mps2, output.dt, target_periods_s, damping)
    elapsed = perf_counter() - start
    metrics = _result_metrics(motion, output, final_spectrum["psa_g"].to_numpy(), target_sa_g, elapsed)
    return ScalingResult(
        method="frequency",
        input_motion=motion,
        output_motion=output,
        periods_s=np.asarray(target_periods_s, dtype=float),
        target_sa_g=np.asarray(target_sa_g, dtype=float),
        initial_spectrum=linear.initial_spectrum,
        final_spectrum=final_spectrum,
        metrics=metrics,
        details={"pre_scale_factor": linear.details["scale_factor"], "blend_exponent": blend_exponent},
    )


def tapered_cosine_wavelet(
    npts: int,
    dt: float,
    center_s: float,
    period_s: float,
    *,
    cycles: float = 4.0,
) -> np.ndarray:
    time = np.arange(npts, dtype=float) * dt
    x = time - center_s
    half_width = max(1.5 * dt, 0.5 * cycles * period_s)
    inside = np.abs(x) <= half_width
    wavelet = np.zeros(npts, dtype=float)
    window = 0.5 * (1.0 + np.cos(np.pi * x[inside] / half_width))
    wavelet[inside] = window * np.cos(2.0 * np.pi * x[inside] / period_s)
    wavelet -= np.mean(wavelet)
    peak = np.max(np.abs(wavelet))
    if peak > EPS:
        wavelet /= peak
    return wavelet


def wavelet_match(
    motion: GroundMotion,
    target_periods_s: np.ndarray,
    target_sa_g: np.ndarray,
    *,
    damping: float = 0.05,
    period_range_s: tuple[float, float] | None = None,
    tolerance: float = 0.05,
    max_iterations: int = 18,
    max_waves_per_iteration: int = 8,
    max_control_periods: int = 60,
    relaxation: float = 0.75,
    off_diagonal_reduction: float = 0.7,
    post_baseline_correction: bool = True,
) -> ScalingResult:
    start = perf_counter()
    periods, target = _control_periods(target_periods_s, target_sa_g, period_range_s, max_control_periods)
    initial_linear = scale_linear(motion, periods, target, damping=damping, period_range_s=period_range_s)
    acc = initial_linear.output_motion.acceleration_mps2.copy()
    initial_spectrum = response_spectrum(motion.acceleration_mps2, motion.dt, periods, damping)
    history: list[dict[str, float]] = []
    iterations = 0
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        base = response_spectrum(acc, motion.dt, periods, damping)
        current = np.maximum(base["psa_g"].to_numpy(), EPS)
        log_error = np.log(current / np.maximum(target, EPS))
        max_error = float(np.max(np.abs(current / target - 1.0)))
        rms_error = float(np.sqrt(np.mean(log_error**2)))
        history.append({"iteration": iterations, "max_abs_relative_error": max_error, "rms_log_error": rms_error})
        if max_error <= tolerance:
            break
        deficit = np.flatnonzero(current < target * (1.0 - 0.5 * tolerance))
        if deficit.size:
            selected_pool = deficit[np.argsort(np.abs(log_error[deficit]))]
        else:
            selected_pool = np.argsort(np.abs(log_error))
        selected = selected_pool[-max_waves_per_iteration:]
        selected = selected[np.argsort(periods[selected])]
        selected_periods = periods[selected]
        selected_target = target[selected]
        selected_current = current[selected]
        response = oscillator_response(acc, motion.dt, selected_periods, damping, keep_history=False)
        peak_times = response["peak_index"] * motion.dt
        basis = np.column_stack(
            [
                tapered_cosine_wavelet(acc.size, motion.dt, float(t0), float(period), cycles=4.0)
                for t0, period in zip(peak_times, selected_periods)
            ]
        )
        eps_amp = max(0.01 * G, 0.03 * float(np.max(np.abs(acc))))
        base_selected = response_spectrum(acc, motion.dt, selected_periods, damping)["psa_g"].to_numpy()
        cmat = np.empty((selected_periods.size, selected_periods.size), dtype=float)
        for j in range(selected_periods.size):
            perturbed = acc + eps_amp * basis[:, j]
            spec_j = response_spectrum(perturbed, motion.dt, selected_periods, damping)["psa_g"].to_numpy()
            cmat[:, j] = (spec_j - base_selected) / eps_amp
        if off_diagonal_reduction != 1.0:
            diag = np.diag(np.diag(cmat))
            cmat = diag + off_diagonal_reduction * (cmat - diag)
        delta = relaxation * (selected_target - selected_current)
        scale = max(float(np.linalg.norm(cmat)), EPS)
        reg = 1.0e-4 * scale
        lhs = np.vstack([cmat, reg * np.eye(cmat.shape[1])])
        rhs = np.concatenate([delta, np.zeros(cmat.shape[1])])
        amps, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        max_amp = max(0.01 * G, 0.08 * float(np.max(np.abs(acc))))
        amps = np.clip(amps, -max_amp, max_amp)
        base_score = rms_error + 0.15 * max_error
        accepted = False
        for step in (1.0, 0.5, 0.25, 0.1):
            trial = acc + basis @ (step * amps)
            trial -= np.mean(trial[: max(3, int(0.05 * trial.size))])
            trial_spec = response_spectrum(trial, motion.dt, periods, damping)["psa_g"].to_numpy()
            trial_fit = spectral_fit_metrics(trial_spec, target)
            trial_score = trial_fit["rms_log_error"] + 0.15 * trial_fit["max_abs_relative_error"]
            if trial_score < base_score:
                acc = trial
                accepted = True
                break
        if not accepted:
            break
    if post_baseline_correction:
        tmp_motion = motion.copy(acceleration_mps2=acc, record_id=f"{motion.record_id}_wavelet_tmp")
        acc = acc - constrained_polynomial_baseline(acc, tmp_motion, order=1)
    output = motion.copy(acceleration_mps2=acc, record_id=f"{motion.record_id}_wavelet")
    final_spectrum = response_spectrum(output.acceleration_mps2, output.dt, periods, damping)
    elapsed = perf_counter() - start
    metrics = _result_metrics(motion, output, final_spectrum["psa_g"].to_numpy(), target, elapsed)
    return ScalingResult(
        method="wavelet",
        input_motion=motion,
        output_motion=output,
        periods_s=periods,
        target_sa_g=target,
        initial_spectrum=initial_spectrum,
        final_spectrum=final_spectrum,
        metrics=metrics,
        details={
            "pre_scale_factor": initial_linear.details["scale_factor"],
            "iterations": iterations,
            "tolerance": tolerance,
            "convergence_history": history,
        },
    )


def compare_scaling_methods(
    motion: GroundMotion,
    target_periods_s: np.ndarray,
    target_sa_g: np.ndarray,
    *,
    methods: Iterable[str] = ("linear", "frequency", "wavelet"),
    damping: float = 0.05,
    period_range_s: tuple[float, float] | None = (0.05, 2.0),
    config: dict[str, Any] | None = None,
) -> ScalingComparison:
    cfg = config or {}
    spectral_cfg = cfg.get("spectral_matching", {})
    tolerance = float(spectral_cfg.get("mismatch_tolerance", 0.05))
    max_iterations = int(min(spectral_cfg.get("max_iterations", 8), 40))
    max_waves = int(spectral_cfg.get("wavelets", {}).get("max_number_of_waves", 8))
    target_periods_s = np.asarray(target_periods_s, dtype=float)
    target_sa_g = np.asarray(target_sa_g, dtype=float)
    mask = _period_mask(target_periods_s, period_range_s)
    periods = target_periods_s[mask]
    target = interpolate_spectrum_loglog(target_periods_s, target_sa_g, periods)

    results: dict[str, ScalingResult] = {}
    for method in methods:
        key = method.lower()
        if key == "linear":
            results[key] = scale_linear(motion, periods, target, damping=damping, period_range_s=period_range_s)
        elif key in {"frequency", "fourier"}:
            results["frequency"] = frequency_domain_match(
                motion,
                periods,
                target,
                damping=damping,
                period_range_s=period_range_s,
            )
        elif key == "wavelet":
            results[key] = wavelet_match(
                motion,
                periods,
                target,
                damping=damping,
                period_range_s=period_range_s,
                tolerance=tolerance,
                max_iterations=max_iterations,
                max_waves_per_iteration=max(1, min(max_waves, 10)),
            )
        else:
            raise ValueError(f"unsupported scaling method: {method}")
    rows = []
    for result in results.values():
        rows.append(
            {
                "method": result.method,
                "record_id": motion.record_id,
                **result.metrics,
                **{k: v for k, v in result.details.items() if isinstance(v, (int, float, str, bool))},
            }
        )
    return ScalingComparison(results=results, summary=pd.DataFrame(rows).sort_values("rms_log_error", ignore_index=True))
