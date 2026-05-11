from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from .constants import EPS
from .metrics import arias_percentile_times, integrate_motion, motion_summary, post_event_slope
from .types import GroundMotion


@dataclass(slots=True)
class WindowSet:
    pre_event: tuple[float, float]
    strong_motion: tuple[float, float]
    post_event: tuple[float, float]

    def mask(self, motion: GroundMotion, name: str) -> np.ndarray:
        start, end = getattr(self, name)
        t = motion.time
        return (t >= start) & (t <= end)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "pre_event": self.pre_event,
            "strong_motion": self.strong_motion,
            "post_event": self.post_event,
        }


def detect_windows(
    motion: GroundMotion,
    *,
    lower_fraction: float = 0.05,
    upper_fraction: float = 0.95,
    pre_event_margin_s: float = 1.0,
    post_event_margin_s: float = 2.0,
) -> WindowSet:
    percent = arias_percentile_times(
        motion.acceleration_mps2,
        motion.dt,
        (lower_fraction * 100.0, upper_fraction * 100.0),
    )
    t5 = percent[lower_fraction * 100.0] + motion.time_start
    t95 = percent[upper_fraction * 100.0] + motion.time_start
    end = motion.time_start + motion.duration
    pre_end = max(motion.time_start + motion.dt, t5 - pre_event_margin_s)
    if pre_end <= motion.time_start + 3 * motion.dt:
        pre_end = min(end, motion.time_start + max(1.0, 0.05 * motion.duration))
    post_start = min(end - motion.dt, t95 + post_event_margin_s)
    if end - post_start <= 3 * motion.dt:
        post_start = max(motion.time_start, end - max(1.0, 0.1 * motion.duration))
    return WindowSet(
        pre_event=(motion.time_start, pre_end),
        strong_motion=(max(motion.time_start, t5), min(end, t95)),
        post_event=(post_start, end),
    )


def _tau(motion: GroundMotion) -> np.ndarray:
    if motion.duration <= 0:
        return np.zeros(motion.npts, dtype=float)
    return (motion.time - motion.time_start) / motion.duration


def _poly_design(tau: np.ndarray, order: int) -> np.ndarray:
    return np.column_stack([tau**k for k in range(order + 1)])


def remove_pre_event_mean(acc: np.ndarray, motion: GroundMotion, windows: WindowSet) -> tuple[np.ndarray, float]:
    mask = windows.mask(motion, "pre_event")
    if np.count_nonzero(mask) < 2:
        mean = float(np.mean(acc))
    else:
        mean = float(np.mean(acc[mask]))
    return acc - mean, mean


def weighted_polynomial_baseline(
    acc: np.ndarray,
    motion: GroundMotion,
    windows: WindowSet,
    order: int,
) -> np.ndarray:
    tau = _tau(motion)
    design = _poly_design(tau, order)
    weights = np.zeros(motion.npts, dtype=float)
    weights[windows.mask(motion, "pre_event")] = 1.0
    weights[windows.mask(motion, "post_event")] = 1.0
    if np.count_nonzero(weights) < order + 2:
        weights[:] = 1.0
        weights[windows.mask(motion, "strong_motion")] = 0.25
    sw = np.sqrt(weights)
    coeffs, *_ = np.linalg.lstsq(design * sw[:, None], acc * sw, rcond=None)
    return design @ coeffs


def constrained_polynomial_baseline(
    acc: np.ndarray,
    motion: GroundMotion,
    *,
    order: int = 1,
    target_final_velocity_mps: float = 0.0,
    target_final_displacement_m: float = 0.0,
) -> np.ndarray:
    velocity, displacement = integrate_motion(acc, motion.dt)
    required = np.array(
        [
            velocity[-1] - target_final_velocity_mps,
            displacement[-1] - target_final_displacement_m,
        ],
        dtype=float,
    )
    tau = _tau(motion)
    design = _poly_design(tau, order)
    contribution = np.empty((2, order + 1), dtype=float)
    for j in range(order + 1):
        vj, dj = integrate_motion(design[:, j], motion.dt)
        contribution[0, j] = vj[-1]
        contribution[1, j] = dj[-1]
    coeffs, *_ = np.linalg.lstsq(contribution, required, rcond=None)
    return design @ coeffs


def velocity_trend_derivative_baseline(
    acc: np.ndarray,
    motion: GroundMotion,
    windows: WindowSet,
    *,
    velocity_order: int = 2,
) -> np.ndarray:
    velocity, _ = integrate_motion(acc, motion.dt)
    tau = _tau(motion)
    design = _poly_design(tau, velocity_order)
    mask = windows.mask(motion, "post_event")
    if np.count_nonzero(mask) < velocity_order + 2:
        mask = np.ones(motion.npts, dtype=bool)
    coeffs, *_ = np.linalg.lstsq(design[mask], velocity[mask], rcond=None)
    baseline = np.zeros_like(acc)
    for k in range(1, velocity_order + 1):
        baseline += coeffs[k] * k * tau ** (k - 1) / max(motion.duration, EPS)
    return baseline


def piecewise_offset_baseline(
    acc: np.ndarray,
    motion: GroundMotion,
    breakpoint_s: tuple[float, float],
    *,
    target_final_velocity_mps: float = 0.0,
    target_final_displacement_m: float = 0.0,
) -> np.ndarray:
    t = motion.time
    t1, t2 = breakpoint_s
    basis = np.column_stack([(t >= t1) & (t < t2), t >= t2]).astype(float)
    velocity, displacement = integrate_motion(acc, motion.dt)
    required = np.array(
        [velocity[-1] - target_final_velocity_mps, displacement[-1] - target_final_displacement_m],
        dtype=float,
    )
    contribution = np.empty((2, 2), dtype=float)
    for j in range(2):
        vj, dj = integrate_motion(basis[:, j], motion.dt)
        contribution[0, j] = vj[-1]
        contribution[1, j] = dj[-1]
    coeffs, *_ = np.linalg.lstsq(contribution, required, rcond=None)
    return basis @ coeffs


def build_baseline_candidates(
    motion: GroundMotion,
    windows: WindowSet,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    acc0, removed_mean = remove_pre_event_mean(motion.acceleration_mps2, motion, windows)
    candidates: list[dict[str, Any]] = [
        {
            "name": "pre_event_mean_only",
            "baseline_mps2": np.zeros_like(acc0),
            "removed_mean_mps2": removed_mean,
            "complexity": 0,
        },
        {
            "name": "constant_velocity_final",
            "baseline_mps2": constrained_polynomial_baseline(acc0, motion, order=0),
            "removed_mean_mps2": removed_mean,
            "complexity": 1,
        },
    ]
    for order in (0, 1, 2):
        candidates.append(
            {
                "name": f"weighted_polynomial_order_{order}",
                "baseline_mps2": weighted_polynomial_baseline(acc0, motion, windows, order),
                "removed_mean_mps2": removed_mean,
                "complexity": 1 + order,
            }
        )
    candidates.append(
        {
            "name": "constrained_polynomial_vu_final",
            "baseline_mps2": constrained_polynomial_baseline(acc0, motion, order=1),
            "removed_mean_mps2": removed_mean,
            "complexity": 3,
        }
    )
    candidates.append(
        {
            "name": "velocity_trend_derivative",
            "baseline_mps2": velocity_trend_derivative_baseline(acc0, motion, windows, velocity_order=2),
            "removed_mean_mps2": removed_mean,
            "complexity": 3,
        }
    )
    return candidates


def _candidate_score(metrics: dict[str, float], complexity: int) -> tuple[float, bool]:
    pgv = max(metrics["PGV_mps"], EPS)
    pgd = max(metrics["PGD_m"], EPS)
    fv = abs(metrics["final_velocity_mps"]) / max(pgv, 1.0e-4)
    fd = abs(metrics["final_displacement_m"]) / max(pgd, 1.0e-4)
    slope = abs(metrics["post_event_velocity_slope_mps2"]) / max(metrics["PGA_mps2"], 1.0e-4)
    score = 2.0 * fv + 2.0 * fd + 10.0 * slope + 0.15 * complexity
    passed = fv < 0.12 and fd < 0.12 and abs(metrics["post_event_velocity_slope_mps2"]) < 1.0e-3
    return float(score), bool(passed)


def select_baseline_candidate(
    motion: GroundMotion,
    windows: WindowSet,
    config: dict[str, Any] | None = None,
) -> tuple[GroundMotion, dict[str, Any], list[dict[str, Any]]]:
    acc0, _ = remove_pre_event_mean(motion.acceleration_mps2, motion, windows)
    evaluated: list[dict[str, Any]] = []
    for candidate in build_baseline_candidates(motion, windows, config):
        corrected = acc0 - candidate["baseline_mps2"]
        metrics = motion_summary(corrected, motion.dt)
        score, passed = _candidate_score(metrics, candidate["complexity"])
        evaluated.append({**candidate, "metrics": metrics, "score": score, "passed": passed})
    passed = [item for item in evaluated if item["passed"]]
    selected = min(passed or evaluated, key=lambda item: (item["score"], item["complexity"]))
    corrected = acc0 - selected["baseline_mps2"]
    return motion.copy(acceleration_mps2=corrected, record_id=f"{motion.record_id}_baseline"), selected, evaluated


def make_taper(npts: int, fraction_total: float = 0.05) -> np.ndarray:
    if npts <= 2 or fraction_total <= 0:
        return np.ones(npts, dtype=float)
    alpha = min(1.0, max(0.0, float(fraction_total)))
    try:
        return signal.windows.tukey(npts, alpha=alpha)
    except AttributeError:
        edge = max(1, int(0.5 * alpha * npts))
        window = np.ones(npts, dtype=float)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, edge)))
        window[:edge] = ramp
        window[-edge:] = ramp[::-1]
        return window


def butterworth_filter(
    acceleration_mps2: np.ndarray,
    dt: float,
    *,
    highpass_hz: float | None = None,
    lowpass_hz: float | None = None,
    order: int = 4,
    zero_phase: bool = True,
    taper_fraction_total: float = 0.02,
) -> np.ndarray:
    acc = np.asarray(acceleration_mps2, dtype=float)
    fs = 1.0 / dt
    nyquist = 0.5 * fs
    high = None if highpass_hz is None or highpass_hz <= 0 else float(highpass_hz)
    low = None if lowpass_hz is None or lowpass_hz >= nyquist else float(lowpass_hz)
    if high is not None and high >= nyquist:
        raise ValueError("highpass_hz must be below Nyquist")
    if high is not None and low is not None and high >= low:
        raise ValueError("highpass_hz must be lower than lowpass_hz")
    if high is None and low is None:
        return acc.copy()
    if high is not None and low is not None:
        btype = "bandpass"
        wn: float | tuple[float, float] = (high, low)
    elif high is not None:
        btype = "highpass"
        wn = high
    else:
        btype = "lowpass"
        wn = low if low is not None else nyquist * 0.95
    tapered = (acc - np.mean(acc[: max(3, min(acc.size, int(2.0 / dt))) ])) * make_taper(
        acc.size, taper_fraction_total
    )
    sos = signal.butter(order, wn, btype=btype, fs=fs, output="sos")
    padlen = min(3 * (2 * sos.shape[0] + 1), max(0, acc.size - 2))
    if zero_phase:
        return signal.sosfiltfilt(sos, tapered, padlen=padlen)
    return signal.sosfilt(sos, tapered)


def _logspace_grid(min_hz: float, max_hz: float, num: int) -> np.ndarray:
    return np.geomspace(max(min_hz, 1.0e-4), max_hz, int(num))


def recommend_filter(
    baseline_motion: GroundMotion,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    filtering = cfg.get("noise_filtering", {})
    hp_cfg = filtering.get("highpass", {})
    lp_cfg = filtering.get("lowpass", {})
    fs = baseline_motion.sampling_rate_hz
    nyquist = 0.5 * fs
    lowpass = lp_cfg.get("frequency_hz")
    if lowpass is None:
        lowpass = min(
            float(lp_cfg.get("maximum_frequency_hz", 40.0)),
            float(lp_cfg.get("default_fraction_of_nyquist", 0.75)) * nyquist,
        )
    highpass = hp_cfg.get("initial_frequency_hz")
    thresholds = cfg.get("quality_control", {}).get("acceptance_thresholds", {})
    max_abs_final_velocity = float(thresholds.get("max_abs_final_velocity_m_s", 0.02))
    max_final_displacement_ratio = float(thresholds.get("max_final_displacement_to_pgd_ratio", 0.12))
    order = int(filtering.get("order_per_pass", filtering.get("order", 4)))
    if highpass is not None:
        return {"highpass_hz": float(highpass), "lowpass_hz": float(lowpass), "order": order, "reason": "manual"}

    grid_cfg = hp_cfg.get("search_grid_hz", {})
    grid = _logspace_grid(
        float(grid_cfg.get("min_hz", 0.02)),
        min(float(grid_cfg.get("max_hz", 0.5)), max(0.51, lowpass * 0.8)),
        int(grid_cfg.get("num", 24)),
    )
    tmax = hp_cfg.get("period_range_constraint", {}).get("analysis_tmax_s")
    ratio = float(hp_cfg.get("period_range_constraint", {}).get("minimum_cutoff_period_to_tmax_ratio", 1.5))
    if tmax:
        grid = grid[grid <= 1.0 / (ratio * float(tmax))]
        if grid.size == 0:
            grid = np.array([1.0 / (ratio * float(tmax))])
    best: dict[str, Any] | None = None
    for hp in grid:
        filtered = butterworth_filter(
            baseline_motion.acceleration_mps2,
            baseline_motion.dt,
            highpass_hz=float(hp),
            lowpass_hz=float(lowpass),
            order=order,
        )
        velocity, displacement = integrate_motion(filtered, baseline_motion.dt)
        pgd = max(np.max(np.abs(displacement)), EPS)
        score = abs(velocity[-1]) / max_abs_final_velocity + abs(displacement[-1]) / (
            max_final_displacement_ratio * pgd
        )
        passed = abs(velocity[-1]) <= max_abs_final_velocity and abs(displacement[-1]) <= max_final_displacement_ratio * pgd
        item = {
            "highpass_hz": float(hp),
            "lowpass_hz": float(lowpass),
            "order": order,
            "score": float(score),
            "passed": bool(passed),
        }
        if best is None or item["score"] < best["score"]:
            best = item
        if passed:
            item["reason"] = "lowest_highpass_passing_displacement_qc"
            return item
    assert best is not None
    best["reason"] = "best_available_highpass_no_candidate_passed_all_qc"
    return best

