from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import save_dataframe_csv, save_motion_csv
from .metrics import integrate_motion, motion_summary
from .preprocessing import (
    WindowSet,
    build_baseline_candidates,
    butterworth_filter,
    constrained_polynomial_baseline,
    detect_windows,
    piecewise_offset_baseline,
    recommend_filter,
    remove_pre_event_mean,
    select_baseline_candidate,
    velocity_trend_derivative_baseline,
    weighted_polynomial_baseline,
)
from .spectra import make_period_grid, response_spectrum
from .types import GroundMotion, GroundMotionPair


@dataclass(slots=True)
class CorrectionResult:
    original: GroundMotion
    baseline_corrected: GroundMotion
    filtered: GroundMotion
    velocity_mps: np.ndarray
    displacement_m: np.ndarray
    windows: WindowSet
    baseline_parameters: dict[str, Any]
    filter_parameters: dict[str, Any]
    metrics: dict[str, float]
    candidate_table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        keys = [
            "PGA_g",
            "PGV_mps",
            "PGD_m",
            "final_velocity_mps",
            "final_displacement_m",
            "arias_intensity_mps",
            "D_5_95",
            "CAV_mps",
        ]
        return pd.DataFrame(
            {
                "record_id": [self.original.record_id],
                "baseline": [self.baseline_parameters.get("name", "manual")],
                "highpass_hz": [self.filter_parameters.get("highpass_hz")],
                "lowpass_hz": [self.filter_parameters.get("lowpass_hz")],
                **{key: [self.metrics.get(key)] for key in keys},
            }
        )

    def write_outputs(self, output_dir: str | Path, *, suffix: str = "_processed") -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = self.original.record_id
        paths = {
            "acceleration": save_motion_csv(self.filtered, root / f"{stem}{suffix}_acc.csv", acceleration_unit="g"),
            "candidate_table": save_dataframe_csv(self.candidate_table, root / f"{stem}{suffix}_baseline_candidates.csv"),
        }
        series = pd.DataFrame(
            {
                "time_s": self.filtered.time,
                "acceleration_g": self.filtered.acc_g,
                "velocity_m_s": self.velocity_mps,
                "displacement_m": self.displacement_m,
            }
        )
        paths["time_series"] = save_dataframe_csv(series, root / f"{stem}{suffix}_series.csv")
        return paths


@dataclass(slots=True)
class PairCorrectionResult:
    ns: CorrectionResult
    ew: CorrectionResult
    pair_id: str

    def summary(self) -> pd.DataFrame:
        out = pd.concat([self.ns.summary(), self.ew.summary()], ignore_index=True)
        out.insert(0, "pair_id", self.pair_id)
        return out


def _candidate_table(evaluated: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in evaluated:
        metrics = item.get("metrics", {})
        rows.append(
            {
                "name": item["name"],
                "complexity": item.get("complexity", 0),
                "score": item.get("score"),
                "passed": item.get("passed"),
                "removed_mean_mps2": item.get("removed_mean_mps2"),
                "PGA_g": metrics.get("PGA_g"),
                "PGV_mps": metrics.get("PGV_mps"),
                "PGD_m": metrics.get("PGD_m"),
                "final_velocity_mps": metrics.get("final_velocity_mps"),
                "final_displacement_m": metrics.get("final_displacement_m"),
                "post_event_velocity_slope_mps2": metrics.get("post_event_velocity_slope_mps2"),
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "complexity"], ignore_index=True)


def _manual_baseline(
    motion: GroundMotion,
    windows: WindowSet,
    baseline: dict[str, Any] | None,
) -> tuple[GroundMotion, dict[str, Any], pd.DataFrame]:
    params = baseline or {"method": "pre_event_mean_only"}
    method = params.get("method", "pre_event_mean_only")
    acc0, removed_mean = remove_pre_event_mean(motion.acceleration_mps2, motion, windows)
    if method in {"none", "pre_event_mean_only", "mean"}:
        baseline_vector = np.zeros_like(acc0)
        name = "pre_event_mean_only"
    elif method in {"polynomial", "weighted_polynomial"}:
        order = int(params.get("order", 1))
        baseline_vector = weighted_polynomial_baseline(acc0, motion, windows, order)
        name = f"manual_weighted_polynomial_order_{order}"
    elif method in {"constrained", "constrained_polynomial"}:
        order = int(params.get("order", 1))
        baseline_vector = constrained_polynomial_baseline(
            acc0,
            motion,
            order=order,
            target_final_velocity_mps=float(params.get("target_final_velocity_mps", 0.0)),
            target_final_displacement_m=float(params.get("target_final_displacement_m", 0.0)),
        )
        name = f"manual_constrained_polynomial_order_{order}"
    elif method in {"velocity_trend", "velocity_trend_derivative"}:
        order = int(params.get("velocity_order", params.get("order", 2)))
        baseline_vector = velocity_trend_derivative_baseline(acc0, motion, windows, velocity_order=order)
        name = f"manual_velocity_trend_derivative_order_{order}"
    elif method in {"piecewise", "piecewise_offset"}:
        breakpoints = params.get("breakpoints_s")
        if not breakpoints or len(breakpoints) != 2:
            breakpoints = (windows.strong_motion[0], windows.strong_motion[1])
        baseline_vector = piecewise_offset_baseline(acc0, motion, tuple(map(float, breakpoints)))
        name = "manual_piecewise_offset"
    else:
        raise ValueError(f"unsupported baseline method: {method}")
    corrected = acc0 - baseline_vector
    info = {
        "name": name,
        "method": method,
        "removed_mean_mps2": removed_mean,
        "baseline_peak_mps2": float(np.max(np.abs(baseline_vector))),
    }
    table = pd.DataFrame([{**info, **motion_summary(corrected, motion.dt)}])
    return motion.copy(acceleration_mps2=corrected, record_id=f"{motion.record_id}_baseline"), info, table


def _filter_params_from_config(config: dict[str, Any] | None, filtering: dict[str, Any] | None) -> dict[str, Any]:
    if filtering:
        return {
            "highpass_hz": filtering.get("highpass_hz"),
            "lowpass_hz": filtering.get("lowpass_hz"),
            "order": int(filtering.get("order", 4)),
            "reason": "manual",
        }
    cfg = config or {}
    noise = cfg.get("noise_filtering", {})
    hp = noise.get("highpass", {}).get("initial_frequency_hz")
    lp = noise.get("lowpass", {}).get("frequency_hz")
    if hp is None and lp is None:
        return {"highpass_hz": None, "lowpass_hz": None, "order": int(noise.get("order_per_pass", 4)), "reason": "none"}
    return {
        "highpass_hz": hp,
        "lowpass_hz": lp,
        "order": int(noise.get("order_per_pass", noise.get("order", 4))),
        "reason": "config",
    }


def _apply_filter_and_metrics(
    baseline_motion: GroundMotion,
    filter_params: dict[str, Any],
) -> tuple[GroundMotion, np.ndarray, np.ndarray, dict[str, float]]:
    filtered_acc = butterworth_filter(
        baseline_motion.acceleration_mps2,
        baseline_motion.dt,
        highpass_hz=filter_params.get("highpass_hz"),
        lowpass_hz=filter_params.get("lowpass_hz"),
        order=int(filter_params.get("order", 4)),
    )
    filtered = baseline_motion.copy(acceleration_mps2=filtered_acc, record_id=baseline_motion.record_id + "_filtered")
    velocity, displacement = integrate_motion(filtered_acc, baseline_motion.dt)
    metrics = motion_summary(filtered_acc, baseline_motion.dt)
    return filtered, velocity, displacement, metrics


def process_motion(
    motion: GroundMotion,
    *,
    config: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    filtering: dict[str, Any] | None = None,
    recommend: bool = True,
) -> CorrectionResult:
    windows_cfg = (config or {}).get("windows", {}).get("arias_detection", {})
    windows = detect_windows(
        motion,
        lower_fraction=float(windows_cfg.get("lower_fraction", 0.05)),
        upper_fraction=float(windows_cfg.get("upper_fraction", 0.95)),
        pre_event_margin_s=float(windows_cfg.get("pre_event_margin_s", 1.0)),
        post_event_margin_s=float(windows_cfg.get("post_event_margin_s", 2.0)),
    )
    if recommend:
        baseline_motion, selected, evaluated = select_baseline_candidate(motion, windows, config)
        baseline_params = {
            "name": selected["name"],
            "removed_mean_mps2": selected.get("removed_mean_mps2"),
            "score": selected.get("score"),
            "passed": selected.get("passed"),
        }
        table = _candidate_table(evaluated)
        filter_params = recommend_filter(baseline_motion, config)
    else:
        baseline_motion, baseline_params, table = _manual_baseline(motion, windows, baseline)
        filter_params = _filter_params_from_config(config, filtering)
    filtered, velocity, displacement, metrics = _apply_filter_and_metrics(baseline_motion, filter_params)
    return CorrectionResult(
        original=motion,
        baseline_corrected=baseline_motion,
        filtered=filtered,
        velocity_mps=velocity,
        displacement_m=displacement,
        windows=windows,
        baseline_parameters=baseline_params,
        filter_parameters=filter_params,
        metrics=metrics,
        candidate_table=table,
    )


def _refilter_result(result: CorrectionResult, filter_params: dict[str, Any]) -> CorrectionResult:
    filtered, velocity, displacement, metrics = _apply_filter_and_metrics(result.baseline_corrected, filter_params)
    return replace(result, filtered=filtered, velocity_mps=velocity, displacement_m=displacement, metrics=metrics, filter_parameters=filter_params)


def process_pair(
    ns: GroundMotion,
    ew: GroundMotion,
    *,
    config: dict[str, Any] | None = None,
    recommend: bool = True,
    shared_filter: bool = True,
    pair_id: str = "pair",
) -> PairCorrectionResult:
    pair = GroundMotionPair(ns, ew, pair_id=pair_id).aligned()
    ns_result = process_motion(pair.ns, config=config, recommend=recommend)
    ew_result = process_motion(pair.ew, config=config, recommend=recommend)
    if shared_filter:
        high_values = [
            x
            for x in (ns_result.filter_parameters.get("highpass_hz"), ew_result.filter_parameters.get("highpass_hz"))
            if x is not None
        ]
        low_values = [
            x
            for x in (ns_result.filter_parameters.get("lowpass_hz"), ew_result.filter_parameters.get("lowpass_hz"))
            if x is not None
        ]
        common = {
            "highpass_hz": max(high_values) if high_values else None,
            "lowpass_hz": min(low_values) if low_values else None,
            "order": max(int(ns_result.filter_parameters.get("order", 4)), int(ew_result.filter_parameters.get("order", 4))),
            "reason": "shared_pair_filter",
        }
        ns_result = _refilter_result(ns_result, common)
        ew_result = _refilter_result(ew_result, common)
    return PairCorrectionResult(ns=ns_result, ew=ew_result, pair_id=pair_id)


def correction_spectrum(result: CorrectionResult, periods_s: np.ndarray | None = None, damping: float = 0.05) -> pd.DataFrame:
    periods = make_period_grid(0.05, 5.0, 100) if periods_s is None else periods_s
    return response_spectrum(result.filtered.acceleration_mps2, result.filtered.dt, periods, damping)

