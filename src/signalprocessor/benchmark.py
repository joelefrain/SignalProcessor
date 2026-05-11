from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from .correction import process_motion
from .io import read_motion_csv, read_seismomatch_txt, read_smc, read_target_spectrum_csv
from .metrics import motion_summary
from .scaling import compare_scaling_methods, spectral_fit_metrics
from .spectra import make_period_grid, response_spectrum


def _safe_ratio(value: float, reference: float) -> float:
    return float(value / reference) if abs(reference) > 1.0e-12 else float("nan")


def benchmark_correction(
    examples_root: str | Path = "examples",
    *,
    components: Iterable[str] = ("HNE", "HNN", "HNZ"),
    periods_s: np.ndarray | None = None,
) -> pd.DataFrame:
    root = Path(examples_root)
    unc_dir = root / "data" / "benchmark" / "uncorrected_motion"
    ref_dir = root / "data" / "benchmark" / "corrected_motion"
    periods = periods_s if periods_s is not None else make_period_grid(0.05, 3.0, 60)
    rows = []
    for component in components:
        uncorrected = unc_dir / f"CCSP.{component}.._u.smc"
        reference = ref_dir / f"CCSP.{component}.._a.smc"
        if not uncorrected.exists() or not reference.exists():
            continue
        motion = read_smc(uncorrected)
        ref_motion = read_smc(reference)
        n = min(motion.npts, ref_motion.npts)
        motion = motion.copy(acceleration_mps2=motion.acceleration_mps2[:n], record_id=component)
        ref_motion = ref_motion.copy(acceleration_mps2=ref_motion.acceleration_mps2[:n], record_id=f"{component}_reference")
        start = perf_counter()
        result = process_motion(
            motion,
            baseline={"method": "polynomial", "order": 1},
            filtering={"highpass_hz": 0.04, "lowpass_hz": 35.0, "order": 4},
            recommend=False,
        )
        runtime = perf_counter() - start
        ours_summary = motion_summary(result.filtered.acceleration_mps2, result.filtered.dt)
        ref_summary = motion_summary(ref_motion.acceleration_mps2, ref_motion.dt)
        ours_spec = response_spectrum(result.filtered.acceleration_mps2, result.filtered.dt, periods)
        ref_spec = response_spectrum(ref_motion.acceleration_mps2, ref_motion.dt, periods)
        fit = spectral_fit_metrics(ours_spec["psa_g"].to_numpy(), np.maximum(ref_spec["psa_g"].to_numpy(), 1.0e-12))
        rows.append(
            {
                "benchmark": "correction",
                "component": component,
                "runtime_s": runtime,
                "baseline": result.baseline_parameters["name"],
                "highpass_hz": result.filter_parameters["highpass_hz"],
                "lowpass_hz": result.filter_parameters["lowpass_hz"],
                "PGA_reference_g": ref_summary["PGA_g"],
                "PGA_signalprocessor_g": ours_summary["PGA_g"],
                "PGA_ratio_to_reference": _safe_ratio(ours_summary["PGA_g"], ref_summary["PGA_g"]),
                "PGV_ratio_to_reference": _safe_ratio(ours_summary["PGV_mps"], ref_summary["PGV_mps"]),
                "PGD_ratio_to_reference": _safe_ratio(ours_summary["PGD_m"], ref_summary["PGD_m"]),
                "spectrum_rms_log_error_to_reference": fit["rms_log_error"],
                "spectrum_max_relative_error_to_reference": fit["max_abs_relative_error"],
            }
        )
    return pd.DataFrame(rows)


def benchmark_scaling(
    examples_root: str | Path = "examples",
    *,
    records: Iterable[str] = ("LIMANS", "LIMAEW"),
    methods: Iterable[str] = ("linear", "frequency", "wavelet"),
    period_range_s: tuple[float, float] = (0.05, 2.0),
) -> pd.DataFrame:
    root = Path(examples_root)
    target = read_target_spectrum_csv(root / "data" / "target_response_spectrum" / "EPU_475.csv")
    rows = []
    for record_id in records:
        motion_path = root / "data" / "motion" / f"{record_id}.csv"
        if not motion_path.exists():
            continue
        motion = read_motion_csv(motion_path, acceleration_unit="g", record_id=record_id)
        comparison = compare_scaling_methods(
            motion,
            target.periods_s,
            target.sa_g,
            methods=methods,
            period_range_s=period_range_s,
        )
        summary = comparison.summary.copy()
        summary.insert(0, "benchmark", "scaling")
        reference_file = root / "data" / "benchmark" / "scaled_motion" / f"{record_id}.txt"
        if reference_file.exists():
            reference = read_seismomatch_txt(reference_file, acceleration_unit="g")
            periods = comparison.results[next(iter(comparison.results))].periods_s
            target_sa = comparison.results[next(iter(comparison.results))].target_sa_g
            ref_spec = response_spectrum(reference.acceleration_mps2, reference.dt, periods)
            ref_fit = spectral_fit_metrics(ref_spec["psa_g"].to_numpy(), target_sa)
            summary["reference_rms_log_error_to_target"] = ref_fit["rms_log_error"]
            summary["reference_max_relative_error_to_target"] = ref_fit["max_abs_relative_error"]
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def benchmark_all(examples_root: str | Path = "examples") -> dict[str, pd.DataFrame]:
    return {
        "correction": benchmark_correction(examples_root),
        "scaling": benchmark_scaling(examples_root),
    }

