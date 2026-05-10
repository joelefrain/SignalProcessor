from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_motion, read_target_spectrum
from .matching import MatchingConfig, match_spectrum
from .metrics import compute_ground_motion_parameters
from .processing import CorrectionConfig, correct_record
from .recommendation import recommend_correction_method
from .scaling import linear_scale, spectral_misfit
from .spectra import response_spectrum


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    name: str
    pga_ratio: float
    rms_acc_error: float
    notes: str = ""


def compare_correction_to_usgs(
    uncorrected_path: str | Path,
    corrected_path: str | Path,
    *,
    config: CorrectionConfig | None = None,
) -> BenchmarkRow:
    raw = read_motion(uncorrected_path)
    if config is None:
        correction = recommend_correction_method(raw).best
        ours = correction.result.record.as_units("cm/s^2")
        notes = correction.name
    else:
        ours = correct_record(raw, config).record.as_units("cm/s^2")
        notes = "custom_config"
    reference = read_motion(corrected_path, units="cm/s^2")
    n = min(ours.npts, reference.npts)
    ref = reference.acceleration[:n]
    got = ours.acceleration[:n]
    scale = max(float(np.max(np.abs(ref))), np.finfo(float).eps)
    rms = float(np.sqrt(np.mean((got - ref) ** 2)) / scale)
    pga_ratio = float(np.max(np.abs(got)) / scale)
    return BenchmarkRow(
        name=Path(str(uncorrected_path)).stem,
        pga_ratio=pga_ratio,
        rms_acc_error=rms,
        notes=notes,
    )


def batch_usgs_benchmark(root: str | Path) -> pd.DataFrame:
    base = Path(root)
    rows = []
    for raw in sorted((base / "uncorrected_motion").glob("*_u.smc")):
        component = raw.name.replace("_u.smc", "")
        reference = base / "corrected_motion" / f"{component}_a.smc"
        if reference.exists():
            rows.append(asdict(compare_correction_to_usgs(raw, reference)))
    return pd.DataFrame(rows)


def compare_scaling_to_seismomatch(
    motion_path: str | Path,
    target_path: str | Path,
    seismomatch_path: str | Path | None = None,
    *,
    t_min: float = 0.2,
    t_max: float = 2.0,
    matched: bool = False,
) -> dict[str, float | str]:
    record = read_motion(motion_path)
    target = read_target_spectrum(target_path)
    if matched:
        match = match_spectrum(
            record,
            target,
            MatchingConfig(
                max_iterations=15, relaxation=0.35, t_min=t_min, t_max=t_max
            ),
        )
        scaled_record = match.record
        scaled_spectrum = match.spectrum
        factor = float("nan")
        max_abs_error = spectral_misfit(
            scaled_spectrum, target, t_min=t_min, t_max=t_max
        )["max_abs_error"]
        rms_log_error = spectral_misfit(
            scaled_spectrum, target, t_min=t_min, t_max=t_max
        )["rms_log_error"]
        method = "matched"
    else:
        result = linear_scale(record, target, t_min=t_min, t_max=t_max)
        scaled_record = result.record
        scaled_spectrum = result.scaled_spectrum
        factor = result.factor
        max_abs_error = result.max_abs_error
        rms_log_error = result.rms_log_error
        method = "linear"
    row: dict[str, float | str] = {
        "name": Path(str(motion_path)).stem,
        "method": method,
        "factor": factor,
        "max_abs_error": max_abs_error,
        "rms_log_error": rms_log_error,
    }
    if seismomatch_path is not None and Path(seismomatch_path).exists():
        sm = read_motion(seismomatch_path)
        spec_sm = response_spectrum(
            sm, target.periods, output_units=target.units, damping=target.damping
        )
        misfit_sm = spectral_misfit(spec_sm, target, t_min=t_min, t_max=t_max)
        ours = scaled_record.as_units("g")
        n = min(ours.npts, sm.npts)
        row["seismomatch_rms_log_error"] = misfit_sm["rms_log_error"]
        row["time_series_rms_vs_seismomatch"] = float(
            np.sqrt(np.mean((ours.acceleration[:n] - sm.acceleration[:n]) ** 2))
        )
    return row


def summarize_records(paths) -> pd.DataFrame:
    rows = []
    for path in paths:
        rec = read_motion(path)
        params = compute_ground_motion_parameters(rec)
        rows.append(
            {
                "name": Path(path).stem,
                "dt": rec.dt,
                "npts": rec.npts,
                "duration": rec.duration,
                "pga_g": params.pga / 9.80665,
                "pgv_mps": params.pgv,
                "arias_mps": params.arias_intensity,
                "d5_95_s": params.d5_95,
            }
        )
    return pd.DataFrame(rows)
