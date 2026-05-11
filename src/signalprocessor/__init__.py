from __future__ import annotations

from .benchmark import benchmark_all, benchmark_correction, benchmark_scaling
from .constants import G
from .correction import CorrectionResult, PairCorrectionResult, correction_spectrum, process_motion, process_pair
from .io import (
    load_json,
    read_motion_csv,
    read_seismomatch_txt,
    read_smc,
    read_target_spectrum_csv,
    save_dataframe_csv,
    save_motion_csv,
)
from .metrics import (
    arias_intensity,
    arias_percentile_times,
    cumulative_absolute_velocity,
    cumulative_trapezoid,
    fourier_amplitude_spectrum,
    integrate_motion,
    motion_summary,
    peak_ground_values,
    significant_durations,
)
from .scaling import (
    ScalingComparison,
    ScalingResult,
    compare_scaling_methods,
    frequency_domain_match,
    linear_scale_factor,
    scale_linear,
    scale_to_pga,
    spectral_fit_metrics,
    wavelet_match,
)
from .spectra import interpolate_spectrum_loglog, make_period_grid, oscillator_response, response_spectrum
from .types import GroundMotion, GroundMotionPair, TargetSpectrum

__all__ = [
    "GroundMotion",
    "GroundMotionPair",
    "TargetSpectrum",
    "CorrectionResult",
    "PairCorrectionResult",
    "ScalingResult",
    "ScalingComparison",
    "read_motion_csv",
    "read_target_spectrum_csv",
    "read_seismomatch_txt",
    "read_smc",
    "load_json",
    "save_motion_csv",
    "save_dataframe_csv",
    "process_motion",
    "process_pair",
    "correction_spectrum",
    "scale_linear",
    "scale_to_pga",
    "frequency_domain_match",
    "wavelet_match",
    "compare_scaling_methods",
    "linear_scale_factor",
    "spectral_fit_metrics",
    "response_spectrum",
    "oscillator_response",
    "make_period_grid",
    "interpolate_spectrum_loglog",
    "cumulative_trapezoid",
    "integrate_motion",
    "arias_intensity",
    "arias_percentile_times",
    "significant_durations",
    "cumulative_absolute_velocity",
    "peak_ground_values",
    "fourier_amplitude_spectrum",
    "motion_summary",
    "benchmark_all",
    "benchmark_correction",
    "benchmark_scaling",
    "G",
]
