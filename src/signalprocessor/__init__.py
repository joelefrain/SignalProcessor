"""Fast seismic signal correction and scaling tools."""

from .constants import G0
from .io import read_motion, read_target_spectrum, write_motion_csv
from .metrics import (
    GroundMotionParameters,
    compute_ground_motion_parameters,
    compute_ground_motion_parameters_from_series,
    ground_motion_parameters_to_dict,
)
from .processing import (
    CorrectionConfig,
    CorrectionResult,
    apply_iir_filter,
    butterworth_filter,
    correct_record,
    design_iir_filter,
    polynomial_baseline,
)
from .records import MotionRecord, Spectrum
from .recommendation import (
    DEFAULT_RECOMMENDATION_FILTER_TYPES,
    CorrectionParameterSuggestion,
    CorrectionRecommendation,
    EventWindows,
    normalize_recommendation_filter_types,
    recommend_correction_method,
    recommend_correction_parameters,
)
from .scaling import ScalingResult, linear_scale, linear_scale_factor, spectral_misfit
from .spectra import response_spectrum

__all__ = [
    "G0",
    "CorrectionConfig",
    "CorrectionResult",
    "GroundMotionParameters",
    "MotionRecord",
    "CorrectionParameterSuggestion",
    "CorrectionRecommendation",
    "EventWindows",
    "DEFAULT_RECOMMENDATION_FILTER_TYPES",
    "ScalingResult",
    "Spectrum",
    "apply_iir_filter",
    "butterworth_filter",
    "compute_ground_motion_parameters",
    "compute_ground_motion_parameters_from_series",
    "ground_motion_parameters_to_dict",
    "correct_record",
    "design_iir_filter",
    "linear_scale",
    "linear_scale_factor",
    "polynomial_baseline",
    "normalize_recommendation_filter_types",
    "read_motion",
    "recommend_correction_method",
    "recommend_correction_parameters",
    "read_target_spectrum",
    "response_spectrum",
    "spectral_misfit",
    "write_motion_csv",
]
