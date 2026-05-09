"""Fast seismic signal correction and scaling tools."""

from .constants import G0
from .io import read_motion, read_target_spectrum, write_motion_csv
from .metrics import GroundMotionParameters, compute_ground_motion_parameters
from .processing import CorrectionConfig, CorrectionResult, correct_record
from .records import MotionRecord, Spectrum
from .scaling import ScalingResult, linear_scale, linear_scale_factor, spectral_misfit
from .spectra import response_spectrum

__all__ = [
    "G0",
    "CorrectionConfig",
    "CorrectionResult",
    "GroundMotionParameters",
    "MotionRecord",
    "ScalingResult",
    "Spectrum",
    "compute_ground_motion_parameters",
    "correct_record",
    "linear_scale",
    "linear_scale_factor",
    "read_motion",
    "read_target_spectrum",
    "response_spectrum",
    "spectral_misfit",
    "write_motion_csv",
]
