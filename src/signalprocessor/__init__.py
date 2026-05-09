"""Fast seismic signal processing utilities."""

from .constants import G0
from .motion import Motion
from .io import read_motion_csv, read_spectrum_csv, write_motion_csv, write_spectrum_csv
from .processing import ProcessConfig, process_motion
from .spectra import response_spectrum
from .scaling import (
    Scaleouput,
    frequency_domain_spectral_match,
    scale_motion_to_target,
    scale_suite_to_target,
)

__all__ = [
    "G0",
    "Motion",
    "ProcessConfig",
    "Scaleouput",
    "frequency_domain_spectral_match",
    "process_motion",
    "read_motion_csv",
    "read_spectrum_csv",
    "response_spectrum",
    "scale_motion_to_target",
    "scale_suite_to_target",
    "write_motion_csv",
    "write_spectrum_csv",
]
