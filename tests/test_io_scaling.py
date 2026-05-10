from __future__ import annotations

from pathlib import Path

import numpy as np

from signalprocessor.io import read_motion, read_target_spectrum
from signalprocessor.matching import MatchingConfig, match_spectrum
from signalprocessor.scaling import linear_scale
from signalprocessor.scaling import spectral_misfit


ROOT = Path(__file__).resolve().parents[1]


def test_read_motion_csv_example():
    rec = read_motion(ROOT / "examples" / "data" / "motion" / "ATICOEW.csv")
    assert rec.npts > 100
    assert np.isclose(rec.dt, 0.02)
    assert rec.units == "g"


def test_linear_scale_against_target_runs():
    rec = read_motion(ROOT / "examples" / "data" / "motion" / "ATICOEW.csv")
    target = read_target_spectrum(
        ROOT / "examples" / "data" / "response_spectrum" / "EPU_475.csv"
    )
    result = linear_scale(rec, target, t_min=0.2, t_max=2.0)
    assert result.factor > 0.0
    assert result.scaled_spectrum.sa.shape == target.sa.shape


def test_hybrid_wavelet_matching_reaches_seismomatch_like_misfit():
    rec = read_motion(
        ROOT / "examples" / "data" / "benchmark" / "unscaled_motion" / "ATICOEW.csv"
    )
    target = read_target_spectrum(
        ROOT / "examples" / "data" / "response_spectrum" / "EPU_475.csv"
    )

    result = match_spectrum(
        rec,
        target,
        MatchingConfig(
            method="hybrid",
            max_iterations=15,
            relaxation=0.35,
            t_min=0.2,
            t_max=2.0,
        ),
    )
    misfit = spectral_misfit(result.spectrum, target, t_min=0.2, t_max=2.0)

    assert result.converged
    assert misfit["rms_log_error"] < 0.035
    assert misfit["max_abs_error"] < 0.06
