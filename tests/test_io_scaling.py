from __future__ import annotations

from pathlib import Path

import numpy as np

from signalprocessor.io import read_motion, read_target_spectrum
from signalprocessor.matching import gaussian_cosine_wavelet
from signalprocessor.matching import MatchingConfig, match_spectrum
from signalprocessor.metrics import integrate_motion
from signalprocessor.scaling import linear_scale
from signalprocessor.scaling import linear_scale_factor
from signalprocessor.scaling import spectral_misfit
from signalprocessor.records import Spectrum


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


def test_linear_scale_factor_supports_linear_and_log_objectives():
    periods = np.asarray([0.2, 1.0])
    spectrum = Spectrum(periods=periods, sa=np.asarray([1.0, 4.0]), units="g")
    target = Spectrum(periods=periods, sa=np.asarray([2.0, 4.0]), units="g")

    linear_factor = linear_scale_factor(spectrum, target, method="linear")
    log_factor = linear_scale_factor(spectrum, target, method="log")

    assert np.isclose(linear_factor, 18.0 / 17.0)
    assert np.isclose(log_factor, np.sqrt(2.0))


def test_gaussian_cosine_wavelet_has_no_terminal_drift():
    dt = 0.01
    time = np.arange(0.0, 20.0, dt)
    wave = gaussian_cosine_wavelet(time, center=3.0, period=1.5)

    velocity, displacement = integrate_motion(wave, dt)

    assert abs(velocity[-1]) < 1.0e-10
    assert abs(displacement[-1]) < 1.0e-10


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
