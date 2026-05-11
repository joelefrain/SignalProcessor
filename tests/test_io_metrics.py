from __future__ import annotations

import numpy as np

import signalprocessor as sp


def test_read_motion_csv_and_summary():
    motion = sp.read_motion_csv("examples/data/motion/LIMANS.csv", acceleration_unit="g")
    summary = sp.motion_summary(motion.acceleration_mps2, motion.dt)
    assert motion.npts > 1000
    assert 0.019 < motion.dt < 0.021
    assert 0.15 < summary["PGA_g"] < 0.30
    assert summary["arias_intensity_mps"] > 0


def test_response_spectrum_shape_and_positive_values():
    dt = 0.01
    time = np.arange(0.0, 12.0, dt)
    acc = 0.05 * 9.80665 * np.sin(2 * np.pi * time)
    periods = np.geomspace(0.05, 2.0, 20)
    spectrum = sp.response_spectrum(acc, dt, periods)
    assert list(spectrum.columns)[:2] == ["period_s", "sd_m"]
    assert len(spectrum) == len(periods)
    assert np.all(np.isfinite(spectrum["psa_g"]))
    assert spectrum["psa_g"].max() > 0


def test_smc_parser_reads_benchmark_record():
    motion = sp.read_smc("examples/data/benchmark/uncorrected_motion/CCSP.HNN.._u.smc")
    assert motion.npts > 10000
    assert abs(motion.dt - 0.01) < 1.0e-9
    assert np.isfinite(motion.acceleration_mps2).all()
