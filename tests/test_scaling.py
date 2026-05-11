from __future__ import annotations

import numpy as np

import signalprocessor as sp
from signalprocessor.scaling import spectral_fit_metrics


def _short_motion():
    motion = sp.read_motion_csv("examples/data/motion/LIMANS.csv", acceleration_unit="g")
    return motion.slice_seconds(0.0, 35.0, record_id="LIMANS_short")


def test_linear_scaling_improves_mean_log_bias():
    motion = _short_motion()
    target = sp.read_target_spectrum_csv("examples/data/target_response_spectrum/EPU_475.csv")
    periods = target.periods_s[(target.periods_s >= 0.05) & (target.periods_s <= 2.0)]
    target_sa = sp.interpolate_spectrum_loglog(target.periods_s, target.sa_g, periods)
    initial = sp.response_spectrum(motion.acceleration_mps2, motion.dt, periods)
    initial_fit = spectral_fit_metrics(initial["psa_g"].to_numpy(), target_sa)
    result = sp.scale_linear(motion, periods, target_sa, period_range_s=(0.05, 2.0))
    assert abs(result.metrics["mean_log_bias"]) < abs(initial_fit["mean_log_bias"])
    assert result.details["scale_factor"] > 0


def test_scaling_comparison_returns_all_methods():
    motion = _short_motion()
    target = sp.read_target_spectrum_csv("examples/data/target_response_spectrum/EPU_475.csv")
    comparison = sp.compare_scaling_methods(
        motion,
        target.periods_s,
        target.sa_g,
        methods=("linear", "frequency", "wavelet"),
        period_range_s=(0.05, 1.5),
        config={"spectral_matching": {"mismatch_tolerance": 0.15, "max_iterations": 3, "wavelets": {"max_number_of_waves": 4}}},
    )
    assert set(comparison.results) == {"linear", "frequency", "wavelet"}
    assert np.isfinite(comparison.summary["rms_log_error"]).all()
