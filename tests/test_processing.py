from __future__ import annotations

import numpy as np

from signalprocessor.metrics import integrate_motion
from signalprocessor.processing import (
    CorrectionConfig,
    apply_iir_filter,
    correct_record,
    polynomial_baseline,
)
from signalprocessor.records import MotionRecord


def test_linear_baseline_constraint_reduces_final_velocity():
    dt = 0.01
    time = np.arange(0.0, 20.0, dt)
    physical = 0.5 * np.sin(2.0 * np.pi * 1.5 * time) * np.exp(-0.08 * time)
    biased = physical + 0.02
    rec = MotionRecord(time=time, acceleration=biased, units="m/s^2")
    raw_velocity, _ = integrate_motion(rec.acceleration_si(), dt)
    result = correct_record(
        rec,
        CorrectionConfig(
            remove_mean=False,
            baseline_order=1,
            constrain_final_velocity=True,
            constrain_final_displacement=False,
            despike=False,
            taper_fraction=0.0,
            highpass_hz=None,
            lowpass_hz=None,
        ),
    )
    assert abs(result.velocity[-1]) < abs(raw_velocity[-1]) * 0.01


def test_polynomial_baseline_constrains_velocity_and_displacement_discretely():
    dt = 0.01
    time = np.arange(0.0, 20.0, dt)
    acc = (
        0.01
        + 0.001 * time
        + 0.15 * np.sin(2.0 * np.pi * 1.0 * time) * np.exp(-0.05 * time)
    )

    baseline, coeffs = polynomial_baseline(
        acc,
        dt,
        1,
        constrain_velocity=True,
        constrain_displacement=True,
    )
    velocity, displacement = integrate_motion(acc - baseline, dt)

    assert coeffs.size == 2
    assert abs(velocity[-1]) < 1.0e-10
    assert abs(displacement[-1]) < 1.0e-10


def test_supported_iir_filter_families_return_finite_arrays():
    dt = 0.01
    time = np.arange(0.0, 10.0, dt)
    acc = np.sin(2.0 * np.pi * 1.0 * time) + 0.1 * np.sin(2.0 * np.pi * 15.0 * time)

    for filter_type in ("butterworth", "cheby1", "cheby2", "ellip", "bessel"):
        filtered, meta = apply_iir_filter(
            acc,
            dt,
            highpass_hz=0.05,
            lowpass_hz=25.0,
            order=4,
            filter_type=filter_type,
            zero_phase=True,
        )
        assert filtered.shape == acc.shape
        assert np.all(np.isfinite(filtered))
        assert meta["filter_type"] in {
            "butterworth",
            "cheby1",
            "cheby2",
            "ellip",
            "bessel",
        }


def test_post_filter_polynomial_removes_terminal_displacement_drift():
    dt = 0.01
    time = np.arange(0.0, 40.0, dt)
    pulse = 0.20 * np.sin(2.0 * np.pi * 1.3 * time) * np.exp(-0.06 * time)
    drift = 0.008 + 0.00015 * time
    rec = MotionRecord(time=time, acceleration=pulse + drift, units="m/s^2")

    result = correct_record(
        rec,
        CorrectionConfig(
            remove_mean=False,
            baseline_order=0,
            constrain_final_velocity=True,
            despike=False,
            taper_fraction=0.01,
            highpass_hz=0.05,
            lowpass_hz=None,
            filter_type="butterworth",
            post_filter_baseline_order=1,
            post_filter_constrain_final_velocity=True,
            post_filter_constrain_final_displacement=True,
        ),
    )

    assert abs(result.velocity[-1]) < 1.0e-10
    assert abs(result.displacement[-1]) < 1.0e-10


def test_explicit_baseline_coefficients_override_estimated_baseline():
    dt = 0.01
    time = np.arange(0.0, 10.0, dt)
    tau = np.linspace(0.0, 1.0, time.size)
    physical = 0.10 * np.sin(2.0 * np.pi * 1.0 * time)
    baseline = 0.02 - 0.01 * tau
    rec = MotionRecord(time=time, acceleration=physical + baseline, units="m/s^2")

    result = correct_record(
        rec,
        CorrectionConfig(
            remove_mean=False,
            baseline_order=0,
            baseline_coefficients=(0.02, -0.01),
            constrain_final_velocity=False,
            despike=False,
            taper_fraction=0.0,
            highpass_hz=None,
            lowpass_hz=None,
        ),
    )

    assert result.diagnostics["pre_filter_baseline_source"] == "user_coefficients"
    np.testing.assert_allclose(
        result.diagnostics["pre_filter_baseline_coefficients"], [0.02, -0.01]
    )
    np.testing.assert_allclose(result.record.acceleration_si(), physical, atol=1.0e-12)


def test_explicit_post_filter_coefficients_override_estimated_post_baseline():
    dt = 0.01
    time = np.arange(0.0, 10.0, dt)
    tau = np.linspace(0.0, 1.0, time.size)
    physical = 0.10 * np.sin(2.0 * np.pi * 1.0 * time)
    post_baseline = -0.01 + 0.004 * tau
    rec = MotionRecord(time=time, acceleration=physical + post_baseline, units="m/s^2")

    result = correct_record(
        rec,
        CorrectionConfig(
            remove_mean=False,
            baseline_order=-1,
            despike=False,
            taper_fraction=0.0,
            highpass_hz=None,
            lowpass_hz=None,
            post_filter_baseline_coefficients=(-0.01, 0.004),
            post_filter_constrain_final_velocity=False,
            post_filter_constrain_final_displacement=False,
        ),
    )

    assert result.diagnostics["post_filter_baseline_source"] == "user_coefficients"
    np.testing.assert_allclose(
        result.diagnostics["post_filter_baseline_coefficients"], [-0.01, 0.004]
    )
    np.testing.assert_allclose(result.record.acceleration_si(), physical, atol=1.0e-12)
