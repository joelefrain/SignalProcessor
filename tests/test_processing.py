from __future__ import annotations

import numpy as np

from signalprocessor.metrics import integrate_motion
from signalprocessor.processing import CorrectionConfig, correct_record
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
