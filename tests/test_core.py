from __future__ import annotations

import numpy as np
import pytest

from signalprocessor.core import trapezoid_integrate
from signalprocessor.records import MotionRecord
from signalprocessor.spectra import response_spectrum


def test_trapezoid_integrate_constant_acceleration():
    acc = np.ones(101, dtype=np.float64) * 2.0
    vel = trapezoid_integrate(acc, 0.1, 0.0)
    assert np.isclose(vel[-1], 20.0)


def test_response_spectrum_scales_linearly():
    dt = 0.01
    time = np.arange(0.0, 10.0, dt)
    acc = 0.1 * np.sin(2.0 * np.pi * time)
    rec = MotionRecord(time=time, acceleration=acc, units="g")
    spec1 = response_spectrum(rec, [1.0])
    spec2 = response_spectrum(rec.with_acceleration(2.0 * acc, units="g"), [1.0])
    assert np.isclose(spec2.sa[0] / spec1.sa[0], 2.0, rtol=1e-4)


def test_motion_record_rejects_nonuniform_sampling():
    time = np.asarray([0.0, 0.01, 0.021, 0.03])
    acc = np.zeros_like(time)

    with pytest.raises(ValueError, match="uniformly sampled"):
        MotionRecord(time=time, acceleration=acc, units="m/s^2")
