import unittest

import numpy as np

from signalprocessor.constants import G0
from signalprocessor.integration import integrate_motion
from signalprocessor.motion import Motion
from signalprocessor.scaling import scale_factor
from signalprocessor.spectra import response_spectrum


class CoreTests(unittest.TestCase):
    def test_integration_constant_acceleration(self):
        t = np.arange(0.0, 1.01, 0.01)
        m = Motion.from_arrays(t, np.ones_like(t), unit="m/s2")
        v, u = integrate_motion(m)
        self.assertAlmostEqual(v[-1], 1.0, places=10)
        self.assertAlmostEqual(u[-1], 0.5, places=3)

    def test_response_spectrum_zero_period_is_pga(self):
        t = np.arange(0.0, 2.0, 0.01)
        acc_g = 0.1 * np.sin(2.0 * np.pi * 2.0 * t)
        m = Motion.from_arrays(t, acc_g, unit="g")
        spec = response_spectrum(m, np.array([0.0, 0.5]), use_numba=False)
        self.assertAlmostEqual(spec["sa_g"][0], np.max(np.abs(acc_g)), places=8)

    def test_log_scale_factor(self):
        periods = np.array([0.1, 0.2, 1.0])
        record = np.array([0.2, 0.3, 0.1])
        target = 2.0 * record
        factor = scale_factor(record, target, periods=periods, method="log_least_squares")
        self.assertAlmostEqual(factor, 2.0, places=10)


if __name__ == "__main__":
    unittest.main()
