from __future__ import annotations

import numpy as np
from .core import newmark_response_spectrum
from .records import MotionRecord, Spectrum
from .units import acceleration_from_si, acceleration_to_si


def default_periods(
    min_period: float = 0.01, max_period: float = 6.0, count: int = 80
) -> np.ndarray:
    return np.geomspace(min_period, max_period, count)


def response_spectrum(
    record: MotionRecord,
    periods=None,
    *,
    damping: float = 0.05,
    output_units: str = "g",
    return_peak_times: bool = False,
) -> Spectrum | tuple[Spectrum, np.ndarray]:
    per = (
        default_periods() if periods is None else np.asarray(periods, dtype=np.float64)
    )
    if per.ndim != 1 or per.size == 0:
        raise ValueError("periods must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(per)) or np.any(per <= 0.0):
        raise ValueError("all response-spectrum periods must be finite and positive")
    if np.any(np.diff(per) <= 0.0):
        raise ValueError("response-spectrum periods must be strictly increasing")
    acc = record.acceleration_si()
    sd, psv, psa_si, saa_si, peak_index = newmark_response_spectrum(
        acc, record.dt, per, damping
    )
    del sd, psv, saa_si
    psa = acceleration_from_si(psa_si, output_units)
    spectrum = Spectrum(
        periods=per,
        sa=psa,
        units=output_units,
        damping=damping,
        metadata={
            "source": record.metadata.get("source"),
            "kind": "pseudo_acceleration",
        },
    )
    if return_peak_times:
        return spectrum, record.time[np.clip(peak_index, 0, record.npts - 1)]
    return spectrum


def fourier_amplitude_spectrum(record: MotionRecord) -> tuple[np.ndarray, np.ndarray]:
    acc = record.acceleration_si()
    freqs = np.fft.rfftfreq(record.npts, d=record.dt)
    amp = np.abs(np.fft.rfft(acc)) * record.dt
    return freqs, amp


def pseudo_velocity_from_sa(periods, sa, *, units: str = "g") -> np.ndarray:
    periods = np.asarray(periods, dtype=np.float64)
    if np.any(periods <= 0.0):
        raise ValueError("periods must be positive")
    sa_si = acceleration_to_si(sa, units)
    omega = 2.0 * np.pi / periods
    return sa_si / omega
