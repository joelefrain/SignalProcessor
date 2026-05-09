from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .motion import Motion


@dataclass(slots=True)
class Filterouput:
    motion: Motion
    info: dict[str, float | int | str | bool | None]


def cosine_taper(npts: int, fraction: float) -> np.ndarray:
    fraction = float(fraction)
    window = np.ones(int(npts), dtype=np.float64)
    if fraction <= 0.0:
        return window
    n_taper = int(round(0.5 * fraction * npts))
    if n_taper <= 1:
        return window
    phase = np.linspace(0.0, np.pi, n_taper, dtype=np.float64)
    edge = 0.5 * (1.0 - np.cos(phase))
    window[:n_taper] = edge
    window[-n_taper:] = edge[::-1]
    return window


def apply_taper(accel: np.ndarray, fraction: float) -> np.ndarray:
    return np.asarray(accel, dtype=np.float64) * cosine_taper(accel.size, fraction)


def butterworth_filter(
    motion: Motion,
    *,
    highpass_hz: float | None = None,
    lowpass_hz: float | None = None,
    order: int = 4,
    zero_phase: bool = True,
    taper_fraction: float = 0.02,
    pad_seconds: float = 5.0,
    pad_mode: str = "edge",
) -> Filterouput:
    hp = None if highpass_hz in (None, 0) else float(highpass_hz)
    lp = None if lowpass_hz in (None, 0) else float(lowpass_hz)
    if hp is None and lp is None:
        return Filterouput(motion, {"type": "none"})
    if hp is not None and hp <= 0.0:
        raise ValueError("highpass_hz must be positive.")
    if lp is not None and lp <= 0.0:
        raise ValueError("lowpass_hz must be positive.")
    if hp is not None and hp >= motion.nyquist:
        raise ValueError("highpass_hz must be lower than Nyquist.")
    if lp is not None and lp >= motion.nyquist:
        raise ValueError("lowpass_hz must be lower than Nyquist.")
    if hp is not None and lp is not None and hp >= lp:
        raise ValueError("highpass_hz must be lower than lowpass_hz.")

    if hp is not None and lp is not None:
        btype = "bandpass"
        wn: float | list[float] = [hp, lp]
    elif hp is not None:
        btype = "highpass"
        wn = hp
    else:
        btype = "lowpass"
        wn = lp  # type: ignore[assignment]

    sos = signal.butter(int(order), wn, btype=btype, fs=motion.fs, output="sos")
    data = apply_taper(motion.accel, taper_fraction)
    n_pad = max(0, int(round(float(pad_seconds) / motion.dt)))
    n_pad = min(n_pad, max(0, motion.npts // 2 - 1))
    if n_pad:
        padded = np.pad(data, (n_pad, n_pad), mode=pad_mode)
    else:
        padded = data

    if zero_phase:
        filtered = signal.sosfiltfilt(sos, padded)
    else:
        filtered = signal.sosfilt(sos, padded)
    if n_pad:
        filtered = filtered[n_pad:-n_pad]

    out = motion.with_accel(
        filtered,
        name=f"{motion.name}_{btype}",
        meta={
            "filter_type": btype,
            "filter_order": int(order),
            "highpass_hz": hp,
            "lowpass_hz": lp,
            "zero_phase": bool(zero_phase),
        },
    )
    info: dict[str, float | int | str | bool | None] = {
        "type": btype,
        "order": int(order),
        "highpass_hz": hp,
        "lowpass_hz": lp,
        "zero_phase": bool(zero_phase),
        "taper_fraction": float(taper_fraction),
        "pad_seconds": float(pad_seconds),
        "pad_samples": int(n_pad),
    }
    return Filterouput(out, info)
