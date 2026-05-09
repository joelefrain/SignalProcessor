from __future__ import annotations

import numpy as np

from .constants import G0


def normalize_units(units: str) -> str:
    text = units.strip().lower().replace(" ", "")
    aliases = {
        "g": "g",
        "gal": "cm/s^2",
        "cm/s/s": "cm/s^2",
        "cm/sec^2": "cm/s^2",
        "cm/s2": "cm/s^2",
        "cms-2": "cm/s^2",
        "m/s/s": "m/s^2",
        "m/sec^2": "m/s^2",
        "m/s2": "m/s^2",
        "ms-2": "m/s^2",
    }
    return aliases.get(text, text)


def acceleration_to_si(values, units: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    unit = normalize_units(units)
    if unit == "m/s^2":
        return arr.copy()
    if unit == "g":
        return arr * G0
    if unit == "cm/s^2":
        return arr * 0.01
    raise ValueError(f"Unsupported acceleration units: {units!r}")


def acceleration_from_si(values, units: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    unit = normalize_units(units)
    if unit == "m/s^2":
        return arr.copy()
    if unit == "g":
        return arr / G0
    if unit == "cm/s^2":
        return arr * 100.0
    raise ValueError(f"Unsupported acceleration units: {units!r}")
