from __future__ import annotations

G = 9.80665
EPS = 1.0e-12

ACCELERATION_TO_MPS2 = {
    "m/s2": 1.0,
    "m/s^2": 1.0,
    "mps2": 1.0,
    "si": 1.0,
    "g": G,
    "gal": 0.01,
    "cm/s2": 0.01,
    "cm/s^2": 0.01,
    "cms2": 0.01,
}

MPS2_TO_ACCELERATION = {name: 1.0 / factor for name, factor in ACCELERATION_TO_MPS2.items()}

