from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .constants import G0


Array = NDArray[np.float64]


def _as_float_array(values: Any) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    if arr.size < 2:
        raise ValueError("A motion needs at least two samples.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Array contains NaN or infinite values.")
    return arr


def to_mps2(accel: Any, unit: str) -> Array:
    arr = _as_float_array(accel)
    norm = unit.strip().lower()
    if norm in {"m/s2", "m/s^2", "mps2", "si"}:
        return arr
    if norm in {"g", "grav", "gravity"}:
        return arr * G0
    if norm in {"cm/s2", "cm/s^2", "gal", "gals"}:
        return arr / 100.0
    raise ValueError(f"Unsupported acceleration unit: {unit}")


def from_mps2(accel_mps2: Any, unit: str) -> Array:
    arr = _as_float_array(accel_mps2)
    norm = unit.strip().lower()
    if norm in {"m/s2", "m/s^2", "mps2", "si"}:
        return arr
    if norm in {"g", "grav", "gravity"}:
        return arr / G0
    if norm in {"cm/s2", "cm/s^2", "gal", "gals"}:
        return arr * 100.0
    raise ValueError(f"Unsupported acceleration unit: {unit}")


@dataclass(slots=True)
class Motion:
    """Ground motion stored internally in SI acceleration units."""

    time: Array
    accel: Array
    name: str = "motion"
    source_unit: str = "m/s2"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = _as_float_array(self.time)
        self.accel = _as_float_array(self.accel)
        if self.time.size != self.accel.size:
            raise ValueError("time and accel must have the same length.")
        dt = np.diff(self.time)
        if np.any(dt <= 0.0):
            raise ValueError("time must be strictly increasing.")
        median_dt = float(np.median(dt))
        if not np.allclose(dt, median_dt, rtol=1e-4, atol=1e-9):
            raise ValueError("time step is not uniform enough for this processor.")

    @classmethod
    def from_arrays(
        cls,
        time: Any,
        accel: Any,
        *,
        unit: str = "g",
        name: str = "motion",
        meta: dict[str, Any] | None = None,
    ) -> "Motion":
        return cls(
            time=_as_float_array(time),
            accel=to_mps2(accel, unit),
            name=name,
            source_unit=unit,
            meta=dict(meta or {}),
        )

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time)))

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def nyquist(self) -> float:
        return 0.5 * self.fs

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def npts(self) -> int:
        return int(self.time.size)

    def accel_as(self, unit: str) -> Array:
        return from_mps2(self.accel, unit)

    def with_accel(
        self,
        accel_mps2: Any,
        *,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "Motion":
        merged = dict(self.meta)
        if meta:
            merged.update(meta)
        return replace(
            self,
            accel=_as_float_array(accel_mps2),
            name=name or self.name,
            source_unit="m/s2",
            meta=merged,
        )

    def scaled(self, factor: float, *, name: str | None = None) -> "Motion":
        return self.with_accel(
            self.accel * float(factor),
            name=name or f"{self.name}_x{factor:.4g}",
            meta={"scale_factor": float(factor)},
        )
