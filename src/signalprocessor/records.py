from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .units import acceleration_from_si, acceleration_to_si, normalize_units


@dataclass(frozen=True, slots=True)
class MotionRecord:
    """A single uniformly sampled acceleration time history."""

    time: np.ndarray
    acceleration: np.ndarray
    units: str = "m/s^2"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=np.float64)
        acc = np.asarray(self.acceleration, dtype=np.float64)
        if time.ndim != 1 or acc.ndim != 1:
            raise ValueError("time and acceleration must be one-dimensional arrays")
        if time.size != acc.size:
            raise ValueError("time and acceleration arrays must have the same length")
        if time.size < 2:
            raise ValueError("a record needs at least two samples")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "acceleration", acc)
        object.__setattr__(self, "units", normalize_units(self.units))

    @classmethod
    def from_acceleration(
        cls,
        acceleration,
        dt: float,
        *,
        units: str = "m/s^2",
        start: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> "MotionRecord":
        acc = np.asarray(acceleration, dtype=np.float64)
        time = start + np.arange(acc.size, dtype=np.float64) * float(dt)
        return cls(time=time, acceleration=acc, units=units, metadata=metadata or {})

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time)))

    @property
    def npts(self) -> int:
        return int(self.acceleration.size)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def name(self) -> str:
        source = self.metadata.get("source")
        if source:
            return Path(str(source)).stem
        return str(self.metadata.get("name", "record"))

    def acceleration_si(self) -> np.ndarray:
        return acceleration_to_si(self.acceleration, self.units)

    def as_units(self, units: str) -> "MotionRecord":
        acc_si = self.acceleration_si()
        return replace(self, acceleration=acceleration_from_si(acc_si, units), units=units)

    def with_acceleration(
        self,
        acceleration,
        *,
        units: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MotionRecord":
        new_meta = dict(self.metadata)
        if metadata:
            new_meta.update(metadata)
        return MotionRecord(
            time=self.time.copy(),
            acceleration=np.asarray(acceleration, dtype=np.float64),
            units=units or self.units,
            metadata=new_meta,
        )


@dataclass(frozen=True, slots=True)
class Spectrum:
    periods: np.ndarray
    sa: np.ndarray
    units: str = "g"
    damping: float = 0.05
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        periods = np.asarray(self.periods, dtype=np.float64)
        sa = np.asarray(self.sa, dtype=np.float64)
        if periods.ndim != 1 or sa.ndim != 1:
            raise ValueError("periods and sa must be one-dimensional arrays")
        if periods.size != sa.size:
            raise ValueError("periods and sa arrays must have the same length")
        if np.any(periods <= 0.0):
            raise ValueError("all spectral periods must be positive")
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "sa", sa)
        object.__setattr__(self, "units", normalize_units(self.units))

    def as_units(self, units: str) -> "Spectrum":
        from .units import acceleration_from_si, acceleration_to_si

        si = acceleration_to_si(self.sa, self.units)
        return replace(self, sa=acceleration_from_si(si, units), units=units)

    def interpolate(self, periods, *, loglog: bool = True) -> np.ndarray:
        x = np.asarray(periods, dtype=np.float64)
        if loglog:
            lx = np.log(x)
            xp = np.log(self.periods)
            fp = np.log(np.maximum(self.sa, np.finfo(float).tiny))
            return np.exp(np.interp(lx, xp, fp))
        return np.interp(x, self.periods, self.sa)
