from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import G


@dataclass(slots=True)
class GroundMotion:
    """Uniformly sampled ground acceleration stored internally in m/s2."""

    acceleration_mps2: np.ndarray
    dt: float
    record_id: str = "motion"
    component: str | None = None
    time_start: float = 0.0
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.acceleration_mps2 = np.asarray(self.acceleration_mps2, dtype=float)
        if self.acceleration_mps2.ndim != 1:
            raise ValueError("acceleration_mps2 must be a 1D array")
        if self.acceleration_mps2.size < 3:
            raise ValueError("a ground motion needs at least 3 samples")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be a positive finite number")

    @property
    def npts(self) -> int:
        return int(self.acceleration_mps2.size)

    @property
    def duration(self) -> float:
        return (self.npts - 1) * self.dt

    @property
    def sampling_rate_hz(self) -> float:
        return 1.0 / self.dt

    @property
    def time(self) -> np.ndarray:
        return self.time_start + np.arange(self.npts, dtype=float) * self.dt

    @property
    def acc_g(self) -> np.ndarray:
        return self.acceleration_mps2 / G

    def copy(
        self,
        *,
        acceleration_mps2: np.ndarray | None = None,
        record_id: str | None = None,
        component: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "GroundMotion":
        return GroundMotion(
            acceleration_mps2=np.array(
                self.acceleration_mps2 if acceleration_mps2 is None else acceleration_mps2,
                dtype=float,
                copy=True,
            ),
            dt=self.dt,
            record_id=self.record_id if record_id is None else record_id,
            component=self.component if component is None else component,
            time_start=self.time_start,
            source_path=self.source_path,
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def slice_seconds(self, start_s: float, end_s: float, record_id: str | None = None) -> "GroundMotion":
        i0 = max(0, int(round((start_s - self.time_start) / self.dt)))
        i1 = min(self.npts, int(round((end_s - self.time_start) / self.dt)) + 1)
        return GroundMotion(
            acceleration_mps2=self.acceleration_mps2[i0:i1],
            dt=self.dt,
            record_id=record_id or f"{self.record_id}_{start_s:g}_{end_s:g}",
            component=self.component,
            time_start=self.time_start + i0 * self.dt,
            source_path=self.source_path,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class GroundMotionPair:
    ns: GroundMotion
    ew: GroundMotion
    pair_id: str = "pair"

    def aligned(self) -> "GroundMotionPair":
        if abs(self.ns.dt - self.ew.dt) > 1.0e-10:
            raise ValueError("paired components must have the same dt")
        n = min(self.ns.npts, self.ew.npts)
        return GroundMotionPair(
            self.ns.copy(acceleration_mps2=self.ns.acceleration_mps2[:n]),
            self.ew.copy(acceleration_mps2=self.ew.acceleration_mps2[:n]),
            pair_id=self.pair_id,
        )


@dataclass(slots=True)
class TargetSpectrum:
    periods_s: np.ndarray
    sa_g: np.ndarray
    damping: float = 0.05
    source_path: str | None = None

    def __post_init__(self) -> None:
        self.periods_s = np.asarray(self.periods_s, dtype=float)
        self.sa_g = np.asarray(self.sa_g, dtype=float)
        if self.periods_s.shape != self.sa_g.shape:
            raise ValueError("period and Sa arrays must have the same shape")
        order = np.argsort(self.periods_s)
        self.periods_s = self.periods_s[order]
        self.sa_g = self.sa_g[order]
        if np.any(self.periods_s <= 0) or np.any(self.sa_g <= 0):
            raise ValueError("target periods and Sa values must be positive")

    @property
    def path(self) -> Path | None:
        return None if self.source_path is None else Path(self.source_path)

