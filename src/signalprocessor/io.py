from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .motion import Motion


def _load_numeric_csv(path: str | Path, delimiter: str = ",") -> NDArray[np.float64]:
    path = Path(path)
    try:
        data = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    except ValueError:
        data = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64, names=True)
        if data.dtype.names:
            data = np.column_stack([data[name] for name in data.dtype.names])
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.size == 0:
        raise ValueError(f"Empty CSV: {path}")
    return data


def read_motion_csv(
    path: str | Path,
    *,
    unit: str = "g",
    delimiter: str = ",",
    dt: float | None = None,
    name: str | None = None,
) -> Motion:
    data = _load_numeric_csv(path, delimiter=delimiter)
    if data.shape[1] >= 2:
        time = data[:, 0]
        accel = data[:, 1]
    elif dt is not None:
        accel = data[:, 0]
        time = np.arange(accel.size, dtype=np.float64) * float(dt)
    else:
        raise ValueError("CSV must contain time,accel columns or dt must be provided.")
    return Motion.from_arrays(time, accel, unit=unit, name=name or Path(path).stem)


def write_motion_csv(
    path: str | Path,
    motion: Motion,
    *,
    unit: str = "g",
    delimiter: str = ",",
    header: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([motion.time, motion.accel_as(unit)])
    hdr = f"time_s,accel_{unit}" if header else ""
    np.savetxt(path, data, delimiter=delimiter, fmt="%.10g", header=hdr, comments="")


def read_spectrum_csv(
    path: str | Path,
    *,
    delimiter: str = ",",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = _load_numeric_csv(path, delimiter=delimiter)
    if data.shape[1] < 2:
        raise ValueError("Spectrum CSV must contain period,sa columns.")
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def write_spectrum_csv(
    path: str | Path,
    period: NDArray[np.float64],
    sa_g: NDArray[np.float64],
    *,
    extra: dict[str, NDArray[np.float64]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [np.asarray(period, dtype=np.float64), np.asarray(sa_g, dtype=np.float64)]
    names = ["period_s", "sa_g"]
    for key, values in (extra or {}).items():
        columns.append(np.asarray(values, dtype=np.float64))
        names.append(key)
    data = np.column_stack(columns)
    np.savetxt(
        path, data, delimiter=",", fmt="%.10g", header=",".join(names), comments=""
    )


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
