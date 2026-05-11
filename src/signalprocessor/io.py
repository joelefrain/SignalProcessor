from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ACCELERATION_TO_MPS2, MPS2_TO_ACCELERATION
from .types import GroundMotion, TargetSpectrum

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def _unit_factor_to_mps2(unit: str) -> float:
    key = unit.strip().lower().replace(" ", "")
    if key not in ACCELERATION_TO_MPS2:
        raise ValueError(f"unsupported acceleration unit: {unit!r}")
    return ACCELERATION_TO_MPS2[key]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_motion_csv(
    path: str | Path,
    *,
    acceleration_unit: str = "g",
    time_column: int = 0,
    acceleration_column: int = 1,
    record_id: str | None = None,
    component: str | None = None,
) -> GroundMotion:
    file = Path(path)
    data = np.genfromtxt(file, delimiter=",", comments="#")
    if data.ndim == 1:
        if data.size < 3:
            raise ValueError(f"{file} does not contain enough samples")
        time = np.arange(data.size, dtype=float)
        acc = data
        dt = 1.0
    else:
        valid = np.all(np.isfinite(data[:, [time_column, acceleration_column]]), axis=1)
        data = data[valid]
        time = data[:, time_column]
        acc = data[:, acceleration_column]
        dt_values = np.diff(time)
        dt = float(np.median(dt_values))
        if not np.allclose(dt_values, dt, rtol=1.0e-3, atol=max(1.0e-9, abs(dt) * 1.0e-3)):
            raise ValueError(f"{file} is not uniformly sampled")
    rid = record_id or file.stem
    return GroundMotion(
        acceleration_mps2=acc * _unit_factor_to_mps2(acceleration_unit),
        dt=dt,
        record_id=rid,
        component=component,
        time_start=float(time[0]),
        source_path=str(file),
    )


def read_target_spectrum_csv(
    path: str | Path,
    *,
    period_column: int = 0,
    sa_column: int = 1,
    acceleration_unit: str = "g",
    damping: float = 0.05,
) -> TargetSpectrum:
    file = Path(path)
    data = np.genfromtxt(file, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{file} must contain period and spectral acceleration columns")
    period = data[:, period_column]
    sa = data[:, sa_column] * _unit_factor_to_mps2(acceleration_unit) * MPS2_TO_ACCELERATION["g"]
    return TargetSpectrum(period, sa, damping=damping, source_path=str(file))


def read_seismomatch_txt(path: str | Path, *, acceleration_unit: str = "g") -> GroundMotion:
    file = Path(path)
    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    dt = None
    samples: list[tuple[float, float]] = []
    for line in lines:
        if "time step" in line.lower():
            match = _FLOAT_RE.search(line)
            if match:
                dt = float(match.group(0))
        floats = [float(x) for x in _FLOAT_RE.findall(line)]
        if len(floats) >= 2 and "time" not in line.lower() and "step" not in line.lower():
            samples.append((floats[0], floats[1]))
    if not samples:
        raise ValueError(f"could not parse time series from {file}")
    time = np.array([x[0] for x in samples], dtype=float)
    acc = np.array([x[1] for x in samples], dtype=float)
    if dt is None:
        dt = float(np.median(np.diff(time)))
    return GroundMotion(
        acceleration_mps2=acc * _unit_factor_to_mps2(acceleration_unit),
        dt=dt,
        record_id=file.stem,
        time_start=float(time[0]),
        source_path=str(file),
    )


def _parse_smc_dt(header_lines: list[str]) -> float:
    values: list[float] = []
    for line in header_lines:
        values.extend(float(x) for x in _FLOAT_RE.findall(line))
    for idx, value in enumerate(values[:-1]):
        candidate = values[idx + 1]
        if abs(value) > 1.0e20 and 20.0 <= candidate <= 1000.0:
            return 1.0 / candidate
    for candidate in values:
        if 20.0 <= candidate <= 1000.0 and abs(candidate - round(candidate)) < 1.0e-6:
            return 1.0 / candidate
    raise ValueError("could not infer SMC sample interval")


def read_smc(path: str | Path, *, acceleration_unit: str = "cm/s2") -> GroundMotion:
    file = Path(path)
    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    comment_indices = [i for i, line in enumerate(lines[:120]) if line.lstrip().startswith("|")]
    data_start = (max(comment_indices) + 1) if comment_indices else 28
    header = lines[:data_start]
    dt = _parse_smc_dt(header)
    text = "\n".join(lines[data_start:])
    values = np.array([float(x) for x in _FLOAT_RE.findall(text)], dtype=float)
    if values.size < 3:
        raise ValueError(f"could not parse acceleration samples from {file}")
    return GroundMotion(
        acceleration_mps2=values * _unit_factor_to_mps2(acceleration_unit),
        dt=dt,
        record_id=file.stem,
        component=None,
        source_path=str(file),
    )


def save_motion_csv(
    motion: GroundMotion,
    path: str | Path,
    *,
    acceleration_unit: str = "g",
) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    factor = MPS2_TO_ACCELERATION[acceleration_unit.strip().lower().replace(" ", "")]
    data = np.column_stack([motion.time, motion.acceleration_mps2 * factor])
    np.savetxt(file, data, delimiter=",", fmt="%.10g")
    return file


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file, index=False)
    return file

