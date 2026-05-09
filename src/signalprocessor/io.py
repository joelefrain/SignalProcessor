from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .records import MotionRecord, Spectrum

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def _floats(line: str) -> list[float]:
    return [float(match.group(0)) for match in FLOAT_RE.finditer(line)]


def _read_two_column(path: Path, *, units: str, metadata: dict | None = None) -> MotionRecord:
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = np.genfromtxt(path, delimiter=None, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a two-column time, acceleration file: {path}")
    meta = {"source": str(path)}
    if metadata:
        meta.update(metadata)
    return MotionRecord(time=data[:, 0], acceleration=data[:, 1], units=units, metadata=meta)


def read_motion_csv(path: str | Path, *, units: str = "g") -> MotionRecord:
    return _read_two_column(Path(path), units=units)


def read_seismomatch_txt(path: str | Path) -> MotionRecord:
    file_path = Path(path)
    time: list[float] = []
    acc: list[float] = []
    dt = None
    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        vals = _floats(line)
        if "Time Step" in line and vals:
            dt = vals[0]
        if len(vals) >= 2:
            if time and vals[0] < time[-1]:
                continue
            time.append(vals[0])
            acc.append(vals[1])
    if len(time) < 2:
        raise ValueError(f"Could not parse SeismoMatch text file: {path}")
    arr_t = np.asarray(time, dtype=np.float64)
    if dt is not None and np.allclose(arr_t[: min(5, arr_t.size)], 0.0):
        arr_t = np.arange(len(acc), dtype=np.float64) * float(dt)
    return MotionRecord(
        time=arr_t,
        acceleration=np.asarray(acc, dtype=np.float64),
        units="g",
        metadata={"source": str(file_path), "format": "seismomatch"},
    )


def _infer_smc_dt(header_lines: Iterable[str]) -> float:
    header_values: list[float] = []
    for line in header_lines:
        header_values.extend(_floats(line))

    common_sps = np.asarray([20.0, 25.0, 40.0, 50.0, 100.0, 200.0, 500.0], dtype=np.float64)
    best_sps = None
    best_err = np.inf
    for value in header_values:
        if 1.0 <= abs(value) <= 1000.0:
            idx = int(np.argmin(np.abs(common_sps - abs(value))))
            err = abs(common_sps[idx] - abs(value))
            if err < best_err:
                best_err = err
                best_sps = common_sps[idx]
    if best_sps is not None and best_err <= 0.05 * best_sps:
        return 1.0 / float(best_sps)

    dt_candidates = [abs(v) for v in header_values if 0.0005 <= abs(v) <= 1.0]
    if dt_candidates:
        return float(dt_candidates[0])
    return 0.01


def read_smc(path: str | Path, *, units: str = "cm/s^2") -> MotionRecord:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    last_comment = -1
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            last_comment = i
    if last_comment < 0:
        numeric_line_count = 0
        start = 0
        for i, line in enumerate(lines):
            if len(_floats(line)) >= 4:
                numeric_line_count += 1
            else:
                numeric_line_count = 0
            if numeric_line_count >= 8:
                start = i + 1
                break
    else:
        start = last_comment + 1

    values: list[float] = []
    for line in lines[start:]:
        values.extend(_floats(line))
    if len(values) < 2:
        raise ValueError(f"Could not parse COSMOS/SMC motion data: {path}")

    dt = _infer_smc_dt(lines[:start])
    acc = np.asarray(values, dtype=np.float64)
    time = np.arange(acc.size, dtype=np.float64) * dt
    return MotionRecord(
        time=time,
        acceleration=acc,
        units=units,
        metadata={"source": str(file_path), "format": "smc", "dt": dt},
    )


def read_motion(path: str | Path, *, units: str | None = None) -> MotionRecord:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return read_motion_csv(file_path, units=units or "g")
    if suffix == ".smc":
        return read_smc(file_path, units=units or "cm/s^2")
    if suffix == ".txt":
        return read_seismomatch_txt(file_path)
    raise ValueError(f"Unsupported motion file extension: {file_path.suffix}")


def read_target_spectrum(path: str | Path, *, units: str = "g", damping: float = 0.05) -> Spectrum:
    file_path = Path(path)
    data = np.genfromtxt(file_path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = np.genfromtxt(file_path, delimiter=None, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a two-column period, Sa spectrum file: {path}")
    return Spectrum(
        periods=data[:, 0],
        sa=data[:, 1],
        units=units,
        damping=damping,
        metadata={"source": str(file_path)},
    )


def write_motion_csv(record: MotionRecord, path: str | Path, *, units: str | None = None) -> Path:
    file_path = Path(path)
    out_record = record.as_units(units) if units else record
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([out_record.time, out_record.acceleration])
    np.savetxt(file_path, data, delimiter=",", fmt="%.10g")
    return file_path


def write_seismomatch_txt(record: MotionRecord, path: str | Path, *, name: str | None = None) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rec = record.as_units("g")
    rows = {
        "Time(sec)": rec.time,
        "Acc(g)": rec.acceleration,
    }
    frame = pd.DataFrame(rows)
    title = name or rec.name
    with file_path.open("w", encoding="utf-8") as f:
        f.write(f"Time Series matched accelerogram: {title}\n\n")
        f.write(f"Time Step: {rec.dt:.8g} s\n\n")
        f.write("     Time(sec)\t        Acc(g)\n")
        for row in frame.itertuples(index=False):
            f.write(f"{row[0]:14.5f}\t{row[1]:14.8f}\n")
    return file_path
