from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from signalprocessor.io import (
    read_motion_csv,
    read_spectrum_csv,
    write_motion_csv,
    write_spectrum_csv,
)
from signalprocessor.motion import Motion
from signalprocessor.scaling import (
    frequency_domain_spectral_match,
    scale_motion_to_target,
)
from signalprocessor.spectra import response_spectrum, significant_duration


Array = NDArray[np.float64]


@dataclass(slots=True)
class BenchmarkRecord:
    name: str
    method: str
    comparison_metric: str
    tolerance: float
    passed: bool
    scale_factor: float
    period_min_s: float
    period_max_s: float
    max_relative_error: float
    mean_relative_error: float
    rms_relative_error: float
    max_target_error_ours: float
    mean_target_error_ours: float
    max_target_error_seismomatch: float
    mean_target_error_seismomatch: float
    raw_pga_g: float
    ours_pga_g: float
    seismomatch_pga_g: float
    time_series_nrmse: float
    npts_ours: int
    npts_seismomatch: int
    motion_path: str
    benchmark_path: str
    duration_raw_s: float
    duration_ours_s: float
    duration_seismomatch_s: float
    duration_variation_pct: float


@dataclass(slots=True)
class BenchmarkRun:
    records: list[BenchmarkRecord]
    spectra: dict[str, dict[str, Array]]
    target_periods: Array
    target_sa_g: Array

    @property
    def all_passed(self) -> bool:
        return all(record.passed for record in self.records)

    def rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


def resolve_examples_path(root: str | Path, *relative_candidates: str) -> Path:
    root = Path(root).resolve()
    for relative in relative_candidates:
        path = root / relative
        if path.exists():
            return path
    joined = ", ".join(relative_candidates)
    raise FileNotFoundError(f"None of these paths exists under {root}: {joined}")


def resolve_motion_dir(root: str | Path) -> Path:
    return resolve_examples_path(root, "examples/motion", "examples/data/motion")


def resolve_response_spectrum_path(root: str | Path) -> Path:
    path = resolve_examples_path(
        root,
        "examples/response_spectrum",
        "examples/data/response_spectrum",
    )
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No spectrum CSV found in {path}")
        return csv_files[0]
    return path


def resolve_benchmark_dir(root: str | Path) -> Path:
    return resolve_examples_path(root, "examples/benchmark", "examples/data/benchmark")


def read_seismomatch_txt(path: str | Path, *, unit: str = "g") -> Motion:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    header_name = None
    rows: list[tuple[float, float]] = []
    numeric = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")
    for line in text.splitlines():
        match = re.search(r"accelerogram:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            header_name = Path(match.group(1).strip()).stem.replace("..", "")
        values = numeric.findall(line)
        if len(values) >= 2:
            rows.append((float(values[0]), float(values[1])))
    if not rows:
        raise ValueError(f"No numeric time series found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return Motion.from_arrays(
        data[:, 0],
        data[:, 1],
        unit=unit,
        name=header_name or path.stem.replace("..", ""),
    )


def discover_benchmark_pairs(
    root: str | Path,
    *,
    motion_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> list[tuple[str, Path, Path]]:
    root = Path(root).resolve()
    motions = (
        Path(motion_dir).resolve()
        if motion_dir is not None
        else resolve_motion_dir(root)
    )
    benchmarks = (
        Path(benchmark_dir).resolve()
        if benchmark_dir is not None
        else resolve_benchmark_dir(root)
    )
    pairs: list[tuple[str, Path, Path]] = []
    for benchmark_path in sorted(benchmarks.glob("*.txt")):
        benchmark_motion = read_seismomatch_txt(benchmark_path)
        name = benchmark_motion.name
        motion_path = motions / f"{name}.csv"
        if not motion_path.exists():
            raise FileNotFoundError(
                f"Benchmark {benchmark_path.name} maps to missing motion file {motion_path}"
            )
        pairs.append((name, motion_path, benchmark_path))
    return pairs


def _period_mask(periods: Array, period_range: tuple[float, float] | None) -> Array:
    if period_range is None:
        return np.ones(periods.size, dtype=bool)
    lo, hi = period_range
    mask = (periods >= float(lo)) & (periods <= float(hi))
    if not np.any(mask):
        raise ValueError(f"No target periods inside range {period_range}")
    return mask


def _relative_error(values: Array, reference: Array) -> Array:
    return np.abs(
        np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ) / np.maximum(np.abs(reference), 1e-12)


def _time_series_nrmse(ours: Motion, reference: Motion) -> float:
    ours_acc_g = ours.accel_as("g")
    ref_acc_g = np.interp(ours.time, reference.time, reference.accel_as("g"))
    denominator = max(float(np.sqrt(np.mean(ref_acc_g * ref_acc_g))), 1e-12)
    return float(np.sqrt(np.mean((ours_acc_g - ref_acc_g) ** 2)) / denominator)


def _comparison_value(record: BenchmarkRecord) -> float:
    if record.comparison_metric == "max_relative_error":
        return record.max_relative_error
    if record.comparison_metric == "rms_relative_error":
        return record.rms_relative_error
    if record.comparison_metric == "mean_relative_error":
        return record.mean_relative_error
    raise ValueError(f"Unsupported comparison metric: {record.comparison_metric}")


def _scale_raw_motion(
    motion: Motion,
    target_periods: Array,
    target_sa_g: Array,
    *,
    method: str,
    period_range: tuple[float, float] | None,
    damping: float,
    iterations: int,
) -> Any:
    method_norm = method.strip().lower()
    if method_norm in {"linear", "log_least_squares", "least_squares", "single"}:
        scaling_method = "log_least_squares" if method_norm == "linear" else method_norm
        return scale_motion_to_target(
            motion,
            target_periods,
            target_sa_g,
            damping=damping,
            method=scaling_method,
            period_range=period_range,
            factor_bounds=None,
        )
    if method_norm in {"frequency_match", "spectral_match", "matching"}:
        return frequency_domain_spectral_match(
            motion,
            target_periods,
            target_sa_g,
            damping=damping,
            iterations=iterations,
            max_factor_per_iteration=1.6,
            smoothing_width=7,
            highpass_hz=None,
            lowpass_hz=None,
        )
    raise ValueError(f"Unsupported benchmark scaling method: {method}")


def run_seismomatch_benchmark(
    root: str | Path = ".",
    *,
    motion_dir: str | Path | None = None,
    response_spectrum_path: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
    tolerance: float = 0.3,
    period_range: tuple[float, float] | None = (0.1, 2.0),
    comparison_metric: str = "mean_relative_error",
    method: str = "frequency_match",
    damping: float = 0.05,
    beta: float = 0.25,
    gamma: float = 0.5,
    num_periods: int = 200,
    iterations: int = 3,
) -> BenchmarkRun:
    root = Path(root).resolve()
    target_path = (
        Path(response_spectrum_path).resolve()
        if response_spectrum_path
        else resolve_response_spectrum_path(root)
    )
    target_periods, target_sa_g = read_spectrum_csv(target_path)
    mask = _period_mask(target_periods, period_range)
    period_min = float(target_periods[mask][0])
    period_max = float(target_periods[mask][-1])
    records: list[BenchmarkRecord] = []
    spectra: dict[str, dict[str, Array]] = {}

    for name, motion_path, benchmark_path in discover_benchmark_pairs(
        root, motion_dir=motion_dir, benchmark_dir=benchmark_dir
    ):
        raw_motion = read_motion_csv(motion_path, unit="g", name=name)
        seismomatch_motion = read_seismomatch_txt(benchmark_path)
        scaled = _scale_raw_motion(
            raw_motion,
            target_periods,
            target_sa_g,
            method=method,
            period_range=period_range,
            damping=damping,
            iterations=iterations,
        )

        # Calculate response spectra with high resolution
        ours_spectrum = response_spectrum(
            scaled.motion,
            None,
            damping=damping,
            beta=beta,
            gamma=gamma,
            num_periods=num_periods,
        )
        seismomatch_spectrum = response_spectrum(
            seismomatch_motion,
            None,
            damping=damping,
            beta=beta,
            gamma=gamma,
            num_periods=num_periods,
        )

        # Interpolate to target periods for comparison
        ours_sa_g = np.interp(
            target_periods, ours_spectrum["period_s"], ours_spectrum["sa_g"]
        )
        seismomatch_sa_g = np.interp(
            target_periods,
            seismomatch_spectrum["period_s"],
            seismomatch_spectrum["sa_g"],
        )

        rel = _relative_error(ours_sa_g, seismomatch_sa_g)
        target_rel_ours = _relative_error(ours_sa_g, target_sa_g)
        target_rel_seismomatch = _relative_error(seismomatch_sa_g, target_sa_g)

        max_rel = float(np.max(rel[mask]))
        mean_rel = float(np.mean(rel[mask]))
        rms_rel = float(np.sqrt(np.mean(rel[mask] ** 2)))

        # Calculate significant durations
        duration_raw = significant_duration(raw_motion)
        duration_ours = significant_duration(scaled.motion)
        duration_seismomatch = significant_duration(seismomatch_motion)

        # Calculate duration variation percentage
        # Use geometric mean of the two scaled motions as reference
        mean_duration = (duration_ours + duration_seismomatch) / 2.0
        if mean_duration > 0:
            duration_variation = (
                abs(duration_ours - duration_seismomatch) / mean_duration * 100.0
            )
        else:
            duration_variation = 0.0

        provisional = BenchmarkRecord(
            name=name,
            method=method,
            comparison_metric=comparison_metric,
            tolerance=float(tolerance),
            passed=False,
            scale_factor=float(scaled.factor),
            period_min_s=period_min,
            period_max_s=period_max,
            max_relative_error=max_rel,
            mean_relative_error=mean_rel,
            rms_relative_error=rms_rel,
            max_target_error_ours=float(np.max(target_rel_ours[mask])),
            mean_target_error_ours=float(np.mean(target_rel_ours[mask])),
            max_target_error_seismomatch=float(np.max(target_rel_seismomatch[mask])),
            mean_target_error_seismomatch=float(np.mean(target_rel_seismomatch[mask])),
            raw_pga_g=float(np.max(np.abs(raw_motion.accel_as("g")))),
            ours_pga_g=float(np.max(np.abs(scaled.motion.accel_as("g")))),
            seismomatch_pga_g=float(np.max(np.abs(seismomatch_motion.accel_as("g")))),
            time_series_nrmse=_time_series_nrmse(scaled.motion, seismomatch_motion),
            npts_ours=scaled.motion.npts,
            npts_seismomatch=seismomatch_motion.npts,
            motion_path=str(motion_path),
            benchmark_path=str(benchmark_path),
            duration_raw_s=float(duration_raw),
            duration_ours_s=float(duration_ours),
            duration_seismomatch_s=float(duration_seismomatch),
            duration_variation_pct=float(duration_variation),
        )
        provisional.passed = _comparison_value(provisional) <= float(tolerance)
        records.append(provisional)
        spectra[name] = {
            "period_s": target_periods,
            "target_sa_g": target_sa_g,
            "ours_sa_g": ours_sa_g,
            "seismomatch_sa_g": seismomatch_sa_g,
            "relative_error": rel,
            "raw_motion": raw_motion,
            "scaled_motion": scaled.motion,
            "seismomatch_motion": seismomatch_motion,
        }

    return BenchmarkRun(
        records=records,
        spectra=spectra,
        target_periods=target_periods,
        target_sa_g=target_sa_g,
    )


def write_benchmark_report(run: BenchmarkRun, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run.rows()
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "spectra_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "record",
                "period_s",
                "target_sa_g",
                "ours_sa_g",
                "seismomatch_sa_g",
                "relative_error",
            ]
        )
        for name, spec in run.spectra.items():
            for row in zip(
                spec["period_s"],
                spec["target_sa_g"],
                spec["ours_sa_g"],
                spec["seismomatch_sa_g"],
                spec["relative_error"],
            ):
                writer.writerow([name, *[float(value) for value in row]])


def write_scaled_benchmark_motions(
    run: BenchmarkRun,
    output_dir: str | Path,
    root: str | Path = ".",
    beta: float = 0.25,
    gamma: float = 0.5,
    num_periods: int = 200,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in run.records:
        raw_motion = read_motion_csv(record.motion_path, unit="g", name=record.name)
        target_periods = run.target_periods
        target_sa_g = run.target_sa_g
        scaled = _scale_raw_motion(
            raw_motion,
            target_periods,
            target_sa_g,
            method=record.method,
            period_range=(record.period_min_s, record.period_max_s),
            damping=0.05,
            iterations=3,
        )
        write_motion_csv(
            output_dir / f"{record.name}_{record.method}.csv", scaled.motion, unit="g"
        )

        # Calculate spectrum with high resolution
        scaled_spectrum = response_spectrum(
            scaled.motion, None, beta=beta, gamma=gamma, num_periods=num_periods
        )
        spectrum_sa_g = np.interp(
            target_periods, scaled_spectrum["period_s"], scaled_spectrum["sa_g"]
        )
        write_spectrum_csv(
            output_dir / f"{record.name}_{record.method}_spectrum.csv",
            target_periods,
            spectrum_sa_g,
        )


def plot_benchmark_spectra(run: BenchmarkRun, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in run.records:
        spec = run.spectra[record.name]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # Left subplot: Response spectra
        ax1.loglog(
            spec["period_s"], spec["target_sa_g"], label="Objetivo", linewidth=1.5
        )
        ax1.loglog(
            spec["period_s"],
            spec["seismomatch_sa_g"],
            label="SeismoMatch",
            linewidth=1.5,
        )
        ax1.loglog(
            spec["period_s"], spec["ours_sa_g"], label="SignalProcessor", linewidth=1.5
        )
        ax1.axvspan(
            record.period_min_s,
            record.period_max_s,
            color="0.8",
            alpha=0.25,
            label="Rango comparado",
        )
        ax1.set_xlabel("T (s)")
        ax1.set_ylabel("Sa (g)")
        ax1.set_title(f"Espectros de Respuesta - {record.name}")
        ax1.grid(True, which="both", alpha=0.25)
        ax1.legend()

        # Right subplot: Time series with original motion
        raw_motion = spec["raw_motion"]
        scaled_motion = spec["scaled_motion"]
        seismomatch_motion = spec["seismomatch_motion"]

        time_raw = raw_motion.time
        time_scaled = scaled_motion.time
        time_seismo = seismomatch_motion.time

        accel_raw = raw_motion.accel_as("g")
        accel_scaled = scaled_motion.accel_as("g")
        accel_seismo = seismomatch_motion.accel_as("g")

        ax2.plot(time_raw, accel_raw, label="Original", linewidth=0.8, alpha=0.7)
        ax2.plot(
            time_scaled, accel_scaled, label="SignalProcessor", linewidth=0.8, alpha=0.8
        )
        ax2.plot(
            time_seismo, accel_seismo, label="SeismoMatch", linewidth=0.8, alpha=0.8
        )
        ax2.set_xlabel("Tiempo (s)")
        ax2.set_ylabel("Aceleración (g)")
        ax2.set_title(
            f"Series Temporales - {record.name}\nDuración Sig.: Original={record.duration_raw_s:.2f}s, SP={record.duration_ours_s:.2f}s, SM={record.duration_seismomatch_s:.2f}s (Var={record.duration_variation_pct:.1f}%)"
        )
        ax2.grid(True, alpha=0.25)
        ax2.legend()

        # Overall title with metric
        fig.suptitle(
            f"{record.name}: {record.comparison_metric}={_comparison_value(record):.3f}",
            fontsize=12,
            fontweight="bold",
        )

        fig.savefig(output_dir / f"{record.name}_spectra.png", dpi=160)
        plt.close(fig)
