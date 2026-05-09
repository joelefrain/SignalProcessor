from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import BaselineResult, correct_baseline
from .filtering import FilterResult, butterworth_filter
from .integration import integrate_motion
from .metrics import motion_metrics
from .motion import Motion
from .spectra import logspace_periods, response_spectrum


@dataclass(slots=True)
class ProcessConfig:
    baseline_method: str = "polynomial"
    baseline_order: int = 1
    baseline_pre_event_end: float | None = None
    baseline_windows: list[tuple[float, float]] | None = None
    enforce_zero_end: bool = True
    highpass_hz: float | None = 0.05
    lowpass_hz: float | None = 20.0
    filter_order: int = 4
    zero_phase: bool = True
    taper_fraction: float = 0.02
    pad_seconds: float = 5.0
    spectrum_min_period: float = 0.02
    spectrum_max_period: float = 5.0
    spectrum_points: int = 120
    damping: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessConfig":
        baseline = data.get("baseline", {})
        filt = data.get("filter", {})
        spectrum = data.get("spectrum", {})
        windows = baseline.get("windows")
        if windows is not None:
            windows = [(float(start), float(end)) for start, end in windows]
        return cls(
            baseline_method=baseline.get("method", data.get("baseline_method", cls.baseline_method)),
            baseline_order=int(baseline.get("order", data.get("baseline_order", cls.baseline_order))),
            baseline_pre_event_end=baseline.get("pre_event_end_s"),
            baseline_windows=windows,
            enforce_zero_end=bool(baseline.get("enforce_zero_end", data.get("enforce_zero_end", True))),
            highpass_hz=filt.get("highpass_hz", data.get("highpass_hz", cls.highpass_hz)),
            lowpass_hz=filt.get("lowpass_hz", data.get("lowpass_hz", cls.lowpass_hz)),
            filter_order=int(filt.get("order", data.get("filter_order", cls.filter_order))),
            zero_phase=bool(filt.get("zero_phase", data.get("zero_phase", True))),
            taper_fraction=float(filt.get("taper_fraction", data.get("taper_fraction", cls.taper_fraction))),
            pad_seconds=float(filt.get("pad_seconds", data.get("pad_seconds", cls.pad_seconds))),
            spectrum_min_period=float(spectrum.get("min_period_s", cls.spectrum_min_period)),
            spectrum_max_period=float(spectrum.get("max_period_s", cls.spectrum_max_period)),
            spectrum_points=int(spectrum.get("points", cls.spectrum_points)),
            damping=float(spectrum.get("damping", cls.damping)),
        )


@dataclass(slots=True)
class ProcessResult:
    raw: Motion
    baseline: BaselineResult
    filtered: FilterResult
    velocity_m_s: np.ndarray
    displacement_m: np.ndarray
    metrics: dict[str, float]
    spectrum: dict[str, np.ndarray]
    config: ProcessConfig


def process_motion(motion: Motion, config: ProcessConfig | dict[str, Any] | None = None) -> ProcessResult:
    cfg = config if isinstance(config, ProcessConfig) else ProcessConfig.from_dict(config or {})
    baseline = correct_baseline(
        motion,
        method=cfg.baseline_method,
        order=cfg.baseline_order,
        pre_event_end=cfg.baseline_pre_event_end,
        windows=cfg.baseline_windows,
        enforce_zero_end=cfg.enforce_zero_end,
    )
    filtered = butterworth_filter(
        baseline.motion,
        highpass_hz=cfg.highpass_hz,
        lowpass_hz=cfg.lowpass_hz,
        order=cfg.filter_order,
        zero_phase=cfg.zero_phase,
        taper_fraction=cfg.taper_fraction,
        pad_seconds=cfg.pad_seconds,
    )
    velocity, displacement = integrate_motion(filtered.motion)
    metrics = motion_metrics(filtered.motion)
    periods = logspace_periods(cfg.spectrum_min_period, cfg.spectrum_max_period, cfg.spectrum_points)
    spectrum = response_spectrum(filtered.motion, periods, damping=cfg.damping)
    return ProcessResult(motion, baseline, filtered, velocity, displacement, metrics, spectrum, cfg)


def period_grid_from_config(data: dict[str, Any]) -> np.ndarray:
    spectrum = data.get("spectrum", {})
    return logspace_periods(
        float(spectrum.get("min_period_s", 0.02)),
        float(spectrum.get("max_period_s", 5.0)),
        int(spectrum.get("points", 120)),
    )


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute() or base is None:
        return path
    return Path(base) / path
