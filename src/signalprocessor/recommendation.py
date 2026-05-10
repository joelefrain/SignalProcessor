from __future__ import annotations

from dataclasses import dataclass, replace
from math import log
from typing import Iterable

import numpy as np

from .core import central_difference
from .metrics import (
    compute_ground_motion_parameters,
    cumulative_arias,
    integrate_motion,
)
from .processing import CorrectionConfig, CorrectionResult, correct_record
from .records import MotionRecord
from .spectra import response_spectrum


@dataclass(frozen=True, slots=True)
class EventWindows:
    pre_event_seconds: float | None
    event_start_seconds: float
    strong_start_seconds: float
    strong_end_seconds: float
    post_event_start_seconds: float
    d5_95_seconds: float

    def to_row(self) -> dict[str, float | None]:
        return {
            "pre_event_seconds": self.pre_event_seconds,
            "event_start_seconds": self.event_start_seconds,
            "strong_start_seconds": self.strong_start_seconds,
            "strong_end_seconds": self.strong_end_seconds,
            "post_event_start_seconds": self.post_event_start_seconds,
            "d5_95_seconds": self.d5_95_seconds,
        }


@dataclass(frozen=True, slots=True)
class CorrectionParameterSuggestion:
    windows: EventWindows
    remove_mean: bool
    despike: bool
    spike_sigma: float
    taper_fraction: float
    baseline_orders: tuple[int, ...]
    highpass_hz_candidates: tuple[float | None, ...]
    lowpass_hz: float | None
    filter_order: int
    filter_types: tuple[str, ...]
    zero_phase: bool
    constrain_final_velocity: bool
    include_final_displacement_constraint: bool
    drift_severity: str
    snr_available: bool
    snr_threshold: float
    notes: tuple[str, ...]

    def to_row(self) -> dict[str, str | float | bool | None]:
        return {
            "pre_event_seconds": self.windows.pre_event_seconds,
            "baseline_orders": ", ".join(str(v) for v in self.baseline_orders),
            "highpass_hz_candidates": ", ".join(
                "None" if v is None else f"{v:g}" for v in self.highpass_hz_candidates
            ),
            "lowpass_hz": self.lowpass_hz,
            "taper_fraction": self.taper_fraction,
            "filter_order": self.filter_order,
            "filter_types": ", ".join(self.filter_types),
            "despike": self.despike,
            "spike_sigma": self.spike_sigma,
            "constrain_final_velocity": self.constrain_final_velocity,
            "include_final_displacement_constraint": self.include_final_displacement_constraint,
            "drift_severity": self.drift_severity,
            "snr_available": self.snr_available,
            "snr_threshold": self.snr_threshold,
        }


@dataclass(frozen=True, slots=True)
class CorrectionCandidate:
    name: str
    description: str
    config: CorrectionConfig


def _format_coefficients_for_row(coefficients) -> str:
    if coefficients is None:
        return ""
    arr = np.asarray(coefficients, dtype=np.float64).ravel()
    if arr.size == 0:
        return ""
    return ", ".join(f"{value:.10g}" for value in arr)


@dataclass(frozen=True, slots=True)
class CorrectionCandidateEvaluation:
    name: str
    description: str
    config: CorrectionConfig
    result: CorrectionResult
    score: float
    final_velocity_ratio: float
    final_displacement_ratio: float
    final_displacement_pgv_seconds: float
    pgd_pgv_seconds: float
    post_event_velocity_drift_ratio: float
    post_event_displacement_range_ratio: float
    spectral_rms_log_change: float
    pga_log_change: float
    arias_log_change: float
    cav_log_change: float
    complexity_penalty: float
    filter_penalty: float
    notes: tuple[str, ...]

    def to_row(self) -> dict[str, float | str | bool | None]:
        cfg = self.config
        return {
            "method": self.name,
            "score": self.score,
            "baseline_order": cfg.baseline_order,
            "highpass_hz": cfg.highpass_hz,
            "lowpass_hz": cfg.lowpass_hz,
            "filter_type": cfg.filter_type,
            "taper_fraction": cfg.taper_fraction,
            "post_filter_baseline_order": cfg.post_filter_baseline_order,
            "baseline_coefficients_mps2": _format_coefficients_for_row(
                self.result.diagnostics.get("pre_filter_baseline_coefficients")
            ),
            "baseline_coefficients_source": self.result.diagnostics.get(
                "pre_filter_baseline_source", ""
            ),
            "post_filter_baseline_coefficients_mps2": _format_coefficients_for_row(
                self.result.diagnostics.get("post_filter_baseline_coefficients")
            ),
            "post_filter_baseline_coefficients_source": self.result.diagnostics.get(
                "post_filter_baseline_source", ""
            ),
            "final_velocity_constraint": cfg.constrain_final_velocity,
            "final_displacement_constraint": cfg.constrain_final_displacement,
            "post_filter_displacement_constraint": cfg.post_filter_constrain_final_displacement,
            "final_velocity_ratio": self.final_velocity_ratio,
            "final_displacement_ratio": self.final_displacement_ratio,
            "final_displacement_pgv_seconds": self.final_displacement_pgv_seconds,
            "pgd_pgv_seconds": self.pgd_pgv_seconds,
            "post_event_velocity_drift_ratio": self.post_event_velocity_drift_ratio,
            "post_event_displacement_range_ratio": self.post_event_displacement_range_ratio,
            "spectral_rms_log_change": self.spectral_rms_log_change,
            "pga_log_change": self.pga_log_change,
            "arias_log_change": self.arias_log_change,
            "cav_log_change": self.cav_log_change,
            "complexity_penalty": self.complexity_penalty,
            "filter_penalty": self.filter_penalty,
        }


@dataclass(frozen=True, slots=True)
class CorrectionRecommendation:
    parameter_suggestion: CorrectionParameterSuggestion
    best: CorrectionCandidateEvaluation
    candidates: tuple[CorrectionCandidateEvaluation, ...]
    periods: np.ndarray
    decision_notes: tuple[str, ...]

    def to_rows(self) -> list[dict[str, float | str | bool | None]]:
        return [candidate.to_row() for candidate in self.candidates]


DEFAULT_RECOMMENDATION_FILTER_TYPES = (
    "butterworth",
    "cheby1",
    "cheby2",
    "ellip",
    "bessel",
)

_FILTER_TYPE_ALIASES = {
    "all": "__all__",
    "todos": "__all__",
    "todas": "__all__",
    "*": "__all__",
    "butter": "butterworth",
    "butterworth": "butterworth",
    "cheby1": "cheby1",
    "chebyshev": "cheby1",
    "chebyshev1": "cheby1",
    "chebyshev_i": "cheby1",
    "chevyshev": "cheby1",
    "chevyshev1": "cheby1",
    "chevyshev_i": "cheby1",
    "cheby2": "cheby2",
    "chebyshev2": "cheby2",
    "chebyshev_ii": "cheby2",
    "chevyshev2": "cheby2",
    "chevyshev_ii": "cheby2",
    "ellip": "ellip",
    "elliptic": "ellip",
    "eliptico": "ellip",
    "elíptico": "ellip",
    "cauer": "ellip",
    "bessel": "bessel",
    "bessel_thomson": "bessel",
}


def normalize_recommendation_filter_types(
    filter_types: str | Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Normalize the filter families used by the automatic recommendation.

    ``None``, ``"all"``, ``"todos"`` or ``"*"`` means all supported IIR
    families. A string can contain one family or a comma/semicolon-separated
    list, for example ``"bessel, cheby2"``. The returned names are canonical
    and de-duplicated while preserving order.
    """

    if filter_types is None:
        return DEFAULT_RECOMMENDATION_FILTER_TYPES

    if isinstance(filter_types, str):
        raw_text = filter_types.strip()
        if not raw_text:
            return DEFAULT_RECOMMENDATION_FILTER_TYPES
        raw_values = [part.strip() for part in raw_text.replace(";", ",").split(",")]
    else:
        raw_values = list(filter_types)

    normalized: list[str] = []
    for value in raw_values:
        if value is None:
            continue
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        try:
            family = _FILTER_TYPE_ALIASES[key]
        except KeyError as exc:
            valid = ", ".join(DEFAULT_RECOMMENDATION_FILTER_TYPES)
            aliases = ", ".join(
                sorted(k for k, v in _FILTER_TYPE_ALIASES.items() if v != "__all__")
            )
            raise ValueError(
                f"Unsupported recommendation filter type {value!r}. "
                f"Valid canonical families: {valid}. Accepted aliases: {aliases}."
            ) from exc
        if family == "__all__":
            return DEFAULT_RECOMMENDATION_FILTER_TYPES
        if family not in normalized:
            normalized.append(family)

    if not normalized:
        raise ValueError(
            "At least one filter family must be provided, or use filter_types='all'."
        )
    return tuple(normalized)


def _time_at_fraction(
    time: np.ndarray, cumulative: np.ndarray, fraction: float
) -> float:
    total = float(cumulative[-1])
    if total <= 0.0:
        return float(time[0])
    target = fraction * total
    idx = int(np.searchsorted(cumulative, target, side="left"))
    if idx <= 0:
        return float(time[0])
    if idx >= cumulative.size:
        return float(time[-1])
    y0 = cumulative[idx - 1]
    y1 = cumulative[idx]
    if y1 == y0:
        return float(time[idx])
    alpha = (target - y0) / (y1 - y0)
    return float(time[idx - 1] + alpha * (time[idx] - time[idx - 1]))


def estimate_event_windows(record: MotionRecord) -> EventWindows:
    arias = cumulative_arias(record.acceleration_si(), record.dt)
    t1 = _time_at_fraction(record.time, arias, 0.01)
    t5 = _time_at_fraction(record.time, arias, 0.05)
    t75 = _time_at_fraction(record.time, arias, 0.75)
    t95 = _time_at_fraction(record.time, arias, 0.95)
    dt = record.dt
    available_pre = max(0.0, t5 - float(record.time[0]))
    pre_event_seconds = None
    if available_pre >= max(1.0, 20.0 * dt):
        pre_event_seconds = min(available_pre, 20.0)
    post_start = min(float(record.time[-1]), t95 + max(1.0, 0.05 * record.duration))
    return EventWindows(
        pre_event_seconds=pre_event_seconds,
        event_start_seconds=t1,
        strong_start_seconds=t5,
        strong_end_seconds=t75,
        post_event_start_seconds=post_start,
        d5_95_seconds=max(0.0, t95 - t5),
    )


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _windowed_fas(
    segment: np.ndarray, dt: float, nfft: int
) -> tuple[np.ndarray, np.ndarray]:
    if segment.size < 8:
        return np.zeros(0), np.zeros(0)
    x = segment.astype(np.float64, copy=True)
    x -= float(np.mean(x))
    x *= np.hanning(x.size)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    amp = np.abs(np.fft.rfft(x, n=nfft)) * dt
    return freqs, amp


def _estimate_snr_filter_corners(
    record: MotionRecord,
    windows: EventWindows,
    *,
    snr_threshold: float,
) -> tuple[float | None, float | None, bool, tuple[str, ...]]:
    acc = record.acceleration_si()
    dt = record.dt
    nyquist = 0.5 / dt
    notes: list[str] = []

    if windows.pre_event_seconds is None:
        notes.append(
            "Sin ventana pre-evento suficiente; cortes por reglas conservadoras."
        )
        return 0.05, min(25.0, 0.80 * nyquist), False, tuple(notes)

    pre_n = int(max(8, round(windows.pre_event_seconds / dt)))
    pre_n = min(pre_n, acc.size // 3)
    start = int(max(0, round(windows.strong_start_seconds / dt)))
    end = int(min(acc.size, round(windows.strong_end_seconds / dt)))
    if end - start < pre_n:
        end = min(acc.size, start + pre_n)
    if pre_n < 16 or end - start < 16:
        notes.append("Ventanas SNR demasiado cortas; cortes por reglas conservadoras.")
        return 0.05, min(25.0, 0.80 * nyquist), False, tuple(notes)

    nfft = 1 << int(np.ceil(np.log2(max(pre_n, end - start))))
    freqs, noise = _windowed_fas(acc[:pre_n], dt, nfft)
    _, signal = _windowed_fas(acc[start:end], dt, nfft)
    if freqs.size == 0:
        notes.append("No se pudo calcular FAS/SNR; cortes por reglas conservadoras.")
        return 0.05, min(25.0, 0.80 * nyquist), False, tuple(notes)

    ratio = signal / np.maximum(
        noise, np.percentile(noise[noise > 0.0], 10) if np.any(noise > 0.0) else 1.0e-16
    )
    ratio = _moving_average(ratio, max(3, ratio.size // 80))
    valid = (freqs > max(1.0 / max(record.duration, dt), 0.01)) & (
        freqs < 0.90 * nyquist
    )
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        notes.append("Rango de frecuencia valido insuficiente para SNR.")
        return 0.05, min(25.0, 0.80 * nyquist), False, tuple(notes)

    good = ratio >= snr_threshold
    hp = None
    for pos in idx:
        stop = min(pos + 5, good.size)
        if stop > pos and np.mean(good[pos:stop]) >= 0.80:
            hp = float(freqs[pos])
            break
    if hp is None:
        hp = 0.05
        notes.append(
            "SNR no cruza umbral a baja frecuencia; se usa high-pass base 0.05 Hz."
        )
    else:
        hp = float(np.clip(hp, 0.02, 0.20))
        notes.append(f"High-pass estimado por SNR: {hp:.3g} Hz.")

    above_hp = idx[freqs[idx] >= hp]
    good_after = above_hp[good[above_hp]]
    if good_after.size:
        lp = float(freqs[good_after[-1]])
        lp = min(lp, 25.0, 0.80 * nyquist)
        if lp <= hp * 2.0:
            lp = min(25.0, 0.80 * nyquist)
            notes.append(
                "Low-pass SNR demasiado cercano al high-pass; se usa limite conservador."
            )
        else:
            notes.append(f"Low-pass estimado por SNR: {lp:.3g} Hz.")
    else:
        lp = min(25.0, 0.80 * nyquist)
        notes.append("SNR no permite low-pass claro; se usa limite conservador.")
    return hp, lp, True, tuple(notes)


def _drift_severity(record: MotionRecord) -> tuple[str, float, float]:
    acc = record.acceleration_si()
    velocity, displacement = integrate_motion(acc, record.dt)
    pgv = max(float(np.max(np.abs(velocity))), 1.0e-12)
    pgd = max(float(np.max(np.abs(displacement))), 1.0e-12)
    vend_ratio = abs(float(velocity[-1])) / pgv
    dend_ratio = abs(float(displacement[-1])) / pgd
    severity_score = max(vend_ratio / 0.05, dend_ratio / 0.25)
    if severity_score < 1.0:
        label = "baja"
    elif severity_score < 4.0:
        label = "media"
    else:
        label = "alta"
    return label, vend_ratio, dend_ratio


def _estimate_spike_need(record: MotionRecord, sigma: float = 8.0) -> tuple[bool, int]:
    acc = record.acceleration_si()
    deriv = central_difference(acc, record.dt)
    med = float(np.median(deriv))
    mad = float(np.median(np.abs(deriv - med)))
    scale = 1.4826 * mad if mad > 0.0 else float(np.std(deriv))
    if scale <= 0.0:
        return False, 0
    count = int(np.count_nonzero(np.abs(deriv - med) > sigma * scale))
    return count > 0, count


def recommend_correction_parameters(
    record: MotionRecord,
    *,
    snr_threshold: float = 3.0,
    filter_types: str | Iterable[str] | None = None,
) -> CorrectionParameterSuggestion:
    selected_filter_types = normalize_recommendation_filter_types(filter_types)
    windows = estimate_event_windows(record)
    hp, lp, snr_available, snr_notes = _estimate_snr_filter_corners(
        record, windows, snr_threshold=snr_threshold
    )
    drift_label, vend_ratio, dend_ratio = _drift_severity(record)
    despike, spike_count = _estimate_spike_need(record)

    if drift_label == "baja":
        baseline_orders = (0, 1)
    elif drift_label == "media":
        baseline_orders = (0, 1, 2)
    else:
        baseline_orders = (1, 2, 0)

    hp_candidates: list[float | None] = [None]
    if hp is not None:
        multipliers = (0.6, 1.0, 1.6)
        hp_candidates.extend(
            float(np.clip(hp * factor, 0.02, 0.20)) for factor in multipliers
        )
    if drift_label in {"media", "alta"} or vend_ratio > 0.05 or dend_ratio > 0.25:
        hp_candidates.extend((0.05, 0.08, 0.10))
    hp_candidates = list(
        dict.fromkeys(
            None if value is None else round(float(value), 4) for value in hp_candidates
        )
    )

    taper_fraction = float(
        np.clip(max(0.01, 1.0 / max(record.duration, 1.0)), 0.01, 0.05)
    )
    include_displacement_constraint = drift_label == "alta" and dend_ratio > 0.85

    notes = [
        f"Ventana pre-evento estimada: {windows.pre_event_seconds if windows.pre_event_seconds is not None else 'no disponible'} s.",
        f"Deriva cruda: velocidad final/PGV={vend_ratio:.3g}, desplazamiento final/PGD={dend_ratio:.3g}; severidad {drift_label}.",
        f"Despiking {'activado' if despike else 'no requerido'}; candidatos detectados={spike_count}.",
        f"Familias de filtro evaluadas: {', '.join(selected_filter_types)}.",
        *snr_notes,
        "Se evaluan ordenes bajos de baseline, sensibilidad de high-pass y candidatos de control de deriva de largo periodo.",
    ]

    return CorrectionParameterSuggestion(
        windows=windows,
        remove_mean=True,
        despike=despike,
        spike_sigma=8.0,
        taper_fraction=taper_fraction,
        baseline_orders=baseline_orders,
        highpass_hz_candidates=tuple(hp_candidates),
        lowpass_hz=lp,
        filter_order=4,
        filter_types=selected_filter_types,
        zero_phase=True,
        constrain_final_velocity=True,
        include_final_displacement_constraint=include_displacement_constraint,
        drift_severity=drift_label,
        snr_available=snr_available,
        snr_threshold=snr_threshold,
        notes=tuple(notes),
    )


def _candidate_filter_suffix(filter_type: str) -> str:
    key = filter_type.lower().replace("-", "_")
    if key in {"butter", "butterworth"}:
        return ""
    return f"_{key}"


def build_correction_candidates(
    record: MotionRecord,
    *,
    parameters: CorrectionParameterSuggestion | None = None,
    filter_types: str | Iterable[str] | None = None,
) -> tuple[CorrectionCandidate, ...]:
    if parameters is None:
        params = recommend_correction_parameters(record, filter_types=filter_types)
    elif filter_types is not None:
        params = replace(
            parameters, filter_types=normalize_recommendation_filter_types(filter_types)
        )
    else:
        params = parameters
    base = CorrectionConfig(
        remove_mean=params.remove_mean,
        pre_event_seconds=params.windows.pre_event_seconds,
        despike=params.despike,
        spike_sigma=params.spike_sigma,
        taper_fraction=params.taper_fraction,
        filter_order=params.filter_order,
        zero_phase=params.zero_phase,
        constrain_final_velocity=params.constrain_final_velocity,
    )
    candidates: list[CorrectionCandidate] = []
    seen: set[tuple] = set()

    def add_candidate(name: str, description: str, cfg: CorrectionConfig) -> None:
        key = (
            cfg.baseline_order,
            cfg.highpass_hz,
            cfg.lowpass_hz,
            cfg.filter_type,
            cfg.constrain_final_displacement,
            cfg.baseline_coefficients,
            cfg.post_filter_baseline_order,
            cfg.post_filter_baseline_coefficients,
            cfg.post_filter_constrain_final_velocity,
            cfg.post_filter_constrain_final_displacement,
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            CorrectionCandidate(name=name, description=description, config=cfg)
        )

    filter_types = params.filter_types or ("butterworth",)
    use_post_filter_drift_control = (
        params.include_final_displacement_constraint
        or params.drift_severity in {"media", "alta"}
    )

    for order in params.baseline_orders:
        for hp in params.highpass_hz_candidates:
            families = ("butterworth",) if hp is None else filter_types
            filter_label = "sin_filtro" if hp is None else f"hp_{hp:g}"
            for filter_type in families:
                suffix = _candidate_filter_suffix(filter_type) if hp is not None else ""
                cfg = replace(
                    base,
                    baseline_order=order,
                    highpass_hz=hp,
                    lowpass_hz=params.lowpass_hz if hp is not None else None,
                    filter_type=filter_type,
                    constrain_final_displacement=False,
                    post_filter_baseline_order=None,
                    post_filter_constrain_final_velocity=True,
                    post_filter_constrain_final_displacement=False,
                )
                add_candidate(
                    f"baseline_{order}_{filter_label}{suffix}",
                    f"Baseline orden {order}, {filter_label}, filtro {filter_type} de fase cero y parametros estimados automaticamente.",
                    cfg,
                )

                if hp is not None and use_post_filter_drift_control:
                    post_cfg = replace(
                        cfg,
                        post_filter_baseline_order=1,
                        post_filter_constrain_final_velocity=True,
                        post_filter_constrain_final_displacement=True,
                    )
                    add_candidate(
                        f"baseline_{order}_{filter_label}{suffix}_postpoly_1_vf_df",
                        "Baseline y filtro con correccion polinomial posterior para anular velocidad y desplazamiento final.",
                        post_cfg,
                    )

    if params.include_final_displacement_constraint:
        for order in params.baseline_orders:
            if order >= 1:
                cfg = replace(
                    base,
                    baseline_order=order,
                    highpass_hz=None,
                    lowpass_hz=None,
                    filter_type="butterworth",
                    constrain_final_displacement=True,
                    post_filter_baseline_order=1,
                    post_filter_constrain_final_velocity=True,
                    post_filter_constrain_final_displacement=True,
                )
                add_candidate(
                    f"baseline_{order}_vf_df",
                    "Baseline con velocidad y desplazamiento final objetivo; revisar si puede haber desplazamiento permanente real.",
                    cfg,
                )

    return tuple(candidates)


def _safe_log_ratio(value: float, reference: float) -> float:
    eps = np.finfo(float).tiny
    return float(abs(np.log(max(value, eps) / max(reference, eps))))


def _post_event_velocity_drift_ratio(
    velocity: np.ndarray, dt: float, fraction: float = 0.20
) -> float:
    n = velocity.size
    if n < 8:
        return 0.0
    start = max(0, int((1.0 - fraction) * n))
    y = velocity[start:]
    if y.size < 4:
        return 0.0
    x = np.arange(y.size, dtype=np.float64) * dt
    slope = float(np.polyfit(x, y, 1)[0])
    drift = abs(slope) * max(float(x[-1] - x[0]), dt)
    pgv = max(float(np.max(np.abs(velocity))), np.finfo(float).eps)
    return drift / pgv


def _post_event_displacement_range_ratio(
    displacement: np.ndarray,
    time: np.ndarray,
    windows: EventWindows,
    *,
    fraction_fallback: float = 0.20,
) -> float:
    n = displacement.size
    if n < 8:
        return 0.0
    start = int(np.searchsorted(time, windows.post_event_start_seconds, side="left"))
    if start >= n - 4:
        start = max(0, int((1.0 - fraction_fallback) * n))
    y = displacement[start:]
    if y.size < 4:
        return 0.0
    post_range = float(np.max(y) - np.min(y))
    pgd = max(float(np.max(np.abs(displacement))), np.finfo(float).eps)
    return post_range / pgd


def _spectral_change(
    record: MotionRecord, result: CorrectionResult, periods: np.ndarray, damping: float
) -> float:
    raw_spec = response_spectrum(record, periods, damping=damping, output_units="g")
    corrected_spec = response_spectrum(
        result.record, periods, damping=damping, output_units="g"
    )
    raw_sa = np.maximum(raw_spec.sa, np.finfo(float).tiny)
    corrected_sa = np.maximum(corrected_spec.sa, np.finfo(float).tiny)
    err = np.log(corrected_sa / raw_sa)
    return float(np.sqrt(np.mean(err * err)))


def _complexity_penalty(config: CorrectionConfig) -> float:
    order = max(config.baseline_order, 0)
    penalty = 0.05 * order
    if config.constrain_final_displacement:
        penalty += 0.25
    if (
        config.post_filter_baseline_order is not None
        and config.post_filter_baseline_order >= 0
    ):
        penalty += 0.06 * max(config.post_filter_baseline_order, 0) + 0.08
        if config.post_filter_constrain_final_displacement:
            penalty += 0.08
    if config.despike:
        penalty += 0.03
    return penalty


def _filter_penalty(config: CorrectionConfig, t_max: float) -> float:
    penalty = 0.0
    if config.highpass_hz:
        hp = float(config.highpass_hz)
        penalty += 0.08 * (hp / 0.05)
        cutoff_period = 1.0 / hp
        if t_max > 0.50 * cutoff_period:
            penalty += 0.30 * (t_max / cutoff_period)
    if config.lowpass_hz:
        penalty += 0.03
    family = config.filter_type.lower().replace("-", "_")
    if family in {"cheby1", "chebyshev", "chebyshev1", "chebyshev_i"}:
        penalty += 0.03
    elif family in {
        "cheby2",
        "chebyshev2",
        "chebyshev_ii",
        "ellip",
        "elliptic",
        "cauer",
    }:
        penalty += 0.05
    if not config.zero_phase:
        penalty += 0.20
    return penalty


def _score_candidate(
    record: MotionRecord,
    candidate: CorrectionCandidate,
    raw_metrics,
    periods: np.ndarray,
    windows: EventWindows,
    *,
    damping: float,
    t_max: float,
) -> CorrectionCandidateEvaluation:
    result = correct_record(record, candidate.config)
    metrics = result.metrics

    final_velocity_ratio = abs(metrics.final_velocity) / max(metrics.pgv, 1.0e-9)
    final_displacement_ratio = abs(metrics.final_displacement) / max(
        metrics.pgd, 1.0e-9
    )
    final_displacement_pgv_seconds = abs(metrics.final_displacement) / max(
        metrics.pgv, 1.0e-9
    )
    pgd_pgv_seconds = metrics.pgd / max(metrics.pgv, 1.0e-9)
    post_event_ratio = _post_event_velocity_drift_ratio(result.velocity, record.dt)
    post_event_displacement_ratio = _post_event_displacement_range_ratio(
        result.displacement, record.time, windows
    )
    spectral_change = _spectral_change(record, result, periods, damping)
    pga_change = _safe_log_ratio(metrics.pga, raw_metrics.pga)
    arias_change = _safe_log_ratio(metrics.arias_intensity, raw_metrics.arias_intensity)
    cav_change = _safe_log_ratio(metrics.cav, raw_metrics.cav)
    complexity = _complexity_penalty(candidate.config)
    filter_penalty = _filter_penalty(candidate.config, t_max)

    displacement_time_scale = max(
        0.75, min(2.50, 0.04 * max(windows.d5_95_seconds, record.dt))
    )
    terminal_velocity_score = min(final_velocity_ratio / 0.05, 10.0)
    terminal_displacement_score = min(final_displacement_ratio / 0.80, 5.0)
    terminal_displacement_pgv_score = min(
        final_displacement_pgv_seconds / displacement_time_scale, 10.0
    )
    pgd_pgv_score = min(pgd_pgv_seconds / displacement_time_scale, 10.0)
    post_event_score = min(post_event_ratio / 0.08, 10.0)
    post_event_displacement_score = min(post_event_displacement_ratio / 0.20, 10.0)
    spectral_score = min(spectral_change / log(1.35), 8.0)
    pga_score = min(pga_change / log(1.25), 5.0)
    arias_score = min(arias_change / log(1.50), 5.0)
    cav_score = min(cav_change / log(1.50), 5.0)

    score = (
        0.18 * terminal_velocity_score
        + 0.04 * terminal_displacement_score
        + 0.20 * terminal_displacement_pgv_score
        + 0.24 * pgd_pgv_score
        + 0.10 * post_event_score
        + 0.12 * post_event_displacement_score
        + 0.06 * spectral_score
        + 0.03 * arias_score
        + 0.02 * pga_score
        + 0.01 * cav_score
        + complexity
        + filter_penalty
    )

    notes: list[str] = []
    if final_velocity_ratio <= 0.05:
        notes.append("velocidad final controlada")
    if final_displacement_ratio <= 0.30:
        notes.append("desplazamiento final estable")
    if pgd_pgv_seconds <= displacement_time_scale:
        notes.append("desplazamiento integrado en rango fisico")
    if post_event_ratio <= 0.08:
        notes.append("sin tendencia post-evento dominante")
    if post_event_displacement_ratio <= 0.20:
        notes.append("sin ondulacion post-evento dominante")
    if spectral_change <= log(1.35):
        notes.append("cambio espectral moderado")
    if candidate.config.highpass_hz:
        notes.append(f"high-pass {candidate.config.highpass_hz:g} Hz")
    if candidate.config.filter_type.lower() not in {"butter", "butterworth"}:
        notes.append(f"filtro {candidate.config.filter_type}")
    if candidate.config.post_filter_baseline_order is not None:
        notes.append(
            f"polinomio post-filtro orden {candidate.config.post_filter_baseline_order}"
        )
    if (
        candidate.config.constrain_final_displacement
        or candidate.config.post_filter_constrain_final_displacement
    ):
        notes.append("impone desplazamiento final; revisar fling-step")

    result.diagnostics["recommendation_method"] = candidate.name
    result.diagnostics["recommendation_score"] = float(score)

    return CorrectionCandidateEvaluation(
        name=candidate.name,
        description=candidate.description,
        config=candidate.config,
        result=result,
        score=float(score),
        final_velocity_ratio=float(final_velocity_ratio),
        final_displacement_ratio=float(final_displacement_ratio),
        final_displacement_pgv_seconds=float(final_displacement_pgv_seconds),
        pgd_pgv_seconds=float(pgd_pgv_seconds),
        post_event_velocity_drift_ratio=float(post_event_ratio),
        post_event_displacement_range_ratio=float(post_event_displacement_ratio),
        spectral_rms_log_change=float(spectral_change),
        pga_log_change=float(pga_change),
        arias_log_change=float(arias_change),
        cav_log_change=float(cav_change),
        complexity_penalty=float(complexity),
        filter_penalty=float(filter_penalty),
        notes=tuple(notes),
    )


def recommend_correction_method(
    record: MotionRecord,
    *,
    candidates: Iterable[CorrectionCandidate] | None = None,
    periods=None,
    t_min: float = 0.05,
    t_max: float = 3.0,
    damping: float = 0.05,
    snr_threshold: float = 3.0,
    filter_types: str | Iterable[str] | None = None,
) -> CorrectionRecommendation:
    """Recommend correction parameters and the best correction result.

    The workflow mirrors the theoretical processing sequence: detect useful
    windows from Arias intensity, estimate noise-controlled filter corners from
    Fourier SNR when possible, build low-order baseline alternatives, test
    sensitivity, and select the option with stable velocity/displacement while
    preserving physically relevant intensity and spectral measures.
    """

    per = (
        np.geomspace(t_min, t_max, 40)
        if periods is None
        else np.asarray(periods, dtype=np.float64)
    )
    parameter_suggestion = recommend_correction_parameters(
        record,
        snr_threshold=snr_threshold,
        filter_types=filter_types,
    )
    raw_metrics = compute_ground_motion_parameters(record)
    candidate_list = (
        tuple(candidates)
        if candidates is not None
        else build_correction_candidates(
            record,
            parameters=parameter_suggestion,
        )
    )
    evaluations = [
        _score_candidate(
            record,
            candidate,
            raw_metrics,
            per,
            parameter_suggestion.windows,
            damping=damping,
            t_max=t_max,
        )
        for candidate in candidate_list
    ]
    ranked = tuple(
        sorted(
            evaluations,
            key=lambda item: (item.score, item.complexity_penalty, item.filter_penalty),
        )
    )
    best = ranked[0]
    best_pre_coeffs = _format_coefficients_for_row(
        best.result.diagnostics.get("pre_filter_baseline_coefficients")
    )
    best_post_coeffs = _format_coefficients_for_row(
        best.result.diagnostics.get("post_filter_baseline_coefficients")
    )
    coefficient_notes = [
        f"Coeficientes recomendados de baseline pre-filtro [m/s^2, tau 0-1]: {best_pre_coeffs or 'sin correccion'}.",
    ]
    if best_post_coeffs:
        coefficient_notes.append(
            f"Coeficientes recomendados de baseline post-filtro [m/s^2, tau 0-1]: {best_post_coeffs}."
        )
    decision_notes = (
        f"Metodo recomendado: {best.name}.",
        f"Familias de filtro consideradas: {', '.join(parameter_suggestion.filter_types)}.",
        "Los parametros se estimaron con ventanas Arias, SNR Fourier, drift terminal y sensibilidad de baseline/filtro.",
        *coefficient_notes,
        "La seleccion penaliza desplazamiento/velocidad de largo periodo no fisicos, deriva y tendencia post-evento sin distorsionar excesivamente PGA, Arias, CAV ni espectro.",
        "Si el registro puede contener desplazamiento permanente fisico, revise manualmente candidatos con desplazamiento final impuesto.",
    )
    return CorrectionRecommendation(
        parameter_suggestion=parameter_suggestion,
        best=best,
        candidates=ranked,
        periods=per,
        decision_notes=decision_notes,
    )
