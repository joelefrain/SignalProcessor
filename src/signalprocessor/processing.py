from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy import signal

from .core import central_difference
from .metrics import (
    GroundMotionParameters,
    compute_ground_motion_parameters,
    integrate_motion,
)
from .records import MotionRecord

FilterType = Literal[
    "butterworth",
    "butter",
    "cheby1",
    "chebyshev1",
    "chebyshev",
    "cheby2",
    "chebyshev2",
    "ellip",
    "elliptic",
    "bessel",
]

BaselineFitMethod = Literal["global", "quiet_windows"]


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    remove_mean: bool = True
    pre_event_seconds: float | None = None
    baseline_order: int = 1
    baseline_fit_method: str = "global"
    baseline_fit_post_event_start_seconds: float | None = None
    constrain_final_velocity: bool = True
    constrain_final_displacement: bool = False
    target_final_velocity: float = 0.0
    target_final_displacement: float = 0.0
    baseline_coefficients: tuple[float, ...] | None = None
    despike: bool = True
    spike_sigma: float = 8.0
    taper_fraction: float = 0.02
    highpass_hz: float | None = 0.05
    lowpass_hz: float | None = None
    filter_order: int = 4
    filter_type: str = "butterworth"
    filter_ripple_db: float = 0.5
    filter_attenuation_db: float = 40.0
    bessel_norm: str = "phase"
    zero_phase: bool = True
    post_filter_baseline_order: int | None = None
    post_filter_baseline_fit_method: str = "global"
    post_filter_baseline_fit_post_event_start_seconds: float | None = None
    post_filter_baseline_coefficients: tuple[float, ...] | None = None
    post_filter_constrain_final_velocity: bool = True
    post_filter_constrain_final_displacement: bool = False


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    record: MotionRecord
    velocity: np.ndarray
    displacement: np.ndarray
    baseline: np.ndarray
    config: CorrectionConfig
    metrics: GroundMotionParameters
    diagnostics: dict[str, Any] = field(default_factory=dict)


def remove_mean(
    acc: np.ndarray, dt: float, seconds: float | None = None
) -> tuple[np.ndarray, float]:
    n = acc.size if seconds is None else max(1, min(acc.size, int(round(seconds / dt))))
    offset = float(np.mean(acc[:n]))
    return acc - offset, offset


def _constraint_rows(design: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return exact discrete trapezoidal rows for final velocity/displacement.

    The project integrates with cumulative trapezoidal integration. Using the
    same discrete operator for polynomial constraints avoids residual terminal
    drift caused by mixing continuous polynomial integrals with discrete
    integration.
    """

    order_plus_one = design.shape[1]
    velocity_row = np.empty(order_plus_one, dtype=np.float64)
    displacement_row = np.empty(order_plus_one, dtype=np.float64)
    for i in range(order_plus_one):
        velocity, displacement = integrate_motion(design[:, i], dt)
        velocity_row[i] = velocity[-1]
        displacement_row[i] = displacement[-1]
    return velocity_row, displacement_row


def _normalize_baseline_fit_method(method: str | None) -> str:
    key = (method or "global").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "global": "global",
        "all": "global",
        "full": "global",
        "quiet": "quiet_windows",
        "quiet_window": "quiet_windows",
        "quiet_windows": "quiet_windows",
        "ventanas_quietas": "quiet_windows",
        "ventanas_tranquilas": "quiet_windows",
        "pre_post": "quiet_windows",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "baseline_fit_method must be 'global' or 'quiet_windows'"
        ) from exc


def baseline_fit_weights(
    npts: int,
    dt: float,
    *,
    method: str = "global",
    pre_event_seconds: float | None = None,
    post_event_start_seconds: float | None = None,
    signal_weight: float = 0.05,
    min_quiet_samples: int = 8,
) -> tuple[np.ndarray, dict[str, float | int | str | None]]:
    """Return weights for fitting an acceleration-domain baseline.

    ``global`` gives every sample equal weight. ``quiet_windows`` follows the
    theoretical baseline workflow: high weight is assigned to pre-event and
    post-event quiet windows, while the strong-motion interval receives a small
    non-zero weight to keep the least-squares problem numerically stable. If
    not enough quiet samples are available, the function falls back to a global
    fit and records that choice in the returned metadata.
    """

    n = int(npts)
    if n < 2:
        raise ValueError("npts must be at least 2")
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    requested = _normalize_baseline_fit_method(method)
    if requested == "global":
        weights = np.ones(n, dtype=np.float64)
        return weights, {
            "requested_method": requested,
            "effective_method": "global",
            "quiet_sample_count": n,
            "signal_weight": 1.0,
            "pre_event_seconds": pre_event_seconds,
            "post_event_start_seconds": post_event_start_seconds,
        }

    signal_weight = float(np.clip(signal_weight, 1.0e-6, 1.0))
    weights = np.full(n, signal_weight, dtype=np.float64)
    quiet = np.zeros(n, dtype=bool)

    if pre_event_seconds is not None and pre_event_seconds > 0.0:
        pre_n = int(np.floor(float(pre_event_seconds) / dt)) + 1
        pre_n = max(0, min(n, pre_n))
        quiet[:pre_n] = True

    if post_event_start_seconds is not None and post_event_start_seconds >= 0.0:
        post_idx = int(np.ceil(float(post_event_start_seconds) / dt))
        post_idx = max(0, min(n, post_idx))
        quiet[post_idx:] = True

    quiet_count = int(np.count_nonzero(quiet))
    if quiet_count < max(1, int(min_quiet_samples)):
        weights[:] = 1.0
        return weights, {
            "requested_method": requested,
            "effective_method": "global",
            "quiet_sample_count": quiet_count,
            "signal_weight": 1.0,
            "pre_event_seconds": pre_event_seconds,
            "post_event_start_seconds": post_event_start_seconds,
            "fallback_reason": "insufficient_quiet_samples",
        }

    weights[quiet] = 1.0
    return weights, {
        "requested_method": requested,
        "effective_method": "quiet_windows",
        "quiet_sample_count": quiet_count,
        "signal_weight": signal_weight,
        "pre_event_seconds": pre_event_seconds,
        "post_event_start_seconds": post_event_start_seconds,
    }


def coerce_polynomial_coefficients(
    coefficients: Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """Return validated polynomial coefficients or ``None``.

    Coefficients are ordered as ``c0, c1, c2, ...`` and are evaluated on the
    normalized time coordinate ``tau`` in ``[0, 1]``. The units are acceleration
    units of the processing pipeline, i.e. SI ``m/s^2`` when used by
    ``correct_record``.
    """

    if coefficients is None:
        return None
    arr = np.asarray(tuple(coefficients), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            "baseline coefficients must be a non-empty one-dimensional sequence"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("baseline coefficients must be finite")
    return arr


def evaluate_polynomial_baseline(
    npts: int, coefficients: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a user-defined acceleration-domain polynomial baseline.

    The polynomial convention is ``baseline(tau) = c0 + c1*tau + c2*tau^2 + ...``
    with ``tau`` linearly normalized from 0 to 1 over the whole record. This is
    the same convention used by ``polynomial_baseline`` when it estimates the
    coefficients automatically.
    """

    coeffs = coerce_polynomial_coefficients(coefficients)
    if coeffs is None:  # pragma: no cover - guarded by caller
        raise ValueError("coefficients are required")
    if npts < 2:
        raise ValueError("npts must be at least 2")
    tau = np.linspace(0.0, 1.0, int(npts), dtype=np.float64)
    design = np.vander(tau, N=coeffs.size, increasing=True)
    return design @ coeffs, coeffs


def polynomial_baseline(
    acceleration: np.ndarray,
    dt: float,
    order: int,
    *,
    constrain_velocity: bool = False,
    constrain_displacement: bool = False,
    target_final_velocity: float = 0.0,
    target_final_displacement: float = 0.0,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an acceleration-domain polynomial baseline and optional constraints.

    Parameters are expressed in SI acceleration units. The returned baseline is
    the polynomial acceleration that should be subtracted from the input signal.
    Constraints are imposed with the same trapezoidal integration operator used
    elsewhere in the package, so velocity and displacement terminal targets are
    met to numerical precision when the polynomial order has enough degrees of
    freedom.
    """

    acc = np.asarray(acceleration, dtype=np.float64)
    if order < 0:
        return np.zeros_like(acc), np.zeros(0, dtype=np.float64)
    if acc.ndim != 1 or acc.size < 2:
        raise ValueError(
            "acceleration must be a one-dimensional array with at least two samples"
        )

    order = int(order)
    tau = np.linspace(0.0, 1.0, acc.size, dtype=np.float64)
    design = np.vander(tau, N=order + 1, increasing=True)
    if sample_weights is None:
        weights = np.ones(acc.size, dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != acc.shape:
            raise ValueError("sample_weights must have the same shape as acceleration")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weights must be finite")
        weights = np.maximum(weights, 0.0)
        if not np.any(weights > 0.0):
            raise ValueError("sample_weights must contain at least one positive value")
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_acc = acc * sqrt_weights
    lhs = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_acc

    velocity, displacement = integrate_motion(acc, dt)
    velocity_row, displacement_row = _constraint_rows(design, dt)
    constraints: list[np.ndarray] = []
    targets: list[float] = []

    if constrain_velocity:
        constraints.append(velocity_row)
        targets.append(float(velocity[-1] - target_final_velocity))
    if constrain_displacement:
        constraints.append(displacement_row)
        targets.append(float(displacement[-1] - target_final_displacement))

    if constraints:
        cmat = np.vstack(constraints)
        if cmat.shape[0] <= order + 1 and np.linalg.matrix_rank(cmat) == cmat.shape[0]:
            zeros = np.zeros((cmat.shape[0], cmat.shape[0]), dtype=np.float64)
            kkt = np.block([[lhs, cmat.T], [cmat, zeros]])
            krhs = np.concatenate([rhs, np.asarray(targets, dtype=np.float64)])
            coeffs = np.linalg.lstsq(kkt, krhs, rcond=None)[0][: order + 1]
        else:
            # Over-constrained low-order models are solved as high-weighted
            # least squares instead of silently ignoring the requested targets.
            weight = max(np.linalg.norm(lhs, ord=2), 1.0) * 1.0e6
            augmented_lhs = np.vstack([weighted_design, np.sqrt(weight) * cmat])
            augmented_rhs = np.concatenate(
                [weighted_acc, np.sqrt(weight) * np.asarray(targets, dtype=np.float64)]
            )
            coeffs = np.linalg.lstsq(augmented_lhs, augmented_rhs, rcond=None)[0]
    else:
        coeffs = np.linalg.lstsq(weighted_design, weighted_acc, rcond=None)[0]

    baseline = design @ coeffs
    return baseline, coeffs


def despike_array(
    acceleration: np.ndarray, dt: float, *, sigma: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    acc = np.asarray(acceleration, dtype=np.float64).copy()
    deriv = central_difference(acc, float(dt))
    med = float(np.median(deriv))
    mad = float(np.median(np.abs(deriv - med)))
    scale = 1.4826 * mad if mad > 0.0 else float(np.std(deriv))
    if scale <= 0.0:
        return acc, np.zeros(0, dtype=np.int64)
    spike_idx = np.flatnonzero(np.abs(deriv - med) > sigma * scale)
    spike_idx = spike_idx[(spike_idx > 0) & (spike_idx < acc.size - 1)]
    if spike_idx.size == 0:
        return acc, spike_idx
    mask = np.ones(acc.size, dtype=bool)
    mask[spike_idx] = False
    acc[spike_idx] = np.interp(spike_idx, np.flatnonzero(mask), acc[mask])
    return acc, spike_idx


def cosine_taper(acceleration: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return acceleration.copy()
    fraction = float(min(fraction, 0.5))
    window = signal.windows.tukey(acceleration.size, alpha=2.0 * fraction)
    return acceleration * window


def _normalized_filter_type(filter_type: str) -> str:
    key = filter_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
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
        "cauer": "ellip",
        "bessel": "bessel",
        "bessel_thomson": "bessel",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported filter_type={filter_type!r}. Valid filter families: {valid}"
        ) from exc


def _filter_band(
    dt: float,
    highpass_hz: float | None,
    lowpass_hz: float | None,
) -> tuple[float | list[float] | None, str | None, dict[str, float | None]]:
    fs = 1.0 / float(dt)
    nyq = 0.5 * fs
    hp = highpass_hz if highpass_hz and highpass_hz > 0.0 else None
    lp = lowpass_hz if lowpass_hz and lowpass_hz > 0.0 else None
    if lp is not None:
        lp = min(float(lp), 0.98 * nyq)
    if hp is not None:
        hp = min(float(hp), 0.95 * nyq)
    if hp is None and lp is None:
        return (
            None,
            None,
            {"fs": fs, "nyquist": nyq, "highpass_hz": None, "lowpass_hz": None},
        )
    if hp is not None and lp is not None and hp >= lp:
        raise ValueError("highpass_hz must be lower than lowpass_hz")
    if hp is not None and lp is not None:
        return (
            [hp, lp],
            "bandpass",
            {"fs": fs, "nyquist": nyq, "highpass_hz": hp, "lowpass_hz": lp},
        )
    if hp is not None:
        return (
            hp,
            "highpass",
            {"fs": fs, "nyquist": nyq, "highpass_hz": hp, "lowpass_hz": None},
        )
    return (
        lp,
        "lowpass",
        {"fs": fs, "nyquist": nyq, "highpass_hz": None, "lowpass_hz": lp},
    )


def design_iir_filter(
    dt: float,
    *,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    order: int,
    filter_type: str = "butterworth",
    ripple_db: float = 0.5,
    attenuation_db: float = 40.0,
    bessel_norm: str = "phase",
) -> tuple[np.ndarray | None, dict[str, float | str | None]]:
    """Design a second-order-section IIR filter for seismic records.

    Supported families are Butterworth, Chebyshev I, Chebyshev II, elliptic and
    Bessel. Frequencies are in Hz and are interpreted with scipy's digital IIR
    design routines using the record sampling rate.
    """

    if order < 1:
        raise ValueError("filter_order must be at least 1")

    wn, btype, meta = _filter_band(dt, highpass_hz, lowpass_hz)
    family = _normalized_filter_type(filter_type)
    meta["filter_type"] = family
    meta["filter_order"] = float(order)
    if wn is None or btype is None:
        return None, meta

    fs = float(meta["fs"])
    if family == "butterworth":
        sos = signal.iirfilter(
            order, wn, btype=btype, ftype="butter", fs=fs, output="sos"
        )
    elif family == "cheby1":
        sos = signal.iirfilter(
            order,
            wn,
            rp=float(ripple_db),
            btype=btype,
            ftype="cheby1",
            fs=fs,
            output="sos",
        )
        meta["filter_ripple_db"] = float(ripple_db)
    elif family == "cheby2":
        sos = signal.iirfilter(
            order,
            wn,
            rs=float(attenuation_db),
            btype=btype,
            ftype="cheby2",
            fs=fs,
            output="sos",
        )
        meta["filter_attenuation_db"] = float(attenuation_db)
    elif family == "ellip":
        sos = signal.iirfilter(
            order,
            wn,
            rp=float(ripple_db),
            rs=float(attenuation_db),
            btype=btype,
            ftype="ellip",
            fs=fs,
            output="sos",
        )
        meta["filter_ripple_db"] = float(ripple_db)
        meta["filter_attenuation_db"] = float(attenuation_db)
    elif family == "bessel":
        sos = signal.bessel(
            order, wn, btype=btype, fs=fs, output="sos", norm=bessel_norm
        )
        meta["bessel_norm"] = bessel_norm
    else:  # pragma: no cover - guarded by _normalized_filter_type
        raise ValueError(f"Unsupported filter family: {family}")
    return sos, meta


def apply_iir_filter(
    acceleration: np.ndarray,
    dt: float,
    *,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    order: int,
    filter_type: str = "butterworth",
    ripple_db: float = 0.5,
    attenuation_db: float = 40.0,
    bessel_norm: str = "phase",
    zero_phase: bool = True,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    sos, meta = design_iir_filter(
        dt,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        order=order,
        filter_type=filter_type,
        ripple_db=ripple_db,
        attenuation_db=attenuation_db,
        bessel_norm=bessel_norm,
    )
    if sos is None:
        return acceleration.copy(), meta
    if zero_phase:
        return signal.sosfiltfilt(sos, acceleration), meta
    return signal.sosfilt(sos, acceleration), meta


def butterworth_filter(
    acceleration: np.ndarray,
    dt: float,
    *,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    order: int,
    zero_phase: bool = True,
) -> np.ndarray:
    """Backward-compatible Butterworth filter wrapper."""

    filtered, _ = apply_iir_filter(
        acceleration,
        dt,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        order=order,
        filter_type="butterworth",
        zero_phase=zero_phase,
    )
    return filtered


def correct_record(
    record: MotionRecord, config: CorrectionConfig | None = None
) -> CorrectionResult:
    cfg = config or CorrectionConfig()
    acc = record.acceleration_si().astype(np.float64, copy=True)
    dt = record.dt
    diagnostics: dict[str, Any] = {}

    if cfg.despike:
        acc, spikes = despike_array(acc, dt, sigma=cfg.spike_sigma)
        diagnostics["spike_count"] = int(spikes.size)
        diagnostics["spike_indices"] = spikes

    if cfg.remove_mean:
        acc, offset = remove_mean(acc, dt, cfg.pre_event_seconds)
        diagnostics["mean_removed_mps2"] = offset

    user_coeffs = coerce_polynomial_coefficients(cfg.baseline_coefficients)
    if user_coeffs is not None:
        baseline, coeffs = evaluate_polynomial_baseline(acc.size, user_coeffs)
        diagnostics["pre_filter_baseline_source"] = "user_coefficients"
        diagnostics["pre_filter_baseline_order"] = int(coeffs.size - 1)
        diagnostics["pre_filter_baseline_fit_method"] = "explicit_coefficients"
        diagnostics["pre_filter_baseline_weighting"] = {
            "requested_method": "explicit_coefficients",
            "effective_method": "explicit_coefficients",
        }
        if cfg.constrain_final_velocity or cfg.constrain_final_displacement:
            diagnostics["pre_filter_constraint_note"] = (
                "terminal constraints are not re-estimated when explicit baseline_coefficients are supplied"
            )
    else:
        constrain_disp = cfg.constrain_final_displacement and cfg.baseline_order >= 1
        baseline_weights, baseline_weight_meta = baseline_fit_weights(
            acc.size,
            dt,
            method=cfg.baseline_fit_method,
            pre_event_seconds=cfg.pre_event_seconds,
            post_event_start_seconds=cfg.baseline_fit_post_event_start_seconds,
            min_quiet_samples=max(8, 2 * (int(cfg.baseline_order) + 1)),
        )
        baseline, coeffs = polynomial_baseline(
            acc,
            dt,
            cfg.baseline_order,
            constrain_velocity=cfg.constrain_final_velocity,
            constrain_displacement=constrain_disp,
            target_final_velocity=cfg.target_final_velocity,
            target_final_displacement=cfg.target_final_displacement,
            sample_weights=baseline_weights,
        )
        diagnostics["pre_filter_baseline_source"] = "estimated"
        diagnostics["pre_filter_baseline_order"] = int(cfg.baseline_order)
        diagnostics["pre_filter_baseline_fit_method"] = baseline_weight_meta[
            "effective_method"
        ]
        diagnostics["pre_filter_baseline_weighting"] = baseline_weight_meta
    acc = acc - baseline
    diagnostics["baseline_coefficients"] = coeffs
    diagnostics["pre_filter_baseline_coefficients"] = coeffs

    if cfg.taper_fraction > 0.0:
        acc = cosine_taper(acc, cfg.taper_fraction)

    acc, filter_meta = apply_iir_filter(
        acc,
        dt,
        highpass_hz=cfg.highpass_hz,
        lowpass_hz=cfg.lowpass_hz,
        order=cfg.filter_order,
        filter_type=cfg.filter_type,
        ripple_db=cfg.filter_ripple_db,
        attenuation_db=cfg.filter_attenuation_db,
        bessel_norm=cfg.bessel_norm,
        zero_phase=cfg.zero_phase,
    )
    diagnostics["filter"] = filter_meta

    post_user_coeffs = coerce_polynomial_coefficients(
        cfg.post_filter_baseline_coefficients
    )
    if post_user_coeffs is not None:
        post_baseline, post_coeffs = evaluate_polynomial_baseline(
            acc.size, post_user_coeffs
        )
        acc = acc - post_baseline
        diagnostics["post_filter_baseline_coefficients"] = post_coeffs
        diagnostics["post_filter_baseline_order"] = int(post_coeffs.size - 1)
        diagnostics["post_filter_baseline_source"] = "user_coefficients"
        diagnostics["post_filter_baseline_fit_method"] = "explicit_coefficients"
        diagnostics["post_filter_baseline_weighting"] = {
            "requested_method": "explicit_coefficients",
            "effective_method": "explicit_coefficients",
        }
        if (
            cfg.post_filter_constrain_final_velocity
            or cfg.post_filter_constrain_final_displacement
        ):
            diagnostics["post_filter_constraint_note"] = (
                "terminal constraints are not re-estimated when explicit post_filter_baseline_coefficients are supplied"
            )
    elif (
        cfg.post_filter_baseline_order is not None
        and cfg.post_filter_baseline_order >= 0
    ):
        post_constrain_disp = (
            cfg.post_filter_constrain_final_displacement
            and cfg.post_filter_baseline_order >= 1
        )
        post_weights, post_weight_meta = baseline_fit_weights(
            acc.size,
            dt,
            method=cfg.post_filter_baseline_fit_method,
            pre_event_seconds=cfg.pre_event_seconds,
            post_event_start_seconds=(
                cfg.post_filter_baseline_fit_post_event_start_seconds
                if cfg.post_filter_baseline_fit_post_event_start_seconds is not None
                else cfg.baseline_fit_post_event_start_seconds
            ),
            min_quiet_samples=max(8, 2 * (int(cfg.post_filter_baseline_order) + 1)),
        )
        post_baseline, post_coeffs = polynomial_baseline(
            acc,
            dt,
            int(cfg.post_filter_baseline_order),
            constrain_velocity=cfg.post_filter_constrain_final_velocity,
            constrain_displacement=post_constrain_disp,
            target_final_velocity=cfg.target_final_velocity,
            target_final_displacement=cfg.target_final_displacement,
            sample_weights=post_weights,
        )
        acc = acc - post_baseline
        diagnostics["post_filter_baseline_coefficients"] = post_coeffs
        diagnostics["post_filter_baseline_order"] = int(cfg.post_filter_baseline_order)
        diagnostics["post_filter_baseline_source"] = "estimated"
        diagnostics["post_filter_baseline_fit_method"] = post_weight_meta[
            "effective_method"
        ]
        diagnostics["post_filter_baseline_weighting"] = post_weight_meta

    velocity, displacement = integrate_motion(acc, dt)
    corrected = record.with_acceleration(
        acc, units="m/s^2", metadata={"processing": "signalprocessor"}
    )
    metrics = compute_ground_motion_parameters(corrected)
    diagnostics["final_velocity_mps"] = float(velocity[-1])
    diagnostics["final_displacement_m"] = float(displacement[-1])
    return CorrectionResult(
        record=corrected,
        velocity=velocity,
        displacement=displacement,
        baseline=baseline,
        config=cfg,
        metrics=metrics,
        diagnostics=diagnostics,
    )
