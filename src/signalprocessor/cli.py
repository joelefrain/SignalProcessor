from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .io import (
    read_motion,
    read_target_spectrum,
    write_motion_csv,
    write_seismomatch_txt,
)
from .matching import MatchingConfig, match_spectrum
from .processing import CorrectionConfig, correct_record
from .recommendation import recommend_correction_method
from .scaling import linear_scale
from .spectra import response_spectrum


def _parse_coefficients(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if not parts:
        return None
    return tuple(float(part) for part in parts)


def _periods(args) -> np.ndarray:
    if args.periods:
        return np.asarray([float(v) for v in args.periods.split(",")], dtype=np.float64)
    return np.geomspace(args.t_min, args.t_max, args.n_periods)


def cmd_correct(args) -> None:
    rec = read_motion(args.input, units=args.units)
    cfg = CorrectionConfig(
        highpass_hz=args.highpass,
        lowpass_hz=args.lowpass,
        baseline_order=args.baseline_order,
        baseline_fit_method=args.baseline_fit_method,
        baseline_fit_post_event_start_seconds=args.baseline_fit_post_event_start,
        constrain_final_velocity=not args.no_final_velocity_constraint,
        constrain_final_displacement=args.final_displacement_constraint,
        baseline_coefficients=_parse_coefficients(args.baseline_coefficients),
        despike=not args.no_despike,
        taper_fraction=args.taper_fraction,
        filter_order=args.filter_order,
        filter_type=args.filter_type,
        filter_ripple_db=args.filter_ripple_db,
        filter_attenuation_db=args.filter_attenuation_db,
        bessel_norm=args.bessel_norm,
        zero_phase=not args.causal_filter,
        post_filter_baseline_order=args.post_filter_baseline_order,
        post_filter_baseline_fit_method=args.post_filter_baseline_fit_method,
        post_filter_baseline_fit_post_event_start_seconds=args.post_filter_baseline_fit_post_event_start,
        post_filter_baseline_coefficients=_parse_coefficients(
            args.post_filter_baseline_coefficients
        ),
        post_filter_constrain_final_velocity=not args.no_post_filter_final_velocity_constraint,
        post_filter_constrain_final_displacement=args.post_filter_final_displacement_constraint,
    )
    result = correct_record(rec, cfg)
    write_motion_csv(result.record, args.output, units=args.output_units)
    print(f"wrote {args.output}")
    print(
        f"PGA={result.metrics.pga / 9.80665:.4g} g PGV={result.metrics.pgv:.4g} m/s PGD={result.metrics.pgd:.4g} m"
    )
    pre_coeffs = result.diagnostics.get("pre_filter_baseline_coefficients")
    post_coeffs = result.diagnostics.get("post_filter_baseline_coefficients")
    if pre_coeffs is not None:
        print(
            "baseline_coefficients_mps2="
            + ",".join(f"{float(v):.10g}" for v in pre_coeffs)
        )
    if post_coeffs is not None:
        print(
            "post_filter_baseline_coefficients_mps2="
            + ",".join(f"{float(v):.10g}" for v in post_coeffs)
        )


def cmd_recommend(args) -> None:
    rec = read_motion(args.input, units=args.units)
    periods = _periods(args)
    recommendation = recommend_correction_method(
        rec,
        periods=periods,
        t_min=float(periods[0]),
        t_max=float(periods[-1]),
        damping=args.damping,
        snr_threshold=args.snr_threshold,
        filter_types=args.filter_types,
    )

    print("\n".join(recommendation.decision_notes))
    print("Notas de parametros:")
    for note in recommendation.parameter_suggestion.notes:
        print(f"- {note}")

    rows = recommendation.to_rows()[: max(1, args.top)]
    fields = [
        "method",
        "score",
        "filter_type",
        "baseline_order",
        "baseline_fit_method",
        "baseline_weighting_effective_method",
        "highpass_hz",
        "lowpass_hz",
        "post_filter_baseline_order",
        "post_filter_baseline_fit_method",
        "post_filter_baseline_weighting_effective_method",
        "baseline_coefficients_mps2",
        "post_filter_baseline_coefficients_mps2",
        "final_displacement_constraint",
        "post_filter_displacement_constraint",
        "final_velocity_ratio",
        "final_displacement_ratio",
        "pgd_pgv_seconds",
        "spectral_rms_log_change",
    ]

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cmd_spectrum(args) -> None:
    rec = read_motion(args.input, units=args.units)
    spec = response_spectrum(
        rec, _periods(args), damping=args.damping, output_units=args.output_units
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output,
        np.column_stack([spec.periods, spec.sa]),
        delimiter=",",
        fmt="%.10g",
    )
    print(f"wrote {args.output}")


def cmd_scale(args) -> None:
    rec = read_motion(args.input, units=args.units)
    target = read_target_spectrum(args.target, units=args.target_units)
    result = linear_scale(rec, target, t_min=args.t_min, t_max=args.t_max)
    write_motion_csv(result.record, args.output, units=args.output_units)
    print(f"wrote {args.output}")
    print(f"factor={result.factor:.6g} rms_log_error={result.rms_log_error:.4g}")


def cmd_match(args) -> None:
    rec = read_motion(args.input, units=args.units)
    target = read_target_spectrum(args.target, units=args.target_units)
    cfg = MatchingConfig(
        method=args.method,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        t_min=args.t_min,
        t_max=args.t_max,
    )
    result = match_spectrum(rec, target, cfg)
    if args.output.lower().endswith(".txt"):
        write_seismomatch_txt(result.record, args.output)
    else:
        write_motion_csv(result.record, args.output, units=args.output_units)
    print(f"wrote {args.output}")
    print(
        f"method={args.method} iterations={result.iterations} converged={result.converged}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signalprocessor")
    sub = parser.add_subparsers(required=True)

    common_periods = argparse.ArgumentParser(add_help=False)
    common_periods.add_argument("--t-min", type=float, default=0.01)
    common_periods.add_argument("--t-max", type=float, default=6.0)
    common_periods.add_argument("--n-periods", type=int, default=80)
    common_periods.add_argument("--periods")

    p = sub.add_parser("correct")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--units")
    p.add_argument("--output-units", default="g")
    p.add_argument("--highpass", type=float, default=0.05)
    p.add_argument("--lowpass", type=float)
    p.add_argument("--baseline-order", type=int, default=1)
    p.add_argument(
        "--baseline-fit-method",
        default="global",
        choices=["global", "quiet_windows"],
        help="Method for estimating baseline coefficients when --baseline-coefficients is not provided.",
    )
    p.add_argument(
        "--baseline-fit-post-event-start",
        type=float,
        help="Start time in seconds of the post-event quiet window used by --baseline-fit-method quiet_windows.",
    )
    p.add_argument(
        "--baseline-coefficients",
        help="Comma-separated acceleration polynomial coefficients c0,c1,c2,... in m/s^2 on normalized tau 0..1. Overrides fitted pre-filter baseline.",
    )
    p.add_argument("--taper-fraction", type=float, default=0.02)
    p.add_argument("--filter-order", type=int, default=4)
    p.add_argument(
        "--filter-type",
        default="butterworth",
        choices=[
            "butterworth",
            "cheby1",
            "cheby2",
            "chebyshev",
            "chebyshev2",
            "chevyshev",
            "chevyshev2",
            "ellip",
            "bessel",
        ],
    )
    p.add_argument("--filter-ripple-db", type=float, default=0.5)
    p.add_argument("--filter-attenuation-db", type=float, default=40.0)
    p.add_argument("--bessel-norm", default="phase", choices=["phase", "delay", "mag"])
    p.add_argument("--causal-filter", action="store_true")
    p.add_argument("--no-final-velocity-constraint", action="store_true")
    p.add_argument("--final-displacement-constraint", action="store_true")
    p.add_argument("--post-filter-baseline-order", type=int)
    p.add_argument(
        "--post-filter-baseline-fit-method",
        default="global",
        choices=["global", "quiet_windows"],
        help="Method for estimating post-filter baseline coefficients when explicit coefficients are not provided.",
    )
    p.add_argument(
        "--post-filter-baseline-fit-post-event-start",
        type=float,
        help="Start time in seconds of the post-event quiet window used by post-filter quiet-window fitting.",
    )
    p.add_argument(
        "--post-filter-baseline-coefficients",
        help="Comma-separated acceleration polynomial coefficients c0,c1,c2,... in m/s^2 on normalized tau 0..1. Overrides fitted post-filter baseline.",
    )
    p.add_argument("--no-post-filter-final-velocity-constraint", action="store_true")
    p.add_argument("--post-filter-final-displacement-constraint", action="store_true")
    p.add_argument("--no-despike", action="store_true")
    p.set_defaults(func=cmd_correct)

    p = sub.add_parser("recommend")
    p.add_argument("input")
    p.add_argument("--units")
    p.add_argument(
        "--filter-types",
        default="all",
        help="all, butterworth, bessel, cheby1, cheby2, ellip or a comma-separated list",
    )
    p.add_argument("--snr-threshold", type=float, default=3.0)
    p.add_argument("--damping", type=float, default=0.05)
    p.add_argument("--t-min", type=float, default=0.05)
    p.add_argument("--t-max", type=float, default=3.0)
    p.add_argument("--n-periods", type=int, default=40)
    p.add_argument("--periods")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--output")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("spectrum", parents=[common_periods])
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--units")
    p.add_argument("--damping", type=float, default=0.05)
    p.add_argument("--output-units", default="g")
    p.set_defaults(func=cmd_spectrum)

    p = sub.add_parser("scale")
    p.add_argument("input")
    p.add_argument("target")
    p.add_argument("output")
    p.add_argument("--units")
    p.add_argument("--target-units", default="g")
    p.add_argument("--output-units", default="g")
    p.add_argument("--t-min", type=float, default=0.2)
    p.add_argument("--t-max", type=float, default=2.0)
    p.set_defaults(func=cmd_scale)

    p = sub.add_parser("match")
    p.add_argument("input")
    p.add_argument("target")
    p.add_argument("output")
    p.add_argument("--units")
    p.add_argument("--target-units", default="g")
    p.add_argument("--output-units", default="g")
    p.add_argument(
        "--method",
        default="hybrid",
        choices=["frequency", "wavelet", "hybrid"],
        help="Spectral matching update: frequency, wavelet or stabilized hybrid.",
    )
    p.add_argument("--max-iterations", type=int, default=15)
    p.add_argument("--relaxation", type=float, default=0.35)
    p.add_argument("--t-min", type=float, default=0.2)
    p.add_argument("--t-max", type=float, default=2.0)
    p.set_defaults(func=cmd_match)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
