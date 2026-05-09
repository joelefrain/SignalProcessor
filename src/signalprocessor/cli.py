from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import read_motion, read_target_spectrum, write_motion_csv, write_seismomatch_txt
from .matching import MatchingConfig, match_spectrum
from .processing import CorrectionConfig, correct_record
from .scaling import linear_scale
from .spectra import response_spectrum


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
        constrain_final_velocity=not args.no_final_velocity_constraint,
        constrain_final_displacement=args.final_displacement_constraint,
        despike=not args.no_despike,
    )
    result = correct_record(rec, cfg)
    write_motion_csv(result.record, args.output, units=args.output_units)
    print(f"wrote {args.output}")
    print(f"PGA={result.metrics.pga / 9.80665:.4g} g PGV={result.metrics.pgv:.4g} m/s PGD={result.metrics.pgd:.4g} m")


def cmd_spectrum(args) -> None:
    rec = read_motion(args.input, units=args.units)
    spec = response_spectrum(rec, _periods(args), damping=args.damping, output_units=args.output_units)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.column_stack([spec.periods, spec.sa]), delimiter=",", fmt="%.10g")
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
    print(f"iterations={result.iterations} converged={result.converged}")


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
    p.add_argument("--no-final-velocity-constraint", action="store_true")
    p.add_argument("--final-displacement-constraint", action="store_true")
    p.add_argument("--no-despike", action="store_true")
    p.set_defaults(func=cmd_correct)

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
