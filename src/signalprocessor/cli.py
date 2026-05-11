from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import benchmark_all
from .correction import process_motion
from .io import load_json, read_motion_csv, read_target_spectrum_csv, save_dataframe_csv
from .scaling import compare_scaling_methods


def _add_common_motion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("motion", help="CSV con columnas tiempo, aceleracion")
    parser.add_argument("--unit", default="g", help="unidad de aceleracion de entrada")
    parser.add_argument("--record-id", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="signalprocessor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_correct = sub.add_parser("correct", help="corregir linea base y filtrar un acelerograma")
    _add_common_motion_args(p_correct)
    p_correct.add_argument("--config", default=None)
    p_correct.add_argument("--output-dir", default="examples/output/correction")
    p_correct.add_argument("--manual", action="store_true", help="usar parametros manuales simples")
    p_correct.add_argument("--highpass", type=float, default=0.26)
    p_correct.add_argument("--lowpass", type=float, default=25.0)
    p_correct.add_argument("--baseline-order", type=int, default=1)

    p_scale = sub.add_parser("scale", help="escalar un acelerograma hacia un espectro objetivo")
    _add_common_motion_args(p_scale)
    p_scale.add_argument("target", help="CSV con columnas periodo, Sa")
    p_scale.add_argument("--config", default=None)
    p_scale.add_argument("--method", choices=["linear", "frequency", "wavelet", "all"], default="wavelet")
    p_scale.add_argument("--output-dir", default="examples/output/scaling")
    p_scale.add_argument("--min-period", type=float, default=0.05)
    p_scale.add_argument("--max-period", type=float, default=2.0)

    p_bench = sub.add_parser("benchmark", help="ejecutar benchmarks de examples/data")
    p_bench.add_argument("--examples-root", default="examples")
    p_bench.add_argument("--output-dir", default="examples/output/benchmark")

    args = parser.parse_args(argv)
    if args.command == "correct":
        config = load_json(args.config) if args.config else None
        motion = read_motion_csv(args.motion, acceleration_unit=args.unit, record_id=args.record_id)
        result = process_motion(
            motion,
            config=config,
            recommend=not args.manual,
            baseline={"method": "polynomial", "order": args.baseline_order},
            filtering={"highpass_hz": args.highpass, "lowpass_hz": args.lowpass, "order": 4},
        )
        paths = result.write_outputs(args.output_dir)
        print(result.summary().to_string(index=False))
        print(f"outputs: {', '.join(str(p) for p in paths.values())}")
        return 0
    if args.command == "scale":
        config = load_json(args.config) if args.config else None
        motion = read_motion_csv(args.motion, acceleration_unit=args.unit, record_id=args.record_id)
        target = read_target_spectrum_csv(args.target)
        methods = ("linear", "frequency", "wavelet") if args.method == "all" else (args.method,)
        comparison = compare_scaling_methods(
            motion,
            target.periods_s,
            target.sa_g,
            methods=methods,
            period_range_s=(args.min_period, args.max_period),
            config=config,
        )
        comparison.write_outputs(args.output_dir)
        print(comparison.summary.to_string(index=False))
        return 0
    if args.command == "benchmark":
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        benches = benchmark_all(args.examples_root)
        for name, df in benches.items():
            save_dataframe_csv(df, output / f"{name}_benchmark.csv")
            print(f"\n{name}")
            print(df.to_string(index=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
