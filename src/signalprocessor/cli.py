from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import as_period_range, load_config
from .io import (
    read_motion_csv,
    read_spectrum_csv,
    write_json,
    write_motion_csv,
    write_spectrum_csv,
)
from .metrics import motion_metrics
from .plotting import save_motion_plot, save_spectrum_plot
from .processing import ProcessConfig, process_motion
from .scaling import (
    frequency_domain_spectral_match,
    scale_motion_to_target,
    scale_suite_to_target,
)
from .spectra import response_spectrum


def _process(args: argparse.Namespace) -> None:
    data = load_config(args.config)
    io_cfg = data.get("io", {})
    motion = read_motion_csv(
        io_cfg["input"],
        unit=io_cfg.get("unit", "g"),
        name=io_cfg.get("name"),
    )
    ouput = process_motion(motion, ProcessConfig.from_dict(data))
    out_dir = Path(io_cfg.get("output_dir", "examples/ouputs/process"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = io_cfg.get("output_stem", motion.name)
    write_motion_csv(
        out_dir / f"{stem}_processed.csv",
        ouput.filtered.motion,
        unit=io_cfg.get("output_unit", "g"),
    )
    write_json(
        out_dir / f"{stem}_metrics.json",
        ouput.metrics
        | {"baseline": ouput.baseline.info, "filter": ouput.filtered.info},
    )
    write_spectrum_csv(
        out_dir / f"{stem}_spectrum.csv",
        ouput.spectrum["period_s"],
        ouput.spectrum["sa_g"],
    )
    save_motion_plot(
        out_dir / f"{stem}_motion.png",
        ouput.filtered.motion,
        ouput.velocity_m_s,
        ouput.displacement_m,
        title=stem,
    )
    save_spectrum_plot(
        out_dir / f"{stem}_spectrum.png",
        ouput.spectrum["period_s"],
        {"processed": ouput.spectrum["sa_g"]},
        title=stem,
    )
    print(f"Wrote processed output to {out_dir}")


def _spectrum(args: argparse.Namespace) -> None:
    motion = read_motion_csv(args.motion, unit=args.unit)
    periods = np.logspace(
        np.log10(args.min_period), np.log10(args.max_period), args.points
    )
    spec = response_spectrum(motion, periods, damping=args.damping)
    write_spectrum_csv(
        args.output,
        spec["period_s"],
        spec["sa_g"],
        extra={"sd_m": spec["sd_m"], "sv_m_s": spec["sv_m_s"]},
    )
    print(f"Wrote spectrum to {args.output}")


def _scale(args: argparse.Namespace) -> None:
    data = load_config(args.config)
    io_cfg = data.get("io", {})
    scale_cfg = data.get("scale", {})
    motion = read_motion_csv(
        io_cfg["input"], unit=io_cfg.get("unit", "g"), name=io_cfg.get("name")
    )
    target_periods, target_sa = read_spectrum_csv(scale_cfg["target_spectrum"])
    ouput = scale_motion_to_target(
        motion,
        target_periods,
        target_sa,
        damping=float(scale_cfg.get("damping", 0.05)),
        method=scale_cfg.get("method", "log_least_squares"),
        period_range=as_period_range(scale_cfg.get("period_range")),
        single_period=scale_cfg.get("single_period_s"),
        factor_bounds=tuple(scale_cfg.get("factor_bounds", [0.1, 10.0])),
    )
    out_dir = Path(io_cfg.get("output_dir", "examples/ouputs/scale"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = io_cfg.get("output_stem", motion.name)
    write_motion_csv(
        out_dir / f"{stem}_scaled.csv",
        ouput.motion,
        unit=io_cfg.get("output_unit", "g"),
    )
    write_spectrum_csv(
        out_dir / f"{stem}_scaled_spectrum.csv", ouput.periods, ouput.scaled_sa_g
    )
    write_json(
        out_dir / f"{stem}_scale.json",
        {
            "factor": ouput.factor,
            "method": ouput.method,
            "metrics": motion_metrics(ouput.motion),
        },
    )
    save_spectrum_plot(
        out_dir / f"{stem}_scale.png",
        ouput.periods,
        {
            "record": ouput.record_sa_g,
            "scaled": ouput.scaled_sa_g,
            "target": ouput.target_sa_g,
        },
        title=f"{stem} factor={ouput.factor:.3g}",
    )
    print(f"Wrote scaled output to {out_dir}; factor={ouput.factor:.6g}")


def _suite(args: argparse.Namespace) -> None:
    data = load_config(args.config)
    suite_cfg = data.get("suite", {})
    target_periods, target_sa = read_spectrum_csv(suite_cfg["target_spectrum"])
    motions = [
        read_motion_csv(
            item["input"],
            unit=item.get("unit", suite_cfg.get("unit", "g")),
            name=item.get("name"),
        )
        for item in suite_cfg["records"]
    ]
    output, geo = scale_suite_to_target(
        motions,
        target_periods,
        target_sa,
        damping=float(suite_cfg.get("damping", 0.05)),
        method=suite_cfg.get("method", "log_least_squares"),
        period_range=as_period_range(suite_cfg.get("period_range")),
        factor_bounds=tuple(suite_cfg.get("factor_bounds", [0.2, 5.0])),
    )
    out_dir = Path(suite_cfg.get("output_dir", "examples/ouputs/suite"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_series = {"target": target_sa, "suite_geo_mean": geo}
    summary = {}
    for ouput in output:
        write_motion_csv(
            out_dir / f"{ouput.motion.name}.csv",
            ouput.motion,
            unit=suite_cfg.get("output_unit", "g"),
        )
        base_name = ouput.motion.name.removesuffix("_scaled")
        plot_series[base_name] = ouput.scaled_sa_g
        summary[base_name] = ouput.factor
    write_spectrum_csv(out_dir / "suite_geomean.csv", target_periods, geo)
    write_json(out_dir / "suite_factors.json", summary)
    save_spectrum_plot(
        out_dir / "suite_scaling.png",
        target_periods,
        plot_series,
        title="Suite scaling",
    )
    print(f"Wrote suite output to {out_dir}")


def _match(args: argparse.Namespace) -> None:
    data = load_config(args.config)
    io_cfg = data.get("io", {})
    match_cfg = data.get("match", {})
    motion = read_motion_csv(
        io_cfg["input"], unit=io_cfg.get("unit", "g"), name=io_cfg.get("name")
    )
    target_periods, target_sa = read_spectrum_csv(match_cfg["target_spectrum"])
    ouput = frequency_domain_spectral_match(
        motion,
        target_periods,
        target_sa,
        damping=float(match_cfg.get("damping", 0.05)),
        iterations=int(match_cfg.get("iterations", 3)),
        max_factor_per_iteration=float(match_cfg.get("max_factor_per_iteration", 1.8)),
        smoothing_width=int(match_cfg.get("smoothing_width", 7)),
        highpass_hz=match_cfg.get("highpass_hz", 0.05),
        lowpass_hz=match_cfg.get("lowpass_hz"),
    )
    out_dir = Path(io_cfg.get("output_dir", "examples/ouputs/match"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = io_cfg.get("output_stem", motion.name)
    write_motion_csv(
        out_dir / f"{stem}_matched.csv",
        ouput.motion,
        unit=io_cfg.get("output_unit", "g"),
    )
    write_spectrum_csv(
        out_dir / f"{stem}_matched_spectrum.csv", ouput.periods, ouput.scaled_sa_g
    )
    write_json(
        out_dir / f"{stem}_match.json",
        {
            "method": ouput.method,
            "final_factor": ouput.factor,
            "metrics": motion_metrics(ouput.motion),
        },
    )
    save_spectrum_plot(
        out_dir / f"{stem}_match.png",
        ouput.periods,
        {"matched": ouput.scaled_sa_g, "target": ouput.target_sa_g},
        title=f"{stem} frequency-domain match",
    )
    print(f"Wrote matched output to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signalprocessor")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "process", help="Correct, filter, integrate and compute metrics."
    )
    p.add_argument("--config", required=True)
    p.set_defaults(func=_process)

    p = sub.add_parser("spectrum", help="Compute an elastic response spectrum.")
    p.add_argument("--motion", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--unit", default="g")
    p.add_argument("--min-period", type=float, default=0.02)
    p.add_argument("--max-period", type=float, default=5.0)
    p.add_argument("--points", type=int, default=120)
    p.add_argument("--damping", type=float, default=0.05)
    p.set_defaults(func=_spectrum)

    p = sub.add_parser("scale", help="Scale one record to a target spectrum.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_scale)

    p = sub.add_parser("suite", help="Scale a suite to a target spectrum.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_suite)

    p = sub.add_parser(
        "match", help="Approximate fast frequency-domain spectral matching."
    )
    p.add_argument("--config", required=True)
    p.set_defaults(func=_match)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
