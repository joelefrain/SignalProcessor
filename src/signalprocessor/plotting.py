from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .correction import CorrectionResult
from .metrics import arias_intensity
from .scaling import ScalingResult


def plot_correction_summary(result: CorrectionResult):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t = result.filtered.time
    axes[0].plot(result.original.time, result.original.acc_g, color="0.65", lw=0.8, label="original", marker=".", markersize=2)
    axes[0].plot(t, result.filtered.acc_g, color="#165a72", lw=0.9, label="procesada")
    axes[0].set_ylabel("a (g)")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, result.velocity_mps, color="#8a4b08", lw=0.9)
    axes[1].set_ylabel("v (m/s)")
    axes[2].plot(t, result.displacement_m, color="#345b2c", lw=0.9)
    axes[2].set_ylabel("u (m)")
    axes[2].set_xlabel("tiempo (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_scaling_summary(result: ScalingResult):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    t = result.output_motion.time
    axes[0].plot(result.input_motion.time, result.input_motion.acc_g, color="0.7", lw=0.8, label="semilla")
    axes[0].plot(t, result.output_motion.acc_g, color="#165a72", lw=0.9, label=result.method)
    axes[0].set_xlabel("tiempo (s)")
    axes[0].set_ylabel("a (g)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)
    axes[1].loglog(result.periods_s, result.target_sa_g, color="black", lw=1.4, label="target")
    axes[1].loglog(
        result.initial_spectrum["period_s"],
        result.initial_spectrum["psa_g"],
        color="0.65",
        lw=0.9,
        label="inicial",
    )
    axes[1].loglog(
        result.final_spectrum["period_s"],
        result.final_spectrum["psa_g"],
        color="#9b2f2f",
        lw=1.0,
        label="final",
    )
    axes[1].set_xlabel("periodo (s)")
    axes[1].set_ylabel("PSa (g)")
    axes[1].legend(loc="best")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_arias_comparison(result: ScalingResult):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for motion, label, color in [
        (result.input_motion, "semilla", "0.55"),
        (result.output_motion, result.method, "#165a72"),
    ]:
        ia = arias_intensity(motion.acceleration_mps2, motion.dt)
        frac = ia / max(float(ia[-1]), 1.0e-12)
        ax.plot(motion.time, 100.0 * frac, color=color, lw=1.0, label=label)
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("Arias acumulado (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax

