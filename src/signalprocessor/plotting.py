from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import G0
from .motion import Motion


def save_motion_plot(
    path: str | Path,
    motion: Motion,
    velocity_m_s: np.ndarray,
    displacement_m: np.ndarray,
    *,
    title: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(motion.time, motion.accel / G0, lw=0.8, color="#1f77b4")
    axes[0].set_ylabel("a (g)")
    axes[1].plot(motion.time, velocity_m_s * 100.0, lw=0.8, color="#2ca02c")
    axes[1].set_ylabel("v (cm/s)")
    axes[2].plot(motion.time, displacement_m * 100.0, lw=0.8, color="#d62728")
    axes[2].set_ylabel("u (cm)")
    axes[2].set_xlabel("t (s)")
    if title:
        fig.suptitle(title)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_spectrum_plot(
    path: str | Path,
    period: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    title: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for label, values in series.items():
        ax.loglog(period, values, lw=1.4, label=label)
    ax.set_xlabel("T (s)")
    ax.set_ylabel("Sa (g)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
