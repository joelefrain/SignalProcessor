from __future__ import annotations

from pathlib import Path

import numpy as np

from signalprocessor.io import read_motion
from signalprocessor.metrics import integrate_motion
from signalprocessor.records import MotionRecord
from signalprocessor.recommendation import (
    DEFAULT_RECOMMENDATION_FILTER_TYPES,
    build_correction_candidates,
    normalize_recommendation_filter_types,
    recommend_correction_method,
    recommend_correction_parameters,
)


def test_parameter_recommendation_returns_candidate_grid():
    dt = 0.01
    time = np.arange(0.0, 30.0, dt)
    pulse = 0.35 * np.sin(2.0 * np.pi * 1.2 * time) * np.exp(-0.07 * time)
    drift = 0.012 + 0.0002 * time
    rec = MotionRecord(time=time, acceleration=pulse + drift, units="m/s^2")

    params = recommend_correction_parameters(rec)

    assert params.remove_mean is True
    assert len(params.baseline_orders) >= 2
    assert len(params.highpass_hz_candidates) >= 2
    assert params.filter_order == 4


def test_recommendation_returns_stable_ranked_result():
    dt = 0.01
    time = np.arange(0.0, 30.0, dt)
    pulse = 0.35 * np.sin(2.0 * np.pi * 1.2 * time) * np.exp(-0.07 * time)
    drift = 0.012 + 0.0002 * time
    rec = MotionRecord(time=time, acceleration=pulse + drift, units="m/s^2")
    raw_vel, _ = integrate_motion(rec.acceleration_si(), rec.dt)

    recommendation = recommend_correction_method(rec, t_min=0.05, t_max=2.0)

    assert recommendation.best.score <= recommendation.candidates[-1].score
    assert recommendation.best.result.metrics.pgv > 0.0
    assert abs(recommendation.best.result.velocity[-1]) < abs(raw_vel[-1])
    assert len(recommendation.to_rows()) >= 4


def test_usgs_hnn_recommendation_rejects_long_period_displacement_drift():
    root = Path(__file__).resolve().parents[1]
    record = read_motion(root / "examples/data/benchmark/uncorrected_motion/CCSP.HNN.._u.smc")

    recommendation = recommend_correction_method(record, t_min=0.05, t_max=3.0)
    no_filter = next(candidate for candidate in recommendation.candidates if candidate.name == "baseline_2_sin_filtro")

    assert recommendation.best.config.highpass_hz is not None
    assert recommendation.best.pgd_pgv_seconds < 1.0
    assert recommendation.best.pgd_pgv_seconds < 0.25 * no_filter.pgd_pgv_seconds
    assert recommendation.best.result.metrics.pgd < 0.50



def test_default_recommendation_uses_all_supported_filter_families():
    dt = 0.01
    time = np.arange(0.0, 20.0, dt)
    pulse = 0.25 * np.sin(2.0 * np.pi * 1.0 * time) * np.exp(-0.06 * time)
    drift = 0.008 + 0.00015 * time
    rec = MotionRecord(time=time, acceleration=pulse + drift, units="m/s^2")

    params = recommend_correction_parameters(rec)

    assert params.filter_types == DEFAULT_RECOMMENDATION_FILTER_TYPES


def test_user_filter_types_restrict_recommendation_candidates():
    dt = 0.01
    time = np.arange(0.0, 20.0, dt)
    pulse = 0.25 * np.sin(2.0 * np.pi * 1.0 * time) * np.exp(-0.06 * time)
    drift = 0.008 + 0.00015 * time
    rec = MotionRecord(time=time, acceleration=pulse + drift, units="m/s^2")

    recommendation = recommend_correction_method(rec, t_min=0.05, t_max=2.0, filter_types="bessel, chebyshev2")
    filtered_families = {
        candidate.config.filter_type
        for candidate in recommendation.candidates
        if candidate.config.highpass_hz is not None
    }

    assert recommendation.parameter_suggestion.filter_types == ("bessel", "cheby2")
    assert filtered_families == {"bessel", "cheby2"}


def test_filter_type_normalization_handles_aliases_and_all():
    assert normalize_recommendation_filter_types("todas") == DEFAULT_RECOMMENDATION_FILTER_TYPES
    assert normalize_recommendation_filter_types(["Butter", "Chevyshev", "elliptic", "Butter"]) == (
        "butterworth",
        "cheby1",
        "ellip",
    )


def test_build_candidates_can_override_filter_types_on_existing_parameters():
    dt = 0.01
    time = np.arange(0.0, 10.0, dt)
    rec = MotionRecord(time=time, acceleration=np.sin(2.0 * np.pi * time), units="m/s^2")
    params = recommend_correction_parameters(rec)

    candidates = build_correction_candidates(rec, parameters=params, filter_types=["ellip"])
    filtered_families = {candidate.config.filter_type for candidate in candidates if candidate.config.highpass_hz is not None}

    assert filtered_families == {"ellip"}
