from __future__ import annotations

import numpy as np

from signalprocessor.metrics import (
    compute_ground_motion_parameters,
    compute_ground_motion_parameters_from_series,
    ground_motion_parameters_to_dict,
)
from signalprocessor.records import MotionRecord
from signalprocessor.spectra import pseudo_velocity_from_sa


def test_ground_motion_parameters_are_peak_absolute_values_in_si():
    dt = 0.1
    time = np.arange(0.0, 1.1, dt)
    acc = np.ones_like(time) * 2.0
    rec = MotionRecord(time=time, acceleration=acc, units="m/s^2")

    params = compute_ground_motion_parameters(rec)

    assert np.isclose(params.pga, 2.0)
    assert np.isclose(params.pgv, 2.0)
    assert np.isclose(params.pgd, 1.0)
    assert np.isclose(params.final_velocity, 2.0)
    assert np.isclose(params.final_displacement, 1.0)


def test_ground_motion_parameters_can_use_external_velocity_and_displacement_channels():
    dt = 0.1
    time = np.arange(0.0, 1.1, dt)
    acc = np.ones_like(time) * 2.0
    velocity = np.linspace(0.0, -3.0, time.size)
    displacement = np.linspace(0.0, 0.25, time.size)

    params = compute_ground_motion_parameters_from_series(
        time,
        acc,
        velocity_si=velocity,
        displacement_si=displacement,
    )

    assert np.isclose(params.pga, 2.0)
    assert np.isclose(params.pgv, 3.0)
    assert np.isclose(params.pgd, 0.25)
    assert np.isclose(params.final_velocity, -3.0)
    assert np.isclose(params.final_displacement, 0.25)


def test_ground_motion_parameters_display_unit_conversion():
    dt = 0.1
    time = np.arange(0.0, 1.1, dt)
    rec = MotionRecord(time=time, acceleration=np.ones_like(time) * 2.0, units="m/s^2")
    params = compute_ground_motion_parameters(rec)

    row = ground_motion_parameters_to_dict(
        params,
        acceleration_units="cm/s^2",
        velocity_units="cm/s",
        displacement_units="cm",
    )

    assert np.isclose(row["pga"], 200.0)
    assert np.isclose(row["pgv"], 200.0)
    assert np.isclose(row["pgd"], 100.0)


def test_pseudo_velocity_from_sa_respects_acceleration_units():
    periods = np.asarray([1.0])
    psv_from_si = pseudo_velocity_from_sa(periods, np.asarray([1.0]), units="m/s^2")
    psv_from_cgs = pseudo_velocity_from_sa(periods, np.asarray([100.0]), units="cm/s^2")

    np.testing.assert_allclose(psv_from_cgs, psv_from_si)
