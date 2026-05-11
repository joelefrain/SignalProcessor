from __future__ import annotations

import numpy as np

import signalprocessor as sp


def test_manual_correction_pipeline_runs():
    motion = sp.read_motion_csv("examples/data/motion/LIMANS.csv", acceleration_unit="g")
    result = sp.process_motion(
        motion,
        baseline={"method": "polynomial", "order": 1},
        filtering={"highpass_hz": 0.05, "lowpass_hz": 20.0, "order": 4},
        recommend=False,
    )
    assert result.filtered.npts == motion.npts
    assert np.isfinite(result.velocity_mps).all()
    assert result.metrics["PGA_g"] > 0
    assert result.baseline_parameters["name"].startswith("manual_weighted")


def test_recommended_correction_pipeline_selects_candidate():
    motion = sp.read_motion_csv("examples/data/motion/LIMANS.csv", acceleration_unit="g")
    config = sp.load_json("examples/config/correction.json")
    result = sp.process_motion(motion, config=config, recommend=True)
    assert not result.candidate_table.empty
    assert result.filter_parameters["highpass_hz"] is not None
    assert result.metrics["PGV_mps"] > 0


def test_pair_correction_uses_shared_filter():
    ns = sp.read_motion_csv("examples/data/motion/LIMANS.csv", acceleration_unit="g", component="NS")
    ew = sp.read_motion_csv("examples/data/motion/LIMAEW.csv", acceleration_unit="g", component="EW")
    pair = sp.process_pair(ns, ew, recommend=True, shared_filter=True, pair_id="LIMA")
    assert pair.ns.filter_parameters["reason"] == "shared_pair_filter"
    assert pair.ew.filter_parameters["highpass_hz"] == pair.ns.filter_parameters["highpass_hz"]
    assert len(pair.summary()) == 2

