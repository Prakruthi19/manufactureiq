"""Calibration (core/calibration.py) and drift monitor (core/drift.py)."""

import numpy as np

from core import drift
from core.calibration import apply_temperature, fit_temperature, reliability


def test_temperature_keeps_the_predicted_class():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(9), size=500)
    for t in (0.5, 1.0, 3.0):
        q = apply_temperature(p, t)
        assert np.allclose(q.sum(1), 1)
        assert (q.argmax(1) == p.argmax(1)).all()


def test_fit_temperature_softens_overconfident_model():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.integers(0, 3, n)
    right = rng.random(n) < 0.7                       # 70% right, but claims ~97%
    pred = np.where(right, y, (y + 1) % 3)
    p = np.full((n, 3), 0.015)
    p[np.arange(n), pred] = 0.97
    t = fit_temperature(p, y)
    assert t > 1
    assert reliability(apply_temperature(p, t), y)["ece"] < reliability(p, y)["ece"]


def test_psi_stable_for_same_distribution_major_for_shift():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5000, 2))
    ref = drift.build_reference(X, ["a", "b"])
    same = drift.compare(ref, rng.normal(size=(2000, 2)), ["a", "b"])
    shifted = drift.compare(ref, rng.normal(loc=1.5, size=(2000, 2)), ["a", "b"])
    assert same["status"] == "stable"
    assert shifted["status"] == "major"


def test_live_report_needs_enough_wafers(tmp_path, monkeypatch):
    monkeypatch.setattr(drift, "DB_PATH", tmp_path / "h.sqlite")
    ref = drift.build_reference(np.random.default_rng(3).normal(size=(1000, 2)), ["a", "b"])
    monkeypatch.setattr(drift, "REFERENCE", tmp_path / "ref.json")
    drift.REFERENCE.write_text(__import__("json").dumps(ref))
    drift.log(np.zeros(2))
    assert drift.live_report(["a", "b"])["status"] == "not_enough_data"
    for v in np.random.default_rng(4).normal(size=(drift.MIN_WAFERS, 2)):
        drift.log(v)
    assert drift.live_report(["a", "b"])["status"] in {"stable", "moderate", "major"}
    drift.clear()
    assert drift.live_report(["a", "b"])["n_wafers"] == 0
