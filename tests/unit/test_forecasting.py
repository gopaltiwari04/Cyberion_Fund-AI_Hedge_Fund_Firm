"""Model bundle behaviour on synthetic data (fast; small boosting rounds)."""

import numpy as np
import pandas as pd
import pytest

from ml_core.models import forecasting as fc


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(fc, "ROUNDS", {"mean": 40, "quantile": 40, "vol": 40})
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-01", periods=700)
    rows = []
    for ticker in ["A", "B", "C", "D", "E"]:
        signal = rng.normal(size=len(dates))
        noise = rng.normal(scale=0.02, size=len(dates))
        frame = pd.DataFrame({"ticker": ticker, "date": dates})
        for feature in fc.MODEL_FEATURES:
            frame[feature] = rng.normal(size=len(dates))
        frame["return_20d"] = signal
        frame["volatility_20d"] = np.abs(rng.normal(0.01, 0.002, len(dates)))
        frame["fwd_return_5d"] = 0.004 * signal + noise
        frame["fwd_vol_5d"] = np.abs(frame["volatility_20d"] * np.sqrt(252) + rng.normal(0, 0.01, len(dates)))
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_bundle_learns_signal_and_intervals_are_ordered(synthetic):
    train = synthetic[synthetic["date"] < "2020-03-01"]
    test = synthetic[synthetic["date"] >= "2020-04-01"]
    fit, cal = fc.split_fit_cal(train, 5)
    bundle = fc.fit_bundle(fit, cal, 5)
    pred = fc.predict_bundle(bundle, test)
    assert (pred["pred_lower"] <= pred["pred_return"]).all() and (pred["pred_return"] <= pred["pred_upper"]).all()
    # Max attainable correlation is 0.004 / sqrt(0.004^2 + 0.02^2) ~ 0.196; random predictions ~ 0.
    assert np.corrcoef(pred["pred_return"], test["fwd_return_5d"])[0, 1] > 0.1
    coverage = ((test["fwd_return_5d"] >= pred["pred_lower"]) & (test["fwd_return_5d"] <= pred["pred_upper"])).mean()
    assert 0.8 <= coverage <= 0.99
    assert (pred["pred_volatility"] > 0).all()


def test_shap_contributions_sum_to_prediction(synthetic):
    fit, cal = fc.split_fit_cal(synthetic[synthetic["date"] < "2020-03-01"], 5)
    bundle = fc.fit_bundle(fit, cal, 5)
    sample = synthetic.tail(20)
    contributions = fc.shap_contributions(bundle, sample)
    assert contributions.shape == (20, len(fc.MODEL_FEATURES) + 1)
    assert np.allclose(contributions.sum(axis=1), fc.predict_bundle(bundle, sample)["pred_return"], atol=1e-5)


def test_shap_matches_reference_implementation(synthetic):
    shap = pytest.importorskip("shap")
    fit, cal = fc.split_fit_cal(synthetic[synthetic["date"] < "2020-03-01"], 5)
    bundle = fc.fit_bundle(fit, cal, 5)
    sample = synthetic.tail(10)
    reference = shap.TreeExplainer(bundle["mean"]).shap_values(sample[fc.MODEL_FEATURES])
    assert np.allclose(fc.shap_contributions(bundle, sample)[:, :-1], reference, atol=1e-4)


def test_evaluate_reports_baselines_and_calibration(synthetic):
    oos = synthetic.assign(
        realized_return=synthetic["fwd_return_5d"], pred_return=synthetic["fwd_return_5d"] * 0.5,
        pred_lower=synthetic["fwd_return_5d"] - 0.05, pred_upper=synthetic["fwd_return_5d"] + 0.05,
        pred_volatility=0.2, realized_volatility=0.21,
    )
    metrics, calibration = fc.evaluate(oos, 5)
    assert metrics["interval_coverage"] == 1.0
    assert metrics["ic_mean"] > 0.99
    assert "mae_zero_baseline" in metrics and "vol_mae_trailing20d_baseline" in metrics
    assert len(calibration["decile_table"]) == 10
    ok, reasons = fc.activation_gates(metrics)
    assert not ok and any("coverage" in reason for reason in reasons)
