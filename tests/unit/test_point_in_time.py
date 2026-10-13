"""Leakage guards: event timing, labels, folds and external signals."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from common.market_time import effective_trading_dates
from ml_core.feature_store.external_signals import filing_features, macro_features, news_features
from ml_core.models import forecasting as fc

TRADING = [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4), date(2026, 3, 6), date(2026, 3, 9)]


def test_event_before_close_maps_to_same_day():
    ts = pd.Series(pd.to_datetime(["2026-03-03 15:59:00-05:00"]))
    assert effective_trading_dates(ts, TRADING).iloc[0] == date(2026, 3, 3)


def test_event_after_close_rolls_to_next_session_skipping_non_trading_days():
    ts = pd.Series(pd.to_datetime(["2026-03-04 16:00:00-05:00", "2026-03-06 20:30:00-05:00"]))
    mapped = effective_trading_dates(ts, TRADING)
    assert mapped.iloc[0] == date(2026, 3, 6)  # 2026-03-05 is not in the trading calendar
    assert mapped.iloc[1] == date(2026, 3, 9)  # Friday evening -> Monday


def test_event_after_last_close_is_not_usable_yet():
    ts = pd.Series(pd.to_datetime(["2026-03-09 18:00:00+00:00", "2026-03-09 22:00:00+00:00"]))
    mapped = effective_trading_dates(ts, TRADING)
    assert mapped.iloc[0] == date(2026, 3, 9)  # 14:00 New York
    assert mapped.iloc[1] is None              # 18:00 New York, after the last close


def test_naive_timestamps_are_treated_as_utc():
    ts = pd.Series([pd.Timestamp("2026-03-03 20:00:00")])  # 15:00 New York
    assert effective_trading_dates(ts, TRADING).iloc[0] == date(2026, 3, 3)


def _prices(n=40, dividend_day=None):
    dates = pd.bdate_range("2026-01-01", periods=n)
    close = 100 * np.cumprod(1 + np.linspace(-0.01, 0.01, n))
    dividend = np.zeros(n)
    if dividend_day is not None:
        dividend[dividend_day] = 1.0
    return pd.DataFrame({"ticker": "AAA", "date": dates, "close": close, "dividend": dividend})


def test_forward_return_label_starts_after_feature_date():
    frame = fc.add_targets(_prices(), horizons=(1, 5))
    total = frame["total_return_1d"].to_numpy()
    for t in range(10, 20):
        expected = np.prod(1 + total[t + 1:t + 6]) - 1
        assert frame["fwd_return_5d"].iloc[t] == pytest.approx(expected)
        # The legacy label used shift(-4), which included today's return; ensure that is no longer the case.
        leaky = np.prod(1 + total[t:t + 5]) - 1
        assert frame["fwd_return_5d"].iloc[t] != pytest.approx(leaky)
    assert frame["fwd_return_5d"].iloc[-5:].isna().all()


def test_total_return_includes_dividends():
    frame = fc.add_targets(_prices(dividend_day=10), horizons=(1,))
    price_only = frame["close"].iloc[10] / frame["close"].iloc[9] - 1
    assert frame["total_return_1d"].iloc[10] == pytest.approx(price_only + 1.0 / frame["close"].iloc[9])


def test_forward_volatility_uses_only_future_returns():
    frame = fc.add_targets(_prices(), horizons=(5,))
    total = frame["total_return_1d"].to_numpy()
    t = 12
    assert frame["fwd_vol_5d"].iloc[t] == pytest.approx(np.std(total[t + 1:t + 6], ddof=1) * np.sqrt(252))


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_walk_forward_folds_are_purged_and_embargoed(horizon):
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=1400))
    folds = fc.make_folds(dates, horizon)
    assert folds, "expected at least one fold"
    for fold in folds:
        gap = dates.get_loc(fold.test_dates[0]) - dates.get_loc(fold.train_end)
        assert gap > horizon + fc.EMBARGO_DAYS - 1


def test_fit_and_calibration_sets_are_purged():
    dates = pd.bdate_range("2020-01-01", periods=600)
    train = pd.DataFrame({"date": dates})
    fit, cal = fc.split_fit_cal(train, 21)
    gap = dates.get_loc(cal["date"].min()) - dates.get_loc(fit["date"].max())
    assert gap > 21 + fc.EMBARGO_DAYS - 1


def test_macro_features_use_only_prior_observations():
    macro = pd.DataFrame({"series_id": ["VIXCLS", "VIXCLS"], "date": ["2026-03-02", "2026-03-03"], "value": [20.0, 30.0]})
    out = macro_features(pd.Series(pd.to_datetime(["2026-03-03", "2026-03-04"])), macro=macro)
    assert out.set_index("date").loc["2026-03-03", "vix"] == 20.0  # same-day value not yet published
    assert out.set_index("date").loc["2026-03-04", "vix"] == 30.0


def test_news_features_rolling_window_and_unknown_before_coverage():
    frame = pd.DataFrame({"ticker": "AAA", "date": pd.to_datetime(["2026-02-20", "2026-03-02", "2026-03-03", "2026-03-12"])})
    daily = pd.DataFrame({"ticker": "AAA", "date": pd.to_datetime(["2026-03-02", "2026-03-03"]),
                          "article_count": [2, 1], "finbert_sentiment_mean": [0.5, -0.4]})
    out = news_features(frame, daily=daily).set_index("date")
    assert np.isnan(out.loc["2026-02-20", "news_count_7d"])          # before any coverage: unknown, not zero
    assert out.loc["2026-03-03", "news_count_7d"] == 3
    assert out.loc["2026-03-03", "news_sentiment_7d"] == pytest.approx((2 * 0.5 - 0.4) / 3)
    assert out.loc["2026-03-12", "news_count_7d"] == 0                # window expired
    assert np.isnan(out.loc["2026-03-12", "news_sentiment_7d"])


def test_filing_after_close_counts_from_next_session():
    dates = pd.bdate_range("2025-12-01", "2026-03-10")
    frame = pd.DataFrame({"ticker": "AAA", "date": dates})
    filings = pd.DataFrame({
        "ticker": ["AAA", "AAA"], "form_type": ["8-K", "8-K"], "filing_date": [date(2025, 11, 1), date(2026, 3, 4)],
        "accepted_at": [pd.Timestamp("2025-11-01 12:00", tz="UTC"), pd.Timestamp("2026-03-04 21:30", tz="UTC")],
        "finbert_score": [0.1, -0.6],
    })
    out = filing_features(frame, filings=filings, trading_dates=dates.date).set_index("date")
    assert out.loc["2026-03-04", "filing_count_30d"] == 0  # 16:30 New York -> not usable on 3/4
    assert out.loc["2026-03-05", "filing_count_30d"] == 1
    assert out.loc["2026-03-05", "days_since_filing"] == 0
    assert out.loc["2026-03-05", "filing_sentiment_90d"] == pytest.approx(-0.6)
