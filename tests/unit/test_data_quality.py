from datetime import date

import pandas as pd
import polars as pl
import pytest

from data_pipeline.clean_and_load import QualityReport, clean, standardize, to_records, validate
from data_pipeline.macro_ingestion import clean_series
from data_pipeline.news_ingestion import normalize_articles
from data_pipeline.sentiment_scoring import build_daily_sentiment, substantive_passages


def raw_frame(rows):
    return pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"])


def test_standardize_maps_close_and_adj_close_separately():
    raw = raw_frame([["2026-01-05", 10, 11, 9, 10.5, 10.2, 1000, 0, 0]])
    df = standardize(raw, "AAA")
    row = df.row(0, named=True)
    assert row["close"] == 10.5 and row["adj_close"] == 10.2 and row["ticker"] == "AAA"


def test_standardize_rejects_missing_columns():
    with pytest.raises(ValueError):
        standardize(pd.DataFrame({"Date": ["2026-01-05"], "Close": [1.0]}), "AAA")


def test_clean_drops_invalid_duplicates_and_incomplete_sessions():
    raw = raw_frame([
        ["2026-01-05", 10, 11, 9, 10.5, 10.5, 1000, 0, 0],
        ["2026-01-05", 10, 11, 9, 10.6, 10.6, 1100, 0, 0],   # duplicate date, keep last
        ["2026-01-06", 10, 11, 9, -1, -1, 1000, 0, 0],       # invalid close
        ["2026-01-07", 10, 9, 11, 10, 10, 1000, 0, 0],       # high < low
        ["2026-01-08", 10, 11, 9, 10, 10, -5, 0, 0],         # negative volume
        ["2026-01-09", 10, 12, 9, 11, 11, 1000, 0.5, 0],     # valid with dividend
        ["2026-01-12", 10, 12, 9, 11, 11, 1000, 0, 0],       # after last completed session
    ])
    report = QualityReport("AAA")
    df = clean(standardize(raw, "AAA"), report, last_session=date(2026, 1, 9))
    assert df["date"].to_list() == [date(2026, 1, 5), date(2026, 1, 9)]
    assert df.filter(pl.col("date") == date(2026, 1, 5))["close"].item() == 10.6
    assert df["split_ratio"].to_list() == [1.0, 1.0]
    assert report.dropped == {"incomplete_session": 1, "invalid_close": 1, "negative_volume": 1,
                              "inconsistent_ohlc": 1, "duplicate_date": 1}


def test_validate_warns_on_unexplained_jump_but_not_on_split():
    df = pl.DataFrame({
        "date": [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
        "close": [100.0, 30.0, 31.0], "volume": [1.0, 1.0, 1.0], "split_ratio": [1.0, 1.0, 1.0],
    })
    report = QualityReport("AAA")
    validate(df, report, as_of=date(2026, 1, 7))
    assert any("unexplained" in warning for warning in report.warnings)
    split = df.with_columns(pl.Series("split_ratio", [1.0, 4.0, 1.0]))
    report = QualityReport("AAA")
    validate(split, report, as_of=date(2026, 1, 7))
    assert not report.warnings and not report.errors


def test_validate_flags_stale_data():
    df = pl.DataFrame({"date": [date(2026, 1, 5)], "close": [1.0], "volume": [1.0], "split_ratio": [1.0]})
    report = QualityReport("AAA")
    validate(df, report, as_of=date(2026, 2, 5))
    assert any("stale" in warning for warning in report.warnings)


def test_records_are_json_safe_for_postgres():
    raw = raw_frame([["2026-01-05", 10, 11, 9, 10.5, None, 1000.4, 0, 0]])
    records = to_records(clean(standardize(raw, "AAA"), QualityReport("AAA"), last_session=date(2026, 1, 5)))
    assert records[0]["adj_close"] is None and records[0]["volume"] == 1000


def test_fred_missing_markers_become_null():
    raw = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-02"],
                        "value": [".", "", "4.5", "4.4"]})
    cleaned = clean_series(raw, "DGS10")
    assert cleaned.height == 3  # de-duplicated
    assert cleaned["value"].to_list() == [None, 4.4, 4.5]


def test_news_normalisation_filters_universe_and_deduplicates():
    articles = [
        {"uuid": "a", "published_at": "2026-01-05T10:00:00Z", "title": "t",
         "entities": [{"symbol": "AAPL", "sentiment_score": 0.4}, {"symbol": "ZZZZ"}, {"symbol": "AAPL", "sentiment_score": 0.5}]},
        {"uuid": None, "published_at": "2026-01-05T10:00:00Z", "entities": [{"symbol": "AAPL"}]},
    ]
    frame = normalize_articles(articles, {"AAPL"})
    assert len(frame) == 1 and frame.iloc[0]["entity_sentiment"] == 0.5


def test_daily_sentiment_respects_market_close():
    news = pd.DataFrame({
        "uuid": ["1", "2", "3"], "ticker": ["AAPL"] * 3,
        "published_at": pd.to_datetime(["2026-03-02 14:00Z", "2026-03-02 22:00Z", "2026-03-03 15:00Z"]),
        "finbert_score": [0.5, -0.5, 0.3], "entity_sentiment": [None, None, None],
    })
    daily = build_daily_sentiment(news, [date(2026, 3, 2), date(2026, 3, 3)]).set_index("date")
    assert daily.loc[date(2026, 3, 2), "article_count"] == 1
    assert daily.loc[date(2026, 3, 3), "article_count"] == 2
    assert daily.loc[date(2026, 3, 3), "finbert_sentiment_mean"] == pytest.approx(-0.1)


def test_substantive_passages_prefer_exhibits():
    document = "Cover page check the appropriate box below " * 3 + "\n" + "EXHIBIT ex99.htm\n" + ("Revenue grew strongly in the quarter driven by services demand. " * 3 + "\n") * 5
    passages = substantive_passages(document)
    assert passages and all("Revenue grew" in passage for passage in passages)
