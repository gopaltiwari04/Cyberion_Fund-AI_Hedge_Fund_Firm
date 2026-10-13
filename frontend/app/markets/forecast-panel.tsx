"use client";

import { useEffect, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { ContributionChart, KeyValue, SignalGrid, Tabs } from "../components/ui";
import { api, errorMessage } from "../lib/api";
import { formatDate, formatPercent, formatSignedPercent, signalClass } from "../lib/formatters";
import type { ForecastResponse, NewsArticle } from "../types";

export function ForecastPanel({ ticker }: { ticker: string }) {
  const [data, setData] = useState<{ ticker: string; forecast?: ForecastResponse; error?: string } | null>(null);
  const [horizon, setHorizon] = useState(5);

  useEffect(() => {
    let cancelled = false;
    api.forecast(ticker)
      .then((forecast) => { if (!cancelled) setData({ ticker, forecast }); })
      .catch((error) => { if (!cancelled) setData({ ticker, error: errorMessage(error) }); });
    return () => { cancelled = true; };
  }, [ticker]);

  const loading = data?.ticker !== ticker;
  const forecasts = data?.forecast?.forecasts ?? [];
  const selected = forecasts.find((item) => item.horizon_days === horizon) ?? forecasts[0];

  return (
    <section className="panel forecast-panel" aria-label={`${ticker} forecast`}>
      <div className="panel-header"><SectionHeading eyebrow="Model forecast & explanation" title={`${ticker} outlook`} action={forecasts.length ? <Tabs label="Forecast horizon" value={selected?.horizon_days ?? 5} onChange={setHorizon} options={forecasts.map((f) => ({ value: f.horizon_days, label: `${f.horizon_days}D` }))} /> : null} /></div>
      {loading ? <><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></> : data?.error ? <ErrorState message={data.error} /> : !selected ? <EmptyState title="No forecast" message="No live forecast is available for this asset yet." /> : (
        <div className="forecast-body">
          <div className="kv-grid">
            <KeyValue label={`Expected ${selected.horizon_days}-day return`} value={formatSignedPercent(selected.expected_return)} tone={signalClass(selected.expected_return)} />
            <KeyValue label={`${Math.round(selected.interval_level * 100)}% interval (conformal)`} value={`${formatSignedPercent(selected.interval[0])} to ${formatSignedPercent(selected.interval[1])}`} />
            <KeyValue label="Forecast volatility (ann.)" value={formatPercent(selected.volatility_annualized)} />
            <KeyValue label="Holdout rank IC (since 2020)" value={selected.model.holdout_ic != null ? selected.model.holdout_ic.toFixed(3) : "N/A"} />
            <KeyValue label="Holdout interval coverage" value={formatPercent(selected.model.holdout_interval_coverage, 1)} />
            <KeyValue label="Forecast as of" value={formatDate(selected.as_of_date)} />
          </div>
          <h3 className="subheading">Why the model expects this</h3>
          <ContributionChart explanation={selected.explanation} />
          <h3 className="subheading">Sentiment, filing & graph signals</h3>
          <SignalGrid signals={selected.explanation?.signals} />
          <p className="muted-note">{data?.forecast?.disclaimer} Model version {selected.model.version}.</p>
        </div>
      )}
    </section>
  );
}

export function NewsPanel({ ticker }: { ticker: string }) {
  const [state, setState] = useState<{ ticker: string; articles?: NewsArticle[]; error?: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.news(ticker, 12)
      .then((response) => { if (!cancelled) setState({ ticker, articles: response.articles }); })
      .catch((error) => { if (!cancelled) setState({ ticker, error: errorMessage(error) }); });
    return () => { cancelled = true; };
  }, [ticker]);

  return (
    <section className="panel news-panel" aria-label={`${ticker} news`}>
      <div className="panel-header"><SectionHeading eyebrow="News intelligence" title="Recent coverage" action={<span className="as-of">FinBERT sentiment −1…+1</span>} /></div>
      {state?.ticker !== ticker ? <Skeleton className="skeleton-line" /> : state.error ? <ErrorState message={state.error} /> : !state.articles?.length ? <EmptyState title="No recent articles" message="No news has been ingested for this asset. Coverage is limited by the news API plan." /> : (
        <ul className="news-list">
          {state.articles.map((article) => (
            <li key={`${article.uuid}-${article.ticker}`}>
              <div>
                {article.url ? <a href={article.url} target="_blank" rel="noopener noreferrer">{article.title ?? "Untitled"}</a> : <span>{article.title ?? "Untitled"}</span>}
                <small>{article.source ?? "Unknown source"} · {article.published_at ? new Date(article.published_at).toLocaleString() : "Unknown time"}</small>
              </div>
              <span className={signalClass(article.finbert_score)}>{article.finbert_score == null ? "unscored" : article.finbert_score.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
