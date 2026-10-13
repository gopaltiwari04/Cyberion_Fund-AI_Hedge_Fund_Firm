"use client";

import { useEffect, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { Badge, Disclaimer, KeyValue, StatusBadge, Tabs } from "../components/ui";
import { api, errorMessage } from "../lib/api";
import { formatDate, formatNumber, formatPercent } from "../lib/formatters";
import type { MonitoringMetric, RegistryModel, ResearchModelsResponse, ResearchPredictionsResponse } from "../types";
import { CalibrationChart, IcByYearChart } from "./charts";
import { ResearchChart } from "./research-chart";

function better(value: number | undefined, baseline: number | undefined, lowerIsBetter = true) {
  if (value == null || baseline == null) return undefined;
  return (lowerIsBetter ? value < baseline : value > baseline) ? "signal-positive" : "signal-negative";
}

function ModelDetail({ model }: { model: RegistryModel }) {
  const m = model.metrics;
  const holdout = m.holdout_since_2020;
  return (
    <>
      <section className="panel" aria-label="Evaluation">
        <div className="panel-header"><SectionHeading eyebrow={`Walk-forward out-of-sample · ${formatDate(m.test_start)} – ${formatDate(m.test_end)} · ${m.observations.toLocaleString()} predictions`} title={`${model.horizon_days}-day forecaster`} action={<StatusBadge status={model.status} />} /></div>
        {model.notes ? <p className="muted-note padded">Promotion gate notes: {model.notes}</p> : null}
        <div className="table-wrap"><table className="holdings-table compact-table">
          <thead><tr><th>Metric</th><th>Full OOS</th><th>Holdout since 2020</th><th>Baseline</th></tr></thead>
          <tbody>
            <tr><td>Mean daily rank IC (t-stat, overlap-adjusted)</td><td>{formatNumber(m.ic_mean, 4)} ({formatNumber(m.ic_tstat_overlap_adjusted)})</td><td>{formatNumber(holdout?.ic_mean, 4)} ({formatNumber(holdout?.ic_tstat_overlap_adjusted)})</td><td>0 (no skill)</td></tr>
            <tr><td>MAE</td><td className={better(m.mae, m.mae_zero_baseline)}>{formatPercent(m.mae, 3)}</td><td className={better(holdout?.mae, holdout?.mae_zero_baseline)}>{formatPercent(holdout?.mae, 3)}</td><td>{formatPercent(m.mae_zero_baseline, 3)} (zero forecast)</td></tr>
            <tr><td>Out-of-sample R² vs zero forecast</td><td className={m.r2_oos_vs_zero > 0 ? "signal-positive" : "signal-negative"}>{formatNumber(m.r2_oos_vs_zero, 4)}</td><td>{formatNumber(holdout?.r2_oos_vs_zero, 4)}</td><td>0</td></tr>
            <tr><td>Directional accuracy</td><td>{formatPercent(m.directional_accuracy, 1)}</td><td>{formatPercent(holdout?.directional_accuracy, 1)}</td><td>{formatPercent(m.base_rate_positive, 1)} (always &ldquo;up&rdquo;)</td></tr>
            <tr><td>Top − bottom tercile forecast spread</td><td>{formatPercent(m.top_minus_bottom_tercile_return, 3)}</td><td>{formatPercent(holdout?.top_minus_bottom_tercile_return, 3)}</td><td>0</td></tr>
            <tr><td>{Math.round(m.interval_level * 100)}% interval coverage (mean width)</td><td>{formatPercent(m.interval_coverage, 1)} ({formatPercent(m.interval_mean_width, 1)})</td><td>{formatPercent(holdout?.interval_coverage, 1)}</td><td>{formatPercent(m.interval_level, 0)} nominal</td></tr>
            <tr><td>Volatility MAE (annualized)</td><td className={better(m.vol_mae, m.vol_mae_trailing20d_baseline)}>{formatPercent(m.vol_mae, 2)}</td><td className={better(holdout?.vol_mae, holdout?.vol_mae_trailing20d_baseline)}>{formatPercent(holdout?.vol_mae, 2)}</td><td>{formatPercent(m.vol_mae_trailing20d_baseline, 2)} (trailing 20d vol)</td></tr>
          </tbody>
        </table></div>
        <p className="muted-note padded">Purged and embargoed expanding-window walk-forward: each prediction comes from a model trained only on labels realized before the prediction date. The configuration was chosen using pre-2020 folds only, so the 2020+ column is an untouched holdout. Weak or negative skill is reported as measured.</p>
      </section>
      <div className="research-lower-grid">
        <section className="panel" aria-label="Calibration"><div className="panel-header"><SectionHeading eyebrow={`Reliability slope ${formatNumber(model.calibration?.reliability_slope, 2)} (1 = perfectly scaled)`} title="Calibration by forecast decile" /></div>{model.calibration?.decile_table?.length ? <CalibrationChart table={model.calibration.decile_table} /> : <EmptyState title="No calibration data" message="Calibration is computed during training." />}</section>
        <section className="panel" aria-label="IC by year"><div className="panel-header"><SectionHeading eyebrow="Stability" title="Rank IC by year" /></div><IcByYearChart data={m.ic_by_year ?? {}} /></section>
      </div>
      <section className="panel" aria-label="Feature importance"><div className="panel-header"><SectionHeading eyebrow="Global explanation (mean |SHAP|, last 12 months)" title="What drives the forecasts" /></div>
        <div className="form-body">{model.feature_importance.map((item) => <div key={item.feature} className="bar-row"><span>{item.label}<small className="block-note">{item.group}</small></span><span className="bar-track"><span className="bar-fill" style={{ width: `${Math.min(item.importance / (model.feature_importance[0]?.importance || 1) * 100, 100)}%` }} /></span><strong>{formatPercent(item.importance, 1)}</strong></div>)}</div>
      </section>
    </>
  );
}

export default function ResearchPage() {
  const [models, setModels] = useState<ResearchModelsResponse | null>(null);
  const [predictions, setPredictions] = useState<{ horizon: number; data?: ResearchPredictionsResponse; error?: string } | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringMetric[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [horizon, setHorizon] = useState(5);
  const [ticker, setTicker] = useState("SPY");

  useEffect(() => {
    api.researchModels().then(setModels).catch((e) => setError(errorMessage(e)));
    api.monitoring().then((r) => setMonitoring(r.metrics)).catch(() => setMonitoring([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.researchPredictions(300, horizon, ticker)
      .then((data) => { if (!cancelled) setPredictions({ horizon, data }); })
      .catch((e) => { if (!cancelled) setPredictions({ horizon, error: errorMessage(e) }); });
    return () => { cancelled = true; };
  }, [horizon, ticker]);

  const model = models?.models.find((m) => m.horizon_days === horizon);
  const challenger = models?.challengers.find((c) => c.horizon_days === horizon);
  const latestMonitoring = (monitoring ?? []).filter((m) => m.horizon_days === horizon).slice(-6);
  const actualColumn = `actual_return_${horizon}d`;
  const predictedColumn = `predicted_return_${horizon}d`;

  return <>
    <div className="content-header"><div><span className="eyebrow">Model intelligence</span><h1>Models</h1><p>Honest out-of-sample evaluation, calibration, explanations and drift monitoring for every forecast horizon.</p></div>
      <Tabs label="Horizon" value={horizon} onChange={setHorizon} options={[{ value: 1, label: "1 day" }, { value: 5, label: "5 days" }, { value: 21, label: "21 days" }]} /></div>
    {error ? <ErrorState message={error} /> : null}
    {!models && !error ? <><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /></> : model ? <ModelDetail model={model} /> : models ? <EmptyState title="No model for this horizon" message="The training pipeline has not registered a model for this horizon yet." /> : null}

    {challenger ? (
      <section className="panel" aria-label="TFT challenger"><div className="panel-header"><SectionHeading eyebrow={`Same holdout rows · ${challenger.metrics.test_start ?? ""} – ${challenger.metrics.test_end ?? ""} · ${challenger.metrics.observations} predictions`} title="Temporal Fusion Transformer challenger" action={<Badge tone={challenger.metrics.tft_beats_incumbent ? "good" : "neutral"}>{challenger.metrics.tft_beats_incumbent ? "outperformed" : "not promoted"}</Badge>} /></div>
        <div className="table-wrap"><table className="holdings-table compact-table"><thead><tr><th>Metric</th><th>TFT</th><th>XGBoost (active)</th></tr></thead><tbody>
          {["ic_mean", "mae", "r2_oos_vs_zero", "directional_accuracy"].map((key) => <tr key={key}><td>{key.replace(/_/g, " ")}</td><td>{formatNumber(challenger.metrics.tft?.[key], 4)}</td><td>{formatNumber(challenger.metrics.xgboost?.[key], 4)}</td></tr>)}
        </tbody></table></div>
        <p className="muted-note padded">{challenger.notes}</p>
      </section>
    ) : null}

    <section className="panel research-panel" aria-label="Out-of-sample predictions"><div className="panel-header"><SectionHeading eyebrow="Walk-forward predictions vs realized" title="Prediction diagnostics" action={<select className="ticker-select" value={ticker} onChange={(e) => setTicker(e.target.value)} aria-label="Ticker">{["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "JPM", "XOM", "JNJ", "TLT", "GLD"].map((t) => <option key={t}>{t}</option>)}</select>} /></div>
      {predictions?.horizon !== horizon ? <Skeleton className="chart-skeleton" /> : predictions.error ? <ErrorState message={predictions.error} /> : predictions.data?.observations.length ? <ResearchChart observations={predictions.data.observations} actualColumn={actualColumn} predictedColumn={predictedColumn} /> : <EmptyState title="No predictions" message="No realized walk-forward predictions are stored for this selection." />}
    </section>

    <section className="panel" aria-label="Monitoring"><div className="panel-header"><SectionHeading eyebrow="Production monitoring" title="Drift & live performance" /></div>
      {monitoring == null ? <Skeleton className="skeleton-line" /> : latestMonitoring.length === 0 ? <EmptyState title="No monitoring data" message="The monitoring job writes feature drift (PSI) and rolling IC/coverage after each daily run." /> : (
        <div className="kv-grid kv-grid-3">{latestMonitoring.map((item) => <KeyValue key={`${item.metric}-${item.as_of_date}`} label={`${item.metric.replace(/_/g, " ")} · ${formatDate(item.as_of_date)}`} value={item.metric.includes("coverage") ? formatPercent(item.value, 1) : formatNumber(item.value, 3)} tone={item.metric === "feature_psi_max" && (item.value ?? 0) > 0.25 ? "signal-negative" : undefined} />)}</div>
      )}
      <p className="muted-note padded">PSI &lt; 0.10 stable, 0.10–0.25 moderate shift, &gt; 0.25 significant drift (alerts are sent).</p>
    </section>
    <Disclaimer compact />
  </>;
}
