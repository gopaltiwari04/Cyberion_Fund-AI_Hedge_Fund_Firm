"use client";

import { useState } from "react";
import { ErrorState, SectionHeading } from "../components/dashboard";
import { Badge, ContributionChart, KeyValue, SignalGrid, StatusBadge } from "../components/ui";
import { formatCurrency, formatDate, formatNumber, formatPercent, formatSignedPercent, signalClass } from "../lib/formatters";
import type { Recommendation, ScenarioAnalysis } from "../types";

function ScenarioTable({ current, proposed }: { current: ScenarioAnalysis | null; proposed: ScenarioAnalysis | null }) {
  const reference = proposed ?? current;
  if (!reference) return null;
  const percentileKeys = ["p5", "p25", "p50", "p75", "p95"];
  return (
    <div className="scenario-block">
      <div className="table-wrap">
        <table className="holdings-table compact-table">
          <thead><tr><th>Scenario</th><th>Current portfolio</th><th>Proposed portfolio</th></tr></thead>
          <tbody>
            <tr><td>Model expected P/L</td><td>{formatCurrency(current?.model_expected_pnl)}</td><td>{formatCurrency(proposed?.model_expected_pnl)}</td></tr>
            <tr><td>Model 90% band (Gaussian)</td><td>{current ? `${formatCurrency(current.model_band_90[0])} to ${formatCurrency(current.model_band_90[1])}` : "—"}</td><td>{proposed ? `${formatCurrency(proposed.model_band_90[0])} to ${formatCurrency(proposed.model_band_90[1])}` : "—"}</td></tr>
            {percentileKeys.map((key) => <tr key={key}><td>Historical {key.slice(1)}th percentile</td><td className={signalClass(current?.historical_horizon_pnl_percentiles?.[key])}>{formatCurrency(current?.historical_horizon_pnl_percentiles?.[key])}</td><td className={signalClass(proposed?.historical_horizon_pnl_percentiles?.[key])}>{formatCurrency(proposed?.historical_horizon_pnl_percentiles?.[key])}</td></tr>)}
            {reference.stress_tests.map((stress, index) => {
              const cur = current?.stress_tests[index];
              const pro = proposed?.stress_tests[index];
              return <tr key={stress.name}><td>{stress.name}<small className="block-note">{formatDate(stress.start)} – {formatDate(stress.end)}</small></td><td className={signalClass(cur?.pnl)}>{cur?.pnl == null ? (cur?.note ?? "—") : `${formatCurrency(cur.pnl)} (${formatSignedPercent(cur.return, 1)})`}</td><td className={signalClass(pro?.pnl)}>{pro?.pnl == null ? (pro?.note ?? "—") : `${formatCurrency(pro.pnl)} (${formatSignedPercent(pro.return, 1)})`}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
      <p className="muted-note">Historical percentiles use rolling {reference.historical_window.observations ? `${reference.historical_window.observations}` : ""} overlapping windows from {formatDate(reference.historical_window.start)} to {formatDate(reference.historical_window.end)}; stress tests replay actual asset returns over each episode with constant weights. Scenarios are illustrations, not predictions.</p>
    </div>
  );
}

export function RecommendationView({ result }: { result: Recommendation }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const metrics = result.proposal?.metrics;
  const valued = result.portfolio_value != null;

  return (
    <section className="panel recommendation" aria-label="Recommendation">
      <div className="panel-header"><SectionHeading eyebrow={`Generated for ${formatDate(result.as_of_date)} · ${result.profile} profile`} title="Proposed allocation" action={<StatusBadge status={result.status} />} /></div>
      {!result.feasible ? (
        <div className="recommendation-message">
          <ErrorState title="No compliant allocation" message={result.message} />
          {result.diagnostics?.min_achievable_cvar_95_1d != null ? <p className="muted-note">Lowest achievable 1-day CVaR under your other constraints: {formatPercent(result.diagnostics.min_achievable_cvar_95_1d as number)}.{result.diagnostics.min_cvar_without_turnover_cap != null ? ` Without the turnover cap: ${formatPercent(result.diagnostics.min_cvar_without_turnover_cap as number)}.` : ""}</p> : null}
        </div>
      ) : (
        <>
          <div className="kv-grid kv-grid-4">
            <KeyValue label={`Expected ${metrics?.horizon_days}-day return`} value={formatSignedPercent(metrics?.expected_return_horizon)} tone={signalClass(metrics?.expected_return_horizon)} />
            <KeyValue label="Volatility (annualized)" value={formatPercent(metrics?.volatility_annualized)} />
            <KeyValue label="1-day VaR / CVaR 95%" value={`${formatPercent(metrics?.var_95_1d)} / ${formatPercent(metrics?.cvar_95_1d)}`} />
            <KeyValue label="CVaR limit" value={formatPercent(metrics?.cvar_limit_1d)} />
            <KeyValue label="Turnover" value={formatPercent(metrics?.turnover)} />
            <KeyValue label="Est. transaction cost" value={valued ? formatCurrency((metrics?.estimated_transaction_cost ?? 0) * (result.portfolio_value ?? 0), 2) : formatPercent(metrics?.estimated_transaction_cost, 3)} />
            <KeyValue label="Effective # of assets" value={formatNumber(metrics?.effective_number_of_assets, 1)} />
            <KeyValue label="Worst historical day" value={formatPercent(metrics?.worst_scenario_1d)} />
          </div>
          {result.binding_constraints.length ? <p className="muted-note">Binding constraints: {result.binding_constraints.map((c) => <Badge key={c} tone="info">{c.replace(/_/g, " ")}</Badge>)}</p> : null}
        </>
      )}

      <div className="table-wrap">
        <table className="holdings-table">
          <thead><tr><th>Asset</th><th>Current</th><th>Target</th><th>Action</th><th>Trade</th><th>Forecast</th><th>90% interval</th><th>Risk share</th><th /></tr></thead>
          <tbody>
            {result.assets.map((asset) => (
              <FragmentRow key={asset.ticker} asset={asset} valued={valued} expanded={expanded === asset.ticker} onToggle={() => setExpanded(expanded === asset.ticker ? null : asset.ticker)} />
            ))}
            <tr><td className="ticker">CASH</td><td>{formatPercent(result.current_weights.CASH)}</td><td>{formatPercent(result.proposal?.weights.CASH)}</td><td colSpan={6} className="left-cell">Earns {result.risk_free.source} ({formatPercent(result.risk_free.daily * 252)} annualized)</td></tr>
          </tbody>
        </table>
      </div>

      {metrics ? (
        <div className="split-grid">
          <div><h3 className="subheading">Sector exposure</h3>{Object.entries(metrics.sector_exposure).filter(([, w]) => w > 0.0005).sort((a, b) => b[1] - a[1]).map(([sector, weight]) => <div key={sector} className="bar-row"><span>{sector}</span><span className="bar-track"><span className="bar-fill" style={{ width: `${Math.min(weight * 100, 100)}%` }} /></span><strong>{formatPercent(weight, 1)}</strong></div>)}</div>
          <div><h3 className="subheading">Risk contribution</h3>{Object.entries(metrics.risk_contribution).filter(([, r]) => Math.abs(r) > 0.0005).sort((a, b) => b[1] - a[1]).map(([ticker, share]) => <div key={ticker} className="bar-row"><span>{ticker}</span><span className="bar-track"><span className="bar-fill bar-amber" style={{ width: `${Math.min(Math.max(share, 0) * 100, 100)}%` }} /></span><strong>{formatPercent(share, 1)}</strong></div>)}</div>
        </div>
      ) : null}

      <h3 className="subheading padded">Expected P/L scenarios ({result.model.horizon_days}-day horizon{valued ? `, portfolio value ${formatCurrency(result.portfolio_value)}` : ""})</h3>
      {valued ? <ScenarioTable current={result.scenarios.current} proposed={result.scenarios.proposed} /> : <p className="muted-note padded">Add holdings or cash to see P/L in currency.</p>}

      <details className="assumptions padded">
        <summary>Model, assumptions and constraints</summary>
        <ul>
          <li>Model {result.model.version} (data to {formatDate(result.model.data_end)}); forecast weight {formatPercent(result.model.alpha_confidence, 0)} based on holdout skill{result.model.holdout_ic != null ? ` (holdout IC ${result.model.holdout_ic.toFixed(3)})` : ""}.</li>
          {Object.entries(result.assumptions).map(([key, value]) => <li key={key}><strong>{key.replace(/_/g, " ")}:</strong> {String(value)}</li>)}
          {Object.entries(result.constraints).map(([key, value]) => <li key={`c-${key}`}><strong>{key.replace(/_/g, " ")}:</strong> {value == null ? "none" : String(value)}</li>)}
        </ul>
      </details>
      <p className="disclaimer-inline">{result.disclaimer}</p>
    </section>
  );
}

function FragmentRow({ asset, valued, expanded, onToggle }: { asset: Recommendation["assets"][number]; valued: boolean; expanded: boolean; onToggle: () => void }) {
  const actionTone = asset.action === "buy" ? "good" : asset.action === "sell" ? "bad" : "neutral";
  return (
    <>
      <tr>
        <td className="ticker">{asset.ticker}<small className="block-note">{asset.name}</small></td>
        <td>{formatPercent(asset.current_weight)}</td>
        <td>{formatPercent(asset.target_weight)}</td>
        <td>{asset.action ? <Badge tone={actionTone}>{asset.action}</Badge> : "—"}</td>
        <td className={signalClass(asset.trade_value)}>{asset.trade_value == null ? "—" : valued ? `${formatCurrency(asset.trade_value)}${asset.trade_shares != null ? ` (${asset.trade_shares > 0 ? "+" : ""}${asset.trade_shares.toFixed(2)} sh)` : ""}` : formatSignedPercent(asset.trade_value)}</td>
        <td className={signalClass(asset.forecast.expected_return)}>{formatSignedPercent(asset.forecast.expected_return)}</td>
        <td>{formatSignedPercent(asset.forecast.interval[0], 1)} / {formatSignedPercent(asset.forecast.interval[1], 1)}</td>
        <td>{formatPercent(asset.risk_contribution)}</td>
        <td><button className="link-button" onClick={onToggle} aria-expanded={expanded}>{expanded ? "Hide" : "Why?"}</button></td>
      </tr>
      {expanded ? (
        <tr className="explain-row"><td colSpan={9}>
          <div className="explain-grid">
            <div><h4>Forecast drivers (SHAP)</h4><ContributionChart explanation={asset.explanation} /></div>
            <div><h4>Signals</h4><SignalGrid signals={asset.explanation?.signals} /><p className="muted-note">Forecast volatility {formatPercent(asset.forecast.volatility_annualized)} (annualized). Price {formatCurrency(asset.price, 2)} on {formatDate(asset.price_date)}.</p></div>
          </div>
        </td></tr>
      ) : null}
    </>
  );
}
