"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { EmptyState, ErrorState, MetricCard, SectionHeading, Skeleton } from "./components/dashboard";
import { Disclaimer, StatusBadge } from "./components/ui";
import { api, errorMessage } from "./lib/api";
import { formatDate, formatNumber, formatPercent } from "./lib/formatters";
import type { PortfolioResponse, RiskResponse, SystemStatus } from "./types";

const segmentColors = ["#76c7a0", "#6c9fa0", "#b7a473", "#7786a8", "#ba7d73", "#8fb36a", "#c48fb4", "#5fa8c9", "#d7ad6d", "#9a8f80", "#7fb8a4", "#a3a3c2", "#566164"];

export default function Home() {
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [errors, setErrors] = useState<{ portfolio?: string; risk?: string; system?: string }>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([api.portfolio(), api.risk(), api.systemStatus()]).then(([p, r, s]) => {
      if (p.status === "fulfilled") setPortfolio(p.value);
      if (r.status === "fulfilled") setRisk(r.value);
      if (s.status === "fulfilled") setSystem(s.value);
      setErrors({
        portfolio: p.status === "rejected" ? errorMessage(p.reason) : undefined,
        risk: r.status === "rejected" ? errorMessage(r.reason) : undefined,
        system: s.status === "rejected" ? errorMessage(s.reason) : undefined,
      });
      setLoading(false);
    });
  }, []);

  const allocations = (portfolio?.allocations ?? []).filter((allocation) => (allocation.weight ?? 0) > 0.0005);
  const metrics = risk?.metrics;
  const cvarUsage = risk?.cvar_95 != null && risk?.cvar_limit ? risk.cvar_95 / risk.cvar_limit : null;

  return <>
    <div className="content-header"><div><span className="eyebrow">Investment intelligence</span><h1>Portfolio overview</h1><p>CVaR-constrained model portfolio built from walk-forward-validated forecasts.</p></div><span className="as-of">As of {formatDate(portfolio?.date ?? risk?.date)}</span></div>
    {errors.portfolio && errors.risk ? <ErrorState message={errors.portfolio} /> : null}
    <section aria-label="Portfolio summary"><div className="metric-grid metric-grid-4">{loading ? <><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /></> : <>
      <MetricCard label={`Expected return (${risk?.horizon_days ?? 5}d)`} value={formatPercent(metrics?.expected_return_horizon)} detail={`Linear annualized ${formatPercent(risk?.expected_annual_return)}`} accent="green" />
      <MetricCard label="Annualized volatility" value={formatPercent(risk?.expected_annual_volatility)} detail="Ledoit-Wolf covariance" accent="amber" />
      <MetricCard label="1-day CVaR 95%" value={formatPercent(risk?.cvar_95)} detail={risk?.cvar_limit ? `Limit ${formatPercent(risk.cvar_limit)} · ${cvarUsage != null ? `${Math.round(cvarUsage * 100)}% used` : ""}` : "No limit recorded"} />
      <MetricCard label="Model skill weight" value={risk?.alpha_confidence != null ? formatPercent(risk.alpha_confidence, 0) : "N/A"} detail="Forecast weight from holdout IC t-stat" />
    </>}</div></section>
    {risk && risk.feasible === false ? <ErrorState title="Risk limit not attainable" message={risk.message ?? "No allocation satisfied the constraints; no trades were made."} /> : null}
    <div className="dashboard-grid">
      <section className="panel" aria-label="Portfolio allocations"><div className="panel-header"><SectionHeading eyebrow="Model portfolio" title={portfolio?.strategy ?? "Current allocation"} action={<Link className="text-link" href="/portfolio">Personalize →</Link>} /></div>
        {loading ? <Skeleton className="skeleton-line" /> : errors.portfolio ? <ErrorState message={errors.portfolio} /> : allocations.length === 0 ? <EmptyState title="No allocation data" message="The daily portfolio job has not produced an allocation yet." /> : <>
          <div className="allocation-visual" aria-label="Allocation visualization">{allocations.map((allocation, index) => <span key={allocation.ticker} className="allocation-segment" style={{ width: `${Math.max((allocation.weight ?? 0) * 100, 0)}%`, backgroundColor: segmentColors[index % segmentColors.length] }} title={`${allocation.ticker}: ${formatPercent(allocation.weight)}`} />)}</div>
          <div className="table-wrap"><table className="holdings-table"><thead><tr><th>Asset</th><th>Weight</th><th>Forecast ({portfolio?.horizon_days ?? 5}d)</th><th>Forecast vol</th><th>Risk share</th></tr></thead><tbody>{allocations.map((allocation) => <tr key={allocation.ticker}><td className="ticker">{allocation.ticker}</td><td>{formatPercent(allocation.weight)}</td><td>{formatPercent(allocation.expected_return)}</td><td>{formatPercent(allocation.predicted_volatility)}</td><td>{formatPercent(allocation.risk_contribution)}</td></tr>)}</tbody></table></div>
        </>}
      </section>
      <section className="panel risk-panel" aria-label="Risk metrics"><div className="panel-header"><SectionHeading eyebrow="Portfolio diagnostics" title="Risk profile" /></div>
        {loading ? <><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></> : errors.risk ? <ErrorState message={errors.risk} /> : risk ? <div className="risk-list">
          <div className="risk-row highlight"><span>Solver status</span><strong><StatusBadge status={risk.solver_status ?? (risk.feasible === false ? "infeasible" : "optimal")} /></strong></div>
          <div className="risk-row"><span>Sharpe (model, annualized)</span><strong>{formatNumber(risk.sharpe_ratio)}</strong></div>
          <div className="risk-row"><span>VaR 95% (1 day)</span><strong>{formatPercent(risk.var_95)}</strong></div>
          <div className="risk-row"><span>CVaR 95% (1 day)</span><strong>{formatPercent(risk.cvar_95)}</strong></div>
          <div className="risk-row"><span>Turnover / est. cost</span><strong>{formatPercent(risk.turnover)} / {formatPercent(risk.transaction_cost, 3)}</strong></div>
          <div className="risk-row"><span>Effective # of assets</span><strong>{formatNumber(metrics?.effective_number_of_assets, 1)}</strong></div>
          <div className="risk-row"><span>Binding constraints</span><strong>{risk.binding_constraints?.length ? risk.binding_constraints.join(", ") : "None"}</strong></div>
        </div> : <EmptyState title="No risk data" message="Risk diagnostics are not available yet." />}
      </section>
    </div>
    <section className="panel system-panel" aria-label="Data freshness"><div className="panel-header"><SectionHeading eyebrow="Data & pipelines" title="Freshness" action={system ? <StatusBadge status={system.status} /> : null} /></div>
      {errors.system ? <ErrorState message={errors.system} /> : !system ? <Skeleton className="skeleton-line" /> : <div className="freshness-grid">{system.datasets.map((dataset) => <div key={dataset.dataset} className="freshness-cell"><span className="eyebrow">{dataset.dataset.replace(/_/g, " ")}</span><strong>{dataset.latest ? formatDate(dataset.latest.slice(0, 10)) : "—"}</strong><StatusBadge status={dataset.status} /></div>)}</div>}
    </section>
    <Disclaimer />
  </>;
}
