"use client";

import { useEffect, useState } from "react";
import { EmptyState, ErrorState, MetricCard, SectionHeading, Skeleton } from "./components/dashboard";
import { api } from "./lib/api";
import { formatDate, formatNumber, formatPercent } from "./lib/formatters";
import type { HealthResponse, PortfolioResponse, RiskResponse } from "./types";

const segmentColors = ["#76c7a0", "#6c9fa0", "#b7a473", "#7786a8", "#ba7d73"];

export default function Home() {
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [errors, setErrors] = useState({ portfolio: false, risk: false, health: false });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([api.portfolio(), api.risk(), api.health()]).then(([portfolioResult, riskResult, healthResult]) => {
      if (portfolioResult.status === "fulfilled") setPortfolio(portfolioResult.value);
      if (riskResult.status === "fulfilled") setRisk(riskResult.value);
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      setErrors({ portfolio: portfolioResult.status === "rejected", risk: riskResult.status === "rejected", health: healthResult.status === "rejected" });
      setLoading(false);
    });
  }, []);

  const strategy = portfolio?.strategy ?? risk?.strategy ?? "Portfolio strategy";
  const allocations = portfolio?.allocations ?? [];

  return <>
    <div className="content-header"><div><span className="eyebrow">Investment intelligence</span><h1>Portfolio overview</h1><p>Decision support for the current model allocation.</p></div><span className="as-of">As of {formatDate(portfolio?.date ?? risk?.date)}</span></div>
    {errors.portfolio && errors.risk ? <ErrorState message="Portfolio and risk data could not be loaded. Check that the FastAPI service is running on port 8000." /> : null}
    <section aria-label="Portfolio summary"><div className="metric-grid">{loading ? <><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /></> : <><MetricCard label="Strategy" value={strategy} detail={portfolio ? `${allocations.length} holdings` : "No portfolio data"} /><MetricCard label="Expected annual return" value={formatPercent(risk?.expected_annual_return)} detail="Annualized model expectation" accent="green" /><MetricCard label="Expected annual volatility" value={formatPercent(risk?.expected_annual_volatility)} detail="Annualized portfolio risk" accent="amber" /></>}</div></section>
    <div className="dashboard-grid">
      <section className="panel" aria-label="Portfolio allocations"><div className="panel-header"><SectionHeading eyebrow="Capital deployment" title="Current allocation" action={<span className="as-of">{allocations.length ? `${allocations.length} positions` : "No positions"}</span>} /></div>{loading ? <Skeleton className="skeleton-line" /> : errors.portfolio ? <ErrorState message="The latest portfolio allocation is unavailable." /> : allocations.length === 0 ? <EmptyState title="No allocation data" message="The API returned no current positions." /> : <><div className="allocation-visual" aria-label="Allocation visualization">{allocations.map((allocation, index) => <span key={allocation.ticker} className="allocation-segment" style={{ width: `${Math.max((allocation.weight ?? 0) * 100, 0)}%`, backgroundColor: segmentColors[index % segmentColors.length] }} title={`${allocation.ticker}: ${formatPercent(allocation.weight)}`} />)}</div><div className="table-wrap"><table className="holdings-table"><thead><tr><th>Asset</th><th>Weight</th><th>Expected return</th><th>Risk contribution</th></tr></thead><tbody>{allocations.map((allocation) => <tr key={allocation.ticker}><td className="ticker">{allocation.ticker}</td><td>{formatPercent(allocation.weight)}</td><td>{formatPercent(allocation.expected_return)}</td><td>{formatPercent(allocation.risk_contribution)}</td></tr>)}</tbody></table></div></>}</section>
      <section className="panel risk-panel" aria-label="Risk metrics"><div className="panel-header"><SectionHeading eyebrow="Portfolio diagnostics" title="Risk profile" /></div>{loading ? <><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></> : errors.risk ? <ErrorState message="The latest risk metrics are unavailable." /> : risk ? <div className="risk-list"><div className="risk-row highlight"><span>Expected return</span><strong>{formatPercent(risk.expected_annual_return)}</strong></div><div className="risk-row"><span>Annual volatility</span><strong>{formatPercent(risk.expected_annual_volatility)}</strong></div><div className="risk-row"><span>Sharpe ratio</span><strong>{formatNumber(risk.sharpe_ratio)}</strong></div><div className="risk-row"><span>VaR 95%</span><strong>{formatPercent(risk.var_95)}</strong></div><div className="risk-row"><span>CVaR 95%</span><strong>{formatPercent(risk.cvar_95)}</strong></div></div> : <EmptyState title="No risk data" message="Risk diagnostics are not available yet." />}</section>
    </div>
    <section className="panel system-panel" aria-label="System status"><div className="panel-header"><SectionHeading eyebrow="Infrastructure" title="System status" /></div><div className="system-body"><div className="system-cell"><span className="eyebrow">API service</span><strong className="system-value"><span className="live-dot" />{health?.status ?? (errors.health ? "Unavailable" : "Checking")}</strong></div><div className="system-cell"><span className="eyebrow">Database</span><strong>{health?.database ?? (errors.health ? "Unavailable" : "Checking")}</strong></div></div></section>
  </>;
}