"use client";

import { useEffect, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { api } from "../lib/api";
import { formatDate, formatNumber, formatPercent } from "../lib/formatters";
import type { PortfolioResponse, RiskResponse } from "../types";

export default function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    Promise.allSettled([api.portfolio(), api.risk()]).then(([portfolioResult, riskResult]) => {
      if (portfolioResult.status === "fulfilled") setPortfolio(portfolioResult.value);
      if (riskResult.status === "fulfilled") setRisk(riskResult.value);
      setFailed(portfolioResult.status === "rejected" && riskResult.status === "rejected");
      setLoading(false);
    });
  }, []);

  const allocations = portfolio?.allocations ?? [];
  return <>
    <div className="content-header"><div><span className="eyebrow">Capital deployment</span><h1>Portfolio</h1><p>Current model allocation and portfolio diagnostics.</p></div><span className="as-of">As of {formatDate(portfolio?.date ?? risk?.date)}</span></div>
    {failed ? <ErrorState message="Portfolio data is unavailable. Check that the FastAPI service is running on port 8000." /> : null}
    <section className="panel portfolio-summary"><div className="panel-header"><SectionHeading eyebrow="Mandate" title={portfolio?.strategy ?? risk?.strategy ?? "Portfolio strategy"} /></div><div className="portfolio-meta"><div><span className="eyebrow">Latest portfolio date</span><strong>{formatDate(portfolio?.date)}</strong></div><div><span className="eyebrow">Positions</span><strong>{portfolio ? allocations.length : "N/A"}</strong></div></div></section>
    <div className="portfolio-page-grid">
      <section className="panel"><div className="panel-header"><SectionHeading eyebrow="Holdings" title="Allocations" /></div>{loading ? <Skeleton className="skeleton-line" /> : allocations.length === 0 ? <EmptyState title="No allocation data" message="The API returned no current positions." /> : <div className="table-wrap"><table className="holdings-table"><thead><tr><th>Asset</th><th>Weight</th><th>Expected return</th><th>Risk contribution</th></tr></thead><tbody>{allocations.map((allocation) => <tr key={allocation.ticker}><td className="ticker">{allocation.ticker}</td><td>{formatPercent(allocation.weight)}</td><td>{formatPercent(allocation.expected_return)}</td><td>{formatPercent(allocation.risk_contribution)}</td></tr>)}</tbody></table></div>}</section>
      <section className="panel risk-panel"><div className="panel-header"><SectionHeading eyebrow="Diagnostics" title="Risk metrics" /></div>{loading ? <><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></> : risk ? <div className="risk-list"><div className="risk-row highlight"><span>Expected annual return</span><strong>{formatPercent(risk.expected_annual_return)}</strong></div><div className="risk-row"><span>Expected annual volatility</span><strong>{formatPercent(risk.expected_annual_volatility)}</strong></div><div className="risk-row"><span>Sharpe ratio</span><strong>{formatNumber(risk.sharpe_ratio)}</strong></div><div className="risk-row"><span>VaR 95%</span><strong>{formatPercent(risk.var_95)}</strong></div><div className="risk-row"><span>CVaR 95%</span><strong>{formatPercent(risk.cvar_95)}</strong></div></div> : <EmptyState title="No risk data" message="Risk diagnostics are not available yet." />}</section>
    </div>
  </>;
}