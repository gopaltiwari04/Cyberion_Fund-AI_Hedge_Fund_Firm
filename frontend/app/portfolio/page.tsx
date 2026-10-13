"use client";

import { useEffect, useMemo, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { Disclaimer, KeyValue, Tabs } from "../components/ui";
import { api, errorMessage } from "../lib/api";
import { formatCurrency, formatPercent, signalClass } from "../lib/formatters";
import type { HoldingsResponse, MarketSnapshot, Recommendation, UserProfile } from "../types";
import { RecommendationView } from "./recommendation-view";

interface EditableRow { ticker: string; quantity: string; average_cost: string }

const PROFILE_DEFAULTS: Record<UserProfile["risk_profile"], { cvar_limit: number; max_weight: number }> = {
  conservative: { cvar_limit: 0.015, max_weight: 0.2 },
  balanced: { cvar_limit: 0.025, max_weight: 0.3 },
  aggressive: { cvar_limit: 0.04, max_weight: 0.4 },
};

export default function PortfolioPage() {
  const [markets, setMarkets] = useState<MarketSnapshot[]>([]);
  const [holdings, setHoldings] = useState<HoldingsResponse | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [rows, setRows] = useState<EditableRow[]>([]);
  const [cash, setCash] = useState("0");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<"holdings" | "profile" | null>(null);
  const [notice, setNotice] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [horizon, setHorizon] = useState<1 | 5 | 21>(5);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [recommending, setRecommending] = useState(false);
  const [recommendError, setRecommendError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.markets(), api.holdings(), api.profile()])
      .then(([marketResponse, holdingsResponse, profileResponse]) => {
        setMarkets(marketResponse.markets);
        setHoldings(holdingsResponse);
        setProfile(profileResponse);
        setRows(holdingsResponse.holdings.map((row) => ({ ticker: row.ticker, quantity: String(row.quantity), average_cost: row.average_cost == null ? "" : String(row.average_cost) })));
        setCash(String(holdingsResponse.cash_balance ?? 0));
      })
      .catch((error) => setLoadError(errorMessage(error)))
      .finally(() => setLoading(false));
  }, []);

  const prices = useMemo(() => Object.fromEntries(markets.map((m) => [m.ticker, m.close ?? 0])), [markets]);
  const available = markets.map((m) => m.ticker).filter((t) => !rows.some((row) => row.ticker === t));
  const draftValue = rows.reduce((sum, row) => sum + (Number(row.quantity) || 0) * (prices[row.ticker] ?? 0), 0) + (Number(cash) || 0);

  function validateRows(): string | null {
    for (const row of rows) {
      const quantity = Number(row.quantity);
      if (!Number.isFinite(quantity) || quantity < 0) return `Quantity for ${row.ticker} must be a non-negative number.`;
      if (row.average_cost !== "" && (!Number.isFinite(Number(row.average_cost)) || Number(row.average_cost) < 0)) return `Average cost for ${row.ticker} must be a non-negative number.`;
    }
    if (!Number.isFinite(Number(cash)) || Number(cash) < 0) return "Cash must be a non-negative number.";
    return null;
  }

  async function saveHoldings() {
    const invalid = validateRows();
    if (invalid) { setNotice({ tone: "error", text: invalid }); return; }
    setSaving("holdings");
    try {
      const response = await api.saveHoldings(rows.map((row) => ({ ticker: row.ticker, quantity: Number(row.quantity), average_cost: row.average_cost === "" ? null : Number(row.average_cost) })), Number(cash));
      setHoldings(response);
      setNotice({ tone: "ok", text: "Holdings saved." });
    } catch (error) {
      setNotice({ tone: "error", text: errorMessage(error) });
    } finally {
      setSaving(null);
    }
  }

  async function saveProfile() {
    if (!profile) return;
    setSaving("profile");
    try {
      const { risk_profile, cvar_limit, max_weight, max_turnover, transaction_cost_bps } = profile;
      const response = await api.updateProfile({ risk_profile, cvar_limit, max_weight, max_turnover, transaction_cost_bps, cash_balance: Number(cash) || 0 });
      setProfile(response);
      setNotice({ tone: "ok", text: "Risk settings saved." });
    } catch (error) {
      setNotice({ tone: "error", text: errorMessage(error) });
    } finally {
      setSaving(null);
    }
  }

  async function generate() {
    setRecommending(true);
    setRecommendError(null);
    try {
      setRecommendation(await api.recommend({ horizon_days: horizon }));
    } catch (error) {
      setRecommendError(errorMessage(error));
    } finally {
      setRecommending(false);
    }
  }

  const updateProfile = (patch: Partial<UserProfile>) => setProfile((current) => (current ? { ...current, ...patch } : current));

  return <>
    <div className="content-header"><div><span className="eyebrow">Capital deployment</span><h1>My portfolio</h1><p>Record your holdings, set risk limits, and generate CVaR-constrained rebalancing proposals with explanations.</p></div></div>
    {loadError ? <ErrorState message={loadError} /> : null}
    {notice ? <div className={`notice notice-${notice.tone}`} role="status">{notice.text}<button className="link-button" onClick={() => setNotice(null)} aria-label="Dismiss">×</button></div> : null}

    <div className="portfolio-page-grid">
      <section className="panel" aria-label="Holdings editor">
        <div className="panel-header"><SectionHeading eyebrow="Holdings" title="Positions & cash" action={<span className="as-of">Value {formatCurrency(holdings?.total_value)}</span>} /></div>
        {loading ? <><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></> : (
          <div className="form-body">
            {rows.length === 0 ? <EmptyState title="No positions yet" message="Add the assets you hold (supported universe only) and your cash balance." /> : (
              <div className="table-wrap"><table className="holdings-table edit-table">
                <thead><tr><th>Asset</th><th>Shares</th><th>Avg cost</th><th>Price</th><th>Value</th><th>Unrealized P/L</th><th /></tr></thead>
                <tbody>{rows.map((row, index) => {
                  const saved = holdings?.holdings.find((h) => h.ticker === row.ticker);
                  return <tr key={row.ticker}>
                    <td className="ticker">{row.ticker}</td>
                    <td><input aria-label={`${row.ticker} shares`} className="input input-num" inputMode="decimal" value={row.quantity} onChange={(e) => setRows(rows.map((r, i) => (i === index ? { ...r, quantity: e.target.value } : r)))} /></td>
                    <td><input aria-label={`${row.ticker} average cost`} className="input input-num" inputMode="decimal" placeholder="optional" value={row.average_cost} onChange={(e) => setRows(rows.map((r, i) => (i === index ? { ...r, average_cost: e.target.value } : r)))} /></td>
                    <td>{formatCurrency(prices[row.ticker], 2)}</td>
                    <td>{formatCurrency((Number(row.quantity) || 0) * (prices[row.ticker] ?? 0))}</td>
                    <td className={signalClass(saved?.unrealized_pnl)}>{formatCurrency(saved?.unrealized_pnl)}</td>
                    <td><button className="link-button" onClick={() => setRows(rows.filter((_, i) => i !== index))} aria-label={`Remove ${row.ticker}`}>Remove</button></td>
                  </tr>;
                })}</tbody>
              </table></div>
            )}
            <div className="form-row">
              <label className="field"><span>Add asset</span><select className="input" value="" onChange={(e) => e.target.value && setRows([...rows, { ticker: e.target.value, quantity: "0", average_cost: "" }])}><option value="">Select…</option>{available.map((t) => <option key={t} value={t}>{t}</option>)}</select></label>
              <label className="field"><span>Cash (USD)</span><input className="input input-num" inputMode="decimal" value={cash} onChange={(e) => setCash(e.target.value)} /></label>
              <div className="field"><span>Draft value</span><strong>{formatCurrency(draftValue)}</strong></div>
              <button className="button" onClick={saveHoldings} disabled={saving !== null}>{saving === "holdings" ? "Saving…" : "Save holdings"}</button>
            </div>
          </div>
        )}
      </section>

      <section className="panel" aria-label="Risk settings">
        <div className="panel-header"><SectionHeading eyebrow="Risk preferences" title="Constraints" /></div>
        {!profile ? <Skeleton className="skeleton-line" /> : (
          <div className="form-body form-stack">
            <label className="field"><span>Risk profile</span><select className="input" value={profile.risk_profile} onChange={(e) => { const value = e.target.value as UserProfile["risk_profile"]; updateProfile({ risk_profile: value, ...PROFILE_DEFAULTS[value] }); }}><option value="conservative">Conservative</option><option value="balanced">Balanced</option><option value="aggressive">Aggressive</option></select></label>
            <label className="field"><span>Max 1-day CVaR 95% ({formatPercent(profile.cvar_limit, 1)})</span><input type="range" min={0.005} max={0.1} step={0.0025} value={profile.cvar_limit} onChange={(e) => updateProfile({ cvar_limit: Number(e.target.value) })} /></label>
            <label className="field"><span>Max position weight ({formatPercent(profile.max_weight, 0)})</span><input type="range" min={0.05} max={1} step={0.05} value={profile.max_weight} onChange={(e) => updateProfile({ max_weight: Number(e.target.value) })} /></label>
            <label className="field"><span>Max turnover per rebalance ({formatPercent(profile.max_turnover, 0)})</span><input type="range" min={0} max={2} step={0.05} value={profile.max_turnover} onChange={(e) => updateProfile({ max_turnover: Number(e.target.value) })} /></label>
            <label className="field"><span>Transaction cost (bps)</span><input className="input input-num" type="number" min={0} max={100} step={0.5} value={profile.transaction_cost_bps} onChange={(e) => updateProfile({ transaction_cost_bps: Number(e.target.value) })} /></label>
            <button className="button button-secondary" onClick={saveProfile} disabled={saving !== null}>{saving === "profile" ? "Saving…" : "Save risk settings"}</button>
            <p className="muted-note">CVaR is the average loss on the worst 5% of historical days for the proposed weights. It is enforced as a hard limit; if it cannot be met, no trades are proposed.</p>
          </div>
        )}
      </section>
    </div>

    <section className="panel action-panel" aria-label="Generate recommendation">
      <div className="panel-header"><SectionHeading eyebrow="Optimizer" title="Rebalancing proposal" action={<Tabs label="Forecast horizon" value={horizon} onChange={setHorizon} options={[{ value: 1, label: "1D" }, { value: 5, label: "5D" }, { value: 21, label: "21D" }]} />} /></div>
      <div className="form-body form-row">
        <KeyValue label="Uses saved holdings & settings" value={holdings ? `${holdings.holdings.length} positions · ${formatCurrency(holdings.total_value)}` : "—"} />
        <button className="button" onClick={generate} disabled={recommending || loading}>{recommending ? "Optimizing…" : "Generate proposal"}</button>
      </div>
      {recommendError ? <ErrorState message={recommendError} /> : null}
    </section>
    {recommendation ? (recommendation.status === "unavailable" ? <ErrorState message={recommendation.message} /> : <RecommendationView result={recommendation} />) : null}
    <Disclaimer />
  </>;
}
