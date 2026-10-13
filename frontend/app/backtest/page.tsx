"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { Disclaimer, StatusBadge } from "../components/ui";
import { api, errorMessage } from "../lib/api";
import { formatCurrency, formatDate, formatDateTime, formatNumber, formatPercent, signalClass } from "../lib/formatters";
import type { BacktestMetrics, BacktestRun } from "../types";
import { DrawdownChart, EquityChart } from "./charts";

const STRATEGY_LABELS: Record<string, string> = {
  ml_cvar: "ML forecasts + CVaR optimizer",
  min_cvar: "Risk-only (min CVaR, no forecasts)",
  equal_weight: "Equal weight (rebalanced)",
  spy_buy_and_hold: "SPY buy & hold",
};

const METRIC_ROWS: Array<[keyof BacktestMetrics, string, (v: number | null | undefined) => string]> = [
  ["total_return", "Total return", (v) => formatPercent(v)],
  ["cagr", "CAGR", (v) => formatPercent(v)],
  ["volatility", "Volatility (ann.)", (v) => formatPercent(v)],
  ["sharpe", "Sharpe (excess of cash)", (v) => formatNumber(v)],
  ["sortino", "Sortino", (v) => formatNumber(v)],
  ["max_drawdown", "Max drawdown", (v) => formatPercent(v)],
  ["max_drawdown_duration_days", "Longest drawdown (days)", (v) => (v == null ? "N/A" : String(v))],
  ["calmar", "Calmar", (v) => formatNumber(v)],
  ["var_95_1d", "1-day VaR 95%", (v) => formatPercent(v)],
  ["cvar_95_1d", "1-day CVaR 95%", (v) => formatPercent(v)],
  ["turnover_annualized", "Turnover (ann.)", (v) => formatPercent(v, 0)],
  ["costs_pct_of_initial", "Costs (% of initial)", (v) => formatPercent(v, 3)],
  ["trades", "Trades", (v) => (v == null ? "N/A" : String(v))],
  ["positive_months", "Positive months", (v) => formatPercent(v, 0)],
];

const today = new Date().toISOString().slice(0, 10);

export default function BacktestPage() {
  const [form, setForm] = useState({ start: "2020-01-01", end: today, strategy: "ml_cvar", horizon_days: 5, rebalance_every: 5, risk_profile: "balanced", cvar_limit: "", max_weight: "", max_turnover: "", commission_bps: 1, slippage_bps: 5, initial_capital: 1000000 });
  const [runs, setRuns] = useState<BacktestRun[]>([]);
  const [active, setActive] = useState<BacktestRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [tradePage, setTradePage] = useState(0);

  const refreshList = useCallback(() => api.backtests().then((r) => { setRuns(r.runs); setListError(null); }).catch((e) => setListError(errorMessage(e))), []);
  useEffect(() => { refreshList(); }, [refreshList]);

  const activeId = active?.id;
  const activeStatus = active?.status;
  useEffect(() => {
    if (!activeId || (activeStatus !== "queued" && activeStatus !== "running")) return;
    const timer = setInterval(() => {
      api.backtest(activeId).then((run) => {
        setActive(run);
        if (run.status === "completed" || run.status === "failed") refreshList();
      }).catch((e) => setError(errorMessage(e)));
    }, 3000);
    return () => clearInterval(timer);
  }, [activeId, activeStatus, refreshList]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const body = {
        ...form,
        cvar_limit: form.cvar_limit === "" ? null : Number(form.cvar_limit) / 100,
        max_weight: form.max_weight === "" ? null : Number(form.max_weight) / 100,
        max_turnover: form.max_turnover === "" ? null : Number(form.max_turnover) / 100,
      };
      const created = await api.submitBacktest(body);
      setActive({ id: created.id, status: "queued", config: body, config_hash: "", error: null, created_at: new Date().toISOString(), finished_at: null });
      setTradePage(0);
      refreshList();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setSubmitting(false);
    }
  }

  async function open(id: string) {
    setError(null);
    setTradePage(0);
    try { setActive(await api.backtest(id)); } catch (e) { setError(errorMessage(e)); }
  }

  const set = (key: string, value: string | number) => setForm((current) => ({ ...current, [key]: value }));
  const result = active?.status === "completed" ? active.result : null;
  const strategies = result ? Object.keys(result.metrics) : [];
  const trades = result?.trades ?? [];
  const pageSize = 25;

  return <>
    <div className="content-header"><div><span className="eyebrow">Strategy validation</span><h1>Backtesting</h1><p>Event-driven simulation with next-open execution, costs, dividends and only out-of-sample forecasts.</p></div></div>
    <div className="backtest-grid">
      <section className="panel" aria-label="Backtest configuration">
        <div className="panel-header"><SectionHeading eyebrow="Configuration" title="New backtest" /></div>
        <form className="form-body form-grid" onSubmit={submit}>
          <label className="field"><span>Start</span><input className="input" type="date" value={form.start} min="2018-01-01" max={today} onChange={(e) => set("start", e.target.value)} required /></label>
          <label className="field"><span>End</span><input className="input" type="date" value={form.end} max={today} onChange={(e) => set("end", e.target.value)} required /></label>
          <label className="field field-wide"><span>Strategy</span><select className="input" value={form.strategy} onChange={(e) => set("strategy", e.target.value)}>{Object.entries(STRATEGY_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className="field"><span>Forecast horizon</span><select className="input" value={form.horizon_days} onChange={(e) => set("horizon_days", Number(e.target.value))}><option value={1}>1 day</option><option value={5}>5 days</option><option value={21}>21 days</option></select></label>
          <label className="field"><span>Rebalance every (days)</span><input className="input" type="number" min={1} max={63} value={form.rebalance_every} onChange={(e) => set("rebalance_every", Number(e.target.value))} /></label>
          <label className="field"><span>Risk profile</span><select className="input" value={form.risk_profile} onChange={(e) => set("risk_profile", e.target.value)}><option value="conservative">Conservative</option><option value="balanced">Balanced</option><option value="aggressive">Aggressive</option></select></label>
          <label className="field"><span>CVaR limit % (optional)</span><input className="input" type="number" step={0.1} min={0.5} max={10} placeholder="profile default" value={form.cvar_limit} onChange={(e) => set("cvar_limit", e.target.value)} /></label>
          <label className="field"><span>Max weight % (optional)</span><input className="input" type="number" step={1} min={5} max={100} placeholder="profile default" value={form.max_weight} onChange={(e) => set("max_weight", e.target.value)} /></label>
          <label className="field"><span>Max turnover % (optional)</span><input className="input" type="number" step={5} min={0} max={200} placeholder="none" value={form.max_turnover} onChange={(e) => set("max_turnover", e.target.value)} /></label>
          <label className="field"><span>Commission (bps)</span><input className="input" type="number" step={0.5} min={0} max={100} value={form.commission_bps} onChange={(e) => set("commission_bps", Number(e.target.value))} /></label>
          <label className="field"><span>Slippage (bps)</span><input className="input" type="number" step={0.5} min={0} max={200} value={form.slippage_bps} onChange={(e) => set("slippage_bps", Number(e.target.value))} /></label>
          <label className="field"><span>Initial capital</span><input className="input" type="number" step={1000} min={1000} max={1000000000} value={form.initial_capital} onChange={(e) => set("initial_capital", Number(e.target.value))} /></label>
          <button className="button field-wide" type="submit" disabled={submitting || active?.status === "queued" || active?.status === "running"}>{submitting ? "Submitting…" : "Run backtest"}</button>
          {error ? <div className="field-wide"><ErrorState message={error} /></div> : null}
        </form>
      </section>
      <section className="panel" aria-label="Backtest history">
        <div className="panel-header"><SectionHeading eyebrow="History" title="Your runs" /></div>
        {listError ? <ErrorState message={listError} /> : runs.length === 0 ? <EmptyState title="No backtests yet" message="Configure and run a backtest to see results here." /> : (
          <ul className="run-list">{runs.map((run) => <li key={run.id}><button className={`run-item ${active?.id === run.id ? "run-active" : ""}`} onClick={() => open(run.id)}><span>{STRATEGY_LABELS[String(run.config.strategy)] ?? String(run.config.strategy)}<small>{String(run.config.start)} → {String(run.config.end)} · {formatDateTime(run.created_at)}</small></span><span className="run-right"><StatusBadge status={run.status} />{run.summary?.cagr != null ? <small className={signalClass(run.summary.cagr)}>CAGR {formatPercent(run.summary.cagr)}</small> : null}</span></button></li>)}</ul>
        )}
      </section>
    </div>

    {active && (active.status === "queued" || active.status === "running") ? <section className="panel"><div className="panel-header"><SectionHeading eyebrow="Running" title="Simulating…" action={<StatusBadge status={active.status} />} /></div><Skeleton className="chart-skeleton" /></section> : null}
    {active?.status === "failed" ? <ErrorState title="Backtest failed" message={active.error ?? "Unknown error"} /> : null}

    {result ? <>
      <section className="panel" aria-label="Performance">
        <div className="panel-header"><SectionHeading eyebrow={`${formatDate(String(result.metrics[strategies[0]]?.start))} – ${formatDate(String(result.metrics[strategies[0]]?.end))} · config ${result.config_hash}`} title="Equity curve vs benchmarks" /></div>
        <EquityChart curves={result.equity_curves} labels={STRATEGY_LABELS} />
        <h3 className="subheading padded">Drawdown ({STRATEGY_LABELS[strategies[0]]})</h3>
        <DrawdownChart data={result.drawdown} />
      </section>
      <section className="panel" aria-label="Metrics">
        <div className="panel-header"><SectionHeading eyebrow="Statistics" title="Strategy vs benchmarks" /></div>
        <div className="table-wrap"><table className="holdings-table"><thead><tr><th>Metric</th>{strategies.map((s) => <th key={s}>{STRATEGY_LABELS[s] ?? s}</th>)}</tr></thead><tbody>
          {METRIC_ROWS.map(([key, label, fmt]) => <tr key={key}><td>{label}</td>{strategies.map((s) => <td key={s}>{fmt(result.metrics[s]?.[key] as number | null)}</td>)}</tr>)}
          <tr><td>Beta of strategy vs benchmark</td>{strategies.map((s, i) => <td key={s}>{i === 0 ? "—" : formatNumber(result.metrics[s]?.vs_strategy?.beta)}</td>)}</tr>
          <tr><td>Strategy alpha vs benchmark (ann.)</td>{strategies.map((s, i) => <td key={s}>{i === 0 ? "—" : formatPercent(result.metrics[s]?.vs_strategy?.alpha_annualized)}</td>)}</tr>
          <tr><td>Information ratio</td>{strategies.map((s, i) => <td key={s}>{i === 0 ? "—" : formatNumber(result.metrics[s]?.vs_strategy?.information_ratio)}</td>)}</tr>
        </tbody></table></div>
        <p className="muted-note padded">{result.model ? `Forecasts: model ${result.model.version}, forecast weight ${formatPercent(result.model.alpha_confidence, 0)}. ` : ""}Market data through {formatDate(result.data_snapshot.market_data_end)}. Rerunning the same configuration on the same data and model version reproduces these numbers.</p>
      </section>
      <div className="backtest-grid">
        <section className="panel" aria-label="Trades">
          <div className="panel-header"><SectionHeading eyebrow="Execution" title={`Trade history (${result.trades_total})`} action={<span className="pager"><button className="link-button" disabled={tradePage === 0} onClick={() => setTradePage(tradePage - 1)}>‹ Prev</button><span>{tradePage + 1} / {Math.max(1, Math.ceil(trades.length / pageSize))}</span><button className="link-button" disabled={(tradePage + 1) * pageSize >= trades.length} onClick={() => setTradePage(tradePage + 1)}>Next ›</button></span>} /></div>
          <div className="table-wrap"><table className="holdings-table compact-table"><thead><tr><th>Date</th><th>Ticker</th><th>Side</th><th>Shares</th><th>Fill price</th><th>Notional</th><th>Commission</th></tr></thead><tbody>
            {[...trades].reverse().slice(tradePage * pageSize, (tradePage + 1) * pageSize).map((trade, index) => <tr key={`${trade.date}-${trade.ticker}-${index}`}><td>{trade.date}</td><td className="ticker">{trade.ticker}</td><td className={trade.side === "buy" ? "signal-positive" : "signal-negative"}>{trade.side}</td><td>{trade.shares.toFixed(2)}</td><td>{formatCurrency(trade.price, 2)}</td><td>{formatCurrency(trade.notional)}</td><td>{formatCurrency(trade.commission, 2)}</td></tr>)}
          </tbody></table></div>
        </section>
        <section className="panel" aria-label="Assumptions">
          <div className="panel-header"><SectionHeading eyebrow="Transparency" title="Assumptions & known biases" /></div>
          <ul className="plain-list">{Object.entries(result.assumptions).map(([k, v]) => <li key={k}><strong>{k.replace(/_/g, " ")}:</strong> {String(v)}</li>)}</ul>
          <ul className="plain-list warning-list">{result.known_biases.map((bias) => <li key={bias}>{bias}</li>)}</ul>
          <h3 className="subheading padded">Recent rebalances</h3>
          <ul className="plain-list">{result.rebalances.slice(-8).reverse().map((r) => <li key={r.date}><strong>{r.date}</strong> {r.weights ? Object.entries(r.weights).map(([t, w]) => `${t} ${(w * 100).toFixed(0)}%`).join(", ") : <em>skipped: {r.message ?? r.skipped}</em>}</li>)}</ul>
        </section>
      </div>
    </> : null}
    <Disclaimer />
  </>;
}
