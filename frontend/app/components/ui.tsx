"use client";

import type { ReactNode } from "react";
import { formatNumber, formatPercent, formatSignedPercent } from "../lib/formatters";
import type { Explanation } from "../types";

export const DISCLAIMER =
  "Educational decision-support analytics. Forecasts are uncertain statistical estimates, not guaranteed returns, and nothing here is personalised investment advice or an offer to buy or sell securities. Backtests and past performance do not guarantee future results.";

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <aside className={`disclaimer ${compact ? "disclaimer-compact" : ""}`} role="note">
      <strong>Important</strong>
      <p>{DISCLAIMER}</p>
    </aside>
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "good" | "warn" | "bad" | "neutral" | "info"; children: ReactNode }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function StatusBadge({ status }: { status: string | null | undefined }) {
  const value = (status ?? "unknown").toLowerCase();
  const tone = ["fresh", "healthy", "success", "completed", "optimal", "active", "answered", "ready"].includes(value)
    ? "good"
    : ["stale", "degraded", "success_with_warnings", "running", "queued", "candidate", "extractive", "abandoned"].includes(value)
      ? "warn"
      : ["missing", "failed", "unhealthy", "infeasible", "error", "insufficient_evidence"].includes(value)
        ? "bad"
        : "neutral";
  return <Badge tone={tone}>{value.replace(/_/g, " ")}</Badge>;
}

export function Tabs<T extends string | number>({ value, options, onChange, label }: { value: T; options: Array<{ value: T; label: string }>; onChange: (value: T) => void; label: string }) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {options.map((option) => (
        <button key={String(option.value)} role="tab" aria-selected={option.value === value} className={`tab ${option.value === value ? "tab-active" : ""}`} onClick={() => onChange(option.value)}>
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function KeyValue({ label, value, tone }: { label: string; value: ReactNode; tone?: string }) {
  return (
    <div className="kv">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}

export function ContributionChart({ explanation, limit = 8 }: { explanation: Explanation | null | undefined; limit?: number }) {
  if (!explanation || !explanation.top_contributions?.length) {
    return <p className="muted-note">No explanation is available for this forecast.</p>;
  }
  const items = explanation.top_contributions.slice(0, limit);
  const max = Math.max(...items.map((item) => Math.abs(item.contribution ?? 0)), 1e-9);
  return (
    <div className="shap" aria-label="Feature contributions to the forecast">
      <div className="shap-header">
        <span>Base value {formatSignedPercent(explanation.base_value, 3)}</span>
        <span>Forecast {formatSignedPercent(explanation.prediction, 3)}</span>
      </div>
      {items.map((item) => {
        const width = (Math.abs(item.contribution ?? 0) / max) * 50;
        const positive = (item.contribution ?? 0) >= 0;
        return (
          <div className="shap-row" key={item.feature} title={`${item.label}: value ${formatNumber(item.value, 4)}, contribution ${formatSignedPercent(item.contribution, 3)}`}>
            <span className="shap-label">{item.label}<small>{item.group} · value {formatNumber(item.value, 3)}</small></span>
            <span className="shap-track">
              <span className="shap-axis" />
              <span className={`shap-bar ${positive ? "shap-positive" : "shap-negative"}`} style={{ width: `${width}%`, [positive ? "left" : "right"]: "50%" }} />
            </span>
            <span className={`shap-value ${positive ? "signal-positive" : "signal-negative"}`}>{formatSignedPercent(item.contribution, 3)}</span>
          </div>
        );
      })}
      <p className="muted-note">{explanation.method}. Contributions add up from the base value to the forecast.</p>
    </div>
  );
}

const SIGNAL_LABELS: Record<string, [string, "percent" | "number" | "score"]> = {
  news_sentiment_1d: ["News sentiment (today)", "score"],
  news_sentiment_7d: ["News sentiment (7d)", "score"],
  news_count_7d: ["News articles (7d)", "number"],
  filing_sentiment_90d: ["8-K sentiment (90d)", "score"],
  filing_count_30d: ["SEC filings (30d)", "number"],
  kg_degree: ["Knowledge-graph degree", "score"],
  kg_pagerank: ["Knowledge-graph PageRank", "score"],
  corr_eigenvector_60d: ["Correlation centrality", "score"],
  peer_return_5d: ["Peer 5-day return", "percent"],
};

export function SignalGrid({ signals }: { signals: Record<string, number | null> | undefined }) {
  if (!signals) return null;
  return (
    <div className="signal-grid">
      {Object.entries(SIGNAL_LABELS).map(([key, [label, kind]]) => {
        const value = signals[key];
        const display = value == null ? "No data" : kind === "percent" ? formatPercent(value) : kind === "number" ? value.toFixed(0) : value.toFixed(3);
        return <KeyValue key={key} label={label} value={display} />;
      })}
    </div>
  );
}
