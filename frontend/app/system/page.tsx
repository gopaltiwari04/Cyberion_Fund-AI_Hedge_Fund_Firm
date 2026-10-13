"use client";

import { useCallback, useEffect, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { KeyValue, StatusBadge } from "../components/ui";
import { api, errorMessage } from "../lib/api";
import { formatDateTime } from "../lib/formatters";
import type { HealthResponse, SystemStatus } from "../types";

export default function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchStatus = useCallback(() => {
    Promise.allSettled([api.health(), api.systemStatus()]).then(([h, s]) => {
      setHealth(h.status === "fulfilled" ? h.value : null);
      setStatus(s.status === "fulfilled" ? s.value : null);
      setError(h.status === "rejected" ? errorMessage(h.reason) : s.status === "rejected" ? errorMessage(s.reason) : null);
      setLoading(false);
    });
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);
  const load = () => { setLoading(true); fetchStatus(); };

  return <>
    <div className="content-header"><div><span className="eyebrow">Infrastructure</span><h1>System</h1><p>Service health, data freshness and pipeline observability.</p></div><button className="button button-secondary" onClick={load} disabled={loading}>{loading ? "Refreshing…" : "Refresh"}</button></div>
    {error ? <ErrorState message={error} /> : null}
    <section className="panel" aria-label="Service health"><div className="panel-header"><SectionHeading eyebrow="Live diagnostics" title="Services" action={status ? <StatusBadge status={status.status} /> : null} /></div>
      {loading && !status ? <Skeleton className="skeleton-line" /> : <div className="kv-grid kv-grid-4">
        <KeyValue label="API" value={<StatusBadge status={health?.status ?? "unavailable"} />} />
        <KeyValue label="Database" value={health?.database ?? "unavailable"} />
        <KeyValue label="Failed runs (24h)" value={status?.failed_runs_24h ?? "N/A"} tone={status?.failed_runs_24h ? "signal-negative" : undefined} />
        <KeyValue label="Last completed session" value={status?.last_completed_session ?? "N/A"} />
        <KeyValue label="SEC chunks indexed" value={status ? status.sec_index.chunks.toLocaleString() : "N/A"} />
        <KeyValue label="Chunks with embeddings" value={status ? status.sec_index.embedded_chunks.toLocaleString() : "N/A"} />
        <KeyValue label="Filings downloaded / indexed" value={status ? `${status.sec_index.filings_by_status.downloaded ?? 0} / ${status.sec_index.filings_by_status.indexed ?? 0}` : "N/A"} />
        <KeyValue label="Checked" value={formatDateTime(status?.checked_at)} />
      </div>}
    </section>
    <section className="panel" aria-label="Data freshness"><div className="panel-header"><SectionHeading eyebrow="Freshness SLAs (business days)" title="Datasets" /></div>
      {!status ? <Skeleton className="skeleton-line" /> : <div className="table-wrap"><table className="holdings-table"><thead><tr><th>Dataset</th><th>Description</th><th>Latest</th><th>Lag</th><th>Allowed</th><th>Status</th></tr></thead><tbody>
        {status.datasets.map((d) => <tr key={d.dataset}><td className="ticker">{d.dataset.replace(/_/g, " ")}</td><td className="left-cell">{d.description}</td><td>{d.latest ? formatDateTime(d.latest) : "—"}</td><td>{d.business_days_behind ?? "—"}</td><td>{d.max_allowed_lag}</td><td><StatusBadge status={d.status} /></td></tr>)}
      </tbody></table></div>}
    </section>
    <section className="panel" aria-label="Pipelines"><div className="panel-header"><SectionHeading eyebrow="Airflow tasks (latest run each)" title="Pipelines" /></div>
      {!status ? <Skeleton className="skeleton-line" /> : status.pipelines.length === 0 ? <EmptyState title="No runs recorded" message="Pipeline runs appear here once Airflow tasks execute." /> : <div className="table-wrap"><table className="holdings-table"><thead><tr><th>Task</th><th>Status</th><th>Started</th><th>Duration</th><th>Rows</th><th>Notes</th></tr></thead><tbody>
        {status.pipelines.map((run) => <tr key={`${run.pipeline}.${run.task}`}><td className="ticker">{run.pipeline}.{run.task}</td><td><StatusBadge status={run.status} /></td><td>{formatDateTime(run.started_at)}</td><td>{run.duration ? `${Number(run.duration).toFixed(0)}s` : "—"}</td><td>{run.rows_processed.toLocaleString()}</td><td className="left-cell wrap-cell">{run.error ? run.error.split("\n")[0].slice(0, 160) : run.warnings?.length ? `${run.warnings.length} warning(s): ${String(run.warnings[0]).slice(0, 120)}` : "—"}</td></tr>)}
      </tbody></table></div>}
    </section>
  </>;
}
