"use client";

import { useEffect, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { api } from "../lib/api";
import type { HealthResponse } from "../types";

export default function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => { api.health().then(setHealth).catch(() => setFailed(true)).finally(() => setLoading(false)); }, []);

  return <>
    <div className="content-header"><div><span className="eyebrow">Infrastructure</span><h1>System</h1><p>Service connectivity and platform health.</p></div></div>
    {failed ? <ErrorState message="The health endpoint is unavailable. Check that the FastAPI service is running on port 8000." /> : null}
    <section className="panel system-page-panel"><div className="panel-header"><SectionHeading eyebrow="Live diagnostics" title="System health" /></div>{loading ? <div className="system-body"><Skeleton className="system-skeleton" /><Skeleton className="system-skeleton" /></div> : health ? <div className="system-body"><div className="system-cell"><span className="eyebrow">API service</span><strong className="system-value"><span className="live-dot" />{health.status}</strong></div><div className="system-cell"><span className="eyebrow">Database</span><strong className="system-value"><span className="live-dot" />{health.database}</strong></div></div> : <EmptyState title="No health data" message="The API did not return system status." />}</section>
  </>;
}