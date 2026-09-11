import type { ReactNode } from "react";

export function MetricCard({ label, value, detail, accent = "neutral" }: { label: string; value: string; detail?: string; accent?: "neutral" | "green" | "amber" }) {
  return <article className={`metric-card metric-card-${accent}`}><span className="eyebrow">{label}</span><strong>{value}</strong>{detail ? <span className="metric-detail">{detail}</span> : null}</article>;
}

export function SectionHeading({ eyebrow, title, action }: { eyebrow: string; title: string; action?: ReactNode }) {
  return <div className="section-heading"><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div>{action}</div>;
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <span className={`skeleton ${className}`} aria-label="Loading" />;
}

export function EmptyState({ title, message }: { title: string; message: string }) {
  return <div className="empty-state"><span className="empty-mark">—</span><strong>{title}</strong><p>{message}</p></div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="error-state" role="alert"><span className="error-icon">!</span><div><strong>API unavailable</strong><p>{message}</p></div></div>;
}