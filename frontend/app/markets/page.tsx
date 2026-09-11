import { EmptyState, SectionHeading } from "../components/dashboard";

export default function MarketsPage() {
  return <><div className="content-header"><div><span className="eyebrow">Market intelligence</span><h1>Markets</h1><p>Market context for the investment workflow.</p></div></div><section className="panel module-panel"><div className="panel-header"><SectionHeading eyebrow="Module status" title="Market data module" /></div><EmptyState title="Market data is not connected" message="Live prices, market breadth, and instrument analytics will appear here once a Markets API is available." /></section></>;
}