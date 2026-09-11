import { EmptyState, SectionHeading } from "../components/dashboard";

export default function ResearchPage() {
  return <><div className="content-header"><div><span className="eyebrow">Model intelligence</span><h1>Research</h1><p>Model diagnostics and research analytics for the investment team.</p></div></div><section className="panel module-panel"><div className="panel-header"><SectionHeading eyebrow="Module status" title="Research module" /></div><EmptyState title="Research data is not connected" message="Model diagnostics, experiment results, and research analytics will appear here once a Research API is available." /></section></>;
}