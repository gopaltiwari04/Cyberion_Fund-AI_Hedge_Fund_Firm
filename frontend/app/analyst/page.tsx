"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { Badge, StatusBadge } from "../components/ui";
import { api, errorMessage } from "../lib/api";
import { formatDate } from "../lib/formatters";
import type { AnalystAnswer, AnalystSource, CoverageRow } from "../types";

const TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "JPM", "XOM", "JNJ"];
const FORMS = ["10-K", "10-Q", "8-K"] as const;
const EXAMPLES = [
  "What are Apple's main supply chain and tariff risks?",
  "How did NVIDIA describe data center demand in its latest filings?",
  "What did JPMorgan disclose about credit loss provisions?",
  "Summarize Microsoft's recent 8-K announcements.",
];

interface Turn { id: number; question: string; answer?: AnalystAnswer; error?: string }

function SourceCard({ source, highlighted }: { source: AnalystSource; highlighted: boolean }) {
  return (
    <article id={`source-${source.id}`} className={`source-card ${highlighted ? "source-highlight" : ""}`}>
      <header>
        <Badge tone="info">{source.id}</Badge>
        <strong>{source.ticker} · Form {source.form_type}</strong>
        <span>filed {formatDate(source.filing_date)}</span>
        <a href={source.url} target="_blank" rel="noopener noreferrer">View on SEC.gov ↗</a>
      </header>
      {source.section ? <small className="block-note">{source.section} · accession {source.accession_no}</small> : null}
      <p>{source.excerpt}{source.excerpt.length >= 1200 ? "…" : ""}</p>
      <small className="block-note">Retrieved by {source.retrieval.join(" + ")} search</small>
    </article>
  );
}

function AnswerView({ answer }: { answer: AnalystAnswer }) {
  const [highlight, setHighlight] = useState<string | null>(null);
  const focus = (id: string) => {
    setHighlight(id);
    document.getElementById(`source-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  return (
    <div className="answer">
      <div className="answer-meta"><StatusBadge status={answer.status} /><span>Retrieval: {answer.retrieval.mode}{answer.retrieval.tickers?.length ? ` · ${answer.retrieval.tickers.join(", ")}${answer.retrieval.tickers_inferred ? " (inferred)" : ""}` : ""}</span>{answer.model ? <span>Model: {answer.model}</span> : null}<span>{(answer.latency_ms / 1000).toFixed(1)}s</span></div>
      {answer.message ? <p className="answer-message">{answer.message}</p> : null}
      {answer.summary ? <p className="answer-summary">{answer.summary}</p> : null}
      {answer.statements.length ? (
        <ul className="statements">
          {answer.statements.map((statement, index) => (
            <li key={index} className={`statement statement-${statement.type}`}>
              <Badge tone={statement.type === "fact" ? "good" : "warn"}>{statement.type === "fact" ? "Fact" : "Inference"}</Badge>
              <span>{statement.text}</span>
              <span className="citations">{statement.citations.map((citation) => <button key={citation} className="citation" onClick={() => focus(citation)}>[{citation}]</button>)}</span>
              {statement.verification === "unverified_numbers" ? <small className="warning-note">Numbers not found verbatim in cited text: {statement.unverified_numbers.join(", ")} — verify in the source.</small> : null}
            </li>
          ))}
        </ul>
      ) : null}
      {answer.limitations ? <p className="muted-note"><strong>Limitations:</strong> {answer.limitations}</p> : null}
      {answer.rejected_statements.length ? (
        <details className="assumptions"><summary>{answer.rejected_statements.length} unsupported statement(s) removed</summary><ul>{answer.rejected_statements.map((item, index) => <li key={index}>{item.text} <em>({item.reason})</em></li>)}</ul></details>
      ) : null}
      {answer.sources.length ? <div className="sources"><h3 className="subheading">Sources</h3>{answer.sources.map((source) => <SourceCard key={source.id} source={source} highlighted={highlight === source.id} />)}</div> : null}
      <p className="disclaimer-inline">{answer.disclaimer}</p>
    </div>
  );
}

export default function AnalystPage() {
  const [question, setQuestion] = useState("");
  const [tickers, setTickers] = useState<string[]>([]);
  const [forms, setForms] = useState<string[]>([]);
  const [dateFrom, setDateFrom] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [coverage, setCoverage] = useState<{ coverage: CoverageRow[]; scope_note: string } | null>(null);
  const [coverageError, setCoverageError] = useState<string | null>(null);
  const counter = useRef(0);

  useEffect(() => { api.coverage().then(setCoverage).catch((error) => setCoverageError(errorMessage(error))); }, []);

  async function submit(event?: FormEvent, preset?: string) {
    event?.preventDefault();
    const text = (preset ?? question).trim();
    if (text.length < 5 || busy) return;
    const id = ++counter.current;
    setTurns((current) => [{ id, question: text }, ...current]);
    setQuestion("");
    setBusy(true);
    try {
      const answer = await api.ask({ question: text, tickers: tickers.length ? tickers : null, form_types: forms.length ? forms : null, date_from: dateFrom || null });
      setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, answer } : turn)));
    } catch (error) {
      setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, error: errorMessage(error) } : turn)));
    } finally {
      setBusy(false);
    }
  }

  const toggle = (list: string[], value: string, set: (next: string[]) => void) => set(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);

  return <>
    <div className="content-header"><div><span className="eyebrow">Qualitative research</span><h1>SEC research analyst</h1><p>Answers grounded in retrieved 10-K, 10-Q and 8-K passages, with every statement cited and labelled as fact or inference.</p></div></div>
    <div className="analyst-grid">
      <div>
        <section className="panel" aria-label="Ask a question">
          <form className="form-body form-stack" onSubmit={submit}>
            <label className="field"><span>Question</span><textarea className="input textarea" maxLength={1000} rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. What risks did Apple disclose about its supply chain?" onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); }} /></label>
            <div className="chip-row" role="group" aria-label="Companies">{TICKERS.map((t) => <button type="button" key={t} className={`chip ${tickers.includes(t) ? "chip-on" : ""}`} aria-pressed={tickers.includes(t)} onClick={() => toggle(tickers, t, setTickers)}>{t}</button>)}</div>
            <div className="form-row">
              <div className="chip-row" role="group" aria-label="Form types">{FORMS.map((f) => <button type="button" key={f} className={`chip ${forms.includes(f) ? "chip-on" : ""}`} aria-pressed={forms.includes(f)} onClick={() => toggle(forms, f, setForms)}>{f}</button>)}</div>
              <label className="field field-inline"><span>Filed after</span><input className="input" type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} /></label>
              <button className="button" type="submit" disabled={busy || question.trim().length < 5}>{busy ? "Researching…" : "Ask"}</button>
            </div>
            {turns.length === 0 ? <div className="chip-row">{EXAMPLES.map((example) => <button type="button" key={example} className="chip chip-example" onClick={() => submit(undefined, example)}>{example}</button>)}</div> : null}
          </form>
        </section>
        {turns.map((turn) => (
          <section key={turn.id} className="panel turn" aria-label={turn.question}>
            <div className="panel-header"><SectionHeading eyebrow="Question" title={turn.question} /></div>
            <div className="form-body">{turn.error ? <ErrorState message={turn.error} /> : turn.answer ? <AnswerView answer={turn.answer} /> : <><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></>}</div>
          </section>
        ))}
      </div>
      <aside className="panel coverage-panel" aria-label="Index coverage">
        <div className="panel-header"><SectionHeading eyebrow="Knowledge base" title="Indexed filings" /></div>
        {coverageError ? <ErrorState message={coverageError} /> : !coverage ? <Skeleton className="skeleton-line" /> : coverage.coverage.length === 0 ? <EmptyState title="No filings indexed" message="The SEC pipeline has not indexed any filings yet." /> : (
          <>
            <div className="table-wrap"><table className="holdings-table compact-table"><thead><tr><th>Ticker</th><th>Form</th><th>Filings</th><th>Latest</th></tr></thead><tbody>{coverage.coverage.map((row) => <tr key={`${row.ticker}-${row.form_type}`}><td className="ticker">{row.ticker}</td><td>{row.form_type}</td><td>{row.filings}</td><td>{formatDate(row.last)}</td></tr>)}</tbody></table></div>
            <p className="muted-note padded">{coverage.scope_note}</p>
          </>
        )}
      </aside>
    </div>
  </>;
}
