"use client";

import { signIn } from "next-auth/react";
import Image from "next/image";
import { useEffect, useState } from "react";
import type { MarketSnapshot } from "../types";
import { formatPrice, formatPercent } from "../lib/login-formatters";

function GlobeVisual() {
  return (
    <div className="cyberion-globe" aria-hidden="true">
      <div className="globe-halo" />
      <div className="globe-sphere">
        <span className="globe-latitude globe-latitude-one" />
        <span className="globe-latitude globe-latitude-two" />
        <span className="globe-latitude globe-latitude-three" />
        <span className="globe-longitude globe-longitude-one" />
        <span className="globe-longitude globe-longitude-two" />
        <span className="globe-longitude globe-longitude-three" />
        <span className="globe-contour contour-one" />
        <span className="globe-contour contour-two" />
        <span className="globe-node node-one" />
        <span className="globe-node node-two" />
        <span className="globe-node node-three" />
        <span className="globe-node node-four" />
        <span className="globe-node node-five" />
      </div>
      <span className="globe-orbit globe-orbit-one" />
      <span className="globe-orbit globe-orbit-two" />
      <span className="globe-coordinate coordinate-one">40.7128 N / 74.0060 W</span>
      <span className="globe-coordinate coordinate-two">GLOBAL SIGNAL FIELD</span>
    </div>
  );
}

function GoogleIcon() {
  return (
    <svg width="19" height="19" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M21.35 12.23c0-.79-.07-1.55-.22-2.27H12v4.3h5.24a4.48 4.48 0 0 1-1.94 2.94v2.45h3.14c1.84-1.69 2.91-4.18 2.91-7.42z" />
      <path fill="#34A853" d="M12 21.82c2.63 0 4.84-.87 6.45-2.35l-3.14-2.45c-.87.58-1.98.92-3.31.92-2.54 0-4.69-1.72-5.46-4.03H3.3v2.53A9.74 9.74 0 0 0 12 21.82z" />
      <path fill="#FBBC05" d="M6.54 13.91A5.86 5.86 0 0 1 6.23 12c0-.66.11-1.3.31-1.91V7.56H3.3A9.82 9.82 0 0 0 2.18 12c0 1.58.38 3.08 1.12 4.44l3.24-2.53z" />
      <path fill="#EA4335" d="M12 6.06c1.43 0 2.71.49 3.72 1.45l2.79-2.79C16.84 3.16 14.63 2.18 12 2.18a9.74 9.74 0 0 0-8.7 5.38l3.24 2.53C7.31 7.78 9.46 6.06 12 6.06z" />
    </svg>
  );
}

function SignalRow({ market }: { market: MarketSnapshot }) {
  const trend = market.return_1d == null ? "neutral" : market.return_1d >= 0 ? "positive" : "negative";
  return (
    <div className="signal-row">
      <span className="signal-symbol">{market.ticker}</span>
      <span className={`signal-bars signal-${trend}`} aria-hidden="true"><i /><i /><i /><i /><i /><i /></span>
      <span className={`signal-value signal-${trend}`}>{formatPercent(market.return_1d)}</span>
      <span className="signal-price">{formatPrice(market.close)}</span>
    </div>
  );
}

export default function LoginPage() {
  const [markets, setMarkets] = useState<MarketSnapshot[]>([]);

  useEffect(() => {
    fetch("/api/markets", { cache: "no-store" })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Market data unavailable")))
      .then((data: { markets?: MarketSnapshot[] }) => setMarkets(data.markets ?? []))
      .catch(() => setMarkets([]));
  }, []);

  return (
    <main className="cyberion-login">
      <div className="login-grid" aria-hidden="true" />
      <div className="login-noise" aria-hidden="true" />
      <div className="login-scanline" aria-hidden="true" />
      <div className="login-glow login-glow-left" aria-hidden="true" />
      <div className="login-glow login-glow-right" aria-hidden="true" />
      <div className="login-orbit login-orbit-one" aria-hidden="true" />
      <div className="login-orbit login-orbit-two" aria-hidden="true" />
      <div className="login-orbit login-orbit-three" aria-hidden="true" />

      <header className="login-header">
        <div className="login-brand">
          <Image src="/cyberion-logo.png" alt="CYBERION AI HEDGE FUND" width={1408} height={768} priority className="login-logo-image" />
        </div>
        <div className="system-status"><span className="status-dot" />SYSTEM ONLINE</div>
      </header>

      <div className="login-content">
        <section className="login-left" aria-labelledby="login-title">
          <div className="login-eyebrow"><span className="eyebrow-line" />ARTIFICIAL INTELLIGENCE <b>×</b> FINANCE</div>
          <h1 id="login-title" className="login-title">Intelligence<br /><span>for the markets.</span></h1>
          <GlobeVisual />
          <p className="login-description">Cyberion combines machine learning, quantitative research and systematic portfolio intelligence to transform market data into actionable investment insight.</p>

          <div className="login-features">
            <div className="login-feature"><span>01</span><strong>AI-DRIVEN</strong><p>Machine learning market intelligence</p></div>
            <div className="login-feature"><span>02</span><strong>SYSTEMATIC</strong><p>Data-driven portfolio construction</p></div>
            <div className="login-feature"><span>03</span><strong>GLOBAL</strong><p>Continuous market analysis</p></div>
          </div>

          <div className="market-terminal" aria-label="Market signal monitor">
            <div className="terminal-header"><span>MARKET SIGNALS</span><span className="terminal-ai"><i />AI ENGINE</span></div>
            <div className="terminal-body">{markets.length ? markets.slice(0, 4).map((market) => <SignalRow key={market.ticker} market={market} />) : <div className="terminal-empty">MARKET DATA LINK UNAVAILABLE</div>}</div>
          </div>
        </section>

        <section className="login-right" aria-labelledby="welcome-title">
          <div className="login-panel">
            <span className="panel-corner panel-corner-tl" /><span className="panel-corner panel-corner-tr" /><span className="panel-corner panel-corner-bl" /><span className="panel-corner panel-corner-br" />
            <div className="panel-top"><span>SECURE ACCESS</span><span>01 / 01</span></div>
            <div className="panel-heading"><div className="panel-kicker">WELCOME BACK.</div><h2 id="welcome-title">Enter the<br /><span>intelligence.</span></h2><p>Sign in to access your investment intelligence platform.</p></div>
            <button className="google-login-button" type="button" onClick={() => signIn("google", { callbackUrl: "/" })}><GoogleIcon /><span>Continue with Google</span><span className="google-arrow" aria-hidden="true">→</span></button>
            <div className="secure-note"><span className="secure-icon">◇</span><span>SECURE GOOGLE AUTHENTICATION</span></div>
            <div className="panel-divider"><span /><small>CYBERION</small><span /></div>
            <div className="login-quote"><span className="quote-mark">“</span><p>A more intelligent tomorrow,<br />for a more prosperous world.</p></div>
            <div className="panel-metrics"><div><span>ENGINE</span><strong>ONLINE</strong></div><div><span>LATENCY</span><strong>LOW</strong></div><div><span>SECURITY</span><strong>ACTIVE</strong></div></div>
            <div className="login-panel-caption"><span>AUTHENTICATION</span><span>ENCRYPTED / PROTECTED</span></div>
          </div>
        </section>
      </div>

      <footer className="login-footer"><span>CYBERION AI HEDGE FUND</span><div><span>INTELLIGENCE</span><b>/</b><span>DISCIPLINE</span><b>/</b><span>ALPHA</span></div><span>© 2026 CYBERION</span></footer>
    </main>
  );
}
