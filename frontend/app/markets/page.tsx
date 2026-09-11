"use client";

import { useEffect, useState } from "react";
import { EmptyState, ErrorState, SectionHeading, Skeleton } from "../components/dashboard";
import { api } from "../lib/api";
import { formatDate, formatNumber, formatPercent } from "../lib/formatters";
import type { MarketHistoryPoint, MarketSnapshot, MarketsResponse } from "../types";
import { MarketChart } from "./market-chart";

function formatPrice(value: number | null | undefined) {
  return value == null || !Number.isFinite(value)
    ? "N/A"
    : value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function signalClass(value: number | null | undefined) {
  if (value == null || value === 0) return "signal-neutral";
  return value > 0 ? "signal-positive" : "signal-negative";
}

function SignalValue({ value }: { value: number | null }) {
  return <span className={signalClass(value)}>{formatPercent(value)}</span>;
}

function MarketTable({ markets, selectedTicker, onSelect }: { markets: MarketSnapshot[]; selectedTicker: string; onSelect: (ticker: string) => void }) {
  return (
    <div className="table-wrap">
      <table className="markets-table">
        <thead><tr><th>Asset</th><th>Data as of</th><th>Latest close</th><th>1D return</th><th>5D return</th><th>RSI</th><th>20D volatility</th><th>Regime</th></tr></thead>
        <tbody>
          {markets.map((market) => (
            <tr key={market.ticker} className={market.ticker === selectedTicker ? "selected-row" : ""} onClick={() => onSelect(market.ticker)}>
              <td className="ticker">{market.ticker}</td><td>{formatDate(market.date)}</td><td className="price-cell">{formatPrice(market.close)}</td><td><SignalValue value={market.return_1d} /></td><td><SignalValue value={market.return_5d} /></td><td>{formatNumber(market.rsi_14)}</td><td>{formatPercent(market.volatility_20d)}</td><td>{market.regime == null ? "N/A" : `Regime: ${market.regime}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DetailMetric({ label, value, className = "" }: { label: string; value: string; className?: string }) {
  return <div className="market-detail-metric"><span className="eyebrow">{label}</span><strong className={className}>{value}</strong></div>;
}

export default function MarketsPage() {
  const [markets, setMarkets] = useState<MarketSnapshot[]>([]);
  const [marketDate, setMarketDate] = useState<string | null>(null);
  const [marketsLoading, setMarketsLoading] = useState(true);
  const [marketsFailed, setMarketsFailed] = useState(false);
  const [selectedTicker, setSelectedTicker] = useState("");
  const [history, setHistory] = useState<MarketHistoryPoint[]>([]);
  const [historyTicker, setHistoryTicker] = useState("");
  const [historyErrorTicker, setHistoryErrorTicker] = useState("");

  useEffect(() => {
    api.markets()
      .then((response: MarketsResponse) => { setMarkets(response.markets); setMarketDate(response.date); })
      .catch(() => setMarketsFailed(true))
      .finally(() => setMarketsLoading(false));
  }, []);

  const activeTicker = markets.some((market) => market.ticker === selectedTicker) ? selectedTicker : markets[0]?.ticker ?? "";
  const selectedMarket = markets.find((market) => market.ticker === activeTicker) ?? null;

  useEffect(() => {
    if (!activeTicker) return;
    let cancelled = false;
    api.marketHistory(activeTicker)
      .then((response) => {
        if (!cancelled) {
          setHistory(response.history);
          setHistoryTicker(activeTicker);
          setHistoryErrorTicker("");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHistoryTicker(activeTicker);
          setHistoryErrorTicker(activeTicker);
        }
      });
    return () => { cancelled = true; };
  }, [activeTicker]);

  const historyLoading = Boolean(activeTicker) && historyTicker !== activeTicker;
  const historyFailed = historyErrorTicker === activeTicker;

  return (
    <>
      <div className="content-header"><div><span className="eyebrow">Market intelligence</span><h1>Markets</h1><p>Live market context and model-derived market signals.</p></div><span className="as-of">Data as of {formatDate(marketDate ?? undefined)}</span></div>
      {marketsFailed ? <ErrorState message="Market data unavailable. Check that the FastAPI service is running and the market endpoints are reachable." /> : null}
      <section className="panel markets-overview-panel" aria-label="Market overview">
        <div className="panel-header"><SectionHeading eyebrow="Cross-asset monitor" title="Market overview" action={markets.length ? <span className="as-of">{markets.length} instruments</span> : null} /></div>
        {marketsLoading ? <div className="markets-skeleton"><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></div> : markets.length === 0 ? <EmptyState title="No market data available" message="The API returned no market observations from the connected database." /> : <MarketTable markets={markets} selectedTicker={activeTicker} onSelect={setSelectedTicker} />}
      </section>
      {selectedMarket ? (
        <section className="markets-detail-grid" aria-label={`${selectedMarket.ticker} market detail`}>
          <div className="panel market-detail-panel">
            <div className="panel-header"><SectionHeading eyebrow="Instrument detail" title={selectedMarket.ticker} action={<select className="ticker-select" value={activeTicker} onChange={(event) => setSelectedTicker(event.target.value)} aria-label="Select ticker">{markets.map((market) => <option key={market.ticker} value={market.ticker}>{market.ticker}</option>)}</select>} /></div>
            <div className="market-price-block"><span className="eyebrow">Latest price</span><strong>{formatPrice(selectedMarket.close)}</strong><span className="as-of">Data as of {formatDate(selectedMarket.date)}</span></div>
            <div className="market-detail-grid">
              <DetailMetric label="1D return" value={formatPercent(selectedMarket.return_1d)} className={signalClass(selectedMarket.return_1d)} /><DetailMetric label="5D return" value={formatPercent(selectedMarket.return_5d)} className={signalClass(selectedMarket.return_5d)} /><DetailMetric label="RSI" value={formatNumber(selectedMarket.rsi_14)} /><DetailMetric label="MACD" value={formatNumber(selectedMarket.macd)} /><DetailMetric label="20D volatility" value={formatPercent(selectedMarket.volatility_20d)} /><DetailMetric label="Beta vs SPY" value={formatNumber(selectedMarket.beta_60d)} /><DetailMetric label="Correlation vs SPY" value={formatNumber(selectedMarket.correlation_spy_60d)} /><DetailMetric label="Relative return vs market" value={formatPercent(selectedMarket.relative_return_5d)} className={signalClass(selectedMarket.relative_return_5d)} /><DetailMetric label="Regime" value={selectedMarket.regime == null ? "N/A" : `Regime: ${selectedMarket.regime}`} />
            </div>
          </div>
          <div className="panel market-chart-panel">
            <div className="panel-header"><SectionHeading eyebrow="Historical prices" title={`${selectedMarket.ticker} close`} action={<span className="as-of">{history.length ? `${history.length} observations` : "No history"}</span>} /></div>
            {historyLoading ? <Skeleton className="chart-skeleton" /> : historyFailed ? <ErrorState message="Historical market data unavailable for this ticker." /> : history.length === 0 ? <EmptyState title="No price history" message="The API returned no historical close prices for this ticker." /> : <MarketChart data={history} />}
          </div>
        </section>
      ) : null}
    </>
  );
}
