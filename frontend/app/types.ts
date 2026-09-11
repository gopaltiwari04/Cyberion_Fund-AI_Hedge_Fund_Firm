export interface HealthResponse {
  status: string;
  database: string;
}

export interface PortfolioAllocation {
  ticker: string;
  weight: number | null;
  expected_return: number | null;
  risk_contribution: number | null;
}

export interface PortfolioResponse {
  date: string;
  strategy: string;
  allocations: PortfolioAllocation[];
}

export interface RiskResponse {
  date: string;
  strategy: string;
  expected_annual_return: number | null;
  expected_annual_volatility: number | null;
  sharpe_ratio: number | null;
  var_95: number | null;
  cvar_95: number | null;
}

export interface MarketSnapshot {
  ticker: string;
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  return_1d: number | null;
  return_5d: number | null;
  rsi_14: number | null;
  macd: number | null;
  volatility_20d: number | null;
  regime: number | null;
  relative_return_5d: number | null;
  beta_60d: number | null;
  correlation_spy_60d: number | null;
}

export interface MarketsResponse {
  date: string | null;
  markets: MarketSnapshot[];
}

export interface MarketHistoryPoint {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
}

export interface MarketHistoryResponse {
  ticker: string;
  days: number;
  history: MarketHistoryPoint[];
}