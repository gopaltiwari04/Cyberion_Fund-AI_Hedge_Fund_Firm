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