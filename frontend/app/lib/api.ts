import type { HealthResponse, MarketHistoryResponse, MarketsResponse, PortfolioResponse, ResearchFeaturesResponse, ResearchModelResponse, ResearchPredictionsResponse, RiskResponse } from "../types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
  return response.json() as Promise<T>;
}

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  portfolio: () => getJson<PortfolioResponse>("/api/portfolio"),
  risk: () => getJson<RiskResponse>("/api/risk"),
  markets: () => getJson<MarketsResponse>("/api/markets"),
  marketHistory: (ticker: string) => getJson<MarketHistoryResponse>(`/api/markets/${encodeURIComponent(ticker)}/history`),
  researchModel: () => getJson<ResearchModelResponse>("/api/research/model"),
  researchPredictions: (limit = 100) => getJson<ResearchPredictionsResponse>(`/api/research/predictions?limit=${limit}`),
  researchFeatures: () => getJson<ResearchFeaturesResponse>("/api/research/features"),
};