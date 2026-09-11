import type { HealthResponse, PortfolioResponse, RiskResponse } from "../types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
  return response.json() as Promise<T>;
}

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  portfolio: () => getJson<PortfolioResponse>("/api/portfolio"),
  risk: () => getJson<RiskResponse>("/api/risk"),
};