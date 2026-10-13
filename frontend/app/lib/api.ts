import type {
  AnalystAnswer, BacktestRun, CoverageRow, ForecastResponse, HealthResponse, HoldingsResponse, MarketHistoryResponse,
  MarketsResponse, MonitoringMetric, NewsArticle, PortfolioResponse, Recommendation, ResearchFeaturesResponse,
  ResearchModelResponse, ResearchModelsResponse, ResearchPredictionsResponse, RiskResponse, SystemStatus, UserProfile,
} from "../types";

export class ApiError extends Error {
  constructor(public status: number, message: string, public details?: unknown) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { cache: "no-store", ...init, headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } });
  } catch {
    throw new ApiError(0, "Network error: the analytics service could not be reached.");
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    let details: unknown;
    try {
      const body = await response.json();
      details = body;
      if (typeof body?.detail === "string") message = body.detail;
      if (Array.isArray(body?.errors) && body.errors.length) {
        message = body.errors.map((error: { field: string; message: string }) => `${error.field || "request"}: ${error.message}`).join("; ");
      }
    } catch {
      /* non-JSON error body */
    }
    if (response.status === 401) message = "Your session has expired. Please sign in again.";
    if (response.status === 429) message = "Too many requests. Please wait a moment and try again.";
    throw new ApiError(response.status, message, details);
  }
  return response.json() as Promise<T>;
}

const get = <T>(path: string) => request<T>(path);
const send = <T>(method: "POST" | "PUT", path: string, body: unknown) => request<T>(path, { method, body: JSON.stringify(body) });

export const api = {
  health: () => get<HealthResponse>("/api/health"),
  portfolio: () => get<PortfolioResponse>("/api/portfolio"),
  risk: () => get<RiskResponse>("/api/risk"),
  markets: () => get<MarketsResponse>("/api/markets"),
  marketHistory: (ticker: string, days = 252) => get<MarketHistoryResponse>(`/api/markets/${encodeURIComponent(ticker)}/history?days=${days}`),
  forecast: (ticker: string) => get<ForecastResponse>(`/api/markets/${encodeURIComponent(ticker)}/forecast`),
  news: (ticker?: string, limit = 20) => get<{ articles: NewsArticle[] }>(`/api/news?limit=${limit}${ticker ? `&ticker=${encodeURIComponent(ticker)}` : ""}`),
  researchModel: (horizon = 5) => get<ResearchModelResponse>(`/api/research/model?horizon=${horizon}`),
  researchModels: () => get<ResearchModelsResponse>("/api/research/models"),
  researchPredictions: (limit = 100, horizon = 5, ticker?: string) =>
    get<ResearchPredictionsResponse>(`/api/research/predictions?limit=${limit}&horizon=${horizon}${ticker ? `&ticker=${ticker}` : ""}`),
  researchFeatures: (horizon = 5) => get<ResearchFeaturesResponse>(`/api/research/features?horizon=${horizon}`),
  profile: () => get<UserProfile>("/api/me/profile"),
  updateProfile: (profile: Partial<UserProfile>) => send<UserProfile>("PUT", "/api/me/profile", profile),
  holdings: () => get<HoldingsResponse>("/api/me/holdings"),
  saveHoldings: (holdings: Array<{ ticker: string; quantity: number; average_cost: number | null }>, cash_balance: number) =>
    send<HoldingsResponse>("PUT", "/api/me/holdings", { holdings, cash_balance }),
  recommend: (body: Record<string, unknown>) => send<Recommendation>("POST", "/api/me/recommendations", body),
  ask: (body: Record<string, unknown>) => send<AnalystAnswer>("POST", "/api/analyst/ask", body),
  coverage: () => get<{ coverage: CoverageRow[]; scope_note: string }>("/api/analyst/coverage"),
  submitBacktest: (body: Record<string, unknown>) => send<{ id: string; status: string }>("POST", "/api/backtests", body),
  backtests: () => get<{ runs: BacktestRun[] }>("/api/backtests"),
  backtest: (id: string) => get<BacktestRun>(`/api/backtests/${encodeURIComponent(id)}`),
  systemStatus: () => get<SystemStatus>("/api/system/status"),
  monitoring: () => get<{ since: string; metrics: MonitoringMetric[] }>("/api/system/monitoring"),
};

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unexpected error";
}
