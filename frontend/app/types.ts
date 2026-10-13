export interface HealthResponse {
  status: string;
  database: string;
}

export interface PortfolioAllocation {
  ticker: string;
  weight: number | null;
  expected_return: number | null;
  risk_contribution: number | null;
  predicted_volatility?: number | null;
}

export interface PortfolioResponse {
  date: string;
  strategy: string;
  horizon_days?: number | null;
  allocations: PortfolioAllocation[];
}

export interface StressTest {
  name: string;
  start: string;
  end: string;
  return: number | null;
  pnl: number | null;
  note?: string;
}

export interface ScenarioAnalysis {
  model_expected_pnl: number;
  model_band_90: [number, number];
  historical_horizon_pnl_percentiles: Record<string, number>;
  historical_window: { start: string; end: string; observations: number };
  stress_tests: StressTest[];
}

export interface PortfolioMetrics {
  horizon_days: number;
  expected_return_horizon: number;
  expected_return_annualized_linear: number;
  volatility_daily: number;
  volatility_horizon: number;
  volatility_annualized: number;
  sharpe_annualized_model: number | null;
  var_95_1d: number;
  cvar_95_1d: number;
  cvar_limit_1d: number;
  parametric_var_95_1d: number;
  worst_scenario_1d: number;
  turnover: number;
  estimated_transaction_cost: number;
  effective_number_of_assets: number | null;
  max_position: number;
  risk_contribution: Record<string, number>;
  sector_exposure: Record<string, number>;
}

export interface RiskResponse {
  date: string;
  strategy: string;
  expected_annual_return: number | null;
  expected_annual_volatility: number | null;
  sharpe_ratio: number | null;
  var_95: number | null;
  cvar_95: number | null;
  cvar_limit?: number | null;
  feasible?: boolean | null;
  solver_status?: string | null;
  turnover?: number | null;
  transaction_cost?: number | null;
  horizon_days?: number | null;
  message?: string | null;
  binding_constraints?: string[] | null;
  alpha_confidence?: number | null;
  metrics?: PortfolioMetrics | null;
  scenarios?: ScenarioAnalysis | null;
  assumptions?: Record<string, unknown> | null;
}

export interface MarketSnapshot {
  ticker: string;
  name?: string;
  sector?: string;
  asset_class?: string;
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
  news_sentiment_7d?: number | null;
  news_count_7d?: number | null;
  filing_sentiment_90d?: number | null;
  forecast_date?: string | null;
  forecast_return_5d?: number | null;
  forecast_lower_5d?: number | null;
  forecast_upper_5d?: number | null;
  forecast_volatility?: number | null;
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

export interface Contribution {
  feature: string;
  label: string;
  group: string;
  value: number | null;
  contribution: number | null;
}

export interface Explanation {
  method: string;
  base_value: number | null;
  prediction: number | null;
  top_contributions: Contribution[];
  signals: Record<string, number | null>;
}

export interface Forecast {
  horizon_days: number;
  as_of_date: string;
  expected_return: number;
  interval: [number, number];
  interval_level: number;
  volatility_annualized: number | null;
  explanation: Explanation | null;
  model: {
    id: number;
    version: string;
    trained_at: string;
    holdout_ic: number | null;
    holdout_interval_coverage: number | null;
    holdout_directional_accuracy: number | null;
  };
}

export interface ForecastResponse {
  ticker: string;
  name: string;
  sector: string;
  forecasts: Forecast[];
  disclaimer: string;
}

export interface NewsArticle {
  uuid: string;
  ticker: string;
  title: string | null;
  description: string | null;
  url: string | null;
  source: string | null;
  published_at: string | null;
  finbert_score: number | null;
  entity_sentiment: number | null;
}

export interface ResearchModelResponse {
  model: {
    name: string | null;
    type: string | null;
    target: string | null;
    horizon_days: number | null;
    version?: string;
    status?: string;
    trained_at?: string;
    notes?: string | null;
  };
  dataset: {
    rows: number | null;
    target_rows: number | null;
    train_rows: number | null;
    validation_rows: number | null;
    test_rows: number | null;
    train_start?: string | null;
    train_end?: string | null;
    oos_start?: string | null;
    oos_end?: string | null;
  };
  metrics: {
    mae: number | null;
    rmse: number | null;
    directional_accuracy: number | null;
    prediction_correlation: number | null;
  };
  evaluation?: EvaluationMetrics;
  calibration?: Calibration | null;
  feature_count: number | null;
  validation?: string;
}

export interface EvaluationMetrics {
  observations: number;
  test_start: string;
  test_end: string;
  mae: number;
  rmse: number;
  mae_zero_baseline: number;
  r2_oos_vs_zero: number;
  directional_accuracy: number;
  base_rate_positive: number;
  prediction_correlation: number;
  ic_mean: number;
  ic_std: number;
  ic_tstat_overlap_adjusted: number;
  ic_hit_rate: number;
  ic_by_year: Record<string, number>;
  top_minus_bottom_tercile_return: number;
  interval_level: number;
  interval_coverage: number;
  interval_mean_width: number;
  vol_mae: number;
  vol_mae_trailing20d_baseline: number;
  vol_correlation: number;
  holdout_since_2020?: Partial<EvaluationMetrics>;
}

export interface Calibration {
  interval_level: number;
  interval_coverage: number;
  reliability_slope: number;
  reliability_intercept: number;
  decile_table: Array<{ decile: number; predicted: number; realized: number; count: number }>;
}

export interface RegistryModel {
  id: number;
  horizon_days: number;
  version: string;
  status: string;
  trained_at: string;
  metrics: EvaluationMetrics;
  calibration: Calibration;
  feature_importance: Array<{ feature: string; label: string; group: string; importance: number }>;
  notes: string | null;
}

export interface Challenger {
  id: number;
  model_name: string;
  algorithm: string;
  horizon_days: number;
  version: string;
  status: string;
  trained_at: string;
  notes: string | null;
  metrics: {
    observations: number;
    tft?: Record<string, number | null>;
    xgboost?: Record<string, number | null>;
    mae_zero_baseline?: number;
    test_start?: string;
    test_end?: string;
    tft_beats_incumbent?: boolean;
  };
}

export interface ResearchModelsResponse {
  models: RegistryModel[];
  challengers: Challenger[];
}

export type ResearchPredictionValue = string | number | null;

export interface ResearchPredictionsResponse {
  available_columns: string[];
  observations: Array<Record<string, ResearchPredictionValue>>;
}

export interface ResearchFeaturesResponse {
  feature_count: number;
  features: string[];
}

export interface UserProfile {
  email: string;
  display_name: string | null;
  risk_profile: "conservative" | "balanced" | "aggressive";
  cvar_limit: number;
  max_weight: number;
  max_turnover: number;
  transaction_cost_bps: number;
  cash_balance: number;
  updated_at: string;
}

export interface HoldingRow {
  ticker: string;
  quantity: number;
  average_cost: number | null;
  price: number | null;
  price_date: string | null;
  market_value: number | null;
  unrealized_pnl: number | null;
  weight: number | null;
}

export interface HoldingsResponse {
  cash_balance: number;
  total_value: number;
  holdings: HoldingRow[];
}

export interface RecommendationAsset {
  ticker: string;
  name: string;
  sector: string;
  price: number;
  price_date: string;
  current_weight: number;
  target_weight: number | null;
  trade_value: number | null;
  trade_shares: number | null;
  action: "buy" | "sell" | "hold" | null;
  forecast: { horizon_days: number; expected_return: number; interval: [number, number]; interval_level: number; volatility_annualized: number | null };
  risk_contribution: number | null;
  explanation: Explanation | null;
}

export interface Recommendation {
  id?: string;
  status: "optimal" | "infeasible" | "error" | "unavailable";
  feasible: boolean;
  message: string;
  as_of_date: string;
  portfolio_value: number | null;
  profile: string;
  constraints: Record<string, number | null>;
  model: { id: number; version: string; horizon_days: number; trained_at: string; data_end: string; alpha_confidence: number; holdout_ic: number | null };
  risk_free: { daily: number; source: string };
  current_weights: Record<string, number>;
  proposal: { weights: Record<string, number>; metrics: PortfolioMetrics } | null;
  assets: RecommendationAsset[];
  binding_constraints: string[];
  diagnostics: Record<string, unknown>;
  assumptions: Record<string, unknown>;
  scenarios: { current: ScenarioAnalysis | null; proposed: ScenarioAnalysis | null };
  disclaimer: string;
}

export interface AnalystSource {
  id: string;
  ticker: string;
  form_type: string;
  filing_date: string;
  section: string | null;
  accession_no: string;
  url: string;
  excerpt: string;
  retrieval: string[];
  score: number;
}

export interface AnalystStatement {
  text: string;
  type: "fact" | "inference";
  citations: string[];
  invalid_citations: string[];
  verification: "cited" | "unverified_numbers";
  unverified_numbers: string[];
}

export interface AnalystAnswer {
  question: string;
  status: "answered" | "insufficient_evidence" | "extractive";
  summary: string | null;
  statements: AnalystStatement[];
  rejected_statements: Array<{ text: string; reason: string }>;
  limitations?: string | null;
  sources: AnalystSource[];
  retrieval: { mode: string; tickers: string[] | null; tickers_inferred: boolean };
  model?: string;
  message: string | null;
  disclaimer: string;
  latency_ms: number;
}

export interface CoverageRow {
  ticker: string;
  form_type: string;
  filings: number;
  first: string;
  last: string;
  chunks: number;
}

export interface BacktestMetrics {
  start: string;
  end: string;
  years: number;
  total_return: number;
  cagr: number | null;
  volatility: number;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number;
  max_drawdown_duration_days: number;
  calmar: number | null;
  var_95_1d: number;
  cvar_95_1d: number;
  turnover_annualized: number;
  costs_paid: number;
  costs_pct_of_initial: number;
  trades: number;
  rebalances_executed: number;
  rebalances_skipped: number;
  positive_months: number | null;
  vs_strategy?: Record<string, number | null>;
}

export interface BacktestResult {
  config: Record<string, unknown>;
  config_hash: string;
  model: { id: number; version: string; alpha_confidence: number } | null;
  data_snapshot: { market_data_end: string; tickers: string[] };
  metrics: Record<string, BacktestMetrics>;
  equity_curves: Record<string, Array<{ date: string; value: number }>>;
  drawdown: Array<{ date: string; value: number }>;
  trades: Array<{ date: string; ticker: string; side: string; shares: number; price: number; notional: number; commission: number }>;
  trades_total: number;
  rebalances: Array<{ date: string; weights: Record<string, number> | null; turnover?: number; status?: string; message?: string; skipped?: string }>;
  assumptions: Record<string, unknown>;
  known_biases: string[];
}

export interface BacktestRun {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  config: Record<string, unknown>;
  config_hash: string;
  code_version?: string;
  result?: BacktestResult | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  summary?: BacktestMetrics | null;
}

export interface DatasetFreshness {
  dataset: string;
  description: string;
  latest: string | null;
  business_days_behind: number | null;
  max_allowed_lag: number;
  status: "fresh" | "stale" | "missing";
}

export interface PipelineRun {
  pipeline: string;
  task: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  rows_processed: number;
  error: string | null;
  warnings: string[] | null;
  duration?: string | null;
}

export interface SystemStatus {
  status: "healthy" | "degraded" | "unhealthy";
  checked_at: string;
  last_completed_session: string;
  datasets: DatasetFreshness[];
  pipelines: PipelineRun[];
  failed_runs_24h: number;
  sec_index: { filings_by_status: Record<string, number>; chunks: number; embedded_chunks: number };
}

export interface MonitoringMetric {
  as_of_date: string;
  metric: string;
  value: number | null;
  details: Record<string, unknown>;
  horizon_days: number;
  version: string;
}
