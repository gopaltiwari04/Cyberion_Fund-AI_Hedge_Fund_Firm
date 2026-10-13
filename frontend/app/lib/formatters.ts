export function formatPercent(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "N/A" : `${(value * 100).toFixed(digits)}%`;
}

export function formatSignedPercent(value: number | null | undefined, digits = 2) {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? "N/A" : value.toFixed(digits);
}

export function formatCurrency(value: number | null | undefined, digits = 0) {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function formatDate(value?: string | null) {
  if (!value) return "No date available";
  const date = new Date(value.length === 10 ? `${value}T00:00:00` : value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function formatDateTime(value?: string | null) {
  if (!value) return "N/A";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function signalClass(value: number | null | undefined) {
  if (value == null || value === 0 || !Number.isFinite(value)) return "signal-neutral";
  return value > 0 ? "signal-positive" : "signal-negative";
}
