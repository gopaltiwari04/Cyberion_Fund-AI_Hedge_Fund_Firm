export function formatPercent(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "N/A" : `${(value * 100).toFixed(2)}%`;
}

export function formatNumber(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "N/A" : value.toFixed(2);
}

export function formatDate(value?: string) {
  if (!value) return "No date available";
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}