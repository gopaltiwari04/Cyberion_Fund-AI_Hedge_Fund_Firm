export function formatInteger(value: number | null | undefined) {
  return value == null ? "N/A" : value.toLocaleString("en-US");
}

export function formatMetric(value: number | null | undefined, digits: number) {
  return value == null || !Number.isFinite(value) ? "N/A" : value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "N/A" : `${(value * 100).toFixed(2)}%`;
}
