export function formatPrice(value: number | null | undefined) {
  return value == null || !Number.isFinite(value)
    ? "N/A"
    : value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function formatPercent(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "N/A" : `${(value * 100).toFixed(2)}%`;
}
