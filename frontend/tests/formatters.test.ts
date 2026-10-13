import { describe, expect, it } from "vitest";
import { formatCurrency, formatPercent, formatSignedPercent, signalClass } from "../app/lib/formatters";

describe("formatters", () => {
  it("formats missing and non-finite values as N/A", () => {
    expect(formatPercent(null)).toBe("N/A");
    expect(formatPercent(Number.NaN)).toBe("N/A");
    expect(formatCurrency(undefined)).toBe("N/A");
  });

  it("formats percentages and signs", () => {
    expect(formatPercent(0.1234)).toBe("12.34%");
    expect(formatSignedPercent(0.01)).toBe("+1.00%");
    expect(formatSignedPercent(-0.015, 1)).toBe("-1.5%");
  });

  it("maps values to signal classes", () => {
    expect(signalClass(0.2)).toBe("signal-positive");
    expect(signalClass(-0.2)).toBe("signal-negative");
    expect(signalClass(0)).toBe("signal-neutral");
    expect(signalClass(null)).toBe("signal-neutral");
  });
});
