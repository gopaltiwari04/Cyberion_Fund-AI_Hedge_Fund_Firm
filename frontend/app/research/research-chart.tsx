"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ResearchPredictionValue } from "../types";

interface ChartPoint {
  index: number;
  label: string;
  actual: number | null;
  predicted: number | null;
}

function numericValue(value: ResearchPredictionValue | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function ResearchChart({ observations, actualColumn, predictedColumn }: { observations: Array<Record<string, ResearchPredictionValue>>; actualColumn: string; predictedColumn: string }) {
  const chartData: ChartPoint[] = observations
    .map((observation, index) => ({
      index,
      label: typeof observation.date === "string" ? observation.date : `Observation ${index + 1}`,
      actual: numericValue(observation[actualColumn]),
      predicted: numericValue(observation[predictedColumn]),
    }))
    .filter((point) => point.actual != null || point.predicted != null)
    .reverse();

  return (
    <div className="research-chart" aria-label="Actual versus predicted out-of-sample returns chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 12, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="#273034" strokeDasharray="2 6" vertical={false} />
          <XAxis dataKey="label" tick={{ fill: "#849092", fontSize: 10 }} tickLine={false} axisLine={false} minTickGap={36} />
          <YAxis tick={{ fill: "#849092", fontSize: 10 }} tickLine={false} axisLine={false} width={58} tickFormatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
          <Tooltip
            contentStyle={{ background: "#141a1d", border: "1px solid #273034", color: "#e8eceb", fontSize: 12 }}
            labelStyle={{ color: "#849092", marginBottom: 4 }}
            formatter={(value, name) => [typeof value === "number" ? `${(value * 100).toFixed(3)}%` : "N/A", name === "actual" ? "Actual return" : "Predicted return"]}
          />
          <Legend wrapperStyle={{ color: "#849092", fontSize: 11 }} />
          <Line type="monotone" dataKey="actual" name="Actual" stroke="#d7ad6d" strokeWidth={1.5} dot={false} connectNulls={false} />
          <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#76c7a0" strokeWidth={1.5} dot={false} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
