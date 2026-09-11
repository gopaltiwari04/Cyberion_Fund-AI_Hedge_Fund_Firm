"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MarketHistoryPoint } from "../types";

export function MarketChart({ data }: { data: MarketHistoryPoint[] }) {
  const chartData = data.filter((point) => point.close != null);

  return (
    <div className="market-chart" aria-label="Historical closing price chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 12, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="#273034" strokeDasharray="2 6" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: "#849092", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            minTickGap={36}
            tickFormatter={(value: string) => value.slice(5)}
          />
          <YAxis
            tick={{ fill: "#849092", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={58}
            tickFormatter={(value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            domain={["auto", "auto"]}
          />
          <Tooltip
            contentStyle={{ background: "#141a1d", border: "1px solid #273034", color: "#e8eceb", fontSize: 12 }}
            labelStyle={{ color: "#849092", marginBottom: 4 }}
            labelFormatter={(value) => `Date: ${value}`}
            formatter={(value) => [typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "N/A", "Close"]}
          />
          <Line type="monotone" dataKey="close" stroke="#76c7a0" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: "#76c7a0", stroke: "#0a0d0f", strokeWidth: 2 }} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
