"use client";

import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const axis = { tick: { fill: "#849092", fontSize: 10 }, tickLine: false, axisLine: false } as const;
const tooltip = { contentStyle: { background: "#141a1d", border: "1px solid #273034", color: "#e8eceb", fontSize: 12 }, labelStyle: { color: "#849092" } };

export function CalibrationChart({ table }: { table: Array<{ decile: number; predicted: number; realized: number }> }) {
  return (
    <div className="chart-box chart-small" aria-label="Calibration by forecast decile">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={table} margin={{ top: 10, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#273034" strokeDasharray="2 6" vertical={false} />
          <XAxis dataKey="decile" {...axis} />
          <YAxis {...axis} width={52} tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
          <Tooltip {...tooltip} formatter={(value, name) => [typeof value === "number" ? `${(value * 100).toFixed(3)}%` : "N/A", name === "predicted" ? "Mean forecast" : "Mean realized"]} labelFormatter={(d) => `Forecast decile ${d}`} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="predicted" name="predicted" fill="#76c7a0" />
          <Bar dataKey="realized" name="realized" fill="#d7ad6d" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function IcByYearChart({ data }: { data: Record<string, number> }) {
  const rows = Object.entries(data).map(([year, ic]) => ({ year, ic }));
  return (
    <div className="chart-box chart-small" aria-label="Rank IC by year">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 10, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#273034" strokeDasharray="2 6" vertical={false} />
          <XAxis dataKey="year" {...axis} />
          <YAxis {...axis} width={46} />
          <Tooltip {...tooltip} formatter={(value) => [typeof value === "number" ? value.toFixed(4) : "N/A", "Mean daily rank IC"]} />
          <Bar dataKey="ic">{rows.map((row) => <Cell key={row.year} fill={row.ic >= 0 ? "#76c7a0" : "#d47d75"} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
