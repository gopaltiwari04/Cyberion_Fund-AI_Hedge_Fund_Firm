"use client";

import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const COLORS = ["#76c7a0", "#d7ad6d", "#7786a8", "#ba7d73"];
const axis = { tick: { fill: "#849092", fontSize: 10 }, tickLine: false, axisLine: false } as const;
const tooltip = { contentStyle: { background: "#141a1d", border: "1px solid #273034", color: "#e8eceb", fontSize: 12 }, labelStyle: { color: "#849092" } };

export function EquityChart({ curves, labels }: { curves: Record<string, Array<{ date: string; value: number }>>; labels: Record<string, string> }) {
  const names = Object.keys(curves);
  const byDate = new Map<string, Record<string, number | string>>();
  names.forEach((name) => curves[name].forEach((point) => {
    const row = byDate.get(point.date) ?? { date: point.date };
    row[name] = point.value;
    byDate.set(point.date, row);
  }));
  const data = [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  return (
    <div className="chart-box" aria-label="Equity curves">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#273034" strokeDasharray="2 6" vertical={false} />
          <XAxis dataKey="date" {...axis} minTickGap={48} />
          <YAxis {...axis} width={70} domain={["auto", "auto"]} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
          <Tooltip {...tooltip} formatter={(value, name) => [typeof value === "number" ? `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "N/A", labels[String(name)] ?? String(name)]} />
          <Legend formatter={(value) => labels[String(value)] ?? value} wrapperStyle={{ fontSize: 11 }} />
          {names.map((name, index) => <Line key={name} type="monotone" dataKey={name} stroke={COLORS[index % COLORS.length]} strokeWidth={index === 0 ? 2 : 1.4} dot={false} connectNulls />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DrawdownChart({ data }: { data: Array<{ date: string; value: number }> }) {
  return (
    <div className="chart-box chart-small" aria-label="Drawdown">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="#273034" strokeDasharray="2 6" vertical={false} />
          <XAxis dataKey="date" {...axis} minTickGap={48} />
          <YAxis {...axis} width={50} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip {...tooltip} formatter={(value) => [typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "N/A", "Drawdown"]} />
          <Area type="monotone" dataKey="value" stroke="#d47d75" fill="#d47d7533" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
