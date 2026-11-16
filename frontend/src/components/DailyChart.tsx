"use client";

import {
    Bar,
    ErrorBar,
    BarChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import type { DailyForecast } from "@/src/lib/daily";

export default function DailyChart({ data }: { data: DailyForecast[] }) {
    return (
        <div className="flex h-105 flex-col rounded-2xl border border-slate-200/90 bg-white p-5 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)] sm:h-[460px] sm:p-6">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
                <div>
                    <h2 className="text-lg font-semibold tracking-tight text-slate-900">
                        Next 7 days
                    </h2>
                    <p className="mt-1 text-sm text-slate-500">
                        Predicted daily consumption (sum of 24 hourly values); whiskers = 80% band
                    </p>
                </div>
                <div className="rounded-lg bg-teal-50 px-3 py-1.5 text-xs font-medium text-teal-800 ring-1 ring-teal-700/10">
                    {data.length} days
                </div>
            </div>

            <div className="min-h-0 flex-1">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                    >
                        <defs>
                            <linearGradient
                                id="dailyFill"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                            >
                                <stop
                                    offset="0%"
                                    stopColor="rgb(13, 148, 136)"
                                    stopOpacity={0.9}
                                />
                                <stop
                                    offset="100%"
                                    stopColor="rgb(13, 148, 136)"
                                    stopOpacity={0.45}
                                />
                            </linearGradient>
                        </defs>
                        <CartesianGrid
                            strokeDasharray="3 6"
                            stroke="#e2e8f0"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="label"
                            tick={{ fill: "#64748b", fontSize: 11 }}
                            tickLine={false}
                            axisLine={{ stroke: "#e2e8f0" }}
                            dy={6}
                        />
                        <YAxis
                            tick={{ fill: "#64748b", fontSize: 11 }}
                            tickLine={false}
                            axisLine={false}
                            width={52}
                            tickFormatter={(v) => v.toFixed(1)}
                        />
                        <Tooltip
                            cursor={{ fill: "rgba(13, 148, 136, 0.06)" }}
                            contentStyle={{
                                borderRadius: "12px",
                                border: "1px solid #e2e8f0",
                                boxShadow:
                                    "0 10px 40px -10px rgba(15, 23, 42, 0.15)",
                            }}
                            labelFormatter={(_, payload) => {
                                const d = payload?.[0]?.payload as
                                    | DailyForecast
                                    | undefined;
                                return d
                                    ? `${d.weekday} ${d.date} · peak ${d.peak.toFixed(2)} at ${d.peakTime} · 80% band ${d.totalLo.toFixed(0)}–${d.totalHi.toFixed(0)}`
                                    : "";
                            }}
                            formatter={(value) => {
                                const v = value as number | string | undefined;
                                const text =
                                    typeof v === "number"
                                        ? `${v.toFixed(2)} kWh`
                                        : v != null
                                          ? String(v)
                                          : "—";
                                return [text, "Total"];
                            }}
                        />
                        <Bar
                            dataKey="total"
                            name="Total"
                            fill="url(#dailyFill)"
                            radius={[8, 8, 0, 0]}
                            maxBarSize={56}
                        >
                            <ErrorBar dataKey="err" width={6} strokeWidth={1.5} stroke="#0f766e" />
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
