"use client";

import {
    Area,
    AreaChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import type { ForecastPoint } from "@/routes/UploadApi";

function formatHour(iso: string) {
    return iso.slice(11, 16);
}

export default function Chart({ data }: { data: ForecastPoint[] }) {
    const rows = data.map((d) => ({ ...d, range: [d.lo, d.hi] as [number, number] }));
    const minV = Math.min(...data.map((d) => d.lo));
    const maxV = Math.max(...data.map((d) => d.hi));
    const pad = (maxV - minV) * 0.08 || 0.1;

    return (
        <div className="flex h-105 flex-col rounded-2xl border border-slate-200/90 bg-white p-5 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)] sm:h-[460px] sm:p-6">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
                <div>
                    <h2 className="text-lg font-semibold tracking-tight text-slate-900">
                        Next 24 hours
                    </h2>
                    <p className="mt-1 text-sm text-slate-500">
                        Predicted load (hourly) with 80% prediction band
                    </p>
                </div>
                <div className="rounded-lg bg-teal-50 px-3 py-1.5 text-xs font-medium text-teal-800 ring-1 ring-teal-700/10">
                    {data.length} points
                </div>
            </div>

            <div className="min-h-0 flex-1">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                        data={rows}
                        margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                    >
                        <defs>
                            <linearGradient
                                id="forecastFill"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                            >
                                <stop
                                    offset="0%"
                                    stopColor="rgb(13, 148, 136)"
                                    stopOpacity={0.35}
                                />
                                <stop
                                    offset="100%"
                                    stopColor="rgb(13, 148, 136)"
                                    stopOpacity={0}
                                />
                            </linearGradient>
                        </defs>
                        <CartesianGrid
                            strokeDasharray="3 6"
                            stroke="#e2e8f0"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="time"
                            tickFormatter={formatHour}
                            tick={{ fill: "#64748b", fontSize: 11 }}
                            tickLine={false}
                            axisLine={{ stroke: "#e2e8f0" }}
                            dy={6}
                        />
                        <YAxis
                            domain={[minV - pad, maxV + pad]}
                            tick={{ fill: "#64748b", fontSize: 11 }}
                            tickLine={false}
                            axisLine={false}
                            width={44}
                            tickFormatter={(v) => v.toFixed(1)}
                        />
                        <Tooltip
                            contentStyle={{
                                borderRadius: "12px",
                                border: "1px solid #e2e8f0",
                                boxShadow:
                                    "0 10px 40px -10px rgba(15, 23, 42, 0.15)",
                            }}
                            labelFormatter={(label) =>
                                typeof label === "string"
                                    ? label.replace("T", " ").slice(0, 16)
                                    : String(label)
                            }
                            formatter={(value) => {
                                const v = value as number | string | undefined;
                                const text = Array.isArray(v)
                                    ? `${Number(v[0]).toFixed(2)} – ${Number(v[1]).toFixed(2)}`
                                    : typeof v === "number"
                                      ? v.toFixed(2)
                                      : v != null
                                        ? String(v)
                                        : "—";
                                return [text, Array.isArray(v) ? "80% band (lo–hi)" : "Predicted"];
                            }}
                        />
                        <Area
                            type="monotone"
                            dataKey="range"
                            name="80% band"
                            stroke="none"
                            fill="rgb(13, 148, 136)"
                            fillOpacity={0.15}
                            dot={false}
                            activeDot={false}
                            isAnimationActive={false}
                        />
                        <Area
                            type="monotone"
                            dataKey="value"
                            name="Prediction"
                            stroke="rgb(13, 148, 136)"
                            strokeWidth={2.5}
                            fill="transparent"
                            dot={false}
                            activeDot={{
                                r: 5,
                                fill: "rgb(13, 148, 136)",
                                stroke: "#fff",
                                strokeWidth: 2,
                            }}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}