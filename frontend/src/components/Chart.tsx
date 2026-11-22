"use client";

import {
    Area,
    CartesianGrid,
    ComposedChart,
    Legend,
    Line,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

import type { ComparisonMetrics, ForecastPoint } from "@/routes/UploadApi";

function formatHour(iso: string) {
    return iso.slice(11, 16);
}

export default function Chart({
    data,
    comparisonMetrics,
    tierTitle,
}: {
    data: ForecastPoint[];
    comparisonMetrics?: ComparisonMetrics | null;
    tierTitle?: string;
}) {
    const hasActual = data.some((d) => d.actual != null);
    const rows = data.map((d) => ({
        ...d,
        range: [d.lo, d.hi] as [number, number],
    }));

    const allValues = data.flatMap((d) =>
        d.actual != null ? [d.lo, d.hi, d.value, d.actual] : [d.lo, d.hi, d.value]
    );
    const minV = Math.min(...allValues);
    const maxV = Math.max(...allValues);
    const pad = (maxV - minV) * 0.08 || 1;

    return (
        <div className="flex h-115 flex-col rounded-2xl border border-slate-200/90 bg-white p-5 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)] sm:h-[500px] sm:p-6">
            <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                    <div className="flex flex-wrap items-center gap-2">
                        <h2 className="text-lg font-semibold tracking-tight text-slate-900">
                            {hasActual
                                ? "Dự báo 24 giờ tới & Đối chiếu Thực tế (AI vs Thực tế)"
                                : "Dự báo phụ tải 24 giờ tới"}
                        </h2>
                        {tierTitle && (
                            <span className="rounded-full bg-teal-50 px-2.5 py-0.5 text-xs font-medium text-teal-800 ring-1 ring-teal-700/15">
                                {tierTitle}
                            </span>
                        )}
                    </div>
                    <p className="mt-1 text-sm text-slate-500">
                        {hasActual
                            ? "Đường nét đứt màu xanh: Dự đoán AI (C) · Đường nét liền màu tím: Giá trị thực tế (B) · Vùng mờ: Dải tin cậy 80%"
                            : "Đường dự báo phụ tải theo giờ kèm dải tin cậy 80% (phân vị 10% – 90%)"}
                    </p>
                </div>

                {hasActual && comparisonMetrics ? (
                    <div className="flex flex-wrap items-center gap-2.5 rounded-xl bg-slate-50 px-3.5 py-2 text-xs ring-1 ring-slate-200/80">
                        <div>
                            <span className="text-slate-500">MAE (24h): </span>
                            <span className="font-mono font-semibold text-slate-900">
                                {comparisonMetrics.mae.toFixed(2)} kWh
                            </span>
                        </div>
                        <span className="text-slate-300">|</span>
                        <div>
                            <span className="text-slate-500">RMSE: </span>
                            <span className="font-mono font-semibold text-slate-900">
                                {comparisonMetrics.rmse.toFixed(2)} kWh
                            </span>
                        </div>
                        <span className="text-slate-300">|</span>
                        <div>
                            <span className="text-slate-500">Sai số WAPE: </span>
                            <span className="font-mono font-semibold text-teal-700">
                                {comparisonMetrics.wape.toFixed(2)}%
                            </span>
                        </div>
                    </div>
                ) : (
                    <div className="rounded-lg bg-teal-50 px-3 py-1.5 text-xs font-medium text-teal-800 ring-1 ring-teal-700/10">
                        {data.length} giờ
                    </div>
                )}
            </div>

            <div className="min-h-0 flex-1">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                        data={rows}
                        margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
                    >
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
                            domain={[Math.max(0, minV - pad), maxV + pad]}
                            tick={{ fill: "#64748b", fontSize: 11 }}
                            tickLine={false}
                            axisLine={false}
                            width={52}
                            tickFormatter={(v) => v.toFixed(1)}
                        />
                        <Tooltip
                            contentStyle={{
                                borderRadius: "12px",
                                border: "1px solid #e2e8f0",
                                boxShadow:
                                    "0 10px 40px -10px rgba(15, 23, 42, 0.15)",
                            }}
                            labelFormatter={(label, payload) => {
                                const timeStr =
                                    typeof label === "string"
                                        ? label.replace("T", " ").slice(0, 16)
                                        : String(label);
                                const point = payload?.[0]?.payload as
                                    | ForecastPoint
                                    | undefined;
                                if (point && point.actual != null) {
                                    const diff = point.value - point.actual;
                                    const pct =
                                        point.actual > 1e-6
                                            ? (diff / point.actual) * 100
                                            : 0;
                                    const sign = diff >= 0 ? "+" : "";
                                    return `${timeStr} · Lệch: ${sign}${diff.toFixed(2)} kWh (${sign}${pct.toFixed(1)}%)`;
                                }
                                return timeStr;
                            }}
                            formatter={(value, name) => {
                                const v = value as number | string | undefined;
                                if (Array.isArray(v)) {
                                    return [
                                        `${Number(v[0]).toFixed(2)} – ${Number(v[1]).toFixed(2)} kWh`,
                                        "Dải dự báo 80% (lo–hi)",
                                    ];
                                }
                                const text =
                                    typeof v === "number"
                                        ? `${v.toFixed(2)} kWh`
                                        : v != null
                                          ? String(v)
                                          : "—";
                                if (name === "actual") {
                                    return [text, "Thực tế (B - Actual)"];
                                }
                                if (name === "value") {
                                    return [text, "Dự đoán AI (C - Predicted)"];
                                }
                                return [text, String(name)];
                            }}
                        />
                        <Legend
                            verticalAlign="top"
                            height={36}
                            formatter={(value) => {
                                if (value === "actual")
                                    return "Giá trị thực tế (B - Actual)";
                                if (value === "value")
                                    return "Dự đoán của AI (C - Predicted)";
                                return "Dải tin cậy 80% (lo–hi)";
                            }}
                        />
                        <Area
                            type="monotone"
                            dataKey="range"
                            name="range"
                            stroke="none"
                            fill="rgb(13, 148, 136)"
                            fillOpacity={0.14}
                            dot={false}
                            activeDot={false}
                            isAnimationActive={false}
                        />
                        {hasActual && (
                            <Line
                                type="monotone"
                                dataKey="actual"
                                name="actual"
                                stroke="#6366f1"
                                strokeWidth={2.5}
                                dot={{ r: 3, fill: "#6366f1" }}
                                activeDot={{
                                    r: 5,
                                    fill: "#6366f1",
                                    stroke: "#fff",
                                    strokeWidth: 2,
                                }}
                            />
                        )}
                        <Line
                            type="monotone"
                            dataKey="value"
                            name="value"
                            stroke="rgb(13, 148, 136)"
                            strokeWidth={2.5}
                            strokeDasharray={hasActual ? "6 3" : undefined}
                            dot={{ r: 3, fill: "rgb(13, 148, 136)" }}
                            activeDot={{
                                r: 5,
                                fill: "rgb(13, 148, 136)",
                                stroke: "#fff",
                                strokeWidth: 2,
                            }}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
