"use client";

import { useState } from "react";
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

import type { ComparisonPoint } from "@/routes/UploadApi";

function formatTick(iso: string, totalPoints: number) {
    if (totalPoints <= 24) {
        return iso.slice(11, 16);
    }
    return `${iso.slice(8, 10)}/${iso.slice(5, 7)} ${iso.slice(11, 13)}h`;
}

function computeStats(points: ComparisonPoint[]) {
    if (points.length === 0) {
        return { mae: 0, rmse: 0, wape: 0, mape: 0 };
    }
    let absSum = 0;
    let sqSum = 0;
    let actualSum = 0;
    let apeSum = 0;
    let apeCount = 0;

    for (const p of points) {
        const err = Math.abs(p.predicted - p.actual);
        absSum += err;
        sqSum += err * err;
        actualSum += Math.abs(p.actual);
        if (Math.abs(p.actual) > 1e-6) {
            apeSum += err / Math.abs(p.actual);
            apeCount += 1;
        }
    }

    const n = points.length;
    return {
        mae: absSum / n,
        rmse: Math.sqrt(sqSum / n),
        wape: (absSum / Math.max(actualSum, 1e-9)) * 100,
        mape: apeCount > 0 ? (apeSum / apeCount) * 100 : 0,
    };
}

export default function ComparisonChart({
    data,
    historyHours = 168,
}: {
    data: ComparisonPoint[];
    historyHours?: number;
}) {
    const hasMultiDay = data.length > 24;
    const [horizon, setHorizon] = useState<"24h" | "all">("24h");

    if (!data || data.length === 0) {
        return (
            <div className="rounded-2xl border border-dashed border-slate-300/80 bg-white/70 p-6 text-center shadow-xs">
                <h2 className="text-base font-semibold text-slate-800">
                    AI Prediction (C) vs Actual Value (B)
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                    Để xem biểu đồ đối chiếu thực tế, hãy tải lên file CSV có nhiều hơn 168 giờ (ví dụ:{" "}
                    <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-700">
                        A (168h lịch sử) + B (24h–168h thực tế)
                    </code>
                    ).
                </p>
            </div>
        );
    }

    const shown = horizon === "24h" ? data.slice(0, 24) : data;
    const stats = computeStats(shown);

    const allValues = shown.flatMap((d) => [d.actual, d.predicted]);
    const minV = Math.min(...allValues);
    const maxV = Math.max(...allValues);
    const pad = (maxV - minV) * 0.1 || 1;

    return (
        <div className="flex flex-col rounded-2xl border border-slate-200/90 bg-white p-5 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)] sm:p-6">
            <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                    <div className="flex flex-wrap items-center gap-2">
                        <h2 className="text-lg font-semibold tracking-tight text-slate-900">
                            So sánh Dự đoán AI (C) vs Giá trị thực tế (B)
                        </h2>
                        <span className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700 ring-1 ring-indigo-600/15">
                            Input A: {historyHours}h → Predict C vs Actual B ({shown.length}h)
                        </span>
                    </div>
                    <p className="mt-1 text-sm text-slate-500">
                        Cắt {historyHours} giờ đầu (A) đưa vào mô hình LightGBM để sinh dự đoán (C), sau đó đối chiếu trực tiếp với dữ liệu thật (B)
                    </p>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                    <div className="flex items-center gap-3 rounded-xl bg-slate-50 px-3.5 py-2 text-xs ring-1 ring-slate-200/80">
                        <div>
                            <span className="text-slate-500">MAE: </span>
                            <span className="font-mono font-semibold text-slate-900">
                                {stats.mae.toFixed(2)} kWh
                            </span>
                        </div>
                        <span className="text-slate-300">|</span>
                        <div>
                            <span className="text-slate-500">RMSE: </span>
                            <span className="font-mono font-semibold text-slate-900">
                                {stats.rmse.toFixed(2)} kWh
                            </span>
                        </div>
                        <span className="text-slate-300">|</span>
                        <div>
                            <span className="text-slate-500">WAPE: </span>
                            <span className="font-mono font-semibold text-teal-700">
                                {stats.wape.toFixed(2)}%
                            </span>
                        </div>
                    </div>

                    {hasMultiDay && (
                        <div className="inline-flex rounded-xl bg-slate-100 p-1">
                            <button
                                type="button"
                                onClick={() => setHorizon("24h")}
                                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                                    horizon === "24h"
                                        ? "bg-white text-teal-900 shadow-xs ring-1 ring-teal-600/20"
                                        : "text-slate-600 hover:text-slate-900"
                                }`}
                            >
                                24 giờ (1 ngày)
                            </button>
                            <button
                                type="button"
                                onClick={() => setHorizon("all")}
                                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                                    horizon === "all"
                                        ? "bg-white text-teal-900 shadow-xs ring-1 ring-teal-600/20"
                                        : "text-slate-600 hover:text-slate-900"
                                }`}
                            >
                                {data.length} giờ ({Math.round(data.length / 24)} ngày)
                            </button>
                        </div>
                    )}
                </div>
            </div>

            <div className="h-95 w-full sm:h-[420px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                        data={shown}
                        margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
                    >
                        <CartesianGrid
                            strokeDasharray="3 6"
                            stroke="#e2e8f0"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="time"
                            tickFormatter={(iso) => formatTick(iso, shown.length)}
                            tick={{ fill: "#64748b", fontSize: 11 }}
                            tickLine={false}
                            axisLine={{ stroke: "#e2e8f0" }}
                            dy={6}
                            minTickGap={24}
                        />
                        <YAxis
                            domain={[Math.max(0, minV - pad), maxV + pad]}
                            tick={{ fill: "#64748b", fontSize: 11 }}
                            tickLine={false}
                            axisLine={false}
                            width={48}
                            tickFormatter={(v) => v.toFixed(0)}
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
                            formatter={(value, name) => {
                                const v =
                                    typeof value === "number"
                                        ? `${value.toFixed(2)} kWh`
                                        : String(value ?? "—");
                                const label =
                                    name === "actual"
                                        ? "Thực tế (B - Actual)"
                                        : "Dự đoán AI (C - Predicted)";
                                return [v, label];
                            }}
                        />
                        <Legend
                            verticalAlign="top"
                            height={36}
                            formatter={(value) =>
                                value === "actual"
                                    ? "Giá trị thực tế (B - Actual)"
                                    : "Dự đoán của AI (C - Predicted)"
                            }
                        />
                        <Line
                            type="monotone"
                            dataKey="actual"
                            name="actual"
                            stroke="#6366f1"
                            strokeWidth={2.5}
                            dot={shown.length <= 24 ? { r: 3, fill: "#6366f1" } : false}
                            activeDot={{
                                r: 5,
                                fill: "#6366f1",
                                stroke: "#fff",
                                strokeWidth: 2,
                            }}
                        />
                        <Line
                            type="monotone"
                            dataKey="predicted"
                            name="predicted"
                            stroke="#0d9488"
                            strokeWidth={2.5}
                            strokeDasharray="6 3"
                            dot={shown.length <= 24 ? { r: 3, fill: "#0d9488" } : false}
                            activeDot={{
                                r: 5,
                                fill: "#0d9488",
                                stroke: "#fff",
                                strokeWidth: 2,
                            }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
