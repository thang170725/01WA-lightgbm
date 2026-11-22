"use client";

import type { ForecastPoint } from "@/routes/UploadApi";

function formatTime(iso: string) {
    return iso.replace("T", " ").slice(0, 16);
}

export default function Table({ data }: { data: ForecastPoint[] }) {
    const hasActual = data.some((d) => d.actual != null);

    return (
        <div className="flex max-h-[min(500px,70vh)] flex-col overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)]">
            <div className="border-b border-slate-100 bg-linear-to-r from-slate-50 to-teal-50/40 px-5 py-4">
                <h2 className="text-base font-semibold text-slate-900">
                    {hasActual
                        ? "Bảng đối chiếu chi tiết 24 giờ"
                        : "Chi tiết dự báo 24 giờ"}
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                    {hasActual
                        ? "So sánh Thực tế (B) vs Dự đoán AI (C) và Độ lệch (AI − Thực tế)"
                        : "Giá trị dự báo theo giờ (kWh) và dải tin cậy 80%"}
                </p>
            </div>

            <div className="min-h-0 flex-1 overflow-auto">
                <table className="w-full min-w-[340px] border-collapse text-sm">
                    <thead className="sticky top-0 z-10 bg-white/95 shadow-[0_1px_0_0_rgb(226_232_240)] backdrop-blur-sm">
                        <tr className="text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                            <th className="px-4 py-3">Thời gian</th>
                            {hasActual ? (
                                <>
                                    <th className="px-3 py-3 text-right text-indigo-700">
                                        Thực tế (B)
                                    </th>
                                    <th className="px-3 py-3 text-right text-teal-700">
                                        Dự đoán (C)
                                    </th>
                                    <th className="px-4 py-3 text-right">
                                        Độ lệch
                                    </th>
                                </>
                            ) : (
                                <>
                                    <th className="px-3 py-3 text-right">Low</th>
                                    <th className="px-3 py-3 text-right">
                                        Dự đoán
                                    </th>
                                    <th className="px-4 py-3 text-right">
                                        High
                                    </th>
                                </>
                            )}
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((row, idx) => {
                            const diff =
                                row.actual != null
                                    ? row.value - row.actual
                                    : null;
                            return (
                                <tr
                                    key={`${row.time}-${idx}`}
                                    className="border-b border-slate-100 transition-colors hover:bg-teal-50/50"
                                >
                                    <td className="whitespace-nowrap px-4 py-2.5 font-mono text-xs text-slate-700">
                                        {formatTime(row.time)}
                                    </td>
                                    {hasActual && row.actual != null ? (
                                        <>
                                            <td className="px-3 py-2.5 text-right font-mono text-sm font-medium tabular-nums text-indigo-700">
                                                {row.actual.toFixed(2)}
                                            </td>
                                            <td className="px-3 py-2.5 text-right font-mono text-sm font-semibold tabular-nums text-teal-800">
                                                {row.value.toFixed(2)}
                                            </td>
                                            <td
                                                className={`px-4 py-2.5 text-right font-mono text-xs font-medium tabular-nums ${
                                                    diff != null &&
                                                    Math.abs(diff) /
                                                        Math.max(
                                                            row.actual,
                                                            1
                                                        ) >
                                                        0.15
                                                        ? "text-amber-600"
                                                        : "text-slate-600"
                                                }`}
                                            >
                                                {diff != null
                                                    ? `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}`
                                                    : "—"}
                                            </td>
                                        </>
                                    ) : (
                                        <>
                                            <td className="px-3 py-2.5 text-right font-mono text-xs tabular-nums text-slate-500">
                                                {row.lo.toFixed(2)}
                                            </td>
                                            <td className="px-3 py-2.5 text-right font-mono text-sm font-medium tabular-nums text-slate-900">
                                                {row.value.toFixed(2)}
                                            </td>
                                            <td className="px-4 py-2.5 text-right font-mono text-xs tabular-nums text-slate-500">
                                                {row.hi.toFixed(2)}
                                            </td>
                                        </>
                                    )}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
