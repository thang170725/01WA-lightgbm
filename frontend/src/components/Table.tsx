"use client";

import type { ForecastPoint } from "@/routes/UploadApi";

function formatTime(iso: string) {
    return iso.replace("T", " ").slice(0, 19);
}

export default function Table({ data }: { data: ForecastPoint[] }) {
    return (
        <div className="flex max-h-[min(520px,70vh)] flex-col overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)]">
            <div className="border-b border-slate-100 bg-linear-to-r from-slate-50 to-teal-50/40 px-5 py-4">
                <h2 className="text-base font-semibold text-slate-900">
                    Hourly values
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                    Scroll for all predictions
                </p>
            </div>

            <div className="min-h-0 flex-1 overflow-auto">
                <table className="w-full min-w-[260px] border-collapse text-sm">
                    <thead className="sticky top-0 z-10 bg-white/95 shadow-[0_1px_0_0_rgb(226_232_240)] backdrop-blur-sm">
                        <tr className="text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                            <th className="px-5 py-3 pl-5">Time</th>
                            <th className="px-5 py-3 text-right">kW (pred.)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((row, idx) => (
                            <tr
                                key={`${row.time}-${idx}`}
                                className="border-b border-slate-100 transition-colors hover:bg-teal-50/50"
                            >
                                <td className="whitespace-nowrap px-5 py-2.5 font-mono text-xs text-slate-700">
                                    {formatTime(row.time)}
                                </td>
                                <td className="px-5 py-2.5 text-right font-mono text-sm font-medium tabular-nums text-slate-900">
                                    {row.value.toFixed(4)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}