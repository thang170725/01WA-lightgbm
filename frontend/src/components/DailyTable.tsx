"use client";

import type { DailyForecast } from "@/src/lib/daily";

export default function DailyTable({ data }: { data: DailyForecast[] }) {
    return (
        <div className="flex max-h-[min(520px,70vh)] flex-col overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)]">
            <div className="border-b border-slate-100 bg-linear-to-r from-slate-50 to-teal-50/40 px-5 py-4">
                <h2 className="text-base font-semibold text-slate-900">
                    Daily values
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                    Each day = 24 consecutive forecast hours
                </p>
            </div>

            <div className="min-h-0 flex-1 overflow-auto">
                <table className="w-full min-w-[300px] border-collapse text-sm">
                    <thead className="sticky top-0 z-10 bg-white/95 shadow-[0_1px_0_0_rgb(226_232_240)] backdrop-blur-sm">
                        <tr className="text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                            <th className="px-5 py-3">Day</th>
                            <th className="px-3 py-3 text-right">Total (kWh)</th>
                            <th className="px-5 py-3 text-right">Peak</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((d) => (
                            <tr
                                key={d.start}
                                className="border-b border-slate-100 transition-colors hover:bg-teal-50/50"
                            >
                                <td className="whitespace-nowrap px-5 py-2.5">
                                    <span className="block text-sm font-medium text-slate-800">
                                        {d.weekday}
                                    </span>
                                    <span className="block font-mono text-[11px] text-slate-500">
                                        {d.start.slice(5, 16).replace("T", " ")} →{" "}
                                        {d.end.slice(11, 16)}
                                    </span>
                                </td>
                                <td className="px-3 py-2.5 text-right font-mono text-sm font-medium tabular-nums text-slate-900">
                                    {d.total.toFixed(2)}
                                    {d.hours < 24 && (
                                        <span className="block text-[10px] font-normal text-amber-600">
                                            {d.hours}h only
                                        </span>
                                    )}
                                </td>
                                <td className="whitespace-nowrap px-5 py-2.5 text-right">
                                    <span className="block font-mono text-sm tabular-nums text-slate-900">
                                        {d.peak.toFixed(2)}
                                    </span>
                                    <span className="block font-mono text-[11px] text-slate-500">
                                        at {d.peakTime}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
