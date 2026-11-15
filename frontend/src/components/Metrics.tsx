"use client";

import type { EvaluateMetrics } from "@/routes/UploadApi";

const accentRing = [
    "ring-indigo-500/15",
    "ring-teal-500/15",
    "ring-violet-500/15",
    "ring-amber-500/15",
    "ring-sky-500/15",
];

export default function Metrics({ evaluate }: { evaluate: EvaluateMetrics }) {
    const metrics: { title: string; subtitle?: string; value: string }[] = [
        {
            title: "MAE",
            subtitle: "Mean absolute error",
            value: evaluate.mae.toFixed(2),
        },
        ...(evaluate.relative_mae != null
            ? [
                  {
                      title: "Rel. MAE",
                      subtitle: "% of mean load",
                      value: `${evaluate.relative_mae.toFixed(2)}%`,
                  },
              ]
            : []),
        {
            title: "MAPE",
            subtitle: "Mean abs. % error",
            value: `${evaluate.mape.toFixed(2)}%`,
        },
        {
            title: "RMSE",
            subtitle: "Root mean squared",
            value: evaluate.rmse.toFixed(2),
        },
        {
            title: "R²",
            subtitle: "Coefficient of determination",
            value: evaluate.r2.toFixed(4),
        },
    ];

    return (
        <section>
            <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                    Model performance (test set)
                </h2>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                {metrics.map((m, i) => (
                    <div
                        key={m.title}
                        className={`rounded-2xl border border-slate-200/90 bg-white p-4 shadow-[0_2px_16px_-4px_rgba(15,23,42,0.06)] ring-1 ${accentRing[i % accentRing.length]}`}
                    >
                        <p className="text-xs font-semibold text-slate-800">
                            {m.title}
                        </p>
                        {m.subtitle && (
                            <p className="mt-0.5 text-[11px] leading-snug text-slate-500">
                                {m.subtitle}
                            </p>
                        )}
                        <p className="mt-3 font-mono text-xl font-bold tabular-nums tracking-tight text-slate-900 sm:text-2xl">
                            {m.value}
                        </p>
                    </div>
                ))}
            </div>
        </section>
    );
}