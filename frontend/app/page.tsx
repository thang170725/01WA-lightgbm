"use client";

import { useState } from "react";
import Upload from "@/src/components/Upload";
import Chart from "@/src/components/Chart";
import Table from "@/src/components/Table";
import DailyChart from "@/src/components/DailyChart";
import DailyTable from "@/src/components/DailyTable";
import Metrics from "@/src/components/Metrics";
import { toDailyForecast } from "@/src/lib/daily";
import type { PredictResponse } from "@/routes/UploadApi";

type View = "hourly" | "daily";

const VIEWS: { id: View; label: string; description: string }[] = [
    { id: "hourly", label: "Next 24 hours", description: "Hourly forecast" },
    { id: "daily", label: "Next 7 days", description: "Daily forecast" },
];

function formatTime(iso: string) {
    return iso.replace("T", " ").slice(0, 16);
}

export default function Home() {
    const [prediction, setPrediction] = useState<PredictResponse | null>(null);
    const [view, setView] = useState<View>("hourly");

    const hourly = prediction?.result.slice(0, 24) ?? [];
    const daily = toDailyForecast(prediction?.result ?? []);
    const shown = view === "hourly" ? hourly : (prediction?.result ?? []);

    return (
        <main className="page-surface min-h-screen">
            <div className="mx-auto max-w-350 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
                <header className="mb-8 flex flex-col gap-6 lg:mb-10 lg:flex-row lg:items-end lg:justify-between">
                    <div className="max-w-xl">
                        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-teal-700/90">
                            LightGBM · hourly &amp; daily forecast
                        </p>
                        <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                            Electricity load prediction
                        </h1>
                        <p className="mt-3 text-base leading-relaxed text-slate-600">
                            Upload historical CSV (<code className="rounded bg-slate-200/80 px-1.5 py-0.5 text-sm text-slate-800">time</code>,{" "}
                            <code className="rounded bg-slate-200/80 px-1.5 py-0.5 text-sm text-slate-800">value</code>, at least 168 hourly rows) to get the next 24h and 7-day forecast and model metrics.
                        </p>
                    </div>
                    <Upload
                        onResult={(data) => {
                            setPrediction(data);
                            setView("hourly");
                        }}
                    />
                </header>

                {prediction && hourly.length > 0 ? (
                    <div className="space-y-6">
                        <Metrics evaluate={prediction.evaluate} />

                        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div
                                className="grid grid-cols-2 gap-2 rounded-xl bg-slate-100/80 p-1 sm:w-96"
                                role="group"
                                aria-label="Select forecast horizon"
                            >
                                {VIEWS.map((v) => (
                                    <button
                                        key={v.id}
                                        type="button"
                                        onClick={() => setView(v.id)}
                                        className={[
                                            "rounded-lg px-3 py-2.5 text-left transition",
                                            view === v.id
                                                ? "bg-white text-teal-900 shadow-sm ring-1 ring-teal-600/20"
                                                : "text-slate-600 hover:text-slate-900",
                                        ].join(" ")}
                                    >
                                        <span className="block text-sm font-semibold">
                                            {v.label}
                                        </span>
                                        <span className="mt-0.5 block text-[10px] text-slate-500">
                                            {v.description}
                                        </span>
                                    </button>
                                ))}
                            </div>

                            <div className="inline-flex items-center gap-2 self-start rounded-full bg-teal-50 px-4 py-1.5 text-sm font-medium text-teal-800 ring-1 ring-teal-600/15 sm:self-auto">
                                <span className="h-2 w-2 rounded-full bg-current opacity-70" />
                                {formatTime(shown[0].time)} → {formatTime(shown[shown.length - 1].time)}
                            </div>
                        </div>

                        <div className="grid gap-6 lg:grid-cols-12 lg:items-start">
                            <section className="lg:col-span-8">
                                {view === "hourly" ? (
                                    <Chart data={hourly} />
                                ) : (
                                    <DailyChart data={daily} />
                                )}
                            </section>
                            <aside className="flex flex-col gap-6 lg:col-span-4">
                                {view === "hourly" ? (
                                    <Table data={hourly} />
                                ) : (
                                    <DailyTable data={daily} />
                                )}
                            </aside>
                        </div>
                    </div>
                ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300/80 bg-white/60 px-6 py-12 text-center backdrop-blur-sm">
                        <p className="text-sm font-medium text-slate-700">
                            No forecast yet
                        </p>
                        <p className="mt-2 text-sm text-slate-500">
                            Choose a CSV file and run LightGBM to see the 24-hour and 7-day forecast.
                        </p>
                    </div>
                )}
            </div>
        </main>
    );
}
