"use client";

import { useState } from "react";
import Upload from "@/src/components/Upload";
import Chart from "@/src/components/Chart";
import Table from "@/src/components/Table";
import Metrics from "@/src/components/Metrics";
import {
    predictLightGBM,
    type LoadTier,
    type PredictResponse,
} from "@/routes/UploadApi";

type TierConfig = {
    id: LoadTier;
    title: string;
    subtitle: string;
    threshold: string;
    sampleHint: string;
};

const TIERS: TierConfig[] = [
    {
        id: "low",
        title: "Phụ tải thấp",
        subtitle: "Hộ gia đình",
        threshold: "Median ≤ 91.4 kWh/h",
        sampleHint: "user_low_1_MT_032.csv, user_low_2_MT_261.csv",
    },
    {
        id: "medium",
        title: "Phụ tải trung bình",
        subtitle: "Cụm hộ gia đình, văn phòng nhỏ & vừa",
        threshold: "91.4 < Median ≤ 248.2 kWh/h",
        sampleHint: "user_medium_1_MT_172.csv, user_medium_2_MT_311.csv",
    },
    {
        id: "high",
        title: "Phụ tải cao",
        subtitle: "Trung tâm thương mại, văn phòng, công ty lớn",
        threshold: "Median > 248.2 kWh/h",
        sampleHint: "user_high_1_MT_355.csv, user_high_2_MT_231.csv",
    },
];

function formatTime(iso: string) {
    return iso.replace("T", " ").slice(0, 16);
}

export default function Home() {
    const [activeTier, setActiveTier] = useState<LoadTier>("low");
    const [filesByTier, setFilesByTier] = useState<Record<LoadTier, File | null>>({
        low: null,
        medium: null,
        high: null,
    });
    const [predictionsByTier, setPredictionsByTier] = useState<
        Record<LoadTier, PredictResponse | null>
    >({
        low: null,
        medium: null,
        high: null,
    });

    const currentTier = TIERS.find((t) => t.id === activeTier) ?? TIERS[0];
    const prediction = predictionsByTier[activeTier];
    const hourly = prediction?.result.slice(0, 24) ?? [];

    const handleTierSwitch = async (nextTier: LoadTier) => {
        setActiveTier(nextTier);
        const existingFile = filesByTier[nextTier];
        if (existingFile && !predictionsByTier[nextTier]) {
            try {
                const res = await predictLightGBM(existingFile, nextTier);
                setPredictionsByTier((prev) => ({ ...prev, [nextTier]: res }));
            } catch (err) {
                console.error(err);
            }
        }
    };

    const tierNameMap: Record<LoadTier, string> = {
        low: "Phụ tải thấp (Hộ gia đình)",
        medium: "Phụ tải trung bình (Cụm hộ / Văn phòng nhỏ & vừa)",
        high: "Phụ tải cao (TTTM / Công ty lớn)",
    };

    return (
        <main className="page-surface min-h-screen">
            <div className="mx-auto max-w-350 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
                <header className="mb-6 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                    <div className="max-w-2xl">
                        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-teal-700/90">
                            LightGBM · Dự báo phụ tải điện 24 giờ &amp; Đối chiếu thực tế
                        </p>
                        <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                            Dự báo phụ tải điện theo 3 phân tầng
                        </h1>
                        <p className="mt-3 text-base leading-relaxed text-slate-600">
                            Chọn phân tầng mô hình phù hợp và tải lên file CSV (
                            <code className="rounded bg-slate-200/80 px-1.5 py-0.5 text-sm text-slate-800">
                                time
                            </code>
                            ,{" "}
                            <code className="rounded bg-slate-200/80 px-1.5 py-0.5 text-sm text-slate-800">
                                value
                            </code>{" "}
                            gồm <strong>168 giờ lịch sử A</strong> +{" "}
                            <strong>24 giờ thực tế B</strong>) để dự báo 24 giờ tới (C) và vẽ trực tiếp 2 đường Thực tế vs Dự đoán trên cùng một biểu đồ.
                        </p>
                    </div>

                    <Upload
                        tier={activeTier}
                        tierLabel={`${currentTier.title} (${currentTier.subtitle})`}
                        tierBadge={currentTier.threshold}
                        sampleHint={currentTier.sampleHint}
                        file={filesByTier[activeTier]}
                        onFileChange={(f) =>
                            setFilesByTier((prev) => ({ ...prev, [activeTier]: f }))
                        }
                        onResult={(data) =>
                            setPredictionsByTier((prev) => ({
                                ...prev,
                                [activeTier]: data,
                            }))
                        }
                    />
                </header>

                {/* 3 Tabs chọn mô hình phân tầng phụ tải */}
                <div
                    className="mb-8 grid grid-cols-1 gap-3 rounded-2xl bg-slate-200/70 p-1.5 sm:grid-cols-3"
                    role="tablist"
                    aria-label="Chọn phân tầng phụ tải"
                >
                    {TIERS.map((t) => {
                        const isActive = activeTier === t.id;
                        const hasResult = predictionsByTier[t.id] != null;
                        return (
                            <button
                                key={t.id}
                                type="button"
                                role="tab"
                                aria-selected={isActive}
                                onClick={() => handleTierSwitch(t.id)}
                                className={[
                                    "flex flex-col justify-between rounded-xl px-4 py-3.5 text-left transition",
                                    isActive
                                        ? "bg-white text-teal-950 shadow-sm ring-1 ring-teal-600/25"
                                        : "text-slate-600 hover:bg-white/50 hover:text-slate-900",
                                ].join(" ")}
                            >
                                <div className="flex items-center justify-between gap-2">
                                    <span className="text-sm font-bold">
                                        {t.title}
                                    </span>
                                    <span
                                        className={`rounded-full px-2 py-0.5 font-mono text-[10px] font-medium ${
                                            isActive
                                                ? "bg-teal-50 text-teal-800 ring-1 ring-teal-600/20"
                                                : "bg-slate-200/80 text-slate-600"
                                        }`}
                                    >
                                        {t.threshold}
                                    </span>
                                </div>
                                <span className="mt-1 block text-xs text-slate-500">
                                    {t.subtitle}
                                </span>
                                {hasResult && (
                                    <span className="mt-2 inline-flex items-center gap-1.5 text-[11px] font-medium text-teal-700">
                                        <span className="h-1.5 w-1.5 rounded-full bg-teal-600" />
                                        Đã có kết quả dự báo
                                    </span>
                                )}
                            </button>
                        );
                    })}
                </div>

                {prediction && hourly.length > 0 ? (
                    <div className="space-y-6">
                        {prediction.recommended_tier &&
                            prediction.recommended_tier !== activeTier && (
                                <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300/80 bg-amber-50/90 px-4 py-3 text-sm text-amber-900">
                                    <div>
                                        <strong>Lưu ý phân tầng dữ liệu:</strong>{" "}
                                        File vừa tải lên có mức tiêu thụ trung vị{" "}
                                        <code className="font-mono font-semibold">
                                            {prediction.input_median_kwh} kWh/h
                                        </code>{" "}
                                        (trung bình{" "}
                                        <code className="font-mono font-semibold">
                                            {prediction.input_mean_kwh} kWh/h
                                        </code>
                                        ), phù hợp nhất với tab{" "}
                                        <strong>
                                            {tierNameMap[prediction.recommended_tier]}
                                        </strong>
                                        .
                                    </div>
                                    <button
                                        type="button"
                                        onClick={() => {
                                            const rec = prediction.recommended_tier!;
                                            const f = filesByTier[activeTier];
                                            if (f) {
                                                setFilesByTier((prev) => ({
                                                    ...prev,
                                                    [rec]: f,
                                                }));
                                            }
                                            setActiveTier(rec);
                                            if (f) {
                                                predictLightGBM(f, rec)
                                                    .then((res) =>
                                                        setPredictionsByTier(
                                                            (prev) => ({
                                                                ...prev,
                                                                [rec]: res,
                                                            })
                                                        )
                                                    )
                                                    .catch(console.error);
                                            }
                                        }}
                                        className="rounded-lg bg-amber-700 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-amber-800"
                                    >
                                        Chuyển sang tab {TIERS.find((x) => x.id === prediction.recommended_tier)?.title}
                                    </button>
                                </div>
                            )}

                        <Metrics evaluate={prediction.evaluate} />

                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <div className="inline-flex items-center gap-2 rounded-full bg-teal-50 px-4 py-1.5 text-sm font-medium text-teal-800 ring-1 ring-teal-600/15">
                                <span className="h-2 w-2 rounded-full bg-current opacity-70" />
                                Khung giờ dự báo (24h): {formatTime(hourly[0].time)} →{" "}
                                {formatTime(hourly[hourly.length - 1].time)}
                            </div>
                            {prediction.input_mean_kwh != null && (
                                <div className="text-xs text-slate-500">
                                    Phụ tải lịch sử (A = 168h): Trung bình{" "}
                                    <span className="font-mono font-semibold text-slate-800">
                                        {prediction.input_mean_kwh} kWh/h
                                    </span>{" "}
                                    · Trung vị{" "}
                                    <span className="font-mono font-semibold text-slate-800">
                                        {prediction.input_median_kwh} kWh/h
                                    </span>
                                </div>
                            )}
                        </div>

                        <div className="grid gap-6 lg:grid-cols-12 lg:items-start">
                            <section className="lg:col-span-8">
                                <Chart
                                    data={hourly}
                                    comparisonMetrics={prediction.comparison_metrics}
                                    tierTitle={`${currentTier.title} · ${currentTier.subtitle}`}
                                />
                            </section>
                            <aside className="flex flex-col gap-6 lg:col-span-4">
                                <Table data={hourly} />
                            </aside>
                        </div>
                    </div>
                ) : (
                    <div className="rounded-2xl border border-dashed border-slate-300/80 bg-white/60 px-6 py-12 text-center backdrop-blur-sm">
                        <p className="text-base font-semibold text-slate-800">
                            Chưa có dữ liệu dự báo cho tab {currentTier.title} ({currentTier.subtitle})
                        </p>
                        <p className="mt-2 text-sm text-slate-500">
                            Chọn một file CSV thuộc nhóm{" "}
                            <strong>{currentTier.title.toLowerCase()}</strong> ({currentTier.threshold}) và bấm{" "}
                            <strong>Dự báo 24 giờ</strong> để xem đồng thời đường dự đoán AI và đường thực tế trên cùng biểu đồ.
                        </p>
                        <p className="mt-3 text-xs text-slate-400">
                            File mẫu trong{" "}
                            <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-slate-600">
                                v0_0_1/data/0_0_1_1/sample_users/
                            </code>
                            : {currentTier.sampleHint}
                        </p>
                    </div>
                )}
            </div>
        </main>
    );
}
