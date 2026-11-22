"use client";

import { useState } from "react";

import {
    predictLightGBM,
    type LoadTier,
    type PredictResponse,
} from "@/routes/UploadApi";

type Props = {
    tier: LoadTier;
    tierLabel: string;
    tierBadge: string;
    sampleHint: string;
    file: File | null;
    onFileChange: (file: File | null) => void;
    onResult: (data: PredictResponse) => void;
};

export default function Upload({
    tier,
    tierLabel,
    tierBadge,
    sampleHint,
    file,
    onFileChange,
    onResult,
}: Props) {
    const [loading, setLoading] = useState(false);

    const handleUpload = async () => {
        if (!file) return;

        try {
            setLoading(true);
            onResult(await predictLightGBM(file, tier));
        } catch (err) {
            console.error(err);
            alert(err instanceof Error ? err.message : "Upload failed!");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full shrink-0 lg:max-w-xl">
            <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)]">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                        Model: {tierLabel}
                    </p>
                    <span className="rounded-full bg-teal-50 px-2.5 py-0.5 text-[11px] font-medium text-teal-800 ring-1 ring-teal-600/15">
                        {tierBadge}
                    </span>
                </div>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
                    <label className="relative flex flex-1 cursor-pointer flex-col justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50/80 px-4 py-3 transition hover:border-teal-400/60 hover:bg-teal-50/40">
                        <span className="text-xs font-medium text-slate-600">
                            File CSV (A = 168h lịch sử + B = 24h thực tế)
                        </span>
                        <span className="mt-1 truncate text-sm font-medium text-slate-900">
                            {file ? file.name : "Chọn file CSV để dự báo…"}
                        </span>
                        <input
                            type="file"
                            accept=".csv"
                            className="absolute inset-0 cursor-pointer opacity-0"
                            onChange={(e) =>
                                onFileChange(e.target.files?.[0] ?? null)
                            }
                        />
                    </label>

                    <button
                        type="button"
                        onClick={handleUpload}
                        disabled={loading || !file}
                        className="inline-flex items-center justify-center rounded-xl bg-linear-to-br from-teal-600 to-teal-700 px-6 py-3 text-sm font-semibold text-white shadow-md shadow-teal-900/15 transition hover:from-teal-500 hover:to-teal-600 disabled:cursor-not-allowed disabled:opacity-45"
                    >
                        {loading ? (
                            <span className="flex items-center gap-2">
                                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                                Đang dự báo…
                            </span>
                        ) : (
                            "Dự báo 24 giờ"
                        )}
                    </button>
                </div>

                <p className="mt-2.5 text-[11px] text-slate-500">
                    Gợi ý file mẫu:{" "}
                    <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-700">
                        {sampleHint}
                    </code>
                </p>
            </div>
        </div>
    );
}
