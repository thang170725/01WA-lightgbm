const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

// Dự báo 24 giờ tới
export const FORECAST_HOURS = 24;

export type LoadTier = "low" | "medium" | "high";

// value = dự báo AI (C); actual = giá trị thực tế (B) nếu file có A+B; lo / hi = dải dự báo 80%
export type ForecastPoint = {
    time: string;
    value: number;
    lo: number;
    hi: number;
    actual?: number | null;
};

export type ComparisonPoint = {
    time: string;
    actual: number;
    predicted: number;
    lo: number;
    hi: number;
};

export type ComparisonMetrics = {
    mae: number;
    rmse: number;
    wape: number;
    mape: number;
    hours: number;
    history_hours: number;
};

export type EvaluateMetrics = {
    mae: number;
    relative_mae?: number;
    mape?: number;
    rmse: number;
    r2: number;
    wape?: number;
    smape?: number;
    coverage_80?: number;
    band_width?: number;
    band_width_pct?: number;
    horizon?: string;
    train?: Partial<EvaluateMetrics>;
    valid?: Partial<EvaluateMetrics>;
};

export type PredictResponse = {
    tier: LoadTier;
    recommended_tier?: LoadTier;
    input_mean_kwh?: number;
    input_median_kwh?: number;
    result: ForecastPoint[];
    comparison?: ComparisonPoint[];
    comparison_metrics?: ComparisonMetrics | null;
    evaluate: EvaluateMetrics;
    band?: string;
    error?: string;
};

export async function predictLightGBM(
    file: File,
    tier: LoadTier = "low"
): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(
        `${API_BASE}/predict_lightgbm?tier=${tier}&steps=${FORECAST_HOURS}`,
        { method: "POST", body: formData }
    );

    const data: PredictResponse = await res.json();

    if (!res.ok || data.error) {
        throw new Error(data.error ?? "Request failed");
    }

    return data;
}
