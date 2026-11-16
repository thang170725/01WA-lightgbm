const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

// 7 ngày x 24 giờ: 24 giờ đầu dùng cho dự báo theo giờ, toàn bộ dùng cho dự báo theo ngày
export const FORECAST_HOURS = 168;

// value = dự báo trung bình; lo / hi = dải dự báo 80% (phân vị 10% - 90%)
export type ForecastPoint = { time: string; value: number; lo: number; hi: number };

export type EvaluateMetrics = {
    mae: number;
    relative_mae?: number;
    mape: number;
    rmse: number;
    r2: number;
    wape?: number;
    smape?: number;
    coverage_80?: number; // % giá trị thực nằm trong dải 80% trên tập test
    band_width_pct?: number;
    horizon?: string;
    train?: Partial<EvaluateMetrics>;
    valid?: Partial<EvaluateMetrics>;
};

export type PredictResponse = {
    result: ForecastPoint[];
    evaluate: EvaluateMetrics;
    band?: string;
    error?: string;
};

export async function predictLightGBM(file: File): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(
        `${API_BASE}/predict_lightgbm?steps=${FORECAST_HOURS}`,
        { method: "POST", body: formData }
    );

    const data: PredictResponse = await res.json();

    if (!res.ok || data.error) {
        throw new Error(data.error ?? "Request failed");
    }

    return data;
}
