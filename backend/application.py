import io
import os
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Đảm bảo import được 'app' và 'preprocessing_services' dù chạy từ root hay từ backend/
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.models.lightgbm import forecast_v4  # noqa: E402
from app.models.model_manager import ModelManager  # noqa: E402

MIN_HISTORY = 168  # 7 ngày lịch sử (A) để tính lag_7d
MAX_STEPS = 24     # Chỉ dự báo 24 giờ tới

_MODEL_BASE_DIR = "/home/thang/workspace/lightgbm/v0_0_1/result/r0_0_1_1"
MODEL_PATHS = {
    "low": f"{_MODEL_BASE_DIR}/r_low/models/lightgbm_low.pkl",
    "medium": f"{_MODEL_BASE_DIR}/r_medium/models/lightgbm_medium.pkl",
    "high": f"{_MODEL_BASE_DIR}/r_high/models/lightgbm_high.pkl",
}

# Ngưỡng phân tầng theo active_median_kwh từ pipeline phân tầng
TIER_LOW_MAX = 91.3738
TIER_MEDIUM_MAX = 248.2464


def _normalize_evaluate(raw_eval: dict) -> dict:
    if "mae" not in raw_eval and "test" in raw_eval:
        test_eval = {k.lower(): v for k, v in raw_eval["test"].items()}
        return {
            **test_eval,
            "train": {k.lower(): v for k, v in raw_eval.get("train", {}).items()},
            "valid": {k.lower(): v for k, v in raw_eval.get("valid", {}).items()},
        }
    return raw_eval


def _infer_tier(values: np.ndarray) -> str:
    active = values[values > 0]
    med = float(np.median(active)) if len(active) > 0 else float(np.median(values))
    if med <= TIER_LOW_MAX:
        return "low"
    if med <= TIER_MEDIUM_MAX:
        return "medium"
    return "high"


app = FastAPI(title="LightGBM electricity forecast (3 Load Tiers)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager(model_dir=os.path.join(_BACKEND_DIR, "models"))
BUNDLES: dict[str, dict] = {}
EVALUATES: dict[str, dict] = {}

for _tier, _path in MODEL_PATHS.items():
    _b = manager.load_model(name=_path)
    BUNDLES[_tier] = _b
    EVALUATES[_tier] = _normalize_evaluate(_b["evaluate"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tiers": list(MODEL_PATHS.keys()),
        "models": MODEL_PATHS,
    }


@app.post("/predict_lightgbm")
async def predict_lightgbm(
    file: UploadFile = File(...),
    tier: str = Query("low", pattern="^(low|medium|high)$", description="Mức phụ tải: low | medium | high"),
    steps: int = Query(24, ge=1, le=MAX_STEPS, description="Số giờ dự báo (24 giờ tới)"),
):
    # Input : CSV 2 cột time,value (>= 168 giờ liên tiếp); tier = low/medium/high; steps = 24
    # Output: {"result": [{time, value, lo, hi, actual?}], "comparison_metrics": {...}, "evaluate": {...}}
    try:
        if tier not in BUNDLES:
            return {"error": f"Invalid tier '{tier}'. Choose low, medium, or high."}

        bundle = BUNDLES[tier]
        lgb_evaluate = EVALUATES[tier]

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), parse_dates=["time"])
        df = df.sort_values("time").dropna(subset=["value"])
        if len(df) < MIN_HISTORY:
            return {"error": f"Cần tối thiểu {MIN_HISTORY} dòng dữ liệu theo giờ (7 ngày lịch sử)."}
        if (df["value"] < 0).any():
            return {"error": "Giá trị phụ tải (value) không được âm."}

        # Chuẩn hoá về lưới giờ liên tục
        series = df.set_index("time")["value"].astype("float64")
        series = series[~series.index.duplicated(keep="last")]
        series = series.resample("1h").mean().interpolate(limit_direction="both")

        comparison = []
        comparison_metrics = None

        if len(series) > MIN_HISTORY:
            actual_steps = min(len(series) - MIN_HISTORY, steps)
            # Lấy 168 giờ đầu làm phần A (lịch sử), và 24 giờ ngay tiếp theo làm phần B (thực tế)
            series_a = series.iloc[:MIN_HISTORY]
            series_b = series.iloc[MIN_HISTORY : MIN_HISTORY + actual_steps]

            history = series_a.tolist()
            start_time = series_b.index[0]

            forecast_df = forecast_v4(bundle, history, start_time, steps)

            y_true = series_b.to_numpy(dtype="float64")
            y_pred = forecast_df["value"].iloc[:actual_steps].to_numpy(dtype="float64")
            mae_val = float(np.mean(np.abs(y_true - y_pred)))
            rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            wape_val = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-9) * 100.0)
            non_zero = y_true > 1e-6
            mape_val = (
                float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
                if non_zero.any()
                else 0.0
            )

            # Gắn trực tiếp cột actual vào forecast_df để vẽ chung trên 1 biểu đồ chính
            actual_col = [None] * len(forecast_df)
            for i in range(actual_steps):
                actual_col[i] = round(float(y_true[i]), 4)
            forecast_df["actual"] = actual_col

            comparison_df = forecast_df.iloc[:actual_steps].copy()
            comparison_df["time"] = comparison_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            comparison_df["predicted"] = comparison_df["value"].round(4)
            comparison_df["actual"] = np.round(y_true, 4)
            comparison_df["lo"] = comparison_df["lo"].round(4)
            comparison_df["hi"] = comparison_df["hi"].round(4)
            comparison = comparison_df[["time", "actual", "predicted", "lo", "hi"]].to_dict(orient="records")
            comparison_metrics = {
                "mae": round(mae_val, 4),
                "rmse": round(rmse_val, 4),
                "wape": round(wape_val, 4),
                "mape": round(mape_val, 4),
                "hours": int(actual_steps),
                "history_hours": int(len(series_a)),
            }
        else:
            series_a = series
            history = series.tolist()
            start_time = series.index[-1] + pd.Timedelta(hours=1)
            forecast_df = forecast_v4(bundle, history, start_time, steps)
            forecast_df["actual"] = None

        a_vals = series_a.to_numpy(dtype="float64")
        input_mean = float(np.mean(a_vals))
        active_vals = a_vals[a_vals > 0]
        input_median = float(np.median(active_vals)) if len(active_vals) > 0 else float(np.median(a_vals))
        recommended_tier = _infer_tier(a_vals)

        forecast_df["time"] = forecast_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        for col in ("value", "lo", "hi"):
            forecast_df[col] = forecast_df[col].round(4)

        return {
            "tier": tier,
            "recommended_tier": recommended_tier,
            "input_mean_kwh": round(input_mean, 2),
            "input_median_kwh": round(input_median, 2),
            "result": forecast_df[["time", "value", "lo", "hi", "actual"]].to_dict(orient="records"),
            "comparison": comparison,
            "comparison_metrics": comparison_metrics,
            "evaluate": lgb_evaluate,
            "band": "80%",
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
