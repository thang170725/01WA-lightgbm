import io
import os
import sys

import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Đảm bảo import được 'app' và 'preprocessing_services' dù chạy từ root hay từ backend/
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.models.lightgbm import forecast_v4  # noqa: E402
from app.models.model_manager import ModelManager  # noqa: E402

MIN_HISTORY = 168  # 7 ngày
MAX_STEPS = 168    # tối đa 7 ngày dự báo
MODEL_FILE = "/home/thang/workspace/lightgbm/v0_0_1/result/r0_0_1/models/lightgbm_v4.pkl"

app = FastAPI(title="LightGBM electricity forecast")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp model LightGBM (trung bình + phân vị 10% / 90%) cùng bộ chỉ số đánh giá
manager = ModelManager(model_dir=os.path.join(_BACKEND_DIR, "models"))
bundle = manager.load_model(name=MODEL_FILE)
_raw_eval = bundle["evaluate"]
if "mae" not in _raw_eval and "test" in _raw_eval:
    _test_eval = {k.lower(): v for k, v in _raw_eval["test"].items()}
    lgb_evaluate = {
        **_test_eval,
        "train": {k.lower(): v for k, v in _raw_eval.get("train", {}).items()},
        "valid": {k.lower(): v for k, v in _raw_eval.get("valid", {}).items()},
    }
else:
    lgb_evaluate = _raw_eval


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_FILE}


@app.post("/predict_lightgbm")
async def predict_lightgbm(
    file: UploadFile = File(...),
    steps: int = Query(24, ge=1, le=MAX_STEPS, description="Số giờ dự báo (24 = 1 ngày, 168 = 7 ngày)"),
):
    # Input : CSV 2 cột time,value (>= 168 giờ liên tiếp); steps = số giờ cần dự báo
    # Output: {"result": [{time, value, lo, hi}], "evaluate": {...}, "band": "80%"}
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), parse_dates=["time"])
        df = df.sort_values("time").dropna(subset=["value"])
        if len(df) < MIN_HISTORY:
            return {"error": f"Need at least {MIN_HISTORY} rows"}
        if (df["value"] < 0).any():
            return {"error": "value must be non-negative"}

        # Chuẩn hoá về lưới giờ liên tục (giờ thiếu được nội suy) để lag đúng nghĩa
        series = df.set_index("time")["value"].astype("float64")
        series = series[~series.index.duplicated(keep="last")]
        series = series.resample("1h").mean().interpolate(limit_direction="both")

        # Nếu file CSV có A + B (> 168 dòng):
        #   - Cắt A (168 giờ lịch sử) đưa vào AI để dự đoán ra C
        #   - Lấy B (phần dữ liệu thực tế phía sau A, tối đa `steps` giờ) để đối chiếu C với B
        comparison = []
        comparison_metrics = None

        if len(series) > MIN_HISTORY:
            actual_steps = min(len(series) - MIN_HISTORY, steps)
            # Nếu file <= MIN_HISTORY + steps (vd: 7 ngày A + 1..7 ngày B), lấy 168 giờ đầu làm A
            # Nếu file dài hơn, lấy 168 giờ ngay trước đoạn B cuối cùng làm A
            split_idx = len(series) - actual_steps
            series_a = series.iloc[max(0, split_idx - MIN_HISTORY):split_idx]
            series_b = series.iloc[split_idx:split_idx + actual_steps]

            history = series_a.tolist()
            start_time = series_b.index[0]

            forecast_df = forecast_v4(bundle, history, start_time, steps)

            import numpy as np
            y_true = series_b.to_numpy(dtype="float64")
            y_pred = forecast_df["value"].iloc[:actual_steps].to_numpy(dtype="float64")
            mae_val = float(np.mean(np.abs(y_true - y_pred)))
            rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            wape_val = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-9) * 100.0)
            non_zero = y_true > 1e-6
            mape_val = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0) if non_zero.any() else 0.0

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
            history = series.tolist()
            start_time = series.index[-1] + pd.Timedelta(hours=1)
            forecast_df = forecast_v4(bundle, history, start_time, steps)

        forecast_df["time"] = forecast_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "result": forecast_df[["time", "value", "lo", "hi"]].round(4).to_dict(orient="records"),
            "comparison": comparison,
            "comparison_metrics": comparison_metrics,
            "evaluate": lgb_evaluate,
            "band": "80%",
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
