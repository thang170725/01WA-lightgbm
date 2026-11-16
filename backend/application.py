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
MODEL_FILE = "lightgbm_v4.pkl"

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
lgb_evaluate = bundle["evaluate"]


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
        history = series.tolist()
        start_time = series.index[-1] + pd.Timedelta(hours=1)

        forecast_df = forecast_v4(bundle, history, start_time, steps)
        forecast_df["time"] = forecast_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "result": forecast_df[["time", "value", "lo", "hi"]].round(4).to_dict(orient="records"),
            "evaluate": lgb_evaluate,
            "band": "80%",
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
