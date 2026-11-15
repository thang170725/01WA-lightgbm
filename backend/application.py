import io
import os

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from app.models.lightgbm import LightGBMTrainerV2
from app.models.model_manager import ModelManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager()

# API chỉ cần inference -> không load dataset train/valid/test
lightgbm_trainer = LightGBMTrainerV2(load_data=False)

# ưu tiên model v2, nếu chưa có thì dùng model v1
model_name = (
    "lightgbm_model_v2.pkl"
    if os.path.exists(os.path.join(manager.model_dir, "lightgbm_model_v2.pkl"))
    else "lightgbm_model.pkl"
)
lgb_loader = manager.load_model(name=model_name)
lgb_model = lgb_loader["model"]
lgb_evaluate = lgb_loader["evaluate"]

MIN_HISTORY = 168  # 7 ngày (lag_7d)
MAX_STEPS = 168  # tối đa 7 ngày dự báo


@app.post("/predict_lightgbm")
async def predict_lightgbm(
    file: UploadFile = File(...),
    steps: int = Query(24, ge=1, le=MAX_STEPS, description="Số giờ dự báo (24 = 1 ngày, 168 = 7 ngày)"),
):
    try:
        contents = await file.read()

        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            parse_dates=["time"],
        )
        df = df.sort_values("time")

        history = df["value"].tolist()

        if len(history) < MIN_HISTORY:
            return {"error": f"Need at least {MIN_HISTORY} rows"}

        start_time = df["time"].iloc[-1] + pd.Timedelta(hours=1)

        forecast_df = lightgbm_trainer.forecast_future(
            model=lgb_model,
            history=history,
            start_time=start_time,
            steps=steps,
        )

        forecast_df = forecast_df.rename(columns={"prediction": "value"})
        result = forecast_df[["time", "value"]].copy()
        result = result.where(pd.notnull(result), None)

        return {
            "result": result.to_dict(orient="records"),
            "evaluate": lgb_evaluate,
        }
    except Exception as e:
        return {"error": str(e)}
