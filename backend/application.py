from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from app.models.lightgbm import LightGBMTrainerV2, LinearTrainerV2
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
lightgbm_trainer = LightGBMTrainerV2()
# linear_trainer = LinearTrainerV2()

lgb_loader = manager.load_model(name="lightgbm_model_v2.pkl")
lgb_model = lgb_loader["model"]
lgb_evaluate = lgb_loader["evaluate"]

# linear_loader = manager.load_model(name="linear_v2.pkl")
# linear_model = linear_loader["model"]
# linear_evaluate = linear_loader["evaluate"]


async def _predict_from_upload(file: UploadFile, trainer, model, evaluate: dict):
    contents = await file.read()

    df = pd.read_csv(
        io.StringIO(contents.decode("utf-8")),
        parse_dates=["time"],
    )
    df = df.sort_values("time")

    history = df["value"].tolist()

    if len(history) < 168:
        return {"error": "Need at least 168 rows"}

    start_time = df["time"].iloc[-1] + pd.Timedelta(hours=1)

    forecast_df = trainer.forecast_future(
        model=model,
        history=history,
        start_time=start_time,
        steps=24,
    )

    forecast_df = forecast_df.rename(columns={"prediction": "value"})
    result = forecast_df[["time", "value"]].copy()
    result = result.where(pd.notnull(result), None)

    return {
        "result": result.to_dict(orient="records"),
        "evaluate": evaluate,
    }


@app.post("/predict_lightgbm")
async def predict_lightgbm(file: UploadFile = File(...)):
    try:
        return await _predict_from_upload(
            file, lightgbm_trainer, lgb_model, lgb_evaluate
        )
    except Exception as e:
        return {"error": str(e)}


# @app.post("/predict_linear")
# async def predict_linear(file: UploadFile = File(...)):
#     try:
#         return await _predict_from_upload(
#             file, linear_trainer, linear_model, linear_evaluate
#         )
#     except Exception as e:
#         return {"error": str(e)}
