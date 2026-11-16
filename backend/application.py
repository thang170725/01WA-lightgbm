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

# Khởi tạo bộ quản lý model và 2 trainer cho LightGBM & Linear (Ridge)
manager = ModelManager()
lightgbm_trainer = LightGBMTrainerV2()
linear_trainer = LinearTrainerV2()

# Nạp mô hình LightGBM (Model 1) và bộ chỉ số đánh giá từ file .pkl
lgb_loader = manager.load_model(name="lightgbm_model_v2.pkl")
lgb_model = lgb_loader["model"]
lgb_evaluate = lgb_loader["evaluate"]

# Nạp mô hình Ridge Regression (Model 2) và bộ chỉ số đánh giá từ file .pkl
linear_loader = manager.load_model(name="linear_v2.pkl")
linear_model = linear_loader["model"]
linear_evaluate = linear_loader["evaluate"]


async def _predict_from_upload(file: UploadFile, trainer, model, evaluate: dict):
    # Input :
    #   - file (UploadFile) : File CSV người dùng upload lên (cần có 2 cột 'time', 'value' và >= 168 dòng)
    #   - trainer (object)  : Đối tượng LightGBMTrainerV2 hoặc LinearTrainerV2 dùng để gọi forecast_future
    #   - model (object)    : Mô hình học máy đã huấn luyện (lgb_model hoặc linear_model)
    #   - evaluate (dict)   : Bộ chỉ số đánh giá mô hình trên tập Test (mae, relative_mae, mape, rmse, r2)
    # Output:
    #   - dict              : Dictionary trả về cho Frontend gồm {"result": list 24 điểm dự báo, "evaluate": dict metrics}
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
    # Input : file (UploadFile) - File CSV lịch sử tiêu thụ điện gửi qua multipart/form-data
    # Output: dict - Kết quả dự báo 24 giờ tiếp theo và metrics của mô hình LightGBM
    try:
        return await _predict_from_upload(
            file, lightgbm_trainer, lgb_model, lgb_evaluate
        )
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_linear")
async def predict_linear(file: UploadFile = File(...)):
    # Input : file (UploadFile) - File CSV lịch sử tiêu thụ điện gửi qua multipart/form-data
    # Output: dict - Kết quả dự báo 24 giờ tiếp theo và metrics của mô hình Ridge Regression
    try:
        return await _predict_from_upload(
            file, linear_trainer, linear_model, linear_evaluate
        )
    except Exception as e:
        return {"error": str(e)}
