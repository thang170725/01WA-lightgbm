# Chạy toàn bộ pipeline: tiền xử lý -> train LightGBM V4 (mean + dải 80%)
# Dùng:  python -m backend.test
from backend.preprocessing_services.preprocessor import Preprocessor
from backend.app.models.lightgbm import LightGBMTrainerV4

Preprocessor("backend/dataset/hourly_electricity_2013_2014.csv").run_pipeline("backend/dataset/processed_v4")
LightGBMTrainerV4().train_model()
