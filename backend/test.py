from backend.app.utils.preprocessor import Preprocessor
from backend.app.models.lightgbm import LightGBMTrainer, LightGBMTrainerV2, LinearTrainerV2
from backend.app.models.model_manager import ModelManager
from backend.app.utils.data_checker import DataCheckerTXT

# 1. check data
# Khởi tạo bộ kiểm tra dữ liệu gốc (.txt) và chạy toàn bộ các bước kiểm tra chất lượng
data_checker = DataCheckerTXT()
data_checker.run_all()

# 3. preprcess data
# Input : dataset_path (str) - Đường dẫn file CSV điện năng theo giờ
# Output: (train, valid, test) - Lưu 3 file train_ready3.csv, valid_ready3.csv, test_ready3.csv
preprocessor = Preprocessor(dataset_path="backend/dataset/hourly_electricity_filtered_v2.csv")
preprocessor.run_pipeline()

# 4. trainer
# Huấn luyện mô hình 1 (LightGBM) và mô hình 2 (Linear / Ridge)
# Output của train_model(): tuple (model, evaluate) gồm đối tượng model đã train và dict các chỉ số đánh giá
trainer = LightGBMTrainerV2(
    train_path="backend/dataset/train_ready3.csv",
    valid_path="backend/dataset/valid_ready3.csv",
    test_path="backend/dataset/test_ready3.csv"
)
model, evaluate = trainer.train_model()

linear_trainer = LinearTrainerV2(
    train_path="backend/dataset/train_ready3.csv",
    valid_path="backend/dataset/valid_ready3.csv",
    test_path="backend/dataset/test_ready3.csv"
)
linear_model, linear_evaluate = linear_trainer.train_model(save=True)

# 5. load model
# Lưu và kiểm tra nạp lại cả 2 model (.pkl) thông qua ModelManager
model_manager = ModelManager(model_dir='backend/models')
save = model_manager.save_model(
    model,
    evaluate,
    save_path='lightgbm_model_v2.pkl'
)
save_linear = model_manager.save_model(
    linear_model,
    linear_evaluate,
    save_path='linear_v2.pkl'
)
loader = model_manager.load_model(name='lightgbm_model_v2.pkl')
print(loader)
loader_linear = model_manager.load_model(name='linear_v2.pkl')
print(loader_linear)

import pandas as pd

df = pd.read_csv(
    "backend/dataset/LD2011_2014.txt",
    sep=";",
    decimal=",",
    parse_dates=[0]
)

print(df.head(5))
