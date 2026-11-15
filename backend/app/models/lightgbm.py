import os
import sys

# Tránh xung đột tên file 'lightgbm.py' với thư viện cài đặt 'lightgbm' khi chạy trực tiếp bằng lệnh python backend/app/models/lightgbm.py
_current_dir = os.path.abspath(os.path.dirname(__file__))
if _current_dir in [os.path.abspath(p) for p in sys.path]:
    sys.path = [p for p in sys.path if os.path.abspath(p) != _current_dir]

# Đảm bảo import được package 'app' dù chạy từ thư mục gốc dự án hay từ thư mục 'backend/'
sys.path.insert(0, os.path.abspath(os.path.join(_current_dir, "../..")))

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from app.models.feature_builder import FeatureBuilder
from app.models.model_manager import ModelManager
from sklearn.linear_model import LinearRegression, Ridge


def _resolve_csv_path(path: str) -> str:
    # Input : path (str) - Đường dẫn file CSV truyền vào (vd: "dataset/train_ready3.csv" hoặc "backend/dataset/train_ready3.csv")
    # Output: str - Đường dẫn thực tế tồn tại trên hệ thống tệp
    if os.path.exists(path):
        return path
    alt_with_backend = os.path.join("backend", path)
    if os.path.exists(alt_with_backend):
        return alt_with_backend
    if path.startswith("backend/") or path.startswith("backend\\"):
        alt_without_backend = path[len("backend/"):]
        if os.path.exists(alt_without_backend):
            return alt_without_backend
    return path


LAG_AND_ROLLING_COLS = [
    "lag_24h", "lag_48h", "lag_72h", "lag_7d",
    "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"
]


def _sync_log1p_features(df: pd.DataFrame) -> pd.DataFrame:
    # Input : df (pd.DataFrame) - Bảng dữ liệu Train / Valid / Test đọc từ file CSV
    # Output: pd.DataFrame - Bảng dữ liệu đã được đồng bộ biến đổi np.log1p cho 7 cột lag & rolling (khớp với FeatureBuilder khi dự báo trên Web)
    df = df.copy()
    if "lag_24h" in df.columns and df["lag_24h"].max() > 25.0:
        for col in LAG_AND_ROLLING_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0)).astype("float32")
    return df


# ==== model ai/ml lightgbm ====
# Lớp huấn luyện và dự báo phụ tải điện bằng mô hình LightGBM (Phiên bản 1)
class LightGBMTrainer:
    def __init__(self,
        train_path="dataset/train_ready3.csv",  # Input 1: đường dẫn file CSV tập huấn luyện (Train)
        valid_path="dataset/valid_ready3.csv",  # Input 2: đường dẫn file CSV tập kiểm định (Valid)
        test_path="dataset/test_ready3.csv"     # Input 3: đường dẫn file CSV tập kiểm tra (Test)
    ):
        # Input :
        #   - train_path (str): Đường dẫn tới file CSV tập Train
        #   - valid_path (str): Đường dẫn tới file CSV tập Valid
        #   - test_path  (str): Đường dẫn tới file CSV tập Test
        # Output:
        #   - None: Khởi tạo các DataFrame (train, valid, test), danh sách 11 features và cột target
        print("Loading datasets (train, valid, test)...")

        self.train = _sync_log1p_features(pd.read_csv(_resolve_csv_path(train_path)))
        self.valid = _sync_log1p_features(pd.read_csv(_resolve_csv_path(valid_path)))
        self.test = _sync_log1p_features(pd.read_csv(_resolve_csv_path(test_path)))

        # LightGBMTrainer
        # Danh sách 11 cột đặc trưng đầu vào (4 đặc trưng thời gian + 4 độ trễ lag + 3 thống kê trượt rolling)
        self.features = [
            "hour_sin", "hour_cos", "day_of_week", "is_weekend",
            "lag_24h", "lag_48h", "lag_72h", "lag_7d",
            "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"
        ]
        self.target = "target_value"  # Cột giá trị mục tiêu cần dự đoán (đang ở thang đo log1p)

    # data
    def get_data(self):
        # Input : Không có tham số truyền vào (sử dụng self.train, self.valid, self.test, self.features, self.target)
        # Output: Tuple 6 phần tử (X_train, y_train, X_valid, y_valid, X_test, y_test)
        #         trong đó X_* là DataFrame (N, 11) chứa đặc trưng, y_* là Series (N,) chứa nhãn mục tiêu
        print("\nPreparing data...")

        steps = ["Train", "Valid", "Test"]
        data = []

        # tqdm cho quá trình split
        for step in tqdm(steps, desc="Splitting data"):
            if step == "Train":
                data.append((self.train[self.features], self.train[self.target]))
            elif step == "Valid":
                data.append((self.valid[self.features], self.valid[self.target]))
            else:
                data.append((self.test[self.features], self.test[self.target]))

        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data

        return (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        )

    # model
    def build_model(self):
        # Input : Không có tham số truyền vào
        # Output: model (lgb.LGBMRegressor) - Đối tượng mô hình LightGBM đã cấu hình hyperparameters
        # === version 2 ===
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=3000,          # tăng từ 300
            learning_rate=0.005,         # giảm từ 0.05 (kết hợp với early stopping)
            num_leaves=127,              # tăng từ 31 (2^depth - 1)
            max_depth=8,                # thêm mới, kiểm soát overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,       # thêm mới, tránh overfit trên node lá
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        return model

    # metrics
    def evaluate(self,
        y_true,  # Input 1: mảng/Series giá trị thực tế trên tập Test (thang log1p)
        preds    # Input 2: mảng/Series giá trị mô hình dự đoán trên tập Test (thang log1p)
    ):
        # Input :
        #   - y_true (np.ndarray | pd.Series): Nhãn thực tế ở thang log1p
        #   - preds  (np.ndarray | pd.Series): Nhãn dự đoán ở thang log1p
        # Output:
        #   - dict: Từ điển gồm 5 chỉ số đánh giá {"mae", "relative_mae", "mape", "rmse", "r2"} ở đơn vị gốc (kWh)
        # inverse log1p
        y_true = np.expm1(y_true)  # Chuyển ngược từ log(1+y) về giá trị thực tế kWh
        preds = np.expm1(preds)    # Chuyển ngược giá trị dự đoán về kWh

        mae = mean_absolute_error(y_true, preds)
        relative_mae = (
        mae / np.mean(y_true)
        ) * 100

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                preds
            )
        )
        mape = np.mean(
            np.abs(
                (
                    y_true-preds
                )/y_true
            )
        ) * 100
        r2 = r2_score(y_true, preds)

        print(f"MAE: {mae:.4f}")
        print(f"Relative MAE: {relative_mae:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R2 SCORE: {r2:.2f}%")

        return {
            "mae": mae,
            "relative_mae": relative_mae,
            "mape": mape,
            "rmse": rmse,
            "r2": r2
        }

    # train
    def train_model(self):
        # Input : Không có tham số truyền vào
        # Output: Tuple (model, evaluate) gồm:
        #   - model (lgb.LGBMRegressor): Mô hình LightGBM đã huấn luyện tại best_iteration_
        #   - evaluate (dict)          : Dictionary chứa 5 chỉ số đánh giá trên tập Test
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        ) = self.get_data()

        model = self.build_model()

        print("\nTraining model...")

        # tqdm wrapper (LightGBM không expose progress trực tiếp)
        for _ in tqdm(range(1), desc="Fitting LightGBM"):
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l2",
                callbacks=[
                    lgb.log_evaluation(period=100),  # in log mỗi 50 dòng,
                    lgb.early_stopping(stopping_rounds=50)
                ]
            )

        print(f"Best iteration: {model.best_iteration_}")

        print("\nPredicting...")
        preds = model.predict(X_test)

        print("\n=== Test Metrics ===")
        evaluate = self.evaluate(y_test, preds)
        print("evaluate: ", evaluate)

        importance = pd.DataFrame({
            "feature": self.features,
            "importance": model.feature_importances_
        }).sort_values(
            "importance",
            ascending=False
        )

        print("\nFeature importance:\n", importance)

        return model, evaluate

    # ==== predict new data ====
    def preprocess_input(self, input_dict):
        """
        input_dict: dữ liệu user nhập dạng dict
        ví dụ:
        {
            "hour_sin": 0.5,
            "hour_cos": 0.86,
            "day_of_week": 2,
            "lag_24h": 120,
            "lag_7d": 130,
            "rolling_mean_24h": 125
        }
        """
        # Input : input_dict (dict) - Từ điển chứa 11 đặc trưng của 1 mốc thời gian
        # Output: df (pd.DataFrame) - DataFrame 1 dòng, 11 cột sắp xếp đúng thứ tự self.features

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[self.features]

        return df


    def predict_new(self, model, input_dict):
        """
        predict cho 1 sample mới
        """
        # Input :
        #   - model (lgb.LGBMRegressor): Mô hình đã huấn luyện
        #   - input_dict (dict)        : Từ điển chứa 11 đặc trưng của thời điểm cần dự đoán
        # Output:
        #   - preds[0] (float)         : Giá trị điện năng dự báo (kWh) sau khi biến đổi ngược np.expm1

        X_new = self.preprocess_input(input_dict)

        preds = model.predict(X_new)

        # inverse log1p (giống evaluate)
        preds = np.expm1(preds)

        return preds[0]
    
    def forecast_future(self, model, history, start_time, steps):
        """
        history: list giá trị quá khứ (>=168)
        start_time: thời điểm bắt đầu dự đoán
        steps: số bước tương lai (vd: 24 giờ)
        """
        # Input :
        #   - model (lgb.LGBMRegressor)     : Mô hình đã huấn luyện
        #   - history (list[float])         : Danh sách giá trị tiêu thụ lịch sử (>= 168 giờ)
        #   - start_time (str | pd.Timestamp): Mốc thời gian bắt đầu dự đoán
        #   - steps (int)                   : Số giờ tương lai cần dự báo (vd: 24)
        # Output:
        #   - pd.DataFrame                  : Bảng kết quả gồm 2 cột ["time", "prediction"] với `steps` dòng

        results = []
        history = history.copy()

        current_time = pd.to_datetime(start_time)

        for i in range(steps):
            # build feature từ history hiện tại
            features = FeatureBuilder().build_features(
                time_str=current_time,
                history=history
            )

            pred = self.predict_new(model, features)

            results.append({
                "time": current_time,
                "prediction": float(pred)
            })

            # 🔥 cực quan trọng: append prediction vào history
            history.append(pred)

            # tăng thời gian (1 giờ)
            current_time += pd.Timedelta(hours=1)

        return pd.DataFrame(results)

# ==== LIGHTGBM TRAINER ====
# Lớp huấn luyện & dự báo chính cho MÔ HÌNH 1: LightGBM Regressor (Version 2)
class LightGBMTrainerV2:
    def __init__(self,
        train_path="dataset/train_ready3.csv",  # Input 1: đường dẫn file CSV tập Train
        valid_path="dataset/valid_ready3.csv",  # Input 2: đường dẫn file CSV tập Valid
        test_path="dataset/test_ready3.csv"     # Input 3: đường dẫn file CSV tập Test
    ):
        # Input :
        #   - train_path (str): Đường dẫn file CSV tập huấn luyện (mặc định "dataset/train_ready3.csv")
        #   - valid_path (str): Đường dẫn file CSV tập kiểm định (mặc định "dataset/valid_ready3.csv")
        #   - test_path  (str): Đường dẫn file CSV tập kiểm tra (mặc định "dataset/test_ready3.csv")
        # Output:
        #   - None: Khởi tạo đối tượng LightGBMTrainerV2, đọc 3 tập dữ liệu và cấu hình 11 features + target
        print("Loading datasets (train, valid, test)...")

        self.train = _sync_log1p_features(pd.read_csv(_resolve_csv_path(train_path)))
        self.valid = _sync_log1p_features(pd.read_csv(_resolve_csv_path(valid_path)))
        self.test = _sync_log1p_features(pd.read_csv(_resolve_csv_path(test_path)))

        # Danh sách 11 cột đặc trưng đầu vào dùng để huấn luyện và dự báo
        self.features = [
            "hour_sin",         # Thành phần sin của giờ trong ngày (chu kỳ 24h)
            "hour_cos",         # Thành phần cos của giờ trong ngày (chu kỳ 24h)
            "day_of_week",      # Thứ trong tuần (0: Thứ 2 -> 6: Chủ Nhật)
            "is_weekend",       # Cờ đánh dấu cuối tuần (1: Thứ 7/CN, 0: Ngày thường)

            "lag_24h",          # Độ trễ 24 giờ (1 ngày trước cùng giờ)
            "lag_48h",          # Độ trễ 48 giờ (2 ngày trước cùng giờ)
            "lag_72h",          # Độ trễ 72 giờ (3 ngày trước cùng giờ)
            "lag_7d",           # Độ trễ 168 giờ (7 ngày trước cùng giờ)

            "rolling_mean_24h", # Trung bình trượt 24 giờ gần nhất
            "rolling_mean_7d",  # Trung bình trượt 7 ngày (168 giờ) gần nhất
            "rolling_std_24h"   # Độ lệch chuẩn trượt 24 giờ gần nhất
        ]

        self.target = "target_value"  # Cột mục tiêu dự đoán (ở thang đo log1p)

    # ==== data ====
    def get_data(self):
        # Input : Không có tham số truyền vào (sử dụng self.train, self.valid, self.test)
        # Output: Tuple 6 phần tử (X_train, y_train, X_valid, y_valid, X_test, y_test)
        #         - X_train, X_valid, X_test (pd.DataFrame): Ma trận đặc trưng 11 cột của 3 tập
        #         - y_train, y_valid, y_test (pd.Series)   : Vector giá trị mục tiêu của 3 tập
        print("\nPreparing data...")

        steps = ["Train", "Valid", "Test"]

        data = []

        # tqdm cho quá trình split
        for step in tqdm(steps, desc="Splitting data"):
            if step == "Train":
                data.append(
                    (
                        self.train[self.features],
                        self.train[self.target]
                    )
                )

            elif step == "Valid":
                data.append(
                    (
                        self.valid[self.features],
                        self.valid[self.target]
                    )
                )

            else:
                data.append(
                    (
                        self.test[self.features],
                        self.test[self.target]
                    )
                )

        (
            (X_train, y_train),
            (X_valid, y_valid),
            (X_test, y_test)
        ) = data

        return (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        )

    # ==== model ====
    def build_model(self):
        # Input : Không có tham số truyền vào
        # Output: model (lgb.LGBMRegressor) - Mô hình LightGBM Regressor đã thiết lập bộ siêu tham số
        # === version 2 ===
        model = lgb.LGBMRegressor(
            objective="regression",     # Bài toán hồi quy (tối ưu L2 loss)
            n_estimators=3000,          # Số lượng cây tối đa
            learning_rate=0.005,        # Tốc độ học nhỏ kết hợp early stopping
            num_leaves=127,             # Số nút lá tối đa trên 1 cây
            max_depth=8,                # Độ sâu tối đa của cây quyết định
            subsample=0.8,              # Tỷ lệ lấy mẫu dữ liệu cho mỗi cây
            colsample_bytree=0.8,       # Tỷ lệ lấy mẫu đặc trưng cho mỗi cây
            min_child_samples=20,       # Số mẫu tối thiểu trên 1 lá
            random_state=42,            # Cố định seed để tái lập kết quả
            n_jobs=-1,                  # Sử dụng toàn bộ nhân CPU
            verbosity=-1                # Ẩn cảnh báo nội bộ của LightGBM
        )

        return model

    # ==== metrics ====
    def evaluate(
        self,
        y_true,  # Input 1: giá trị thực tế trên tập Test (thang log1p)
        preds    # Input 2: giá trị dự đoán của mô hình trên tập Test (thang log1p)
    ):
        # Input :
        #   - y_true (np.ndarray | pd.Series): Nhãn thực tế trên tập Test (ở thang đo log1p)
        #   - preds  (np.ndarray | pd.Series): Nhãn dự báo trên tập Test (ở thang đo log1p)
        # Output:
        #   - dict: Dictionary chứa 5 chỉ số đánh giá kiểu float {"mae", "relative_mae", "mape", "rmse", "r2"}
        # inverse log1p
        y_true = np.expm1(y_true)  # Đưa nhãn thực tế từ thang log1p về đơn vị kWh gốc

        preds = np.expm1(preds)    # Đưa giá trị dự đoán từ thang log1p về đơn vị kWh gốc

        mae = mean_absolute_error(y_true, preds)  # Sai số tuyệt đối trung bình (kWh)

        relative_mae = (
            mae / np.mean(y_true)
        ) * 100                                   # Tỷ lệ MAE so với mức tiêu thụ trung bình (%)

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                preds
            )
        )                                         # Căn bậc hai sai số toàn phương trung bình (kWh)

        mape = np.mean(
            np.abs(
                (
                    y_true - preds
                ) / y_true
            )
        ) * 100                                   # Sai số phần trăm tuyệt đối trung bình (%)

        r2 = r2_score(
            y_true,
            preds
        )                                         # Hệ số xác định R2 (độ phù hợp của mô hình)

        # ===== debug =====
        print(f"MAE: {mae:.4f}")
        print(f"Relative MAE: {relative_mae:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R2 SCORE: {r2:.4f}")

        return {
            "mae": float(mae),
            "relative_mae": float(relative_mae),
            "mape": float(mape),
            "rmse": float(rmse),
            "r2": float(r2)
        }

    # ==== train ====
    def train_model(self,
        save: bool = False  # Input: cờ tùy chọn có lưu thêm bản raw model hay không (mặc định False)
    ):
        # Input :
        #   - save (bool): Cờ tùy chọn lưu model (mặc định False)
        # Output:
        #   - Tuple (model, evaluate):
        #       + model (lgb.LGBMRegressor): Đối tượng mô hình LightGBM đã huấn luyện xong
        #       + evaluate (dict)          : Dictionary chứa 5 chỉ số đánh giá trên tập Test
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        ) = self.get_data()

        model = self.build_model()

        print("\nTraining model...")

        # tqdm wrapper
        for _ in tqdm(
            range(1),
            desc="Fitting LightGBM"
        ):
            model.fit(
                X_train,
                y_train,

                eval_set=[
                    (
                        X_valid,
                        y_valid
                    )
                ],

                eval_metric="l2",

                callbacks=[
                    lgb.log_evaluation(period=100),

                    lgb.early_stopping(
                        stopping_rounds=50
                    )
                ]
            )

        print(
            f"Best iteration: {model.best_iteration_}"
        )

        print("\nPredicting...")

        preds = model.predict(X_test)

        print("\n=== Test Metrics ===")

        evaluate = self.evaluate(
            y_test,
            preds
        )

        print("evaluate: ", evaluate)

        importance = pd.DataFrame({
            "feature": self.features,
            "importance": model.feature_importances_
        }).sort_values(
            "importance",
            ascending=False
        )

        print(
            "\nFeature importance:\n",
            importance
        )

        # ==== save model ====
        save_dir = "backend/models" if os.path.exists("backend") else "models"
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(
            model,
            os.path.join(save_dir, "lightgbm_v2.pkl")
        )

        print(
            f"\nModel saved: {os.path.join(save_dir, 'lightgbm_v2.pkl')}"
        )

        return model, evaluate

    # ==== predict new data ====
    def preprocess_input(
        self,
        input_dict  # Input: dictionary chứa 11 đặc trưng của 1 mốc thời gian
    ):
        """
        input_dict:
            dữ liệu user nhập dạng dict
        """
        # Input : input_dict (dict) - Dictionary chứa 11 đặc trưng của thời điểm cần dự đoán
        # Output: df (pd.DataFrame) - DataFrame kích thước (1, 11) đúng thứ tự cột self.features

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[self.features]

        return df

    # ==== predict 1 sample ====
    def predict_new(
        self,
        model,       # Input 1: mô hình LightGBM đã huấn luyện
        input_dict   # Input 2: dictionary chứa 11 đặc trưng đầu vào của 1 thời điểm
    ):
        """
        predict cho 1 sample mới
        """
        # Input :
        #   - model (lgb.LGBMRegressor): Mô hình LightGBM đã train
        #   - input_dict (dict)        : 11 đặc trưng đầu vào của điểm thời gian cần dự đoán
        # Output:
        #   - preds[0] (float)         : Giá trị tiêu thụ điện dự đoán (kWh) sau khi biến đổi ngược np.expm1

        X_new = self.preprocess_input(
            input_dict
        )

        preds = model.predict(X_new)

        # inverse log1p
        preds = np.expm1(preds)

        return preds[0]

    # ==== forecast future ====
    def forecast_future(
        self,
        model,       # Input 1: mô hình LightGBM đã huấn luyện
        history,     # Input 2: danh sách giá trị quá khứ (>= 168 giờ)
        start_time,  # Input 3: thời điểm bắt đầu dự đoán
        steps        # Input 4: số bước (số giờ) cần dự báo trong tương lai (vd: 24)
    ):
        """
        history:
            list giá trị quá khứ (>=168)

        start_time:
            thời điểm bắt đầu dự đoán

        steps:
            số bước tương lai
            vd: 24 giờ
        """
        # Input :
        #   - model (lgb.LGBMRegressor)     : Mô hình LightGBM đã train
        #   - history (list[float])         : Chuỗi giá trị lịch sử tiêu thụ (>= 168 điểm)
        #   - start_time (str | pd.Timestamp): Thời điểm bắt đầu dự báo
        #   - steps (int)                   : Số giờ tương lai cần dự báo (vd: 24 giờ)
        # Output:
        #   - pd.DataFrame                  : DataFrame gồm `steps` dòng và 2 cột ["time", "prediction"]

        results = []

        history = history.copy()

        current_time = pd.to_datetime(
            start_time
        )

        feature_builder = FeatureBuilder()

        for i in range(steps):
            # build feature từ history hiện tại
            features = feature_builder.build_features(
                time_str=current_time,
                history=history
            )

            pred = self.predict_new(
                model,
                features
            )

            results.append({
                "time": current_time,
                "prediction": float(pred)
            })

            # 🔥 cực quan trọng:
            # append prediction vào history
            history.append(pred)

            # tăng thời gian (1 giờ)
            current_time += pd.Timedelta(hours=1)

        return pd.DataFrame(results)

    # ==== predict từ file csv mới ====
    def predict_from_csv(
        self,
        model,      # Input 1: mô hình LightGBM đã huấn luyện
        csv_file,   # Input 2: đường dẫn hoặc file-like CSV có 2 cột 'time', 'value' (>= 168 dòng)
        steps=24    # Input 3: số giờ tương lai cần dự đoán (mặc định 24 giờ)
    ):
        """
        csv_file:
            file csv dạng:

            time,value
            2024-01-01 00:00:00,100
            2024-01-01 01:00:00,120
        """
        # Input :
        #   - model (lgb.LGBMRegressor): Mô hình LightGBM đã train
        #   - csv_file (str | file)    : File CSV lịch sử của người dùng (cột time, value)
        #   - steps (int)              : Số bước thời gian (giờ) cần dự báo (mặc định 24)
        # Output:
        #   - forecast_df (pd.DataFrame): Bảng kết quả dự báo gồm 2 cột ["time", "prediction"]

        df = pd.read_csv(
            csv_file,
            parse_dates=["time"]
        )

        df = df.sort_values("time")

        history = df["value"].tolist()

        # cần ít nhất 7 ngày
        if len(history) < 168:
            raise ValueError(
                "Need at least 168 rows"
            )

        # timestamp tiếp theo
        start_time = (
            df["time"].iloc[-1]
            + pd.Timedelta(hours=1)
        )

        forecast_df = self.forecast_future(
            model=model,
            history=history,
            start_time=start_time,
            steps=steps
        )

        return forecast_df


# ==== LINEAR / RIDGE TRAINER ====
# Lớp huấn luyện & dự báo cho MÔ HÌNH 2: Ridge Regression (Version 2 - dùng làm Baseline đối sánh)
class LinearTrainerV2:
    def __init__(
        self,
        train_path="dataset/train_ready3.csv",  # Input 1: đường dẫn file CSV tập Train
        valid_path="dataset/valid_ready3.csv",  # Input 2: đường dẫn file CSV tập Valid
        test_path="dataset/test_ready3.csv"     # Input 3: đường dẫn file CSV tập Test
    ):
        # Input :
        #   - train_path (str): Đường dẫn file CSV tập huấn luyện (mặc định "dataset/train_ready3.csv")
        #   - valid_path (str): Đường dẫn file CSV tập kiểm định (mặc định "dataset/valid_ready3.csv")
        #   - test_path  (str): Đường dẫn file CSV tập kiểm tra (mặc định "dataset/test_ready3.csv")
        # Output:
        #   - None: Khởi tạo đối tượng LinearTrainerV2 với cùng bộ 11 features và target như LightGBM
        print("Loading datasets (train, valid, test)...")

        self.train = _sync_log1p_features(pd.read_csv(_resolve_csv_path(train_path)))
        self.valid = _sync_log1p_features(pd.read_csv(_resolve_csv_path(valid_path)))
        self.test = _sync_log1p_features(pd.read_csv(_resolve_csv_path(test_path)))

        # Sử dụng chung 11 đặc trưng đầu vào với LightGBM để đối sánh kết quả công bằng
        self.features = [
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "is_weekend",

            "lag_24h",
            "lag_48h",
            "lag_72h",
            "lag_7d",

            "rolling_mean_24h",
            "rolling_mean_7d",
            "rolling_std_24h"
        ]

        self.target = "target_value"

    # ==================================================
    # DATA
    # ==================================================
    def get_data(self):
        # Input : Không có tham số truyền vào
        # Output: Tuple 6 phần tử (X_train, y_train, X_valid, y_valid, X_test, y_test)
        #         - X_train, X_valid, X_test (pd.DataFrame): Ma trận 11 đặc trưng của 3 tập
        #         - y_train, y_valid, y_test (pd.Series)   : Vector nhãn mục tiêu (target_value) của 3 tập
        print("\nPreparing data...")

        steps = ["Train", "Valid", "Test"]

        data = []

        for step in tqdm(
            steps,
            desc="Splitting data"
        ):
            if step == "Train":
                data.append(
                    (
                        self.train[self.features],
                        self.train[self.target]
                    )
                )

            elif step == "Valid":
                data.append(
                    (
                        self.valid[self.features],
                        self.valid[self.target]
                    )
                )

            else:
                data.append(
                    (
                        self.test[self.features],
                        self.test[self.target]
                    )
                )

        (
            (X_train, y_train),
            (X_valid, y_valid),
            (X_test, y_test)
        ) = data

        return (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        )

    # ==================================================
    # MODEL
    # ==================================================
    def build_model(self):
        # Input : Không có tham số truyền vào
        # Output: model (sklearn.linear_model.Ridge) - Mô hình hồi quy tuyến tính có điều chuẩn L2 (alpha=1.0)
        return Ridge(
            alpha=1.0,
            random_state=42
        )

    # ==================================================
    # METRICS
    # ==================================================
    def evaluate(
        self,
        y_true,  # Input 1: mảng/Series giá trị thực tế trên tập Test (thang log1p)
        preds    # Input 2: mảng/Series giá trị dự đoán của mô hình Ridge (thang log1p)
    ):
        # Input :
        #   - y_true (np.ndarray | pd.Series): Giá trị thực tế của tập Test (ở thang đo log1p)
        #   - preds  (np.ndarray | pd.Series): Giá trị dự đoán của mô hình Ridge (ở thang đo log1p)
        # Output:
        #   - dict: Dictionary chứa 5 chỉ số đánh giá kiểu float {"mae", "relative_mae", "mape", "rmse", "r2"}
        y_true = np.expm1(y_true)  # Đưa nhãn thực tế về đơn vị gốc (kWh)
        preds = np.expm1(preds)    # Đưa giá trị dự đoán về đơn vị gốc (kWh)

        mae = mean_absolute_error(
            y_true,
            preds
        )

        relative_mae = (
            mae / np.mean(y_true)
        ) * 100

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                preds
            )
        )

        mape = np.mean(
            np.abs(
                (
                    y_true - preds
                ) / y_true
            )
        ) * 100

        r2 = r2_score(
            y_true,
            preds
        )

        print(f"MAE: {mae:.4f}")
        print(f"Relative MAE: {relative_mae:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R2 SCORE: {r2:.4f}")

        return {
            "mae": float(mae),
            "relative_mae": float(relative_mae),
            "mape": float(mape),
            "rmse": float(rmse),
            "r2": float(r2)
        }

    # ==================================================
    # TRAIN
    # ==================================================
    def train_model(
        self,
        save: bool = False  # Input: cờ tùy chọn có lưu trực tiếp model bằng joblib hay không
    ):
        # Input :
        #   - save (bool): Tùy chọn lưu file .pkl trực tiếp trong hàm (mặc định False)
        # Output:
        #   - Tuple (model, evaluate):
        #       + model (Ridge)  : Mô hình Ridge Regression đã huấn luyện xong
        #       + evaluate (dict): Dictionary chứa 5 chỉ số đánh giá trên tập Test
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        ) = self.get_data()

        model = self.build_model()

        print("\nTraining Linear Regression...")

        for _ in tqdm(
            range(1),
            desc="Fitting Linear Regression"
        ):
            model.fit(
                X_train,
                y_train
            )

        print("\nPredicting...")

        preds = model.predict(
            X_test
        )

        print("\n=== Test Metrics ===")

        evaluate = self.evaluate(
            y_test,
            preds
        )

        print(
            "evaluate:",
            evaluate
        )

        importance = pd.DataFrame({
            "feature": self.features,
            "coefficient": model.coef_
        })

        importance["abs_coef"] = (
            importance["coefficient"]
            .abs()
        )

        importance = (
            importance
            .sort_values(
                "abs_coef",
                ascending=False
            )
        )

        print(
            "\nFeature coefficients:\n",
            importance[
                [
                    "feature",
                    "coefficient"
                ]
            ]
        )

        if save:
            save_dir = "backend/models" if os.path.exists("backend") else "models"
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(
                model,
                os.path.join(save_dir, "linear_v2.pkl")
            )

            print(
                f"\nModel saved: {os.path.join(save_dir, 'linear_v2.pkl')}"
            )

        return model, evaluate

    # ==================================================
    # INPUT PREPROCESS
    # ==================================================
    def preprocess_input(
        self,
        input_dict  # Input: dictionary chứa 11 đặc trưng của 1 mốc thời gian
    ):
        # Input : input_dict (dict) - Dictionary chứa 11 đặc trưng của thời điểm cần dự đoán
        # Output: df (pd.DataFrame) - DataFrame 1 dòng, 11 cột đúng thứ tự self.features
        df = pd.DataFrame(
            [input_dict]
        )

        df = df[self.features]

        return df

    # ==================================================
    # PREDICT ONE SAMPLE
    # ==================================================
    def predict_new(
        self,
        model,       # Input 1: mô hình Ridge đã huấn luyện
        input_dict   # Input 2: dictionary chứa 11 đặc trưng của 1 thời điểm
    ):
        # Input :
        #   - model (Ridge)    : Mô hình Ridge đã train
        #   - input_dict (dict): 11 đặc trưng của 1 bước thời gian
        # Output:
        #   - preds[0] (float) : Giá trị tiêu thụ điện dự đoán (kWh) sau khi biến đổi ngược np.expm1
        X_new = self.preprocess_input(
            input_dict
        )

        preds = model.predict(
            X_new
        )

        preds = np.expm1(
            preds
        )

        return preds[0]

    # ==================================================
    # FORECAST FUTURE
    # ==================================================
    def forecast_future(
        self,
        model,       # Input 1: mô hình Ridge đã huấn luyện
        history,     # Input 2: danh sách giá trị quá khứ (>= 168 giờ)
        start_time,  # Input 3: mốc thời gian bắt đầu dự đoán
        steps        # Input 4: số bước tương lai cần dự báo (vd: 24 giờ)
    ):
        # Input :
        #   - model (Ridge)                 : Mô hình Ridge đã train
        #   - history (list[float])         : Chuỗi giá trị lịch sử tiêu thụ (>= 168 giờ)
        #   - start_time (str | pd.Timestamp): Mốc thời gian bắt đầu dự báo
        #   - steps (int)                   : Số giờ tương lai cần dự báo (vd: 24 giờ)
        # Output:
        #   - pd.DataFrame                  : Bảng kết quả dự báo gồm 2 cột ["time", "prediction"]
        results = []

        history = history.copy()

        current_time = pd.to_datetime(
            start_time
        )

        feature_builder = FeatureBuilder()

        for _ in range(steps):
            features = (
                feature_builder
                .build_features(
                    time_str=current_time,
                    history=history
                )
            )

            pred = self.predict_new(
                model,
                features
            )

            results.append({
                "time": current_time,
                "prediction": float(pred)
            })

            history.append(pred)

            current_time += pd.Timedelta(
                hours=1
            )

        return pd.DataFrame(
            results
        )

    # ==================================================
    # PREDICT CSV
    # ==================================================
    def predict_from_csv(
        self,
        model,      # Input 1: mô hình Ridge đã huấn luyện
        csv_file,   # Input 2: đường dẫn hoặc đối tượng file CSV đầu vào ('time', 'value')
        steps=24    # Input 3: số giờ tương lai cần dự báo (mặc định 24)
    ):
        # Input :
        #   - model (Ridge)            : Mô hình Ridge đã train
        #   - csv_file (str | file)    : File CSV lịch sử của người dùng (cột time, value)
        #   - steps (int)              : Số bước thời gian (giờ) cần dự báo (mặc định 24)
        # Output:
        #   - forecast_df (pd.DataFrame): Bảng kết quả dự báo gồm 2 cột ["time", "prediction"]
        df = pd.read_csv(
            csv_file,
            parse_dates=["time"]
        )

        df = df.sort_values(
            "time"
        )

        history = (
            df["value"]
            .tolist()
        )

        if len(history) < 168:
            raise ValueError(
                "Need at least 168 rows"
            )

        start_time = (
            df["time"].iloc[-1]
            + pd.Timedelta(hours=1)
        )

        forecast_df = (
            self.forecast_future(
                model=model,
                history=history,
                start_time=start_time,
                steps=steps
            )
        )

        return forecast_df


if __name__ == "__main__":
    # ===== train model version 2 (Mô hình 1: LightGBM Regressor) =====
    # Input : 3 file CSV đã tiền xử lý (train_ready3.csv, valid_ready3.csv, test_ready3.csv)
    # Output: File model "backend/models/lightgbm_model_v2.pkl" chứa {'model': lgb_model, 'evaluate': lgb_evaluate}
    model = LightGBMTrainerV2(
        train_path="backend/dataset/train_ready3.csv",
        valid_path="backend/dataset/valid_ready3.csv",
        test_path="backend/dataset/test_ready3.csv"
    )
    model, evaluate = model.train_model()
    model_manager = ModelManager(model_dir="backend/models")
    save = model_manager.save_model(model, evaluate, save_path="lightgbm_model_v2.pkl")
    loader = model_manager.load_model(name='lightgbm_model_v2.pkl')
    print(loader)

    # ===== train model linear version 2 (Mô hình 2: Ridge Regression) =====
    # Input : 3 file CSV đã tiền xử lý (train_ready3.csv, valid_ready3.csv, test_ready3.csv)
    # Output: File model "backend/models/linear_v2.pkl" chứa {'model': linear_model, 'evaluate': linear_evaluate}
    linear_trainer = LinearTrainerV2(
        train_path="backend/dataset/train_ready3.csv",
        valid_path="backend/dataset/valid_ready3.csv",
        test_path="backend/dataset/test_ready3.csv"
    )
    linear_model, linear_evaluate = linear_trainer.train_model(save=False)
    model_manager = ModelManager(model_dir="backend/models")
    save_linear = model_manager.save_model(linear_model, linear_evaluate, save_path="linear_v2.pkl")
    loader_linear = model_manager.load_model(name='linear_v2.pkl')
    print(loader_linear)
