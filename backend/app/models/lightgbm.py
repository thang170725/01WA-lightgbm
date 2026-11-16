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
        input_dict,  # Input: dictionary chứa 11 đặc trưng của 1 mốc thời gian
        features=None
    ):
        """
        input_dict:
            dữ liệu user nhập dạng dict
        features:
            danh sách feature theo thứ tự (mặc định self.features)
        """
        # Input : input_dict (dict) - Dictionary chứa 11 đặc trưng của thời điểm cần dự đoán
        # Output: df (pd.DataFrame) - DataFrame kích thước (1, 11) đúng thứ tự cột self.features

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[features or self.features]

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

        # dùng đúng danh sách feature mà model đã được train
        X_new = self.preprocess_input(
            input_dict,
            features=getattr(model, "feature_name_", None)
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


# ==== V4: SCALE-FREE LIGHTGBM (khuyến nghị) ====
# Dự đoán tỷ lệ y / scale (scale = trung bình 168h trước đó của chính khách hàng) rồi nhân lại scale
# => không bị chi phối bởi khách hàng lớn (MT_362 ~48.000 kWh), không cần log1p / clip lag.
import json
import time
from datetime import datetime

_V4_PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, num_threads=-1, verbosity=-1, seed=42,
)


class _Tee:
    # Ghi đồng thời ra màn hình và file log
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")
        self.f.flush()


def regression_metrics(y, p):
    # Input : y, p (np.ndarray) kWh thực / dự đoán. Output: dict chỉ số (MAPE chỉ tính khi y > 1 kWh, tránh chia 0)
    y = np.asarray(y, dtype="float64"); p = np.asarray(p, dtype="float64")
    err = y - p
    m = y > 1.0
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "wape": float(np.abs(err).sum() / y.sum() * 100),
        "relative_mae": float(np.abs(err).mean() / y.mean() * 100),
        "mape": float((np.abs(err[m]) / y[m]).mean() * 100),
        "smape": float((2 * np.abs(err) / (np.abs(y) + np.abs(p) + 1e-6)).mean() * 100),
    }


class LightGBMTrainerV4:
    def __init__(
        self,
        data_dir="backend/dataset/processed_v4",
        log_dir="backend/logs",
        params=None,
        weight_power=0.5,
        num_boost_round=1000,
    ):
        self.data_dir = data_dir
        self.params = {**_V4_PARAMS, **(params or {})}
        self.weight_power = weight_power
        self.num_boost_round = num_boost_round
        # data_dir: thư mục do Preprocessor.run_pipeline() tạo ra (.npy, đọc bằng mmap -> không tốn RAM)
        self.data_dir = data_dir
        self.params = {**_V4_PARAMS, **(params or {})}
        self.weight_power = weight_power          # trọng số mẫu = scale ** power (0.5: cân bằng khách lớn/nhỏ)
        self.num_boost_round = num_boost_round
        self.log = _Tee(os.path.join(log_dir, f"train_v4_{datetime.now():%Y%m%d_%H%M%S}.log"))
        info = json.load(open(os.path.join(data_dir, "preprocessing_metadata.json"), encoding="utf-8"))
        self.features = info["features"]
        self.clients = info["clients"]

    def _load(self, name):
        X = np.load(os.path.join(self.data_dir, f"{name}_X.npy"), mmap_mode="r")
        m = np.load(os.path.join(self.data_dir, f"{name}_meta.npz"))
        return X, m["y"], m["scale"], m["client"]

    def _dataset(self, name, reference=None, cache=True):
        # Dataset nhị phân (.bin) được cache lại lần sau nạp gần như tức thì và rất nhẹ RAM
        X, y, s, _ = self._load(name)
        w = np.power(s, self.weight_power).astype("float32")
        w /= w.mean()
        path = os.path.join(self.data_dir, f"{name}_p{self.weight_power}.bin")
        if cache and os.path.exists(path):
            return lgb.Dataset(path, reference=reference)
        ds = lgb.Dataset(np.asarray(X), label=(y / s).astype("float32"), weight=w,
                         feature_name=self.features, reference=reference, free_raw_data=True)
        ds.construct()
        if cache:
            ds.save_binary(path)
        return ds

    def predict_kwh(self, booster, name, num_iteration=None):
        X, y, s, c = self._load(name)
        out = np.empty(len(y), dtype="float32")
        step = 500_000  # dự đoán theo khối để tiết kiệm RAM
        for i in range(0, len(y), step):
            out[i:i + step] = booster.predict(np.asarray(X[i:i + step]), num_iteration=num_iteration)
        return y, np.maximum(out * s, 0), c

    def _fit(self, params, dtr, dva, label):
        t0 = time.time()

        b = lgb.train(
            params=params,
            train_set=dtr,
            num_boost_round=self.num_boost_round,
            valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[
                lgb.log_evaluation(period=100)
            ]
        )

        self.log(
            f"[{label}] completed {self.num_boost_round} iterations | "
            f"time={time.time() - t0:.0f}s"
        )

        return b

    def train_model(self, save_name="lightgbm_v4", model_dir="backend/models", band=(0.1, 0.9)):
        # Huấn luyện 3 booster: trung bình (dự báo chính) + phân vị 10% & 90% (dải dự báo 80%)
        log = self.log
        log("Params:", json.dumps(self.params), "| weight_power:", self.weight_power)
        t0 = time.time()
        dtr = self._dataset("train")
        dva = self._dataset("valid", reference=dtr)
        
        # Không construct thủ công ở đây.
        # LightGBM sẽ tự construct khi bắt đầu training.
        
        train_rows = len(self._load("train")[1])
        valid_rows = len(self._load("valid")[1])
        
        log(
            f"Datasets ready in {time.time() - t0:.1f}s | "
            f"train rows={train_rows:,} "
            f"valid rows={valid_rows:,}"
        )

        mean_m = self._fit(self.params, dtr, dva, "mean")
        lo_m = self._fit({**self.params, "objective": "quantile", "alpha": band[0], "metric": "quantile"}, dtr, dva, "q10")
        hi_m = self._fit({**self.params, "objective": "quantile", "alpha": band[1], "metric": "quantile"}, dtr, dva, "q90")

        # --- đánh giá train / valid / test để kiểm tra overfit ---
        result = {}
        for name in ("train", "valid", "test"):
            y, p, c = self.predict_kwh(mean_m, name, mean_m.best_iteration)
            _, lo, _ = self.predict_kwh(lo_m, name, lo_m.best_iteration)
            _, hi, _ = self.predict_kwh(hi_m, name, hi_m.best_iteration)
            lo, hi = np.minimum(lo, p), np.maximum(hi, p)
            m = regression_metrics(y, p)
            m["coverage_80"] = float(((y >= lo) & (y <= hi)).mean() * 100)
            m["band_width_pct"] = float((hi - lo).sum() / max(y.sum(), 1e-9) * 100)
            result[name] = m
            log(f"=== {name.upper()} ===", json.dumps(m))
            if name == "test":
                ty, tp, tc = y, p, c
        gap = result["test"]["wape"] - result["train"]["wape"]
        log(f"Overfit check: WAPE train {result['train']['wape']:.2f}% | valid {result['valid']['wape']:.2f}% | "
            f"test {result['test']['wape']:.2f}% (test-train gap {gap:+.2f} điểm %)")

        per = pd.DataFrame({"c": tc, "ae": np.abs(ty - tp), "y": ty}).groupby("c")[["ae", "y"]].sum()
        per["wape"] = per.ae / per.y.clip(lower=1e-9) * 100
        per.index = [self.clients[i] for i in per.index]
        log(f"Per-client WAPE%: median={per.wape.median():.2f} p90={per.wape.quantile(.9):.2f} max={per.wape.max():.2f}")
        imp = pd.DataFrame({"feature": self.features, "gain": mean_m.feature_importance("gain")})
        log("Feature importance (gain):\n" + imp.sort_values("gain", ascending=False).to_string(index=False))

        evaluate = {**result["test"], "train": result["train"], "valid": result["valid"], "horizon": "day-ahead (24h)"}
        os.makedirs(model_dir, exist_ok=True)
        for tag, m in (("", mean_m), ("_lo", lo_m), ("_hi", hi_m)):
            m.save_model(
                os.path.join(model_dir, f"{save_name}{tag}.txt"),
                num_iteration=self.num_boost_round
            )
        joblib.dump({"model": mean_m, "lo_model": lo_m, "hi_model": hi_m, "evaluate": evaluate,
                     "features": self.features, "kind": "ratio_v4", "band": list(band)},
                    os.path.join(model_dir, f"{save_name}.pkl"))
        log(f"Saved: {model_dir}/{save_name}.pkl")
        return mean_m, evaluate



def forecast_v4(bundle, history, start_time, steps):
    # bundle: dict nạp từ lightgbm_v4.pkl
    # history: list kWh (>=168 giờ liên tiếp)
    # Trả DataFrame [time, value, lo, hi] (kWh)

    from preprocessing_services.feature_engineering import (
        build_row_features,
        FEATURE_COLUMNS
    )

    mean_m = bundle["model"]
    lo_m = bundle["lo_model"]
    hi_m = bundle["hi_model"]

    history = list(history)
    t = pd.to_datetime(start_time)
    rows = []

    for k in range(steps):
        f, scale = build_row_features(history, t)

        x = np.array(
            [[f[c] for c in FEATURE_COLUMNS]],
            dtype="float32"
        )

        # Không truyền best_iteration:
        # LightGBM sử dụng toàn bộ cây đã train.
        v = max(float(mean_m.predict(x)[0]) * scale, 0.0)

        lo = float(lo_m.predict(x)[0]) * scale
        hi = float(hi_m.predict(x)[0]) * scale

        # Nới khoảng dự báo theo số ngày dự báo đệ quy
        widen = 1.0 + 0.15 * (k // 24)

        lo = max(min(v - (v - lo) * widen, v), 0.0)
        hi = max(v + (hi - v) * widen, v)

        rows.append({
            "time": t,
            "value": v,
            "lo": lo,
            "hi": hi
        })

        # Đưa giá trị dự báo vào history cho bước tiếp theo
        history.append(v)

        t += pd.Timedelta(hours=1)

    return pd.DataFrame(rows)



if __name__ == "__main__":
    from preprocessing_services.preprocessor import Preprocessor
    # Bước 1: tiền xử lý (bỏ qua nếu đã có cache: xoá thư mục processed_v4 để chạy lại)
    if not os.path.exists("backend/dataset/processed_v4/preprocessing_metadata.json") and "--rebuild" not in sys.argv:
        Preprocessor("backend/dataset/hourly_electricity_2013_2014.csv").run_pipeline("backend/dataset/processed_v4")
    # Bước 2: train LightGBM scale-free
    LightGBMTrainerV4().train_model()
