import os
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

# ==== model ai/ml lightgbm ====
# [Ghi chú mới] Khung huấn luyện và dự báo phụ tải điện bằng mô hình LightGBM (Phiên bản 1)
class LightGBMTrainer:
    def __init__(self,
        train_path="dataset/train_ready3.csv",
        valid_path="dataset/valid_ready3.csv",
        test_path="dataset/test_ready3.csv"
    ):
        # [Input]:
        #   - train_path (str): Đường dẫn tới file CSV tập huấn luyện (Train)
        #   - valid_path (str): Đường dẫn tới file CSV tập kiểm định (Valid)
        #   - test_path  (str): Đường dẫn tới file CSV tập kiểm tra (Test)
        # [Output]:
        #   - None (Khởi tạo các thuộc tính đường dẫn, danh sách features và cột target)
        print("Loading datasets (train, valid, test)...")

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.train = pd.read_csv(train_path) if os.path.exists(train_path) else None
        self.valid = pd.read_csv(valid_path) if os.path.exists(valid_path) else None
        self.test = pd.read_csv(test_path) if os.path.exists(test_path) else None

        # LightGBMTrainer
        # [Ghi chú mới] Danh sách 11 đặc trưng đầu vào (4 đặc trưng thời gian + 4 độ trễ lag + 3 thống kê trượt rolling)
        self.features = [
            "hour_sin", "hour_cos", "day_of_week", "is_weekend",
            "lag_24h", "lag_48h", "lag_72h", "lag_7d",
            "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"
        ]
        self.target = "target_value"  # [Ghi chú mới] Cột mục tiêu cần dự đoán (đang ở thang đo log1p)

    # data
    # [Input] : Không có tham số truyền vào (sử dụng self.train, self.valid, self.test, self.features, self.target)
    # [Output]: Tuple 6 phần tử (X_train, y_train, X_valid, y_valid, X_test, y_test)
    #           trong đó X_* là DataFrame (N, 11) chứa đặc trưng, y_* là Series (N,) chứa nhãn mục tiêu
    def get_data(self):
        if self.train is None or self.valid is None or self.test is None:
            self.train = pd.read_csv(self.train_path)
            self.valid = pd.read_csv(self.valid_path)
            self.test = pd.read_csv(self.test_path)

        print("\nPreparing data...")

        steps = ["Train", "Valid", "Test"]
        data = []

        # tqdm cho quá trình split
        # [Ghi chú mới] Lần lượt tách ma trận đặc trưng X và vector mục tiêu y cho 3 tập Train, Valid, Test
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
    # [Input] : Không có tham số truyền vào
    # [Output]: model (lgb.LGBMRegressor) - Đối tượng mô hình LightGBM đã cấu hình hyperparameters
    def build_model(self):
        # version 1
        # model = lgb.LGBMRegressor(
        #     objective="regression",
        #     n_estimators=300,
        #     learning_rate=0.05,
        #     num_leaves=31,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     random_state=42,
        #     n_jobs=-1,
        #     verbosity=-1
        # )

        # === version 2 ===
        # [Ghi chú mới] Cấu hình siêu tham số phiên bản 2: tăng số cây đi kèm giảm learning_rate và giới hạn độ sâu cây
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
    # [Input] :
    #   - y_true (np.ndarray | pd.Series): Giá trị thực tế của tập Test (ở thang đo log1p)
    #   - preds  (np.ndarray | pd.Series): Giá trị dự đoán của mô hình (ở thang đo log1p)
    # [Output]:
    #   - dict: Từ điển gồm 5 chỉ số đánh giá {"mae", "relative_mae", "mape", "rmse", "r2"} tính trên đơn vị gốc (kWh)
    def evaluate(self,
        y_true,
        preds
    ):
        # inverse log1p
        # [Ghi chú mới] Đưa giá trị từ thang log1p về lại thang đo thực tế bằng công thức exp(x) - 1
        y_true = np.expm1(y_true)
        preds = np.expm1(preds)

        mae = mean_absolute_error(y_true, preds)  # Sai số tuyệt đối trung bình
        relative_mae = (
        mae / np.mean(y_true)
        ) * 100                                   # Tỉ lệ MAE so với mức tiêu thụ trung bình (%)

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                preds
            )
        )                                         # Căn bậc hai sai số bình phương trung bình
        mape = np.mean(
            np.abs(
                (
                    y_true-preds
                )/y_true
            )
        ) * 100                                   # Sai số phần trăm tuyệt đối trung bình (%)
        r2 = r2_score(y_true, preds)              # Hệ số xác định R2

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
    # [Input] : Không có tham số truyền vào
    # [Output]: Tuple (model, evaluate) gồm:
    #   - model (lgb.LGBMRegressor): Mô hình LightGBM đã được huấn luyện tại best_iteration_
    #   - evaluate (dict)          : Dictionary chứa 5 chỉ số đánh giá trên tập Test
    def train_model(self):
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
        # [Ghi chú mới] Huấn luyện mô hình trên tập Train và dùng tập Valid để dừng sớm (Early Stopping sau 50 vòng không cải thiện)
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
        preds = model.predict(X_test)  # [Ghi chú mới] Dự đoán trên tập Test

        print("\n=== Test Metrics ===")
        evaluate = self.evaluate(y_test, preds)
        print("evaluate: ", evaluate)

        # [Ghi chú mới] Thống kê mức độ quan trọng của từng đặc trưng (Feature Importance)
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
    # [Input] : input_dict (dict) - Dictionary chứa 11 đặc trưng của 1 mẫu thời gian
    # [Output]: df (pd.DataFrame) - DataFrame 1 dòng, 11 cột sắp xếp đúng thứ tự self.features
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

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[self.features]

        return df


    # [Input] :
    #   - model (lgb.LGBMRegressor): Mô hình đã huấn luyện
    #   - input_dict (dict)        : Dictionary chứa 11 đặc trưng của thời điểm cần dự đoán
    # [Output]:
    #   - preds[0] (float)         : Giá trị điện năng dự báo (kWh) sau khi biến đổi ngược np.expm1
    def predict_new(self, model, input_dict):
        """
        predict cho 1 sample mới
        """

        X_new = self.preprocess_input(input_dict)

        preds = model.predict(X_new)

        # inverse log1p (giống evaluate)
        preds = np.expm1(preds)

        return preds[0]
    
    # [Input] :
    #   - model (lgb.LGBMRegressor)     : Mô hình đã huấn luyện
    #   - history (list[float])         : Danh sách giá trị tiêu thụ lịch sử (>= 168 giờ)
    #   - start_time (str | pd.Timestamp): Mốc thời gian bắt đầu dự đoán
    #   - steps (int)                   : Số giờ tương lai cần dự báo (vd: 24)
    # [Output]:
    #   - pd.DataFrame                  : Bảng kết quả gồm 2 cột ["time", "prediction"] với `steps` dòng
    def forecast_future(self, model, history, start_time, steps):
        """
        history: list giá trị quá khứ (>=168)
        start_time: thời điểm bắt đầu dự đoán
        steps: số bước tương lai (vd: 24 giờ)
        """

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
            # [Ghi chú mới] Dùng kết quả vừa dự báo làm dữ liệu lịch sử để tính lag/rolling cho giờ kế tiếp
            history.append(pred)

            # tăng thời gian (1 giờ)
            current_time += pd.Timedelta(hours=1)

        return pd.DataFrame(results)

# ==== LIGHTGBM TRAINER ====
# [Ghi chú mới] Khung huấn luyện & dự báo cho MÔ HÌNH 1: LightGBM Regressor (Version 2 - dùng trong API /predict_lightgbm)
class LightGBMTrainerV2:
    # [Input] :
    #   - train_path (str): Đường dẫn tới file CSV tập huấn luyện (mặc định "dataset/train_ready3.csv")
    #   - valid_path (str): Đường dẫn tới file CSV tập kiểm định (mặc định "dataset/valid_ready3.csv")
    #   - test_path  (str): Đường dẫn tới file CSV tập kiểm tra (mặc định "dataset/test_ready3.csv")
    # [Output]:
    #   - None (Khởi tạo đối tượng Trainer cùng danh sách 11 features và cột target)
    def __init__(self,
        train_path="dataset/train_ready3.csv",
        valid_path="dataset/valid_ready3.csv",
        test_path="dataset/test_ready3.csv"
    ):
        print("Loading datasets (train, valid, test)...")

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        # [Ghi chú mới] Chỉ đọc ngay nếu file đã tồn tại; nếu chưa có file (đang chờ tiền xử lý) thì gán None để tránh lỗi khi khởi tạo khung
        self.train = pd.read_csv(train_path) if os.path.exists(train_path) else None
        self.valid = pd.read_csv(valid_path) if os.path.exists(valid_path) else None
        self.test = pd.read_csv(test_path) if os.path.exists(test_path) else None

        # [Ghi chú mới] Danh sách 11 cột đặc trưng đầu vào dùng để huấn luyện và dự báo
        self.features = [
            "hour_sin",         # Thành phần sin của giờ trong ngày (chu kỳ 24h)
            "hour_cos",         # Thành phần cos của giờ trong ngày (chu kỳ 24h)
            "day_of_week",      # Thứ trong tuần (0: Thứ Hai -> 6: Chủ Nhật)
            "is_weekend",       # Ngày cuối tuần (1: Thứ Bảy/CN, 0: Ngày thường)

            "lag_24h",          # Điện năng tiêu thụ cách đó 24 giờ (1 ngày trước)
            "lag_48h",          # Điện năng tiêu thụ cách đó 48 giờ (2 ngày trước)
            "lag_72h",          # Điện năng tiêu thụ cách đó 72 giờ (3 ngày trước)
            "lag_7d",           # Điện năng tiêu thụ cách đó 168 giờ (7 ngày trước)

            "rolling_mean_24h", # Trung bình trượt trong 24 giờ gần nhất
            "rolling_mean_7d",  # Trung bình trượt trong 7 ngày (168 giờ) gần nhất
            "rolling_std_24h"   # Độ lệch chuẩn trượt trong 24 giờ gần nhất
        ]

        self.target = "target_value"  # [Ghi chú mới] Cột nhãn mục tiêu (đã qua biến đổi log1p)

    # ==== data ====
    # [Input] : Không có tham số truyền vào (lấy dữ liệu từ self.train, self.valid, self.test)
    # [Output]: Tuple 6 phần tử (X_train, y_train, X_valid, y_valid, X_test, y_test)
    #           - X_train, X_valid, X_test (pd.DataFrame): Ma trận đặc trưng 11 cột của 3 tập
    #           - y_train, y_valid, y_test (pd.Series)   : Vector giá trị mục tiêu của 3 tập
    def get_data(self):
        # [Ghi chú mới] Nạp dữ liệu từ file CSV nếu ở bước __init__ dữ liệu chưa được tải
        if self.train is None or self.valid is None or self.test is None:
            for path in [self.train_path, self.valid_path, self.test_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Chưa tìm thấy file dữ liệu tiền xử lý: '{path}'. "
                        f"Vui lòng đợi bước tiền xử lý hoàn tất trước khi gọi train_model()."
                    )
            self.train = pd.read_csv(self.train_path)
            self.valid = pd.read_csv(self.valid_path)
            self.test = pd.read_csv(self.test_path)

        print("\nPreparing data...")

        steps = ["Train", "Valid", "Test"]

        data = []

        # tqdm cho quá trình split
        # [Ghi chú mới] Tách bảng dữ liệu thành cặp (X, y) tương ứng cho từng tập Train, Valid, Test
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
    # [Input] : Không có tham số truyền vào
    # [Output]: model (lgb.LGBMRegressor) - Đối tượng mô hình LightGBM Regressor đã thiết lập siêu tham số
    def build_model(self):
        # === version 2 ===
        # [Ghi chú mới] Khởi tạo mô hình Gradient Boosted Decision Trees (LightGBM) cho bài toán hồi quy
        model = lgb.LGBMRegressor(
            objective="regression",     # Hàm mục tiêu hồi quy (L2 loss)
            n_estimators=3000,          # Số cây quyết định tối đa
            learning_rate=0.005,        # Tốc độ học nhỏ giúp tối ưu chính xác hơn
            num_leaves=127,             # Số lá tối đa trên 1 cây
            max_depth=8,                # Độ sâu tối đa của cây để tránh overfitting
            subsample=0.8,              # Tỉ lệ lấy mẫu dòng dữ liệu ở mỗi vòng lặp
            colsample_bytree=0.8,       # Tỉ lệ lấy mẫu đặc trưng khi xây dựng mỗi cây
            min_child_samples=20,       # Số lượng mẫu tối thiểu cần có ở 1 nút lá
            random_state=42,            # Cố định seed để tái lập kết quả
            n_jobs=-1,                  # Sử dụng toàn bộ luồng CPU
            verbosity=-1                # Tắt cảnh báo thừa của LightGBM
        )

        return model

    # ==== metrics ====
    # [Input] :
    #   - y_true (np.ndarray | pd.Series): Giá trị thực tế trên tập Test (ở thang đo log1p)
    #   - preds  (np.ndarray | pd.Series): Giá trị mô hình dự đoán trên tập Test (ở thang đo log1p)
    # [Output]:
    #   - dict: Dictionary chứa 5 chỉ số đánh giá kiểu float:
    #       + "mae"          : Sai số tuyệt đối trung bình (kWh)
    #       + "relative_mae" : Tỉ lệ phần trăm MAE so với trung bình thực tế (%)
    #       + "mape"         : Sai số phần trăm tuyệt đối trung bình (%)
    #       + "rmse"         : Căn bậc hai sai số toàn phương trung bình (kWh)
    #       + "r2"           : Hệ số xác định R² (độ phù hợp của mô hình)
    def evaluate(
        self,
        y_true,
        preds
    ):
        # inverse log1p
        # [Ghi chú mới] Biến đổi ngược từ log1p về giá trị tiêu thụ thực tế (kWh) trước khi tính sai số
        y_true = np.expm1(y_true)

        preds = np.expm1(preds)

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
                    y_true - preds
                ) / y_true
            )
        ) * 100

        r2 = r2_score(
            y_true,
            preds
        )

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
    # [Input] :
    #   - save (bool): Cờ tùy chọn lưu model trực tiếp trong hàm (mặc định False)
    # [Output]:
    #   - Tuple (model, evaluate):
    #       + model (lgb.LGBMRegressor): Mô hình LightGBM đã huấn luyện xong
    #       + evaluate (dict)          : Dictionary kết quả đánh giá trên tập Test
    def train_model(self,
        save: bool = False
    ):
        # [Ghi chú mới] Bước 1: Chuẩn bị dữ liệu Train, Valid, Test
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        ) = self.get_data()

        # [Ghi chú mới] Bước 2: Khởi tạo mô hình LightGBM
        model = self.build_model()

        print("\nTraining model...")

        # tqdm wrapper
        # [Ghi chú mới] Bước 3: Huấn luyện mô hình kết hợp kiểm định trên tập Valid và Early Stopping (dừng nếu 50 vòng liên tiếp không giảm lỗi L2)
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

        # [Ghi chú mới] Bước 4: Dự báo trên tập Test và tính toán các chỉ số đánh giá
        print("\nPredicting...")

        preds = model.predict(X_test)

        print("\n=== Test Metrics ===")

        evaluate = self.evaluate(
            y_test,
            preds
        )

        print("evaluate: ", evaluate)

        # [Ghi chú mới] Bước 5: Xuất bảng xếp hạng mức độ quan trọng của 11 đặc trưng
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
        # [Ghi chú mới] Đảm bảo thư mục tồn tại trước khi lưu file model bằng joblib
        os.makedirs("backend/models", exist_ok=True)
        joblib.dump(
            model,
            "backend/models/lightgbm_v2.pkl"
        )

        print(
            "\nModel saved: backend/models/lightgbm_v2.pkl"
        )

        return model, evaluate

    # ==== predict new data ====
    # [Input] :
    #   - input_dict (dict): Dictionary chứa 11 đặc trưng của thời điểm cần dự đoán
    # [Output]:
    #   - df (pd.DataFrame): DataFrame kích thước (1, 11) đã sắp xếp đúng thứ tự cột trong self.features
    def preprocess_input(
        self,
        input_dict
    ):
        """
        input_dict:
            dữ liệu user nhập dạng dict
        """

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[self.features]

        return df

    # ==== predict 1 sample ====
    # [Input] :
    #   - model (lgb.LGBMRegressor): Mô hình LightGBM đã huấn luyện
    #   - input_dict (dict)        : Dictionary chứa 11 đặc trưng đầu vào của 1 thời điểm
    # [Output]:
    #   - preds[0] (float)         : Giá trị điện năng dự báo (kWh) đã chuyển ngược từ thang log1p
    def predict_new(
        self,
        model,
        input_dict
    ):
        """
        predict cho 1 sample mới
        """

        X_new = self.preprocess_input(
            input_dict
        )

        preds = model.predict(X_new)

        # inverse log1p
        preds = np.expm1(preds)

        return preds[0]

    # ==== forecast future ====
    # [Input] :
    #   - model (lgb.LGBMRegressor)     : Mô hình LightGBM đã huấn luyện
    #   - history (list[float])         : Chuỗi giá trị lịch sử tiêu thụ điện (>= 168 giờ)
    #   - start_time (str | pd.Timestamp): Thời điểm bắt đầu dự đoán (giờ kế tiếp sau chuỗi history)
    #   - steps (int)                   : Số bước (số giờ) cần dự báo trong tương lai (vd: 24 giờ)
    # [Output]:
    #   - pd.DataFrame                  : DataFrame gồm `steps` dòng và 2 cột ["time", "prediction"]
    def forecast_future(
        self,
        model,
        history,
        start_time,
        steps
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

        results = []

        history = history.copy()

        current_time = pd.to_datetime(
            start_time
        )

        feature_builder = FeatureBuilder()

        for i in range(steps):
            # build feature từ history hiện tại
            # [Ghi chú mới] Trích xuất 11 đặc trưng (time + lag + rolling) cho mốc current_time
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
            # [Ghi chú mới] Dự báo cuốn chiếu (Autoregressive): thêm giá trị vừa dự đoán vào history cho bước lặp kế tiếp
            history.append(pred)

            # tăng thời gian (1 giờ)
            current_time += pd.Timedelta(hours=1)

        return pd.DataFrame(results)

    # ==== predict từ file csv mới ====
    # [Input] :
    #   - model (lgb.LGBMRegressor): Mô hình LightGBM đã huấn luyện
    #   - csv_file (str | file-like): Đường dẫn hoặc đối tượng file CSV chứa 2 cột `time`, `value` (>= 168 dòng)
    #   - steps (int)              : Số giờ tiếp theo cần dự báo (mặc định 24 giờ)
    # [Output]:
    #   - forecast_df (pd.DataFrame): DataFrame kết quả dự báo gồm 2 cột ["time", "prediction"]
    def predict_from_csv(
        self,
        model,
        csv_file,
        steps=24
    ):
        """
        csv_file:
            file csv dạng:

            time,value
            2024-01-01 00:00:00,100
            2024-01-01 01:00:00,120
        """

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

# [Ghi chú mới] Khung huấn luyện & dự báo cho MÔ HÌNH 2: Ridge Regression (Version 2 - dùng làm Baseline & API /predict_linear)
class LinearTrainerV2:
    # [Input] :
    #   - train_path (str): Đường dẫn tới file CSV tập huấn luyện (mặc định "dataset/train_ready3.csv")
    #   - valid_path (str): Đường dẫn tới file CSV tập kiểm định (mặc định "dataset/valid_ready3.csv")
    #   - test_path  (str): Đường dẫn tới file CSV tập kiểm tra (mặc định "dataset/test_ready3.csv")
    # [Output]:
    #   - None (Khởi tạo đối tượng LinearTrainerV2 với cùng bộ 11 features và target như LightGBM)
    def __init__(
        self,
        train_path="dataset/train_ready3.csv",
        valid_path="dataset/valid_ready3.csv",
        test_path="dataset/test_ready3.csv"
    ):
        print("Loading datasets (train, valid, test)...")

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        # [Ghi chú mới] Chỉ nạp ngay nếu file CSV đã tồn tại; nếu chưa có file thì giữ None và nạp khi gọi get_data()
        self.train = pd.read_csv(train_path) if os.path.exists(train_path) else None
        self.valid = pd.read_csv(valid_path) if os.path.exists(valid_path) else None
        self.test = pd.read_csv(test_path) if os.path.exists(test_path) else None

        # [Ghi chú mới] Sử dụng chung 11 đặc trưng đầu vào với LightGBM để đối sánh kết quả công bằng
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
    # [Input] : Không có tham số truyền vào
    # [Output]: Tuple 6 phần tử (X_train, y_train, X_valid, y_valid, X_test, y_test)
    #           - X_train, X_valid, X_test (pd.DataFrame): Bảng gồm 11 cột đặc trưng của 3 tập
    #           - y_train, y_valid, y_test (pd.Series)   : Cột giá trị mục tiêu (target_value) của 3 tập
    def get_data(self):
        # [Ghi chú mới] Kiểm tra và đọc 3 file CSV đã tiền xử lý khi bắt đầu chạy train
        if self.train is None or self.valid is None or self.test is None:
            for path in [self.train_path, self.valid_path, self.test_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Chưa tìm thấy file dữ liệu tiền xử lý: '{path}'. "
                        f"Vui lòng đợi bước tiền xử lý hoàn tất trước khi gọi train_model()."
                    )
            self.train = pd.read_csv(self.train_path)
            self.valid = pd.read_csv(self.valid_path)
            self.test = pd.read_csv(self.test_path)

        print("\nPreparing data...")

        steps = ["Train", "Valid", "Test"]

        data = []

        # [Ghi chú mới] Tách ma trận đặc trưng X và vector nhãn y cho từng tập dữ liệu
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
    # def build_model(self):
    #     model = LinearRegression(
    #         fit_intercept=True,
    #         n_jobs=-1
    #     )

    #     return model
    # [Input] : Không có tham số truyền vào
    # [Output]: model (sklearn.linear_model.Ridge) - Mô hình Hồi quy tuyến tính kèm điều chuẩn L2 (alpha=1.0)
    def build_model(self):
        # [Ghi chú mới] Dùng Ridge (L2 regularization) giúp ổn định trọng số khi các biến lag có hiện tượng đa cộng tuyến
        return Ridge(
            alpha=1.0,
            random_state=42
        )

    # ==================================================
    # METRICS
    # ==================================================
    # [Input] :
    #   - y_true (np.ndarray | pd.Series): Giá trị thực tế của tập Test (ở thang đo log1p)
    #   - preds  (np.ndarray | pd.Series): Giá trị dự đoán của mô hình Ridge (ở thang đo log1p)
    # [Output]:
    #   - dict: Dictionary chứa 5 chỉ số đánh giá kiểu float {"mae", "relative_mae", "mape", "rmse", "r2"}
    def evaluate(
        self,
        y_true,
        preds
    ):
        # [Ghi chú mới] Đưa giá trị từ thang log1p về lại đơn vị thực tế (kWh) bằng hàm np.expm1
        y_true = np.expm1(y_true)
        preds = np.expm1(preds)

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
    # [Input] :
    #   - save (bool): Tùy chọn có lưu trực tiếp model bằng joblib hay không (mặc định False)
    # [Output]:
    #   - Tuple (model, evaluate):
    #       + model (Ridge)  : Mô hình Ridge Regression đã huấn luyện
    #       + evaluate (dict): Dictionary chứa 5 chỉ số đánh giá trên tập Test
    def train_model(
        self,
        save: bool = False
    ):
        # [Ghi chú mới] Bước 1: Lấy dữ liệu Train, Valid, Test
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test
        ) = self.get_data()

        # [Ghi chú mới] Bước 2: Khởi tạo mô hình Ridge Regression
        model = self.build_model()

        print("\nTraining Linear Regression...")

        # [Ghi chú mới] Bước 3: Khớp (fit) mô hình tuyến tính trên tập X_train, y_train
        for _ in tqdm(
            range(1),
            desc="Fitting Linear Regression"
        ):
            model.fit(
                X_train,
                y_train
            )

        # [Ghi chú mới] Bước 4: Dự đoán trên tập X_test và tính toán bộ chỉ số đánh giá
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

        # [Ghi chú mới] Bước 5: Thống kê hệ số hồi quy (coefficients) của từng đặc trưng theo độ lớn tuyệt đối
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
            os.makedirs("backend/models", exist_ok=True)
            joblib.dump(
                model,
                "backend/models/linear_v2.pkl"
            )

            print(
                "\nModel saved: backend/models/linear_v2.pkl"
            )

        return model, evaluate

    # ==================================================
    # INPUT PREPROCESS
    # ==================================================
    # [Input] :
    #   - input_dict (dict): Dictionary chứa 11 đặc trưng của thời điểm cần dự báo
    # [Output]:
    #   - df (pd.DataFrame): DataFrame 1 dòng, 11 cột đúng thứ tự self.features
    def preprocess_input(
        self,
        input_dict
    ):
        df = pd.DataFrame(
            [input_dict]
        )

        df = df[self.features]

        return df

    # ==================================================
    # PREDICT ONE SAMPLE
    # ==================================================
    # [Input] :
    #   - model (Ridge)    : Mô hình Ridge đã huấn luyện
    #   - input_dict (dict): Dictionary 11 đặc trưng của 1 bước thời gian
    # [Output]:
    #   - preds[0] (float) : Giá trị điện năng dự báo (kWh) sau khi biến đổi ngược np.expm1
    def predict_new(
        self,
        model,
        input_dict
    ):
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
    # [Input] :
    #   - model (Ridge)                 : Mô hình Ridge đã huấn luyện
    #   - history (list[float])         : Chuỗi giá trị lịch sử tiêu thụ (>= 168 giờ)
    #   - start_time (str | pd.Timestamp): Mốc thời gian bắt đầu dự báo
    #   - steps (int)                   : Số giờ tương lai cần dự báo (vd: 24 giờ)
    # [Output]:
    #   - pd.DataFrame                  : Bảng kết quả dự báo gồm 2 cột ["time", "prediction"]
    def forecast_future(
        self,
        model,
        history,
        start_time,
        steps
    ):
        results = []

        history = history.copy()

        current_time = pd.to_datetime(
            start_time
        )

        feature_builder = FeatureBuilder()

        for _ in range(steps):
            # [Ghi chú mới] Tạo 11 đặc trưng cho giờ hiện tại từ chuỗi history
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

            # [Ghi chú mới] Nối giá trị dự đoán mới vào history để dự báo cuốn chiếu cho giờ kế tiếp
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
    # [Input] :
    #   - model (Ridge)            : Mô hình Ridge đã huấn luyện
    #   - csv_file (str | file-like): Đường dẫn hoặc đối tượng file CSV đầu vào (`time`, `value`)
    #   - steps (int)              : Số bước thời gian (giờ) cần dự báo (mặc định 24)
    # [Output]:
    #   - forecast_df (pd.DataFrame): Bảng kết quả dự báo gồm 2 cột ["time", "prediction"]
    def predict_from_csv(
        self,
        model,
        csv_file,
        steps=24
    ):
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


# ==================================================
# [Ghi chú mới] HÀM KHUNG ĐIỀU PHỐI TRAIN CẢ 2 MODEL
# ==================================================
# [Input] :
#   - train_path (str): Đường dẫn file CSV tập Train đã tiền xử lý
#   - valid_path (str): Đường dẫn file CSV tập Valid đã tiền xử lý
#   - test_path  (str): Đường dẫn file CSV tập Test đã tiền xử lý
#   - model_dir  (str): Thư mục lưu 2 file model `.pkl` đầu ra
# [Output]:
#   - dict: Dictionary chứa model và bộ chỉ số evaluate của cả 2 mô hình:
#       {
#           "lightgbm": {"model": lgb_model, "evaluate": lgb_evaluate},
#           "ridge":    {"model": linear_model, "evaluate": linear_evaluate}
#       }
def run_training_pipeline(
    train_path="backend/dataset/train_ready3.csv",
    valid_path="backend/dataset/valid_ready3.csv",
    test_path="backend/dataset/test_ready3.csv",
    model_dir="backend/models"
):
    model_manager = ModelManager(model_dir=model_dir)

    # 1. Huấn luyện và lưu Mô hình 1: LightGBM (LightGBMTrainerV2)
    lgb_trainer = LightGBMTrainerV2(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path
    )
    lgb_model, lgb_evaluate = lgb_trainer.train_model(save=False)
    model_manager.save_model(
        model=lgb_model,
        evaluate=lgb_evaluate,
        save_path="lightgbm_model_v2.pkl"
    )

    # 2. Huấn luyện và lưu Mô hình 2: Ridge Regression (LinearTrainerV2)
    linear_trainer = LinearTrainerV2(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path
    )
    linear_model, linear_evaluate = linear_trainer.train_model(save=False)
    model_manager.save_model(
        model=linear_model,
        evaluate=linear_evaluate,
        save_path="linear_v2.pkl"
    )

    return {
        "lightgbm": {"model": lgb_model, "evaluate": lgb_evaluate},
        "ridge": {"model": linear_model, "evaluate": linear_evaluate}
    }


if __name__ == "__main__":
    # ===== train model version 2 =====
    # model = LightGBMTrainerV2(
    #     train_path="backend/dataset/train_ready3.csv",
    #     valid_path="backend/dataset/valid_ready3.csv",
    #     test_path="backend/dataset/test_ready3.csv"
    # )
    # model, evaluate = model.train_model()
    # model_manager = ModelManager(model_dir="backend/models")
    # save = model_manager.save_model(model, evaluate, save_path="lightgbm_model_v2.pkl")
    # loader = model_manager.load_model(name='lightgbm_model_v2.pkl')
    # print(loader)

    # ===== train model linear version 2 =====
    # [Ghi chú mới] Tạm comment phần thực thi trực tiếp bên dưới để chờ dữ liệu tiền xử lý.
    # Khi đã có đủ 3 file CSV (train_ready3.csv, valid_ready3.csv, test_ready3.csv),
    # bạn có thể bỏ comment các khối trên hoặc gọi `run_training_pipeline()` để train cả 2 model.
    # model = LinearTrainerV2(
    #     train_path="backend/dataset/train_ready3.csv",
    #     valid_path="backend/dataset/valid_ready3.csv",
    #     test_path="backend/dataset/test_ready3.csv"
    # )
    # model, evaluate = model.train_model(save=False)
    # model_manager = ModelManager(model_dir="backend/models")
    # save = model_manager.save_model(model, evaluate, save_path="linear_v2.pkl")
    # loader = model_manager.load_model(name='linear_v2.pkl')
    # print(loader)

    # ===== Khung chạy gộp cả 2 model khi có dữ liệu =====
    # results = run_training_pipeline(
    #     train_path="backend/dataset/train_ready3.csv",
    #     valid_path="backend/dataset/valid_ready3.csv",
    #     test_path="backend/dataset/test_ready3.csv",
    #     model_dir="backend/models"
    # )
    print("Khung huấn luyện 2 model (LightGBMTrainerV2 & LinearTrainerV2) đã sẵn sàng.")