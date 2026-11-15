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
class LightGBMTrainer:
    def __init__(self,
        train_path="dataset/train_ready3.csv",
        valid_path="dataset/valid_ready3.csv",
        test_path="dataset/test_ready3.csv"
    ):
        print("Loading datasets (train, valid, test)...")

        self.train = pd.read_csv(train_path)
        self.valid = pd.read_csv(valid_path)
        self.test = pd.read_csv(test_path)

        # LightGBMTrainer
        self.features = [
            "hour_sin", "hour_cos", "day_of_week", "is_weekend",
            "lag_24h", "lag_48h", "lag_72h", "lag_7d",
            "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"
        ]
        self.target = "target_value"

    # data
    def get_data(self):
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
        y_true,
        preds
    ):
        # inverse log1p
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

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[self.features]

        return df


    def predict_new(self, model, input_dict):
        """
        predict cho 1 sample mới
        """

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
class LightGBMTrainerV2:
    def __init__(self,
        train_path="dataset/train_ready3.csv",
        valid_path="dataset/valid_ready3.csv",
        test_path="dataset/test_ready3.csv",
        load_data=True
    ):
        # load_data=False: chỉ dùng để inference (API), không cần dataset train/valid/test
        self.train = self.valid = self.test = None

        if load_data:
            print("Loading datasets (train, valid, test)...")

            self.train = pd.read_csv(train_path)
            self.valid = pd.read_csv(valid_path)
            self.test = pd.read_csv(test_path)

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

    # ==== data ====
    def get_data(self):
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
        # === version 2 ===
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=3000,
            learning_rate=0.005,
            num_leaves=127,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        return model

    # ==== metrics ====
    def evaluate(
        self,
        y_true,
        preds
    ):
        # inverse log1p
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
    def train_model(self,
        save: bool = False
    ):
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
        joblib.dump(
            model,
            "backend/models/lightgbm_v2.pkl"
        )

        print(
            "\nModel saved: backend/models/lightgbm_v2.pkl"
        )

        return model, evaluate

    # ==== predict new data ====
    def preprocess_input(
        self,
        input_dict,
        features=None
    ):
        """
        input_dict:
            dữ liệu user nhập dạng dict
        features:
            danh sách feature theo thứ tự (mặc định self.features)
        """

        # convert -> DataFrame 1 dòng
        df = pd.DataFrame([input_dict])

        # đảm bảo đúng thứ tự feature
        df = df[features or self.features]

        return df

    # ==== predict 1 sample ====
    def predict_new(
        self,
        model,
        input_dict
    ):
        """
        predict cho 1 sample mới
        """

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

class LinearTrainerV2:
    def __init__(
        self,
        train_path="dataset/train_ready3.csv",
        valid_path="dataset/valid_ready3.csv",
        test_path="dataset/test_ready3.csv"
    ):
        print("Loading datasets (train, valid, test)...")

        self.train = pd.read_csv(train_path)
        self.valid = pd.read_csv(valid_path)
        self.test = pd.read_csv(test_path)

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
    # def build_model(self):
    #     model = LinearRegression(
    #         fit_intercept=True,
    #         n_jobs=-1
    #     )

    #     return model
    def build_model(self):
        return Ridge(
            alpha=1.0,
            random_state=42
        )

    # ==================================================
    # METRICS
    # ==================================================
    def evaluate(
        self,
        y_true,
        preds
    ):
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
    def train_model(
        self,
        save: bool = False
    ):
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
    model = LinearTrainerV2(
        train_path="backend/dataset/train_ready3.csv",
        valid_path="backend/dataset/valid_ready3.csv",
        test_path="backend/dataset/test_ready3.csv"
    )
    model, evaluate = model.train_model(save=False)
    model_manager = ModelManager(model_dir="backend/models")
    save = model_manager.save_model(model, evaluate, save_path="linear_v2.pkl")
    loader = model_manager.load_model(name='linear_v2.pkl')
    print(loader)