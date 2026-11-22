
import os
import sys
import json
import time
from datetime import datetime

# Tránh xung đột tên file 'lightgbm.py' với thư viện cài đặt 'lightgbm' khi chạy trực tiếp bằng lệnh python backend/app/models/lightgbm.py
_current_dir = os.path.abspath(os.path.dirname(__file__))
if _current_dir in [os.path.abspath(p) for p in sys.path]:
    sys.path = [p for p in sys.path if os.path.abspath(p) != _current_dir]

# Đảm bảo import được package 'app' dù chạy từ thư mục gốc dự án hay từ thư mục 'backend/'
sys.path.insert(0, os.path.abspath(os.path.join(_current_dir, "../..")))

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

class LightGBMTrainerV4:

    def __init__(
        self,
        train_path,
        valid_path,
        test_path,
        output_dir,
        save_name="lightgbm_v4",
        num_boost_round=500,
        early_stopping_rounds=100,
        params=None
    ):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.output_dir = output_dir
        self.save_name = save_name

        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42
        }

        if params:
            self.params.update(params)

        os.makedirs(self.output_dir, exist_ok=True)

        self.model_dir = os.path.join(self.output_dir, "models")
        self.result_dir = os.path.join(self.output_dir, "results")
        self.log_dir = os.path.join(self.output_dir, "logs")

        for directory in [
            self.model_dir,
            self.result_dir,
            self.log_dir
        ]:
            os.makedirs(directory, exist_ok=True)

        self.features = None
        self.models = {}

        self.log_path = os.path.join(
            self.log_dir,
            f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
        )

    # --------------------------------------------------
    # LOG
    # --------------------------------------------------

    def log(self, message):
        message = str(message)
        print(message)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    def load_data(self):

        self.log("Loading datasets...")

        self.train_df = pd.read_csv(self.train_path)
        self.valid_df = pd.read_csv(self.valid_path)
        self.test_df = pd.read_csv(self.test_path)

        self.log(f"Train: {len(self.train_df):,} rows")
        self.log(f"Valid: {len(self.valid_df):,} rows")
        self.log(f"Test : {len(self.test_df):,} rows")

        if "target_value" not in self.train_df.columns:
            raise ValueError("Train thiếu target_value.")

        excluded_columns = [
            "timestamp",
            "client_id",
            "target_value"
        ]

        self.features = [
            col for col in self.train_df.columns
            if col not in excluded_columns
        ]

        if not self.features:
            raise ValueError("Không tìm thấy feature columns.")

        for name, df in [
            ("valid", self.valid_df),
            ("test", self.test_df)
        ]:
            if "target_value" not in df.columns:
                raise ValueError(f"{name} thiếu target_value.")

            missing = set(self.features) - set(df.columns)

            if missing:
                raise ValueError(
                    f"{name} thiếu features: {missing}"
                )

        self.log(f"Features: {len(self.features)}")
        self.log(str(self.features))

    # --------------------------------------------------
    # 2. PREPARE DATA
    # --------------------------------------------------

    def prepare_dataset(self):

        self.dtrain = lgb.Dataset(
            self.train_df[self.features],
            label=self.train_df["target_value"],
            feature_name=self.features
        )

        self.dvalid = lgb.Dataset(
            self.valid_df[self.features],
            label=self.valid_df["target_value"],
            feature_name=self.features,
            reference=self.dtrain
        )

    # --------------------------------------------------
    # 3. TRAIN ONE MODEL
    # --------------------------------------------------

    def _train_one_model(self, params, name):

        self.log(f"\nTraining {name}...")

        start = time.time()

        model = lgb.train(
            params=params,
            train_set=self.dtrain,

            # Giới hạn cứng số vòng boosting
            num_boost_round=self.num_boost_round,

            valid_sets=[self.dvalid],
            valid_names=["valid"],

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=True
                ),
                lgb.log_evaluation(period=50)
            ]
        )

        self.log(
            f"{name} finished | "
            f"best_iteration={model.best_iteration} | "
            f"elapsed={time.time() - start:.1f}s"
        )

        return model

    # --------------------------------------------------
    # 4. TRAIN THREE MODELS
    # --------------------------------------------------

    def train_models(self, band=(0.1, 0.9)):

        self.prepare_dataset()

        # Mean prediction
        mean_model = self._train_one_model(
            self.params,
            "Mean model"
        )

        # Quantile 10%
        lo_params = {
            **self.params,
            "objective": "quantile",
            "metric": "quantile",
            "alpha": band[0]
        }

        lo_model = self._train_one_model(
            lo_params,
            "Quantile 10%"
        )

        # Quantile 90%
        hi_params = {
            **self.params,
            "objective": "quantile",
            "metric": "quantile",
            "alpha": band[1]
        }

        hi_model = self._train_one_model(
            hi_params,
            "Quantile 90%"
        )

        self.models = {
            "model": mean_model,
            "lo_model": lo_model,
            "hi_model": hi_model
        }

    # --------------------------------------------------
    # 5. PREDICT
    # --------------------------------------------------

    def predict(self, model, df):

        pred = model.predict(
            df[self.features],
            num_iteration=model.best_iteration
        )

        return np.maximum(pred, 0)

    # --------------------------------------------------
    # 6. METRICS
    # --------------------------------------------------

    @staticmethod
    def regression_metrics(y_true, y_pred):

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        wape = (
            np.abs(y_true - y_pred).sum()
            / max(np.abs(y_true).sum(), 1e-9)
            * 100
        )

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "WAPE": float(wape)
        }

    # --------------------------------------------------
    # 7. EVALUATE
    # --------------------------------------------------

    def evaluate(self):

        datasets = {
            "train": self.train_df,
            "valid": self.valid_df,
            "test": self.test_df
        }

        results = {}

        mean_model = self.models["model"]
        lo_model = self.models["lo_model"]
        hi_model = self.models["hi_model"]

        for name, df in datasets.items():

            y = df["target_value"].to_numpy()

            pred = self.predict(mean_model, df)
            lo = self.predict(lo_model, df)
            hi = self.predict(hi_model, df)

            lo = np.minimum(lo, pred)
            hi = np.maximum(hi, pred)

            metrics = self.regression_metrics(y, pred)

            metrics["coverage_80"] = float(
                ((y >= lo) & (y <= hi)).mean() * 100
            )

            metrics["band_width"] = float(
                (hi - lo).mean()
            )

            results[name] = metrics

            self.log(f"\n========== {name.upper()} ==========")

            for key, value in metrics.items():
                self.log(f"{key}: {value:.4f}")

        return results

    # --------------------------------------------------
    # 8. FEATURE IMPORTANCE
    # --------------------------------------------------

    def feature_importance(self):

        model = self.models["model"]

        importance = pd.DataFrame({
            "feature": self.features,
            "gain": model.feature_importance(
                importance_type="gain"
            )
        }).sort_values(
            "gain",
            ascending=False
        ).reset_index(drop=True)

        path = os.path.join(
            self.result_dir,
            f"{self.save_name}_feature_importance.csv"
        )

        importance.to_csv(path, index=False)

        self.log("\n========== FEATURE IMPORTANCE ==========")
        self.log(importance.to_string(index=False))

        return importance

    # --------------------------------------------------
    # 9. SAVE MODELS AND RESULTS
    # --------------------------------------------------

    def save_model(self, results, band):

        # Save 3 LightGBM models
        for tag, model in [
            ("", self.models["model"]),
            ("_lo", self.models["lo_model"]),
            ("_hi", self.models["hi_model"])
        ]:

            model.save_model(
                os.path.join(
                    self.model_dir,
                    f"{self.save_name}{tag}.txt"
                ),
                num_iteration=model.best_iteration
            )

        # Save bundle
        bundle = {
            "model": self.models["model"],
            "lo_model": self.models["lo_model"],
            "hi_model": self.models["hi_model"],
            "evaluate": results,
            "features": self.features,
            "kind": "direct_v4",
            "band": list(band)
        }

        joblib.dump(
            bundle,
            os.path.join(
                self.model_dir,
                f"{self.save_name}.pkl"
            )
        )

        # Save evaluation JSON
        with open(
            os.path.join(
                self.result_dir,
                f"{self.save_name}_metrics.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Save evaluation CSV
        pd.DataFrame(results).T.to_csv(
            os.path.join(
                self.result_dir,
                f"{self.save_name}_metrics.csv"
            ),
            index=True,
            index_label="dataset"
        )

        self.log(
            f"\nModels saved to: {self.model_dir}"
        )

        self.log(
            f"Results saved to: {self.result_dir}"
        )

    # --------------------------------------------------
    # 10. RUN
    # --------------------------------------------------

    def train_model(self, band=(0.1, 0.9)):

        self.log("\n========== START TRAINING ==========")

        self.log(
            f"Max boosting rounds: {self.num_boost_round}"
        )

        self.log(
            f"Early stopping rounds: {self.early_stopping_rounds}"
        )

        self.load_data()

        self.train_models(band=band)

        results = self.evaluate()

        self.feature_importance()

        self.save_model(
            results=results,
            band=band
        )

        self.log("\n========== TRAINING COMPLETED ==========")

        return self.models["model"], results


if __name__ == "__main__":

    trainer = LightGBMTrainerV4(
        train_path="/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_1/train_test_valid_high/train_ready.csv",
        valid_path="/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_1/train_test_valid_high/valid_ready.csv",
        test_path="/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_1/train_test_valid_high/test_ready.csv",

        output_dir="/home/thang/workspace/lightgbm/v0_0_1/result/r0_0_1_1/r_high",

        save_name="lightgbm_high",

        num_boost_round=500,
        early_stopping_rounds=100
    )

    model, results = trainer.train_model()
