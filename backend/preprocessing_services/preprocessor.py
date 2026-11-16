"""Leakage-safe feature engineering and temporal splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class Preprocessor:
    """Create lagged client features using only information available in time."""

    LAG_COLUMNS = ["lag_24h", "lag_48h", "lag_72h", "lag_7d"]
    ROLLING_COLUMNS = ["rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"]

    # Khởi tạo bộ tiền xử lý và nạp dữ liệu đầu vào.
    def __init__(self, dataset_path: str = "backend/dataset/hourly_electricity_2013_2014.csv", **kwargs) -> None:
        self.dataset_path = Path(dataset_path)
        self.boundaries = {
            "train_start": pd.Timestamp(kwargs.get("train_start", "2013-01-01")),
            "train_end": pd.Timestamp(kwargs.get("train_end", "2013-09-30 23:00:00")),
            "valid_start": pd.Timestamp(kwargs.get("valid_start", "2013-10-01")),
            "valid_end": pd.Timestamp(kwargs.get("valid_end", "2013-12-31 23:00:00")),
            "test_start": pd.Timestamp(kwargs.get("test_start", "2014-01-01")),
            "test_end": pd.Timestamp(kwargs.get("test_end", "2014-12-31 23:00:00")),
        }
        self.clip_quantile = kwargs.get("clip_quantile", 0.995)
        self.clip_thresholds: dict[str, float] = {}
        self.iqr_thresholds = pd.DataFrame()
        self.df = pd.read_csv(self.dataset_path, parse_dates=["timestamp"])
        self._validate_input()

    # Kiểm tra cấu trúc, thời gian và tính hợp lệ của dữ liệu.
    def _validate_input(self) -> None:
        if "timestamp" not in self.df.columns:
            raise ValueError("Dataset must contain a timestamp column")
        if self.df["timestamp"].duplicated().any():
            raise ValueError("Dataset contains duplicate timestamps")
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        if not self.df["timestamp"].between(self.boundaries["train_start"], self.boundaries["test_end"]).any():
            raise ValueError("Dataset does not overlap the configured 2013-2014 range")

    # Chuyển dữ liệu điện năng từ dạng rộng sang dạng dài.
    def from_wide_to_long(self) -> pd.DataFrame:
        """Keep real zeros and missing values; neither is silently deleted."""
        df = self.df.melt(id_vars="timestamp", var_name="client_id", value_name="target_value")
        df["client_id"] = df["client_id"].astype("category")
        df["target_value"] = pd.to_numeric(df["target_value"], errors="coerce").astype("float32")
        return df.sort_values(["client_id", "timestamp"]).reset_index(drop=True)

    # Tạo các đặc trưng lịch và chu kỳ theo thời gian.
    def build_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["client_id", "timestamp"]).copy()
        day = df["timestamp"].dt.dayofweek
        hour = df["timestamp"].dt.hour
        df["day_of_week"] = day.astype("int8")
        df["is_weekend"] = day.ge(5).astype("int8")
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
        return df

    # Tạo các đặc trưng trễ và thống kê trượt theo từng khách hàng.
    def build_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build true hourly lags per client, tolerating missing timestamps."""
        df = df.sort_values(["client_id", "timestamp"]).copy()
        start, end = df["timestamp"].min(), df["timestamp"].max()
        full_index = pd.date_range(start, end, freq="1h")
        feature_parts = []
        for client_id, group in df.groupby("client_id", observed=True):
            series = group.set_index("timestamp")["target_value"].reindex(full_index)
            previous = series.shift(1)
            features = pd.DataFrame({
                "timestamp": full_index,
                "client_id": client_id,
                "lag_24h": series.shift(24),
                "lag_48h": series.shift(48),
                "lag_72h": series.shift(72),
                "lag_7d": series.shift(168),
                "rolling_mean_24h": previous.rolling(24, min_periods=24).mean(),
                "rolling_mean_7d": previous.rolling(168, min_periods=168).mean(),
                "rolling_std_24h": previous.rolling(24, min_periods=24).std(),
            })
            feature_parts.append(features)
        features = pd.concat(feature_parts, ignore_index=True)
        return df.merge(features, on=["client_id", "timestamp"], how="left", sort=False)

    # Chia dữ liệu theo các khoảng thời gian train, validation và test.
    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        b = self.boundaries
        train = df[df.timestamp.between(b["train_start"], b["train_end"])].copy()
        valid = df[df.timestamp.between(b["valid_start"], b["valid_end"])].copy()
        test = df[df.timestamp.between(b["test_start"], b["test_end"])].copy()
        self._report_splits(train, valid, test)
        return train, valid, test

    @staticmethod
    # In thống kê tóm tắt của các tập dữ liệu đã chia.
    def _report_splits(*splits: pd.DataFrame) -> None:
        for name, part in zip(("train", "valid", "test"), splits):
            print(f"{name}: rows={len(part):,}, clients={part.client_id.nunique()}, timestamps={part.timestamp.nunique()}, range={part.timestamp.min()} -> {part.timestamp.max()}")

    # Cắt các giá trị lớn bất thường dựa trên ngưỡng học từ dữ liệu train.
    def clip_outliers(self, df: pd.DataFrame, train: Optional[pd.DataFrame] = None, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Fit bounds on train and clip input features only; never clip target."""
        columns = columns or self.LAG_COLUMNS + self.ROLLING_COLUMNS
        train = train if train is not None else df
        for column in columns:
            if column in train:
                threshold = train[column].quantile(self.clip_quantile)
                if pd.notna(threshold):
                    self.clip_thresholds[column] = float(threshold)
                    df[column] = df[column].clip(upper=threshold)
        return df

    # Tính ngưỡng IQR theo khách hàng để phát hiện ngoại lệ.
    def fit_iqr_thresholds(self, train: pd.DataFrame, value_column: str = "target_value") -> pd.DataFrame:
        """Fit client-specific IQR bounds on train for detection/reporting only."""
        grouped = train.dropna(subset=[value_column]).groupby("client_id", observed=True)[value_column]
        quantiles = grouped.quantile([0.25, 0.75]).unstack()
        quantiles.columns = ["q1", "q3"]
        quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]
        quantiles["lower"] = quantiles["q1"] - 1.5 * quantiles["iqr"]
        quantiles["upper"] = quantiles["q3"] + 1.5 * quantiles["iqr"]
        self.iqr_thresholds = quantiles
        return quantiles

    # Đánh dấu các quan sát nằm ngoài ngưỡng IQR đã tính.
    def detect_iqr_outliers(
        self, df: pd.DataFrame, thresholds: Optional[pd.DataFrame] = None, value_column: str = "target_value"
    ) -> pd.Series:
        """Flag observations without modifying them or dropping valid spikes."""
        thresholds = thresholds if thresholds is not None else self.iqr_thresholds
        if thresholds.empty:
            raise ValueError("Call fit_iqr_thresholds(train) before detecting outliers")
        bounds = thresholds[["lower", "upper"]].rename(columns={"lower": "_lower", "upper": "_upper"})
        joined = df[["client_id", value_column]].join(bounds, on="client_id")
        return joined[value_column].lt(joined["_lower"]) | joined[value_column].gt(joined["_upper"])

    # Lập báo cáo các đột biến dựa trên thống kê trượt quá khứ.
    def rolling_spike_report(
        self, df: pd.DataFrame, value_column: str = "target_value", z_threshold: float = 5.0, window: int = 24
    ) -> pd.DataFrame:
        """Report trailing rolling spikes using only prior observations."""
        ordered = df.sort_values(["client_id", "timestamp"]).copy()
        groups = ordered.groupby("client_id", observed=True)[value_column]
        previous = groups.shift(1)
        mean = previous.groupby(ordered["client_id"], observed=True).transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        std = previous.groupby(ordered["client_id"], observed=True).transform(
            lambda s: s.rolling(window, min_periods=window).std()
        )
        ordered["rolling_z"] = ((ordered[value_column] - mean) / std.replace(0, np.nan)).abs()
        ordered["is_spike"] = ordered["rolling_z"].gt(z_threshold)
        return ordered.groupby("client_id", observed=True).agg(
            records=("is_spike", "size"), spikes=("is_spike", "sum"), spike_rate=("is_spike", "mean")
        ).reset_index()

    # Biến đổi log1p cho cột mục tiêu không âm.
    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if (df["target_value"].dropna() < 0).any():
            raise ValueError("target_value contains negative values; log1p is undefined here")
        df["target_value"] = np.log1p(df["target_value"]).astype("float32")
        return df

    @staticmethod
    # Chuyển giá trị mục tiêu từ thang log về thang ban đầu.
    def inverse_transform_target(values: pd.Series | np.ndarray) -> np.ndarray:
        return np.expm1(values)

    @staticmethod
    # Trả về danh sách các cột đặc trưng dùng cho mô hình.
    def feature_columns() -> list[str]:
        return ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "lag_24h", "lag_48h", "lag_72h", "lag_7d", "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"]

    # Lưu các tập dữ liệu và metadata tiền xử lý ra tệp.
    def save_splits(self, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, base_path: str = "backend/dataset") -> None:
        destination = Path(base_path)
        destination.mkdir(parents=True, exist_ok=True)
        train.to_csv(destination / "train_ready_2013.csv", index=False)
        valid.to_csv(destination / "valid_ready_2013.csv", index=False)
        test.to_csv(destination / "test_ready_2014.csv", index=False)
        metadata = {
            "boundaries": {k: v.isoformat() for k, v in self.boundaries.items()},
            "clip_quantile": self.clip_quantile,
            "clip_thresholds": self.clip_thresholds,
            "iqr_thresholds": self.iqr_thresholds.astype(float).to_dict(orient="index"),
        }
        (destination / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Chạy toàn bộ quy trình tiền xử lý và lưu kết quả.
    def run_pipeline(self, base_path: str = "backend/dataset"):
        df = self.build_lag_features(self.build_time_features(self.from_wide_to_long()))
        df = df[df["target_value"].notna()].copy()
        train, valid, test = self.split_data(df)
        self.fit_iqr_thresholds(train)
        df = self.clip_outliers(df, train)
        train, valid, test = df.loc[train.index].copy(), df.loc[valid.index].copy(), df.loc[test.index].copy()
        train, valid, test = self.transform_target(train), self.transform_target(valid), self.transform_target(test)
        self.save_splits(train, valid, test, base_path)
        return train, valid, test

if __name__=="__main__":
    preprocessor = Preprocessor()
    preprocessor.run_pipeline()
