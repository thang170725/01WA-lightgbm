from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ElectricityPreprocessor:
    """
    Tiền xử lý dữ liệu điện dạng wide:
        timestamp, MT_001, MT_002, ...

    Đầu ra dạng long:
        timestamp, client_id, target_value, ...

    Quy ước:
    - Giữ nguyên target_value ở đơn vị gốc.
    - Không StandardScaler/MinMaxScaler.
    - Không log1p/expm1.
    - Không clip outlier.
    - Giữ các giá trị 0 nếu đó là mức tiêu thụ thực tế.
    - Lag/rolling được tính riêng theo từng công tơ.
    """

    LAG_HOURS = (24, 48, 72, 168)

    ROLLING_FEATURES = (
        "rolling_mean_24h",
        "rolling_mean_7d",
        "rolling_std_24h",
    )

    FEATURE_COLUMNS = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "lag_24h",
        "lag_48h",
        "lag_72h",
        "lag_7d",
        "rolling_mean_24h",
        "rolling_mean_7d",
        "rolling_std_24h",
    ]

    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "backend/dataset/processed",
        timestamp_col: str = "timestamp",
        train_end: str = "2014-09-30",
        valid_end: str = "2014-11-30",
        expected_frequency: Optional[str] = "1h",
        keep_zero: bool = True,
        target_dtype: str = "float64",
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.timestamp_col = timestamp_col
        self.train_end = pd.Timestamp(train_end)
        self.valid_end = pd.Timestamp(valid_end)
        self.expected_frequency = expected_frequency
        self.keep_zero = keep_zero
        self.target_dtype = target_dtype

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu: {self.dataset_path}"
            )

        if self.train_end >= self.valid_end:
            raise ValueError("train_end phải nhỏ hơn valid_end.")

        if target_dtype not in ("float32", "float64"):
            raise ValueError("target_dtype chỉ nhận 'float32' hoặc 'float64'.")

        self.df_wide: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 1. LOAD + VALIDATE WIDE DATA
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        print(f"Loading: {self.dataset_path}")

        df = pd.read_csv(
            self.dataset_path,
            parse_dates=[self.timestamp_col],
        )

        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"Dataset phải có cột '{self.timestamp_col}'."
            )

        if df.empty:
            raise ValueError("Dataset không có dòng dữ liệu.")

        if df[self.timestamp_col].isna().any():
            raise ValueError("Cột timestamp có giá trị rỗng/không parse được.")

        meter_cols = [c for c in df.columns if c != self.timestamp_col]
        if not meter_cols:
            raise ValueError("Không tìm thấy cột công tơ nào.")

        duplicated = df[self.timestamp_col].duplicated()
        if duplicated.any():
            examples = df.loc[duplicated, self.timestamp_col].head().tolist()
            raise ValueError(
                f"Có timestamp bị trùng. Ví dụ: {examples}"
            )

        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        # Chuyển cột công tơ sang số; giá trị không hợp lệ thành NaN.
        for col in meter_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.expected_frequency:
            diffs = df[self.timestamp_col].diff().dropna()
            expected_delta = pd.Timedelta(self.expected_frequency)
            wrong = diffs != expected_delta
            if wrong.any():
                examples = diffs[wrong].head().tolist()
                raise ValueError(
                    "Timestamp không liên tục theo tần suất "
                    f"{self.expected_frequency}. Chênh lệch ví dụ: {examples}. "
                    "Cần xử lý timestamp thiếu/trùng trước khi tạo lag."
                )

        self.df_wide = df

        print(f"Wide shape: {df.shape}")
        print(f"Number of meters: {len(meter_cols):,}")
        print(
            f"Time range: {df[self.timestamp_col].min()} "
            f"-> {df[self.timestamp_col].max()}"
        )
        return df

    # ------------------------------------------------------------------
    # 2. WIDE -> LONG
    # ------------------------------------------------------------------
    def from_wide_to_long(self) -> pd.DataFrame:
        if self.df_wide is None:
            self.load_data()

        assert self.df_wide is not None

        df = self.df_wide.melt(
            id_vars=[self.timestamp_col],
            var_name="client_id",
            value_name="target_value",
        )

        # Không tự động xóa zero: zero có thể là mức tiêu thụ thực.
        # NaN được giữ đến bước tạo lag, sau đó các dòng thiếu feature
        # sẽ bị loại khỏi tập train/valid/test.
        if not self.keep_zero:
            # Chỉ dùng tùy chọn này khi đã xác nhận zero là sentinel
            # biểu thị dữ liệu không hợp lệ/mất dữ liệu.
            df = df[df["target_value"] != 0].copy()

        df["client_id"] = df["client_id"].astype("category")
        df["target_value"] = df["target_value"].astype(self.target_dtype)

        df = df.sort_values(
            ["client_id", self.timestamp_col]
        ).reset_index(drop=True)

        print(f"Long shape: {df.shape}")
        print(df.head())
        return df

    # ------------------------------------------------------------------
    # 3. TIME FEATURES
    # ------------------------------------------------------------------
    def build_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ts = df[self.timestamp_col]
        df["day_of_week"] = ts.dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

        hour_float = (
            ts.dt.hour
            + ts.dt.minute / 60.0
            + ts.dt.second / 3600.0
        )

        df["hour_sin"] = np.sin(
            2.0 * np.pi * hour_float / 24.0
        ).astype("float32")

        df["hour_cos"] = np.cos(
            2.0 * np.pi * hour_float / 24.0
        ).astype("float32")

        return df

    # ------------------------------------------------------------------
    # 4. LAG + ROLLING FEATURES
    # ------------------------------------------------------------------
    def build_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(
            ["client_id", self.timestamp_col]
        ).copy()

        grouped = df.groupby(
            "client_id",
            observed=True,
            sort=False,
        )["target_value"]

        df["lag_24h"] = grouped.shift(24)
        df["lag_48h"] = grouped.shift(48)
        df["lag_72h"] = grouped.shift(72)
        df["lag_7d"] = grouped.shift(168)

        # transform(lambda ...) bảo đảm rolling được tính riêng trong
        # từng công tơ, không chạy nối tiếp từ công tơ này sang công tơ khác.
        df["rolling_mean_24h"] = grouped.transform(
            lambda s: s.shift(1).rolling(
                window=24,
                min_periods=24,
            ).mean()
        )

        df["rolling_mean_7d"] = grouped.transform(
            lambda s: s.shift(1).rolling(
                window=168,
                min_periods=168,
            ).mean()
        )

        df["rolling_std_24h"] = grouped.transform(
            lambda s: s.shift(1).rolling(
                window=24,
                min_periods=24,
            ).std()
        )

        required = self.FEATURE_COLUMNS + ["target_value"]
        before = len(df)
        df = df.dropna(subset=required).copy()
        dropped = before - len(df)

        print(
            f"After lag/rolling: {df.shape}; "
            f"dropped {dropped:,} rows without complete history."
        )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. TARGET TRANSFORM
    # ------------------------------------------------------------------
    @staticmethod
    def transform_target(df: pd.DataFrame) -> pd.DataFrame:
        """
        Giữ nguyên giá trị thật. Hàm này chỉ copy DataFrame để pipeline
        có điểm mở rộng tương thích; không log, scale hoặc clip.
        """
        return df.copy()

    # ------------------------------------------------------------------
    # 6. TIME-BASED SPLIT
    # ------------------------------------------------------------------
    def split_data(self, df: pd.DataFrame):
        # Dùng mốc exclusive để bao gồm toàn bộ ngày train_end/valid_end,
        # không vô tình chỉ lấy timestamp đúng 00:00:00.
        train_cutoff = self.train_end + pd.Timedelta(days=1)
        valid_cutoff = self.valid_end + pd.Timedelta(days=1)

        train = df[df[self.timestamp_col] < train_cutoff].copy()
        valid = df[
            (df[self.timestamp_col] >= train_cutoff)
            & (df[self.timestamp_col] < valid_cutoff)
        ].copy()
        test = df[df[self.timestamp_col] >= valid_cutoff].copy()

        if train.empty or valid.empty or test.empty:
            raise ValueError(
                "Một trong train/valid/test đang rỗng. "
                "Hãy kiểm tra khoảng thời gian dữ liệu và các mốc split."
            )

        print(f"Train: {train.shape}")
        print(f"Valid: {valid.shape}")
        print(f"Test : {test.shape}")

        for name, part in (
            ("train", train),
            ("valid", valid),
            ("test", test),
        ):
            print(
                f"{name} time range: "
                f"{part[self.timestamp_col].min()} -> "
                f"{part[self.timestamp_col].max()}"
            )

        return train, valid, test

    # ------------------------------------------------------------------
    # 7. SAVE SPLITS
    # ------------------------------------------------------------------
    def save_splits(self, train, valid, test) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "train": self.output_dir / "train_ready.csv",
            "valid": self.output_dir / "valid_ready.csv",
            "test": self.output_dir / "test_ready.csv",
        }

        train.to_csv(paths["train"], index=False)
        valid.to_csv(paths["valid"], index=False)
        test.to_csv(paths["test"], index=False)

        print("Saved split files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    # ------------------------------------------------------------------
    # 8. FULL PIPELINE
    # ------------------------------------------------------------------
    def run_pipeline(self):
        df = self.load_data()
        df = self.from_wide_to_long()
        df = self.build_time_features(df)
        df = self.build_lag_features(df)
        df = self.transform_target(df)

        print(f"Final dataset shape: {df.shape}")

        train, valid, test = self.split_data(df)
        self.save_splits(train, valid, test)

        return train, valid, test

    # ------------------------------------------------------------------
    # 9. LOAD PROCESSED SPLITS
    # ------------------------------------------------------------------
    @staticmethod
    def load_split(path: str) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["timestamp"])

    @staticmethod
    def feature_columns():
        return ElectricityPreprocessor.FEATURE_COLUMNS.copy()

    @staticmethod
    def load_train_data(output_dir="backend/dataset/processed"):
        return ElectricityPreprocessor.load_split(
            str(Path(output_dir) / "train_ready.csv")
        )

    @staticmethod
    def load_valid_data(output_dir="backend/dataset/processed"):
        return ElectricityPreprocessor.load_split(
            str(Path(output_dir) / "valid_ready.csv")
        )

    @staticmethod
    def load_test_data(output_dir="backend/dataset/processed"):
        return ElectricityPreprocessor.load_split(
            str(Path(output_dir) / "test_ready.csv")
        )


class DatasetChecker:
    """Kiểm tra dữ liệu trên thang đo gốc, không dùng expm1."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.df = pd.read_csv(
            self.dataset_path,
            parse_dates=["timestamp"],
        )

        if "target_value" not in self.df.columns:
            raise ValueError("File không có cột target_value.")

    def overview(self):
        y = pd.to_numeric(
            self.df["target_value"],
            errors="coerce",
        ).dropna()

        print("\nTarget value statistics (original scale):")
        print(y.describe())

        print(f"\nP95: {y.quantile(0.95):,.4f}")
        print(f"P99: {y.quantile(0.99):,.4f}")
        print(f"Max: {y.max():,.4f}")
        print(f"Min: {y.min():,.4f}")
        print(f"Zero count: {(y == 0).sum():,}")
        print(f"Negative count: {(y < 0).sum():,}")

        if "client_id" in self.df.columns:
            print(
                f"Number of meters: "
                f"{self.df['client_id'].nunique():,}"
            )

        print(
            f"Time range: {self.df['timestamp'].min()} "
            f"-> {self.df['timestamp'].max()}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess hourly electricity data without scaling/log transform."
    )
    # parser.add_argument(
    #     "--input",
    #     required=True,
    #     help="Đường dẫn CSV wide đầu vào.",
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     default="backend/dataset/processed",
    #     help="Thư mục lưu train/valid/test.",
    # )
    parser.add_argument(
        "--train-end",
        default="2014-09-30",
        help="Ngày cuối của train, được tính bao gồm cả ngày này.",
    )
    parser.add_argument(
        "--valid-end",
        default="2014-11-30",
        help="Ngày cuối của validation, được tính bao gồm cả ngày này.",
    )
    parser.add_argument(
        "--frequency",
        default="1h",
        help="Tần suất timestamp, mặc định 1h. Dùng 'none' để bỏ kiểm tra.",
    )
    parser.add_argument(
        "--drop-zero",
        action="store_true",
        help="Chỉ bật nếu zero là mã biểu thị dữ liệu thiếu, không phải tiêu thụ thật.",
    )

    args = parser.parse_args()

    frequency = None if args.frequency.lower() == "none" else args.frequency

    processor = ElectricityPreprocessor(
        dataset_path=r"/home/thang/workspace/lightgbm/backend/dataset/household.csv",
        output_dir=r"/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_0",
        train_end=args.train_end,
        valid_end=args.valid_end,
        expected_frequency=frequency,
        keep_zero=not args.drop_zero,
        target_dtype="float64",
    )

    processor.run_pipeline()


if __name__ == "__main__":
    main()
