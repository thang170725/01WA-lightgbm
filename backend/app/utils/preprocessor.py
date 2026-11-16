import pandas as pd
import numpy as np

# ===== Sau khi tối ưu dataset thì dùng class này để tiền xử lý =====
class Preprocessor:
    def __init__(self,
        dataset_path="backend/dataset/hourly_electricity_filtered.csv"
    ):
        self.df = pd.read_csv(
            dataset_path,
            parse_dates=["timestamp"]
        )

    # ====== 1. wide -> long =====
    # chuyển dataset từ dạng rộng -> dài
    def from_wide_to_long(self):
        print(self.df.head())
        df = self.df.melt(
            id_vars="timestamp",
            var_name="client_id",
            value_name="target_value"
        )

        # bỏ zero
        df = df[df["target_value"] != 0].copy()

        # 2. ép kiểu dữ liệu để tăng tốc độ tính toán
        df["client_id"] = df["client_id"].astype("category")
        df["target_value"] = df["target_value"].astype("float32")

        print(df.head())
        print("after melt:",df.shape)

        return df

    # 2. time features
    # thêm các đặc trưng về thời gian vào dataset
    def build_time_features(self,
        df
    ):
        df = df.sort_values(["client_id","timestamp"])

        df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype("int8")

        hour = (
            df["timestamp"]
            .dt.hour
        )

        df["hour_sin"] = np.sin(
            2*np.pi*hour/24
        ).astype(
            "float32"
        )

        df["hour_cos"] = np.cos(
            2*np.pi*hour/24
        ).astype(
            "float32"
        )

        return df

    # 3. lag features
    # thêm các đặc trựng về lịch sử 
    # def build_lag_features(self,
    #     df
    # ):
    #     g = df.groupby(
    #         "client_id",
    #         observed=True
    #     )["target_value"]

    #     df["lag_24h"] = (
    #         g.shift(24)
    #         .astype("float32")
    #     )

    #     df["lag_7d"] = (
    #         g.shift(168)
    #         .astype("float32")
    #     )

    #     df["rolling_mean_24h"] = (
    #         g.shift(1)
    #          .rolling(
    #             24,
    #             min_periods=24
    #          )
    #          .mean()
    #          .astype(
    #             "float32"
    #          )
    #     )

    #     df.dropna(
    #         inplace=True
    #     )

    #     print(
    #         "after lag:",
    #         df.shape
    #     )

    #     return df

    def build_lag_features(self, df):
        g = df.groupby("client_id", observed=True)["target_value"]

        # === Lag features hiện có ===
        df["lag_24h"] = g.shift(24).astype("float32")
        df["lag_7d"]  = g.shift(168).astype("float32")

        # === THÊM MỚI ===
        df["lag_48h"] = g.shift(48).astype("float32")   # hôm kia cùng giờ
        df["lag_72h"] = g.shift(72).astype("float32")   # 3 ngày trước

        # Rolling means
        df["rolling_mean_24h"] = (
            g.shift(1).rolling(24, min_periods=24).mean().astype("float32")
        )
        df["rolling_mean_7d"] = (                        # THÊM MỚI - trend tuần
            g.shift(1).rolling(168, min_periods=168).mean().astype("float32")
        )
        df["rolling_std_24h"] = (                        # THÊM MỚI - độ biến động
            g.shift(1).rolling(24, min_periods=24).std().astype("float32")
        )

        df.dropna(inplace=True)
        print("after lag:", df.shape)
        return df

    # 4. target transform
    def transform_target(
        self,
        df
    ):
        df["target_value"] = (
            np.log1p(
                df["target_value"]
            )
            .astype(
                "float32"
            )
        )

        return df

    # 5. split time series
    def split_data(self,
        df,
        train_end="2014-09-30",
        valid_end="2014-11-30"
    ):
        train = df[
            df["timestamp"]
            <= train_end
        ].copy()

        valid = df[
            (
                df["timestamp"] > train_end
            )
            &
            (
                df["timestamp"] <= valid_end
            )
        ].copy()

        test = df[
            df["timestamp"]
            > valid_end
        ].copy()

        print("train:", train.shape)
        print("valid:", valid.shape)
        print("test:", test.shape)

        return (train, valid, test)

    # 6 save split files
    def save_splits(self,
        train,
        valid,
        test,
        base_path="backend/dataset/"
    ):
        train.to_csv(
            f"{base_path}train_ready3.csv",
            index=False
        )

        valid.to_csv(
            f"{base_path}valid_ready3.csv",
            index=False
        )

        test.to_csv(
            f"{base_path}test_ready3.csv",
            index=False
        )

        print("saved split files")

    # 7. load split files
    @staticmethod
    def load_train_data():
        return pd.read_csv(
            "backend/dataset/train_ready3.csv",
            parse_dates=["timestamp"]
        )

    @staticmethod
    def load_valid_data():
        return pd.read_csv(
            "backend/dataset/valid_ready3.csv",
            parse_dates=["timestamp"]
        )

    @staticmethod
    def load_test_data():
        return pd.read_csv(
            "backend/dataset/test_ready3.csv",
            parse_dates=["timestamp"]
        )

    # model columns
    # @staticmethod
    # def feature_columns():
    #     return [
    #         "hour_sin",
    #         "hour_cos",
    #         "day_of_week",

    #         "lag_24h",
    #         "lag_7d",

    #         "rolling_mean_24h"
    #     ]
    # Preprocessor
    @staticmethod
    def feature_columns():
        return [
            "hour_sin", "hour_cos", "day_of_week",
            "lag_24h", "lag_48h", "lag_72h", "lag_7d",
            "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"
        ]
    # 4. clip outliers (THÊM MỚI - trước transform_target)
    def clip_outliers(self, df):
        # Tính threshold CHỈ từ train portion (trước 2014-09-30)
        train_mask = df["timestamp"] <= "2014-09-30"
        threshold = df.loc[train_mask, "target_value"].quantile(0.995)

        print(f"Clip threshold (P99.5 of train): {threshold:.2f}")
        print(f"Điểm bị clip: {(df['target_value'] > threshold).sum()}")

        df["target_value"] = df["target_value"].clip(upper=threshold)
        return df

    # 4. target transform (giữ nguyên, chỉ đổi số thứ tự thành 5)
    def transform_target(self, df):
        # Đồng bộ log1p cho cả biến mục tiêu và 7 biến lag/rolling để khớp với FeatureBuilder lúc dự báo
        lag_rolling_cols = [
            "lag_24h", "lag_48h", "lag_72h", "lag_7d",
            "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"
        ]
        for col in lag_rolling_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0)).astype("float32")
        df["target_value"] = (
            np.log1p(df["target_value"])
            .astype("float32")
        )
        return df

    # full pipeline (cập nhật thêm bước clip)
    def run_pipeline(self):
        df = self.from_wide_to_long()
        df = self.build_time_features(df)
        df = self.build_lag_features(df)
        df = self.clip_outliers(df)   # << THÊM VÀO ĐÂY
        df = self.transform_target(df)

        print("final:", df.shape)

        train, valid, test = self.split_data(df)
        self.save_splits(train, valid, test)

        return (train, valid, test)


    # full pipeline
    # def run_pipeline(self):
    #     df = self.from_wide_to_long()
    #     df = self.build_time_features(df)
    #     df = self.build_lag_features(df)
    #     df = self.transform_target(df)

    #     print("final:", df.shape)

    #     train, valid, test = self.split_data(df)
    #     self.save_splits(train, valid, test)

    #     return ( train, valid, test )

class PreprocessorCSVVersion2:
    def __init__(self,
        dataset_path="backend/dataset/hourly_electricity_filtered.csv"
    ):
        self.df = pd.read_csv(
            dataset_path,
            parse_dates=["timestamp"]
        )

    # ====== 1. wide -> long =====
    # chuyển dataset từ dạng rộng -> dài
    def from_wide_to_long(self):
        print(self.df.head())
        df = self.df.melt(
            id_vars="timestamp",
            var_name="client_id",
            value_name="target_value"
        )

        # bỏ zero
        df = df[df["target_value"] != 0].copy()

        # 2. ép kiểu dữ liệu để tăng tốc độ tính toán
        df["client_id"] = df["client_id"].astype("category")
        df["target_value"] = df["target_value"].astype("float32")

        print(df.head())
        print("after melt:",df.shape)

        return df

    # ======= 2. time features =====
    # thêm các đặc trưng về thời gian vào dataset
    def build_time_features(self,
        df
    ):
        df = df.sort_values(["client_id","timestamp"])

        df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
        hour = df["timestamp"].dt.hour
        df["hour_sin"] = np.sin(2*np.pi*hour/24).astype("float32")
        df["hour_cos"] = np.cos(2*np.pi*hour/24).astype("float32")

        return df

    # ======= 3. lag features ======
    # thêm các đặc trựng về lịch sử 
    def build_lag_features(self, 
        df
    ):
        g = df.groupby("client_id", observed=True)["target_value"]

        # === Lag features hiện có ===
        df["lag_24h"] = g.shift(24).astype("float32")
        df["lag_7d"]  = g.shift(168).astype("float32")

        # === THÊM MỚI ===
        df["lag_48h"] = g.shift(48).astype("float32")   # hôm kia cùng giờ
        df["lag_72h"] = g.shift(72).astype("float32")   # 3 ngày trước

        # Rolling means
        df["rolling_mean_24h"] = (g.shift(1).rolling(24, min_periods=24).mean().astype("float32"))
        df["rolling_mean_7d"] = (g.shift(1).rolling(168, min_periods=168).mean().astype("float32"))
        df["rolling_std_24h"] = (g.shift(1).rolling(24, min_periods=24).std().astype("float32"))

        df.dropna(inplace=True)
        print("after lag:", df.shape)
        return df

    # ======== 4. clip outliers =======
    def clip_outliers(self, df):
        # Tính threshold CHỈ từ train portion (trước 2014-09-30)
        train_mask = df["timestamp"] <= "2014-09-30"
        threshold = df.loc[train_mask, "target_value"].quantile(0.995)

        print(f"Clip threshold (P99.5 of train): {threshold:.2f}")
        print(f"Điểm bị clip: {(df['target_value'] > threshold).sum()}")

        df["target_value"] = df["target_value"].clip(upper=threshold)
        return df
    
    # ======= 5. target transform =======
    def transform_target(self,
        df
    ):
        df["target_value"] = np.log1p(df["target_value"].astype("float32"))

        return df

    # ===== 6. split time series ======
    def split_data(self,
        df,
        train_end="2014-09-30",
        valid_end="2014-11-30"
    ):
        train = df[df["timestamp"]<= train_end].copy()
        valid = df[(df["timestamp"] > train_end) & (df["timestamp"] <= valid_end)].copy()
        test = df[df["timestamp"] > valid_end].copy()

        print("train:", train.shape)
        print("valid:", valid.shape)
        print("test:", test.shape)

        return (train, valid, test)

    # ===== 7 save split files ======
    def save_splits(self,
        train,
        valid,
        test,
        base_path="backend/dataset/"
    ):
        train.to_csv(f"{base_path}train_ready3.csv", index=False)
        valid.to_csv(f"{base_path}valid_ready3.csv",index=False)
        test.to_csv(f"{base_path}test_ready3.csv", index=False)

        print("saved split files")

    # ====== 8. load split files ======
    @staticmethod
    def load_train_data():
        return pd.read_csv("backend/dataset/train_ready.csv", parse_dates=["timestamp"])
    @staticmethod
    def load_valid_data():
        return pd.read_csv("backend/dataset/valid_ready.csv", parse_dates=["timestamp"])
    @staticmethod
    def load_test_data():
        return pd.read_csv("backend/dataset/test_ready.csv",parse_dates=["timestamp"])
    @staticmethod
    def feature_columns():
        return [
            "hour_sin", "hour_cos", "day_of_week",
            "lag_24h", "lag_48h", "lag_72h", "lag_7d",
            "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h"
        ]
    
    # ====== full pipeline ======
    def run_pipeline(self):
        df = self.from_wide_to_long()
        df = self.build_time_features(df)
        df = self.build_lag_features(df)
        df = self.clip_outliers(df)   
        df = self.transform_target(df)

        print("final:", df.shape)

        train, valid, test = self.split_data(df)
        self.save_splits(train, valid, test)

        return (train, valid, test)

class CheckerVersion2:
    def __init__(self, dataset_path: str):
        self.df = pd.read_csv(dataset_path)
    
    def overview(self):
        y = np.expm1(self.df["target_value"])
        print(y.describe())
        print(f"\nP95: {y.quantile(0.95):.1f}")
        print(f"P99: {y.quantile(0.99):.1f}")
        print(f"Max: {y.max():.1f}")
        print(f"\nSố điểm > P99: {(y > y.quantile(0.99)).sum()}")

        
if __name__ == "__main__":
    # ===== Quy trình tiền xử lý dữ liệu =======
    preprocessor = Preprocessor(dataset_path="backend/dataset/hourly_electricity_filtered_v2.csv")
    df = preprocessor.from_wide_to_long()
    print(df.head())
    df_build_feature = preprocessor.build_time_features(df)
    df_build_lag = preprocessor.build_lag_features(df_build_feature)
    print(df_build_lag.head())
    df = preprocessor.clip_outliers(df_build_lag)
    print(df.head())
    df_scale = preprocessor.transform_target(df_build_lag)
    print(df_scale.head())
    preprocessor.split_data(df_scale)
    preprocessor.run_pipeline()

    # === checker version 2 ====
    checker = CheckerVersion2(dataset_path="backend/dataset/train_ready3.csv")
    checker.overview()

    # import pandas as pd
    # import numpy as np

    # train = pd.read_csv("backend/dataset/train_ready.csv")
    # y = np.expm1(train["target_value"])

    # # Xem phân phối theo bucket
    # buckets = [0, 100, 200, 500, 1000, 2000, 5000, 10000]
    # for i in range(len(buckets)-1):
    #     count = ((y >= buckets[i]) & (y < buckets[i+1])).sum()
    #     pct = count / len(y) * 100
    #     print(f"{buckets[i]:>6} - {buckets[i+1]:>6}: {count:>8} điểm ({pct:.1f}%)")

    # # Xem outlier tập trung vào giờ/ngày nào
    # train["value_orig"] = y
    # top1pct = train[y > y.quantile(0.99)]
    # print(f"\nOutlier theo day_of_week:\n{top1pct['day_of_week'].value_counts().sort_index()}")

    # Nếu có cột time gốc thì thêm:
    # print(f"\nOutlier theo hour:\n{top1pct['hour'].value_counts().sort_index()}")

    # import pandas as pd
    # import numpy as np

    # train = pd.read_csv("backend/dataset/train_ready.csv")
    # y = np.expm1(train["target_value"])

    # # So sánh các ngưỡng clip
    # for p in [99, 99.5, 99.9]:
    #     threshold = y.quantile(p/100)
    #     clipped = y.clip(upper=threshold)
    #     n_affected = (y > threshold).sum()
    #     print(f"Clip P{p}: threshold={threshold:.1f}, "
    #           f"affected={n_affected} ({n_affected/len(y)*100:.2f}%), "
    #           f"new_mean={clipped.mean():.1f}, new_std={clipped.std():.1f}")