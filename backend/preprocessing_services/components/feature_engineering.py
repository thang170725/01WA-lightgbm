"""Pure feature-engineering functions used by training and inference."""

import numpy as np
import pandas as pd


# Chuyển dữ liệu từ dạng rộng sang dạng dài theo khách hàng.
def from_wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Melt client columns while preserving zeros and missing values."""
    result = frame.melt(
        id_vars="timestamp", var_name="client_id", value_name="target_value"
    )
    result["client_id"] = result["client_id"].astype("category")
    result["target_value"] = pd.to_numeric(
        result["target_value"], errors="coerce"
    ).astype("float32")
    return result.sort_values(["client_id", "timestamp"]).reset_index(drop=True)


# Tạo các đặc trưng lịch và chu kỳ theo thời gian.
def build_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical hour features."""
    result = frame.sort_values(["client_id", "timestamp"]).copy()
    day = result["timestamp"].dt.dayofweek
    hour = result["timestamp"].dt.hour
    result["day_of_week"] = day.astype("int8")
    result["is_weekend"] = day.ge(5).astype("int8")
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    return result


# Tạo các đặc trưng trễ và thống kê trượt theo khách hàng.
def build_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build true hourly lags per client, tolerating missing timestamps."""
    result = frame.sort_values(["client_id", "timestamp"]).copy()
    start, end = result["timestamp"].min(), result["timestamp"].max()
    full_index = pd.date_range(start, end, freq="1h")
    feature_parts = []
    for client_id, group in result.groupby("client_id", observed=True):
        series = group.set_index("timestamp")["target_value"].reindex(full_index)
        previous = series.shift(1)
        feature_parts.append(pd.DataFrame({
            "timestamp": full_index,
            "client_id": client_id,
            "lag_24h": series.shift(24),
            "lag_48h": series.shift(48),
            "lag_72h": series.shift(72),
            "lag_7d": series.shift(168),
            "rolling_mean_24h": previous.rolling(24, min_periods=24).mean(),
            "rolling_mean_7d": previous.rolling(168, min_periods=168).mean(),
            "rolling_std_24h": previous.rolling(24, min_periods=24).std(),
        }))
    features = pd.concat(feature_parts, ignore_index=True)
    return result.merge(features, on=["client_id", "timestamp"], how="left", sort=False)
