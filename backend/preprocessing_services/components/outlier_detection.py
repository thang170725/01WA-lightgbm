"""Train-fitted outlier detection and feature clipping helpers."""

import numpy as np
import pandas as pd


# Cắt các giá trị ngoại lệ của feature theo ngưỡng học từ train.
def clip_feature_outliers(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    columns: list[str],
    quantile: float,
) -> dict[str, float]:
    """Clip only input features and return thresholds fitted on train."""
    thresholds: dict[str, float] = {}
    for column in columns:
        if column not in train:
            continue
        threshold = train[column].quantile(quantile)
        if pd.notna(threshold):
            thresholds[column] = float(threshold)
            frame[column] = frame[column].clip(upper=threshold)
    return thresholds


# Tính ngưỡng IQR theo khách hàng để báo cáo ngoại lệ.
def fit_iqr_thresholds(
    train: pd.DataFrame, value_column: str = "target_value"
) -> pd.DataFrame:
    """Fit client-specific IQR bounds for reporting only."""
    grouped = train.dropna(subset=[value_column]).groupby(
        "client_id", observed=True
    )[value_column]
    quantiles = grouped.quantile([0.25, 0.75]).unstack()
    quantiles.columns = ["q1", "q3"]
    quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]
    quantiles["lower"] = quantiles["q1"] - 1.5 * quantiles["iqr"]
    quantiles["upper"] = quantiles["q3"] + 1.5 * quantiles["iqr"]
    return quantiles


# Đánh dấu các quan sát nằm ngoài ngưỡng IQR.
def detect_iqr_outliers(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    value_column: str = "target_value",
) -> pd.Series:
    """Flag observations without modifying or dropping them."""
    if thresholds.empty:
        raise ValueError("Fit IQR thresholds before detecting outliers")
    bounds = thresholds[["lower", "upper"]].rename(
        columns={"lower": "_lower", "upper": "_upper"}
    )
    joined = frame[["client_id", value_column]].join(bounds, on="client_id")
    return joined[value_column].lt(joined["_lower"]) | joined[value_column].gt(joined["_upper"])


# Lập báo cáo các đột biến dựa trên thống kê trượt quá khứ.
def rolling_spike_report(
    frame: pd.DataFrame,
    value_column: str = "target_value",
    z_threshold: float = 5.0,
    window: int = 24,
) -> pd.DataFrame:
    """Report trailing rolling spikes using only prior observations."""
    ordered = frame.sort_values(["client_id", "timestamp"]).copy()
    groups = ordered.groupby("client_id", observed=True)[value_column]
    previous = groups.shift(1)
    mean = previous.groupby(ordered["client_id"], observed=True).transform(
        lambda series: series.rolling(window, min_periods=window).mean()
    )
    std = previous.groupby(ordered["client_id"], observed=True).transform(
        lambda series: series.rolling(window, min_periods=window).std()
    )
    ordered["rolling_z"] = (
        (ordered[value_column] - mean) / std.replace(0, np.nan)
    ).abs()
    ordered["is_spike"] = ordered["rolling_z"].gt(z_threshold)
    return ordered.groupby("client_id", observed=True).agg(
        records=("is_spike", "size"),
        spikes=("is_spike", "sum"),
        spike_rate=("is_spike", "mean"),
    ).reset_index()
