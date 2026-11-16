"""Scale-free, leakage-safe feature engineering shared by training and serving.

Why scale-free?
    The 180 clients span 10 kWh .. 48,500 kWh mean load (MT_362 alone is ~6% of
    the total load). Predicting raw / log values and clipping lag features at a
    global quantile destroys the big clients. Instead every row is normalised by
    its own client's recent level ``scale`` (trailing 7-day mean, only past data):

        target  = y_t / scale_t                     (dimensionless ratio)
        lag_*   = lag / scale_t, ...                (dimensionless ratios)
        y_hat   = model(features) * scale_t         (back to kWh)

    The same function ``build_row_features`` is used by forecast_v4 (serving),
    so training and serving cannot drift apart.

Forecast horizon: every feature uses information at least 24 h old relative to
the predicted hour (day-ahead forecasting); rolling stats use data up to t-1
exactly as the previous pipeline did.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-3
MIN_HISTORY = 168  # hours required before a row has all features

TIME_FEATURES = ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
RATIO_FEATURES = [
    "lag_24h", "lag_48h", "lag_72h", "lag_96h", "lag_120h", "lag_144h", "lag_7d",
    "rolling_mean_24h", "rolling_std_24h", "rolling_min_24h", "rolling_max_24h", "rolling_mean_6h",
    "same_hour_mean_7d", "same_hour_std_7d", "same_hour_max_7d", "same_hour_min_7d",
]
LEVEL_FEATURES = ["log_scale", "cv_7d"]
FEATURE_COLUMNS = TIME_FEATURES + RATIO_FEATURES + LEVEL_FEATURES


def time_features(timestamps) -> dict[str, np.ndarray]:
    ts = np.asarray(timestamps, dtype="datetime64[h]")
    hour = (ts.astype("datetime64[h]") - ts.astype("datetime64[D]")).astype(int)
    # 1970-01-01 was Thursday (=3 with Monday=0)
    dow = ((ts.astype("datetime64[D]").astype(int) + 3) % 7).astype(int)
    return {
        "hour": hour.astype("float32"),
        "day_of_week": dow.astype("float32"),
        "is_weekend": (dow >= 5).astype("float32"),
        "hour_sin": np.sin(2 * np.pi * hour / 24).astype("float32"),
        "hour_cos": np.cos(2 * np.pi * hour / 24).astype("float32"),
    }


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """out[t] = a[t-k] along axis 0 (NaN padded)."""
    out = np.full_like(a, np.nan)
    if k < len(a):
        out[k:] = a[:-k] if k else a
    return out


def rolling_stat(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    """Statistic over a[t-w .. t-1] for every t (ddof=0 for std)."""
    T = a.shape[0]
    out = np.full(a.shape, np.nan, dtype="float32")
    if T <= w:
        return out
    view = np.lib.stride_tricks.sliding_window_view(a, w, axis=0)  # idx i -> a[i:i+w]
    # row t = i + w  -> uses a[t-w : t]
    out[w:] = getattr(np, fn)(view, axis=-1)[: T - w]
    return out


HORIZON = 24  # strict day-ahead: every feature only uses data at or before t-24
SCALE_WINDOW = 144  # 6 days, ends at t-24 -> needs history back to t-167 (<= 168)


def _known(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    """Statistic over the w values a[t-24-w+1 .. t-24] (window ends HORIZON hours ago)."""
    return _shift(rolling_stat(a, w, fn), HORIZON - 1)


def compute_feature_arrays(values: np.ndarray, timestamps) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Vectorised day-ahead features for a wide matrix (no information newer than t-24).

    values     : (T, C) float array of consecutive hourly kWh (NaN = missing)
    timestamps : (T,) hourly timestamps
    returns    : ({feature: (T, C) float32}, scale (T, C) float32)
    """
    y = np.asarray(values, dtype="float32")
    if y.ndim == 1:
        y = y[:, None]
    T, C = y.shape

    lag = {h: _shift(y, h) for h in (24, 48, 72, 96, 120, 144, 168)}
    scale = np.maximum(_known(y, SCALE_WINDOW, "mean"), EPS).astype("float32")
    inv = 1.0 / scale

    stack = np.stack([lag[h] for h in (24, 48, 72, 96, 120, 144, 168)], axis=0)  # same hour, past 7 days
    feats: dict[str, np.ndarray] = {}
    for k, v in time_features(timestamps).items():
        feats[k] = np.repeat(v[:, None], C, axis=1)

    for name, h in (("lag_24h", 24), ("lag_48h", 48), ("lag_72h", 72), ("lag_96h", 96),
                    ("lag_120h", 120), ("lag_144h", 144), ("lag_7d", 168)):
        feats[name] = lag[h] * inv
    feats["rolling_mean_24h"] = _known(y, 24, "mean") * inv
    feats["rolling_std_24h"] = _known(y, 24, "std") * inv
    feats["rolling_min_24h"] = _known(y, 24, "min") * inv
    feats["rolling_max_24h"] = _known(y, 24, "max") * inv
    feats["rolling_mean_6h"] = _known(y, 6, "mean") * inv
    feats["same_hour_mean_7d"] = stack.mean(axis=0) * inv
    feats["same_hour_std_7d"] = stack.std(axis=0) * inv
    feats["same_hour_max_7d"] = stack.max(axis=0) * inv
    feats["same_hour_min_7d"] = stack.min(axis=0) * inv
    feats["log_scale"] = np.log1p(scale)
    feats["cv_7d"] = _known(y, SCALE_WINDOW, "std") * inv
    return {k: np.asarray(v, dtype="float32") for k, v in feats.items()}, scale


def build_row_features(history, timestamp) -> tuple[dict[str, float], float]:
    """Features for ONE future hour from a plain list of past hourly values.

    history[-1] is the value at timestamp - 1h. Needs >= 168 values.
    Returns (feature_dict, scale). prediction_kWh = model(features) * scale.
    """
    h = np.asarray(history, dtype="float32")
    if len(h) < MIN_HISTORY:
        raise ValueError("History must contain >= 168 hourly values")
    h = h[-MIN_HISTORY:]
    # append a dummy slot so that row index T is the predicted hour
    padded = np.concatenate([h, [np.nan]]).astype("float32")[:, None]
    ts = np.datetime64(timestamp, "h") - np.arange(len(padded) - 1, -1, -1).astype("timedelta64[h]")
    feats, scale = compute_feature_arrays(padded, ts)
    return {k: float(v[-1, 0]) for k, v in feats.items()}, float(scale[-1, 0])
