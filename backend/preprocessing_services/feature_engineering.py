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
    "rolling_mean_24h", "rolling_std_24h", "rolling_min_24h", "rolling_max_24h",
    "same_hour_mean_7d", "same_hour_std_7d", "same_hour_max_7d", "same_hour_min_7d",
    "last_value", "rolling_mean_6h",
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


def compute_feature_arrays(values: np.ndarray, timestamps) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Vectorised features for a wide matrix.

    values     : (T, C) float array of consecutive hourly kWh (NaN = missing)
    timestamps : (T,) hourly timestamps
    returns    : ({feature: (T, C) float32}, scale (T, C) float32)
    """
    y = np.asarray(values, dtype="float32")
    if y.ndim == 1:
        y = y[:, None]
    T, C = y.shape

    lag = {h: _shift(y, h) for h in (24, 48, 72, 96, 120, 144, 168)}
    scale_raw = rolling_stat(y, 168, "mean")
    scale = np.maximum(scale_raw, EPS).astype("float32")

    # same hour of day over the previous 7 days: lag 24,48,...,168
    stack = np.stack([lag[h] for h in (24, 48, 72, 96, 120, 144, 168)], axis=0)
    sh_mean = stack.mean(axis=0)
    sh_std = stack.std(axis=0)
    sh_max = stack.max(axis=0)
    sh_min = stack.min(axis=0)

    r24_mean = rolling_stat(y, 24, "mean")
    feats: dict[str, np.ndarray] = {}
    tf = time_features(timestamps)
    for k, v in tf.items():
        feats[k] = np.repeat(v[:, None], C, axis=1)

    inv = 1.0 / scale
    feats["lag_24h"] = lag[24] * inv
    feats["lag_48h"] = lag[48] * inv
    feats["lag_72h"] = lag[72] * inv
    feats["lag_96h"] = lag[96] * inv
    feats["lag_120h"] = lag[120] * inv
    feats["lag_144h"] = lag[144] * inv
    feats["lag_7d"] = lag[168] * inv
    feats["rolling_mean_24h"] = r24_mean * inv
    feats["rolling_std_24h"] = rolling_stat(y, 24, "std") * inv
    feats["rolling_min_24h"] = rolling_stat(y, 24, "min") * inv
    feats["rolling_max_24h"] = rolling_stat(y, 24, "max") * inv
    feats["same_hour_mean_7d"] = sh_mean * inv
    feats["same_hour_std_7d"] = sh_std * inv
    feats["same_hour_max_7d"] = sh_max * inv
    feats["same_hour_min_7d"] = sh_min * inv
    feats["last_value"] = _shift(y, 1) * inv
    feats["rolling_mean_6h"] = rolling_stat(y, 6, "mean") * inv
    feats["log_scale"] = np.log1p(scale)
    feats["cv_7d"] = rolling_stat(y, 168, "std") * inv
    feats = {k: np.asarray(v, dtype="float32") for k, v in feats.items()}
    return feats, scale


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
