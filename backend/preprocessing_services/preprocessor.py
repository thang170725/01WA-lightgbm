"""Leakage-safe, scale-free preprocessing for hourly electricity forecasting (v4).

Pipeline (memory friendly, no giant CSVs / pandas melt):
    wide hourly CSV -> float32 matrix (T x clients) -> vectorised features
    (feature_engineering.py) -> temporal split -> compact .npy cache per split.

Fixes vs. the previous version (see backend/logs/CHANGELOG_*.md):
    * features/target are normalised per client (ratio to trailing 7-day mean),
      no global clipping of lag features (it crushed MT_362 / MT_279 / MT_228);
    * target is never log-transformed -> no expm1 blow-up of errors;
    * rows without a full 168 h history are dropped (no NaN reaches any model);
    * rolling std uses ddof=0 exactly like the serving FeatureBuilder;
    * outputs are float32 .npy (about 10x smaller than the CSV files).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # works both as `backend.preprocessing_services...` and plain import
    from .feature_engineering import FEATURE_COLUMNS, MIN_HISTORY, compute_feature_arrays
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from feature_engineering import FEATURE_COLUMNS, MIN_HISTORY, compute_feature_arrays


class Preprocessor:
    """Build train/valid/test caches from the wide hourly dataset."""

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
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path, parse_dates=["timestamp"], dtype={"timestamp": "object"})
        if "timestamp" not in df.columns:
            raise ValueError("Dataset must contain a timestamp column")
        if df["timestamp"].duplicated().any():
            raise ValueError("Dataset contains duplicate timestamps")
        df = df.set_index("timestamp").sort_index().astype("float32")
        # guarantee a continuous hourly grid so shift(k) == k hours
        full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
        if len(full) != len(df):
            print(f"[warn] {len(full) - len(df)} missing hours re-inserted as NaN")
            df = df.reindex(full)
        df.index.name = "timestamp"
        if (df < 0).any().any():
            raise ValueError("Negative consumption found")
        return df

    def _masks(self) -> dict[str, np.ndarray]:
        b, idx = self.boundaries, self.df.index
        first_ok = idx[0] + pd.Timedelta(hours=MIN_HISTORY)  # need 168h of history
        return {
            "train": np.asarray((idx >= max(b["train_start"], first_ok)) & (idx <= b["train_end"])),
            "valid": np.asarray((idx >= b["valid_start"]) & (idx <= b["valid_end"])),
            "test": np.asarray((idx >= b["test_start"]) & (idx <= b["test_end"])),
        }

    def run_pipeline(self, base_path: str = "backend/dataset/processed_v4") -> dict:
        out = Path(base_path)
        out.mkdir(parents=True, exist_ok=True)
        y = self.df.to_numpy(dtype="float32")
        clients = list(self.df.columns)
        print(f"Building features for {y.shape[0]:,} hours x {y.shape[1]} clients ...")
        feats, scale = compute_feature_arrays(y, self.df.index.values)
        n_clients = y.shape[1]
        report = {}
        for name, mask in self._masks().items():
            X = np.stack([feats[c][mask].reshape(-1) for c in FEATURE_COLUMNS], axis=1).astype("float32")
            yy = y[mask].reshape(-1)
            ss = scale[mask].reshape(-1)
            cid = np.tile(np.arange(n_clients, dtype="int16"), int(mask.sum()))
            ts = np.repeat(self.df.index.values[mask], n_clients)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(yy)
            np.save(out / f"{name}_X.npy", X[ok])
            np.savez_compressed(
                out / f"{name}_meta.npz",
                y=yy[ok], scale=ss[ok], client=cid[ok], timestamp=ts[ok].astype("datetime64[h]").astype("int64"),
            )
            report[name] = {"rows": int(ok.sum()), "dropped_nan_rows": int((~ok).sum()),
                            "start": str(ts[ok].min()), "end": str(ts[ok].max())}
            print(f"{name}: {report[name]}")
            del X, yy, ss, cid, ts, ok
        meta = {
            "boundaries": {k: v.isoformat() for k, v in self.boundaries.items()},
            "features": FEATURE_COLUMNS,
            "clients": clients,
            "min_history_hours": MIN_HISTORY,
            "target": "ratio = y / scale, scale = trailing 168h mean (up to t-1)",
            "splits": report,
        }
        (out / "preprocessing_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return report


if __name__ == "__main__":
    Preprocessor().run_pipeline()
