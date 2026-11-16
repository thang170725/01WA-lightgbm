"""Temporal split reporting and CSV persistence."""

import json
from pathlib import Path

import pandas as pd


# Chia dữ liệu thành các tập train, validation và test theo thời gian.
def split_data(
    frame: pd.DataFrame, boundaries: dict[str, pd.Timestamp]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame.timestamp.between(boundaries["train_start"], boundaries["train_end"])].copy()
    valid = frame[frame.timestamp.between(boundaries["valid_start"], boundaries["valid_end"])].copy()
    test = frame[frame.timestamp.between(boundaries["test_start"], boundaries["test_end"])].copy()
    return train, valid, test


# In thống kê tóm tắt của các tập dữ liệu.
def report_splits(*splits: pd.DataFrame) -> None:
    for name, part in zip(("train", "valid", "test"), splits):
        print(
            f"{name}: rows={len(part):,}, clients={part.client_id.nunique()}, "
            f"timestamps={part.timestamp.nunique()}, "
            f"range={part.timestamp.min()} -> {part.timestamp.max()}"
        )


# Lưu các tập dữ liệu và metadata tiền xử lý ra tệp.
def save_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    base_path: str,
    metadata: dict,
) -> None:
    destination = Path(base_path)
    destination.mkdir(parents=True, exist_ok=True)
    train.to_csv(destination / "train_ready_2013.csv", index=False)
    valid.to_csv(destination / "valid_ready_2013.csv", index=False)
    test.to_csv(destination / "test_ready_2014.csv", index=False)
    (destination / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# Đọc một tập dữ liệu đã lưu và chuyển cột thời gian về dạng ngày giờ.
def load_split(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])
