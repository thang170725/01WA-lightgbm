import pandas as pd

from backend.preprocessing_services.preprocessor import Preprocessor
from backend.preprocessing_services.data_checker import DataChecker


def make_dataset(tmp_path):
    timestamps = pd.date_range("2013-01-01", "2014-01-05 23:00", freq="h")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "client_a": range(len(timestamps)),
        "client_b": [0.0] * len(timestamps),
    })
    path = tmp_path / "hourly.csv"
    frame.to_csv(path, index=False)
    return path


def make_preprocessor(path, tmp_path):
    return Preprocessor(
        path,
        train_end="2013-01-03 23:00",
        valid_start="2013-01-04",
        valid_end="2013-01-04 23:00",
        test_end="2014-01-05 23:00",
    )


def test_split_has_no_year_overlap(tmp_path):
    processor = make_preprocessor(make_dataset(tmp_path), tmp_path)
    train, valid, test = processor.run_pipeline(tmp_path / "output")
    assert train.timestamp.dt.year.eq(2013).all()
    assert valid.timestamp.dt.year.eq(2013).all()
    assert test.timestamp.dt.year.eq(2014).all()
    assert set(train.timestamp).isdisjoint(valid.timestamp)
    assert set(valid.timestamp).isdisjoint(test.timestamp)


def test_zero_is_preserved_and_lag_is_per_client(tmp_path):
    processor = make_preprocessor(make_dataset(tmp_path), tmp_path)
    long = processor.from_wide_to_long()
    assert (long.loc[long.client_id == "client_b", "target_value"] == 0).all()
    features = processor.build_lag_features(processor.build_time_features(long))
    row = features[(features.client_id == "client_a") & (features.timestamp == "2013-01-08")].iloc[0]
    assert row.lag_7d == 0


def test_missing_timestamp_does_not_become_a_false_24_hour_lag(tmp_path):
    processor = make_preprocessor(make_dataset(tmp_path), tmp_path)
    frame = processor.from_wide_to_long()
    frame = frame[~((frame.client_id == "client_a") & (frame.timestamp == "2013-01-02 00:00:00"))]
    features = processor.build_lag_features(processor.build_time_features(frame))
    row = features[(features.client_id == "client_a") & (features.timestamp == "2013-01-03")].iloc[0]
    assert pd.isna(row.lag_24h)


def test_iqr_thresholds_are_fit_per_client_and_do_not_modify_target(tmp_path):
    processor = make_preprocessor(make_dataset(tmp_path), tmp_path)
    frame = processor.from_wide_to_long()
    thresholds = processor.fit_iqr_thresholds(frame)
    flags = processor.detect_iqr_outliers(frame, thresholds)
    assert set(thresholds.index) == {"client_a", "client_b"}
    assert frame.target_value.iloc[0] == 0
    assert flags.dtype == bool


def test_dst_report_keeps_naive_source_timestamps(tmp_path):
    timestamps = pd.date_range("2013-03-31", periods=96, freq="15min")
    timestamps = timestamps.append(pd.date_range("2014-03-30", periods=96, freq="15min"))
    raw_path = tmp_path / "raw.txt"
    pd.DataFrame({"timestamp": timestamps, "MT_288": 1.0}).to_csv(raw_path, sep=";", index=False)
    report = DataChecker.dst_report(str(raw_path))
    assert report.raw_records.tolist() == [96, 96]
    assert report.duplicate_timestamps.tolist() == [0, 0]
