"""Read-only quality reports for raw, hourly, and feature datasets."""

from pathlib import Path

import pandas as pd


class DataChecker:
    """Return structured checks and print concise summaries."""

    # Khởi tạo bộ kiểm tra và nạp dataset cần phân tích.
    def __init__(self, dataset_path: str = "backend/dataset/LD2011_2014.txt") -> None:
        path = Path(dataset_path)
        if path.suffix.lower() == ".txt":
            self.df = pd.read_csv(path, sep=";", decimal=",", parse_dates=[0], index_col=0)
            self.df.index.name = "timestamp"
        else:
            self.df = pd.read_csv(path, parse_dates=["timestamp"])

    # Tổng hợp các chỉ số khái quát của dataset.
    def overview(self) -> dict:
        timestamps = self._timestamps()
        result = {
            "rows": len(self.df),
            "clients": len(self._value_columns()),
            "min_timestamp": timestamps.min(),
            "max_timestamp": timestamps.max(),
            "missing_cells": int(self.df[self._value_columns()].isna().sum().sum()),
            "negative_cells": int((self.df[self._value_columns()] < 0).sum().sum()),
        }
        print(result)
        return result

    # Thống kê số lượng giá trị thiếu theo khách hàng.
    def check_missing(self) -> pd.Series:
        return self.df[self._value_columns()].isna().sum().sort_values(ascending=False)

    # Tính tỷ lệ giá trị bằng không theo khách hàng.
    def check_zero_ratio(self) -> pd.Series:
        return self.df[self._value_columns()].eq(0).mean().sort_values(ascending=False)

    # Tìm chuỗi giá trị bằng không liên tiếp dài nhất.
    def max_consecutive_zeros(self) -> pd.Series:
        # Tính độ dài chuỗi số không liên tiếp dài nhất.
        def longest(values: pd.Series) -> int:
            groups = values.ne(0).cumsum()
            return int(values.eq(0).groupby(groups).sum().max())
        return self.df[self._value_columns()].apply(longest).sort_values(ascending=False)

    # Xác định thời điểm mỗi khách hàng bắt đầu có hoạt động.
    def detect_activation_time(self) -> pd.Series:
        values = self.df[self._value_columns()]
        active = values.gt(0)
        return pd.Series({col: self._timestamps()[mask].min() for col, mask in active.items()})

    # Lập báo cáo mức độ thưa và phạm vi giá trị theo khách hàng.
    def sparsity_report(self) -> pd.DataFrame:
        values = self.df[self._value_columns()]
        return pd.DataFrame({
            "missing_ratio": values.isna().mean(),
            "zero_ratio": values.eq(0).mean(),
            "active_ratio": values.gt(0).mean(),
            "min": values.min(),
            "max": values.max(),
        }).sort_values("zero_ratio", ascending=False)

    # Kiểm tra tần suất, bản ghi trùng và khoảng thời gian bị thiếu.
    def check_time_frequency(self, frequency: str | None = None) -> dict:
        timestamps = self._timestamps().sort_values()
        expected = frequency or ("15min" if self._is_raw() else "1h")
        deltas = timestamps.diff().dropna()
        return {
            "expected_frequency": expected,
            "duplicate_timestamps": int(timestamps.duplicated().sum()),
            "off_frequency": int((deltas != pd.Timedelta(expected)).sum()),
            "missing_intervals": int((deltas / pd.Timedelta(expected)).sub(1).clip(lower=0).sum()),
        }

    # Kiểm tra nhanh tính liên tục của trục thời gian.
    def check_time(self) -> dict:
        return self.check_time_frequency()

    # Tạo báo cáo tổng hợp chất lượng dữ liệu.
    def report(self) -> dict:
        return {
            "overview": self.overview(),
            "time": self.check_time_frequency(),
            "missing_by_client": self.check_missing().to_dict(),
            "zero_ratio": self.check_zero_ratio().to_dict(),
            "sparsity": self.sparsity_report().to_dict(orient="index"),
        }

    # Chạy toàn bộ kiểm tra và in kết quả thời gian.
    def run_all(self) -> dict:
        result = self.report()
        print("Time:", result["time"])
        return result

    @staticmethod
    # So sánh dữ liệu thô theo quý giờ với dữ liệu theo giờ.
    def compare_raw_to_hourly(
        raw_path: str,
        hourly_path: str | None = None,
        client_id: str = "MT_288",
    ) -> pd.DataFrame:
        """Compare raw quarter-hour observations with hourly values.

        ``raw_mean`` is the expected hourly value for this kW dataset. The
        function never fills, removes, or overwrites any source value.
        """
        raw = pd.read_csv(
            raw_path,
            sep=";",
            decimal=",",
            usecols=lambda column: column in {"Unnamed: 0", client_id},
            parse_dates=[0],
        ).rename(columns={"Unnamed: 0": "timestamp"}).set_index("timestamp")
        raw = raw.sort_index()
        grouped = raw[client_id].resample("1h").agg(
            raw_count="count", raw_mean="mean", raw_sum="sum", raw_zero_count=lambda s: int(s.eq(0).sum())
        )
        grouped["raw_missing_count"] = 4 - grouped["raw_count"]

        if hourly_path and Path(hourly_path).exists():
            header = pd.read_csv(hourly_path, nrows=0).columns
            if client_id in header:
                hourly = pd.read_csv(
                    hourly_path, usecols=["timestamp", client_id], parse_dates=["timestamp"]
                ).set_index("timestamp").rename(columns={client_id: "existing_hourly_value"})
                grouped = grouped.join(hourly, how="left")
            else:
                grouped["existing_hourly_value"] = pd.NA
        else:
            grouped["existing_hourly_value"] = pd.NA
        grouped["difference"] = grouped["existing_hourly_value"] - grouped["raw_mean"]
        return grouped.reset_index()

    @staticmethod
    # Báo cáo số lượng bản ghi quanh các mốc chuyển giờ mùa hè.
    def dst_report(raw_path: str, dates: tuple[str, ...] = ("2013-03-31", "2014-03-30")) -> pd.DataFrame:
        """Report observed quarter-hour counts around the requested DST dates."""
        raw = pd.read_csv(raw_path, sep=";", usecols=[0], parse_dates=[0])
        timestamps = pd.to_datetime(raw.iloc[:, 0]).sort_values()
        rows = []
        for date in dates:
            day = pd.Timestamp(date)
            window = timestamps[timestamps.between(day, day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15))]
            rows.append({
                "date": date,
                "raw_records": int(len(window)),
                "duplicate_timestamps": int(window.duplicated().sum()),
                "missing_from_96": int(96 - len(window)),
                "min_timestamp": window.min(),
                "max_timestamp": window.max(),
            })
        return pd.DataFrame(rows)

    # Xác định dataset hiện tại có phải dữ liệu thô hay không.
    def _is_raw(self) -> bool:
        return isinstance(self.df.index, pd.DatetimeIndex)

    # Lấy chuỗi timestamp chuẩn hóa từ dataset.
    def _timestamps(self) -> pd.Series:
        if self._is_raw():
            return pd.Series(self.df.index, index=self.df.index)
        return self.df["timestamp"]

    # Lấy danh sách các cột giá trị của khách hàng.
    def _value_columns(self) -> list[str]:
        return list(self.df.columns if self._is_raw() else self.df.columns.drop("timestamp"))

if __name__=="__main__":
    checker = DataChecker("backend/dataset/hourly_electricity_2013_2014.csv")
    checker.run_all()
