"""Build the hourly wide dataset used by the forecasting pipeline."""

from pathlib import Path
from typing import Optional

import pandas as pd


# Chuyển dữ liệu 15 phút thành dữ liệu theo giờ và chọn khách hàng.
def build_hourly_dataset(
    input_path: str = "backend/dataset/LD2011_2014.txt",
    output_path: str = "backend/dataset/hourly_electricity_2013_2014.csv",
    top_clients: int = 180,
    min_mean_load: float = 1.0,
    keep_year: Optional[str] = None,
    selection_year: int = 2013,
    start_year: int = 2013,
    end_year: int = 2014,
    save: bool = False,
) -> pd.DataFrame:
    """Convert 15-minute kW readings to hourly means and select clients.

    Client selection is fitted on 2013 only; 2014 remains untouched for the
    final test set. ``keep_year`` is retained for compatibility, but the new
    default intentionally keeps both years so 2013 history is available.
    """
    if keep_year is not None:
        start_year = end_year = int(keep_year)
    if start_year > selection_year or selection_year > end_year:
        raise ValueError("selection_year must be within start_year..end_year")

    raw = pd.read_csv(
        input_path, sep=";", decimal=",", parse_dates=[0], index_col=0
    )
    raw.index.name = "timestamp"
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce").astype("float32")
    hourly = raw.resample("1h").mean()
    hourly = hourly.loc[f"{start_year}-01-01":f"{end_year}-12-31 23:00:00"]
    if hourly.empty:
        raise ValueError("No observations remain in the requested year range")

    selection = hourly.loc[f"{selection_year}-01-01":f"{selection_year}-12-31 23:00:00"]
    stats = pd.DataFrame({
        "active_ratio": selection.gt(0).mean(),
        "mean_load": selection.mean(),
        "missing_ratio": selection.isna().mean(),
    })
    stats = stats[(stats["mean_load"] >= min_mean_load) & (stats["active_ratio"] > 0)]
    keep_clients = stats.sort_values(
        ["active_ratio", "mean_load"], ascending=False
    ).head(top_clients).index.tolist()
    if not keep_clients:
        raise ValueError("No clients satisfy min_mean_load")

    result = hourly.loc[:, keep_clients].reset_index()
    print(f"Hourly range: {result['timestamp'].min()} -> {result['timestamp'].max()}")
    print(f"Rows, clients: {len(result):,}, {len(keep_clients)}")
    print(f"Selection year: {selection_year}; missing cells: {int(result.isna().sum().sum()):,}")

    if save:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False, float_format="%.4f")
        print(f"Saved: {destination}")
    return result


if __name__ == "__main__":
    build_hourly_dataset(save=True)
