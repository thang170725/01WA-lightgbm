"""
Pipeline tách dữ liệu điện thành 2 nhóm:
- Household: công tơ có mức tiêu thụ trung bình < threshold
- Business: công tơ có mức tiêu thụ trung bình >= threshold

Định nghĩa:
- Với mỗi cột MT_..., tính trung bình các giá trị hợp lệ trên toàn bộ
  khoảng thời gian trong dataset.
- threshold phải cùng đơn vị với giá trị trong các cột MT_...
- Timestamp được giữ lại trong cả hai file đầu ra.

Thiết kế đọc theo chunksize để xử lý CSV lớn mà không cần nạp toàn bộ
file vào RAM cùng lúc.

Ví dụ:
python -m backend.app.utils.split_electricity_dataset \
    --input backend/dataset/hourly_electricity_2013_2014.csv \
    --output-dir backend/dataset/split \
    --threshold 500

Nếu chạy trực tiếp:
python backend/app/utils/split_electricity_dataset.py \
    --input backend/dataset/hourly_electricity_2013_2014.csv \
    --output-dir backend/dataset/split \
    --threshold 500
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd


LOGGER = logging.getLogger("electricity_dataset_splitter")


def _read_header(input_path: Path, timestamp_col: str) -> Tuple[List[str], List[str]]:
    """Đọc header và trả về (toàn bộ cột, các cột công tơ)."""
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    header = [str(col).strip() for col in header]

    if timestamp_col not in header:
        raise ValueError(
            f"Không tìm thấy cột thời gian '{timestamp_col}'. "
            f"Các cột đầu file: {header[:10]}"
        )

    meter_columns = [col for col in header if col != timestamp_col]
    if not meter_columns:
        raise ValueError("Dataset không có cột công tơ nào ngoài timestamp.")

    return header, meter_columns


def calculate_meter_means(
    input_path: Path,
    meter_columns: List[str],
    timestamp_col: str,
    chunksize: int,
) -> pd.Series:
    """
    Pass 1: tính mean của từng công tơ bằng tổng / số lượng giá trị hợp lệ.
    """
    sums = pd.Series(0.0, index=meter_columns, dtype="float64")
    counts = pd.Series(0, index=meter_columns, dtype="int64")

    reader = pd.read_csv(
        input_path,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_index, chunk in enumerate(reader, start=1):
        chunk.columns = chunk.columns.astype(str).str.strip()

        values = chunk[meter_columns].apply(pd.to_numeric, errors="coerce")
        sums = sums.add(values.sum(axis=0, skipna=True), fill_value=0)
        counts = counts.add(values.count(axis=0), fill_value=0)

        LOGGER.info("Pass 1 - đã xử lý chunk %d", chunk_index)

    means = sums.div(counts.replace(0, pd.NA)).astype("float64")

    if means.isna().any():
        empty_meters = means[means.isna()].index.tolist()
        LOGGER.warning(
            "Có %d công tơ không có giá trị số hợp lệ; sẽ không đưa vào hai nhóm. Ví dụ: %s",
            len(empty_meters),
            empty_meters[:10],
        )

    return means


def write_group_csv(
    input_path: Path,
    output_path: Path,
    selected_columns: List[str],
    timestamp_col: str,
    chunksize: int,
) -> int:
    """
    Pass 2: đọc lại file theo chunksize và ghi các cột thuộc một nhóm.
    Trả về tổng số dòng đã ghi.
    """
    output_columns = [timestamp_col] + selected_columns
    if not selected_columns:
        # Tạo file chỉ có header để kết quả luôn có đủ hai file.
        pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)
        return 0

    rows_written = 0
    first_chunk = True

    reader = pd.read_csv(
        input_path,
        usecols=output_columns,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_index, chunk in enumerate(reader, start=1):
        chunk.columns = chunk.columns.astype(str).str.strip()
        chunk = chunk[output_columns]

        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False
        rows_written += len(chunk)
        LOGGER.info(
            "Đang ghi %s - chunk %d, tổng %d dòng",
            output_path.name,
            chunk_index,
            rows_written,
        )

    return rows_written


def split_electricity_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    threshold: float,
    timestamp_col: str = "timestamp",
    chunksize: int = 20_000,
) -> dict:
    """
    Tách dataset wide-format thành household.csv và business.csv.

    Phân loại theo mean của từng công tơ:
        mean < threshold  -> household
        mean >= threshold -> business

    threshold có cùng đơn vị với dữ liệu gốc.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy input CSV: {input_path}")

    if chunksize <= 0:
        raise ValueError("chunksize phải lớn hơn 0.")

    output_dir.mkdir(parents=True, exist_ok=True)

    _, meter_columns = _read_header(input_path, timestamp_col)

    LOGGER.info("Dataset: %s", input_path)
    LOGGER.info("Số công tơ: %d", len(meter_columns))
    LOGGER.info("Ngưỡng phân loại: %s", threshold)
    LOGGER.info("Kích thước chunk: %d", chunksize)

    # Pass 1: thống kê mức tiêu thụ trung bình từng công tơ.
    meter_means = calculate_meter_means(
        input_path=input_path,
        meter_columns=meter_columns,
        timestamp_col=timestamp_col,
        chunksize=chunksize,
    )

    household_columns = meter_means[
        meter_means.notna() & (meter_means < threshold)
    ].index.tolist()

    business_columns = meter_means[
        meter_means.notna() & (meter_means >= threshold)
    ].index.tolist()

    # Pass 2: ghi dữ liệu theo hai nhóm.
    household_path = output_dir / "household.csv"
    business_path = output_dir / "business.csv"

    LOGGER.info(
        "Phân loại xong: %d hộ gia đình, %d doanh nghiệp",
        len(household_columns),
        len(business_columns),
    )

    household_rows = write_group_csv(
        input_path=input_path,
        output_path=household_path,
        selected_columns=household_columns,
        timestamp_col=timestamp_col,
        chunksize=chunksize,
    )

    business_rows = write_group_csv(
        input_path=input_path,
        output_path=business_path,
        selected_columns=business_columns,
        timestamp_col=timestamp_col,
        chunksize=chunksize,
    )

    # Lưu bảng thống kê để kiểm tra ngưỡng và kết quả phân loại.
    summary = pd.DataFrame({
        "meter": meter_means.index,
        "mean_consumption": meter_means.values,
    })
    summary["group"] = summary["mean_consumption"].apply(
        lambda value: (
            "household" if pd.notna(value) and value < threshold
            else "business" if pd.notna(value)
            else "unclassified"
        )
    )
    summary = summary.sort_values("mean_consumption", na_position="last")
    summary_path = output_dir / "meter_classification_summary.csv"
    summary.to_csv(summary_path, index=False)

    result = {
        "threshold": threshold,
        "total_meters": len(meter_columns),
        "household_meters": len(household_columns),
        "business_meters": len(business_columns),
        "unclassified_meters": int(meter_means.isna().sum()),
        "household_rows": household_rows,
        "business_rows": business_rows,
        "household_file": str(household_path),
        "business_file": str(business_path),
        "summary_file": str(summary_path),
    }

    LOGGER.info("=" * 60)
    LOGGER.info("HOÀN TẤT TÁCH DATASET")
    for key, value in result.items():
        LOGGER.info("%s: %s", key, value)
    LOGGER.info("=" * 60)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tách CSV điện năng thành nhóm hộ gia đình và doanh nghiệp."
    )
#     parser.add_argument(
#         "--input",
#         required=True,
#         help="Đường dẫn file CSV đầu vào.",
#     )
#     parser.add_argument(
#         "--output-dir",
#         required=True,
#         help="Thư mục chứa các file CSV đầu ra.",
#     )
    parser.add_argument(
        "--threshold",
        # required=True,
        type=float,
        help="Ngưỡng mean tiêu thụ mỗi công tơ, cùng đơn vị với dữ liệu gốc.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Tên cột thời gian (mặc định: timestamp).",
    )
    parser.add_argument(
        "--chunksize",
        default=20_000,
        type=int,
        help="Số dòng mỗi chunk (mặc định: 20000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Mức log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # split_electricity_dataset(
    #     input_path=args.input,
    #     output_dir=args.output_dir,
    #     threshold=args.threshold,
    #     timestamp_col=args.timestamp_col,
    #     chunksize=args.chunksize,
    # )
    split_electricity_dataset(
            input_path=r"/home/thang/workspace/lightgbm/backend/dataset/hourly_electricity_2013_2014.csv",
            output_dir=r"/home/thang/workspace/lightgbm/backend/dataset",
            threshold=500,
            timestamp_col=args.timestamp_col,
            chunksize=args.chunksize,
        )


if __name__ == "__main__":
    main()
