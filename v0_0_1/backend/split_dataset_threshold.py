import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("LoadTierSplitter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def detect_timestamp_col(df: pd.DataFrame, explicit_col: str | None) -> str:
    if explicit_col:
        if explicit_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột thời gian '{explicit_col}' trong file CSV.")
        return explicit_col

    common_names = ["timestamp", "datetime", "date", "time"]
    for name in common_names:
        if name in df.columns:
            return name
    return df.columns[0]


def max_consecutive_zeros(series: pd.Series) -> int:
    """Tính số giờ (số bản ghi liên tiếp) bằng 0 dài nhất trong chuỗi."""
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy()
    is_zero = arr == 0.0
    if not np.any(is_zero):
        return 0

    max_streak = 0
    current = 0
    for z in is_zero:
        if z:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


def active_median_kwh(series: pd.Series) -> float:
    """Tính median trên các giá trị > 0 (tránh bị kéo thấp bởi chuỗi 0 ngắn/ngày đổi giờ)."""
    vals = pd.to_numeric(series, errors="coerce")
    positive_vals = vals[vals > 0]
    if len(positive_vals) == 0:
        return 0.0
    return float(positive_vals.median())


def plot_visualizations(
    df_hourly: pd.DataFrame,
    ts_col: str,
    stats_df: pd.DataFrame,
    low_thresh: float,
    high_thresh: float,
    max_zero_streak_hours: int,
    fig_dir: Path,
    logger: logging.Logger,
) -> None:
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # 1. Phân bố chuỗi 0 liên tiếp tối đa của các cột
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(stats_df["max_zero_streak_hours"], bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(
        max_zero_streak_hours,
        color="#D62728",
        linestyle="--",
        linewidth=2,
        label=f"Ngưỡng loại bỏ: >= {max_zero_streak_hours}h",
    )
    ax.set_title("Phân bố chuỗi giá trị 0 liên tiếp dài nhất theo từng cột")
    ax.set_xlabel("Số giờ liên tiếp bằng 0 tối đa")
    ax.set_ylabel("Số lượng cột (MT)")
    ax.legend()
    fig.tight_layout()
    p1 = fig_dir / "01_zero_streak_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 2. Đường cong median kWh/h đã sắp xếp và 2 ngưỡng chia 3 nhóm
    kept = stats_df[stats_df["kept_after_zero_filter"]].sort_values("active_median_kwh").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(range(len(kept)), kept["active_median_kwh"], marker=".", markersize=3, linewidth=1.5, label="Median (kWh/h, >0)")
    ax.axhline(low_thresh, color="#2CA02C", linestyle="--", linewidth=1.8, label=f"Ngưỡng Thấp/TB: {low_thresh:.4f} kWh/h")
    ax.axhline(high_thresh, color="#D62728", linestyle="--", linewidth=1.8, label=f"Ngưỡng TB/Cao: {high_thresh:.4f} kWh/h")
    ax.set_yscale("log")
    ax.set_title("Phân bố Median phụ tải (thang log) và ngưỡng chia 3 cấp")
    ax.set_xlabel("Thứ tự cột giữ lại (xếp theo median tăng dần)")
    ax.set_ylabel("Median điện năng (kWh/h, log scale)")
    ax.legend()
    fig.tight_layout()
    p2 = fig_dir / "02_median_sorted_and_thresholds.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # 3. Hồ sơ tiêu thụ trung bình theo giờ trong ngày (0..23h) của 3 nhóm
    hourly_with_hour = df_hourly.copy()
    hourly_with_hour["_dt"] = pd.to_datetime(hourly_with_hour[ts_col], errors="coerce")
    hourly_with_hour["_hour"] = hourly_with_hour["_dt"].dt.hour

    tier_map = dict(zip(kept["column"], kept["tier"]))
    low_cols = [c for c, t in tier_map.items() if t == "low"]
    med_cols = [c for c, t in tier_map.items() if t == "medium"]
    high_cols = [c for c, t in tier_map.items() if t == "high"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    groups_info = [
        ("Thấp (Hộ gia đình)", low_cols, "#2CA02C", axes[0]),
        ("Trung bình (Cụm hộ / VP nhỏ)", med_cols, "#1F77B4", axes[1]),
        ("Cao (Doanh nghiệp / Tòa nhà)", high_cols, "#D62728", axes[2]),
    ]
    for title, cols, color, ax in groups_info:
        if cols:
            prof = hourly_with_hour.groupby("_hour")[cols].mean().mean(axis=1)
            ax.plot(prof.index, prof.values, marker="o", markersize=4, color=color, linewidth=2)
        ax.set_title(f"{title}\n(n={len(cols)} cột)")
        ax.set_xlabel("Giờ trong ngày (0 - 23)")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("Trung bình kWh/h mỗi cột")
    fig.suptitle("Nhịp tiêu thụ trung bình theo khung giờ của 3 nhóm phụ tải", y=1.02)
    fig.tight_layout()
    p3 = fig_dir / "03_group_hourly_profiles.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Số lượng cột mỗi nhóm
    counts = [len(low_cols), len(med_cols), len(high_cols)]
    labels = ["Thấp\n(Hộ gia đình)", "Trung bình\n(Cụm hộ / VP nhỏ)", "Cao\n(Doanh nghiệp / Tòa nhà)"]
    colors = ["#2CA02C", "#1F77B4", "#D62728"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, counts, color=colors, width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts + [1]) * 0.02,
            str(cnt),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Số lượng cột sau khi lọc và phân loại 3 cấp")
    ax.set_ylabel("Số cột (MT)")
    fig.tight_layout()
    p4 = fig_dir / "04_group_column_counts.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    logger.info("Đã xuất 4 biểu đồ trực quan vào thư mục: %s", fig_dir)


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    timestamp_col: str | None,
    max_zero_streak_hours: int,
    q1: float,
    q2: float,
) -> None:
    if not (0 < q1 < q2 < 1):
        raise ValueError("Tham số phân vị phải thỏa mãn 0 < q1 < q2 < 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "charts"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "pipeline.log"
    logger = setup_logger(log_path)

    logger.info("=== BẮT ĐẦU PIPELINE PHÂN TẦNG PHỤ TẢI ĐIỆN NĂNG ===")
    logger.info("File CSV đầu vào: %s", input_csv.resolve())
    logger.info("Thư mục kết quả: %s", output_dir.resolve())
    logger.info("Ngưỡng loại chuỗi 0 liên tiếp: >= %d giờ", max_zero_streak_hours)
    logger.info("Phân vị chia 3 nhóm: q1=%.4f (Thấp/TB), q2=%.4f (TB/Cao)", q1, q2)

    # Bước 1: Đọc dữ liệu
    df = pd.read_csv(input_csv)
    ts_col = detect_timestamp_col(df, timestamp_col)
    feature_cols = [c for c in df.columns if c != ts_col]
    logger.info(
        "Đã đọc file: %d dòng, %d cột dữ liệu (cột thời gian: '%s').",
        len(df),
        len(feature_cols),
        ts_col,
    )

    # Ép kiểu số cho các cột dữ liệu
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Bước 2: Kiểm tra chuỗi 0 liên tiếp và loại bỏ cột có chuỗi 0 dài ngày
    records = []
    for col in feature_cols:
        streak = max_consecutive_zeros(df[col])
        med_active = active_median_kwh(df[col])
        mean_all = float(df[col].mean())
        zero_ratio = float((df[col] == 0.0).mean())
        kept = streak < max_zero_streak_hours
        records.append(
            {
                "column": col,
                "max_zero_streak_hours": streak,
                "zero_ratio": zero_ratio,
                "mean_kwh": mean_all,
                "active_median_kwh": med_active,
                "kept_after_zero_filter": kept,
            }
        )

    stats_df = pd.DataFrame(records)
    dropped_df = stats_df[~stats_df["kept_after_zero_filter"]]
    kept_df = stats_df[stats_df["kept_after_zero_filter"]].copy()

    logger.info(
        "Bước lọc chuỗi 0 liên tiếp (>= %d giờ): giữ lại %d cột, loại bỏ %d cột.",
        max_zero_streak_hours,
        len(kept_df),
        len(dropped_df),
    )
    if len(dropped_df) > 0:
        dropped_summary = ", ".join(
            f"{r.column} (max_0={r.max_zero_streak_hours}h)" for r in dropped_df.itertuples()
        )
        logger.info("Danh sách cột bị loại: %s", dropped_summary)

    if len(kept_df) < 3:
        raise ValueError("Số cột còn lại sau khi lọc quá ít (< 3) để chia thành 3 nhóm.")

    # Bước 3: Tính ngưỡng chia 3 nhóm dựa trên active_median_kwh của các cột giữ lại
    low_thresh = float(kept_df["active_median_kwh"].quantile(q1))
    high_thresh = float(kept_df["active_median_kwh"].quantile(q2))

    logger.info("Ngưỡng phân tầng (tính theo active_median_kwh của các cột giữ lại):")
    logger.info("  - Cấp 1 [Thấp - Hộ gia đình]: median <= %.4f kWh/h", low_thresh)
    logger.info("  - Cấp 2 [Trung bình - Cụm hộ / VP nhỏ]: %.4f < median <= %.4f kWh/h", low_thresh, high_thresh)
    logger.info("  - Cấp 3 [Cao - Doanh nghiệp / Tòa nhà]: median > %.4f kWh/h", high_thresh)

    def assign_tier(val: float) -> str:
        if val <= low_thresh:
            return "low"
        elif val <= high_thresh:
            return "medium"
        return "high"

    kept_df["tier"] = kept_df["active_median_kwh"].apply(assign_tier)
    stats_df = stats_df.merge(kept_df[["column", "tier"]], on="column", how="left")
    stats_df["tier"] = stats_df["tier"].fillna("dropped")

    # Bước 4: Tách và lưu 3 file CSV theo từng nhóm
    tier_configs = [
        ("low", "load_low_household.csv", "Thấp (Hộ gia đình)"),
        ("medium", "load_medium_small_office.csv", "Trung bình (Cụm gia đình / Văn phòng nhỏ)"),
        ("high", "load_high_enterprise.csv", "Cao (Doanh nghiệp / Công ty lớn / Tòa nhà)"),
    ]

    for tier_key, out_filename, label in tier_configs:
        cols_in_tier = kept_df.loc[kept_df["tier"] == tier_key, "column"].tolist()
        sub_df = df[[ts_col] + cols_in_tier]
        out_csv_path = output_dir / out_filename
        sub_df.to_csv(out_csv_path, index=False)

        tier_medians = kept_df.loc[kept_df["tier"] == tier_key, "active_median_kwh"]
        logger.info(
            "Nhóm %-42s -> %s | Số cột: %d | Khoảng median: [%.4f .. %.4f] kWh/h",
            label,
            out_filename,
            len(cols_in_tier),
            float(tier_medians.min()) if len(tier_medians) else 0.0,
            float(tier_medians.max()) if len(tier_medians) else 0.0,
        )

    # Lưu bảng thống kê chi tiết từng cột để tiện đối chiếu trong báo cáo
    summary_table_path = output_dir / "columns_summary.csv"
    stats_df.to_csv(summary_table_path, index=False)
    logger.info("Đã lưu bảng thống kê từng cột: %s", summary_table_path.name)

    # Bước 5: Vẽ biểu đồ chứng minh
    plot_visualizations(
        df_hourly=df,
        ts_col=ts_col,
        stats_df=stats_df,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        max_zero_streak_hours=max_zero_streak_hours,
        fig_dir=fig_dir,
        logger=logger,
    )

    logger.info("=== HOÀN TẤT PIPELINE ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lọc chuỗi 0 dài ngày và tách dữ liệu điện năng theo giờ thành 3 cấp phụ tải."
    )
    parser.add_argument("--input-csv", type=Path, 
        default=Path(r"/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_1/dataset_2013_2014.csv"), help="Đường dẫn file CSV đầu vào (đã gom theo giờ).")
    parser.add_argument("--output-dir", type=Path, 
        default=Path(r"/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_1"), help="Thư mục lưu file CSV kết quả, log và ảnh.")
    parser.add_argument("--timestamp-col", type=str, default=None, help="Tên cột thời gian (mặc định tự nhận diện).")
    parser.add_argument(
        "--max-zero-streak-hours",
        type=int,
        default=72,
        help="Loại bỏ cột nếu có chuỗi giá trị 0 liên tiếp >= số giờ này (mặc định: 72 giờ = 3 ngày).",
    )
    parser.add_argument("--q1", type=float, default=1 / 3, help="Phân vị ranh giới Thấp / Trung bình (mặc định: 0.3333).")
    parser.add_argument("--q2", type=float, default=2 / 3, help="Phân vị ranh giới Trung bình / Cao (mặc định: 0.6667).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        timestamp_col=args.timestamp_col,
        max_zero_streak_hours=args.max_zero_streak_hours,
        q1=args.q1,
        q2=args.q2,
    )