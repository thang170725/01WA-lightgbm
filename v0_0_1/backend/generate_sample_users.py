import json
import random
from pathlib import Path
import pandas as pd


def extract_slice(
    df_tier: pd.DataFrame,
    ts_col: str,
    col_name: str,
    history_days: int = 7,
    actual_days: int = 1,
    prefer_start: str = "2014-12-01 00:00:00",
    rng: random.Random | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Trích xuất chuỗi (A = 7 ngày lịch sử 168h + B = 1 ngày thực tế 24h = 192h) dạng cột [time, value]."""
    total_hours = (history_days + actual_days) * 24

    sub = pd.DataFrame({
        "time": pd.to_datetime(df_tier[ts_col], errors="coerce"),
        "value": pd.to_numeric(df_tier[col_name], errors="coerce").fillna(0.0),
    }).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Ưu tiên cửa sổ từ prefer_start (tập Test tháng 12/2014) và căn theo đầu ngày 00:00:00
    start_ts = pd.to_datetime(prefer_start)
    after_start = sub[sub["time"] >= start_ts].reset_index(drop=True)

    if len(after_start) >= total_hours:
        max_days_offset = (len(after_start) - total_hours) // 24
        offset = (rng.randint(0, max_days_offset) * 24) if (rng and max_days_offset > 0) else 0
        window_df = after_start.iloc[offset : offset + total_hours].copy()
    else:
        if len(sub) < total_hours:
            raise ValueError(f"Cột {col_name} chỉ có {len(sub)} giờ, không đủ {total_hours} giờ.")
        max_days_offset = (len(sub) - total_hours) // 24
        offset = (rng.randint(0, max_days_offset) * 24) if (rng and max_days_offset > 0) else 0
        window_df = sub.iloc[offset : offset + total_hours].copy()

    window_df = window_df.reset_index(drop=True)
    window_df["time"] = window_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    a_end = history_days * 24 - 1
    b_start = history_days * 24
    info_a = f"{window_df['time'].iloc[0]} -> {window_df['time'].iloc[a_end]}"
    info_b = f"{window_df['time'].iloc[b_start]} -> {window_df['time'].iloc[-1]}"
    return window_df[["time", "value"]], info_a, info_b


def generate_6_user_files(
    tier_dir: Path,
    output_dir: Path,
    samples_per_tier: int = 2,
    history_days: int = 7,
    actual_days: int = 1,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tier_files = [
        ("low", "load_low_household.csv"),
        ("medium", "load_medium_small_office.csv"),
        ("high", "load_high_enterprise.csv"),
    ]

    manifest = []

    for tier_name, filename in tier_files:
        csv_path = tier_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

        df_tier = pd.read_csv(csv_path)
        ts_col = "timestamp" if "timestamp" in df_tier.columns else df_tier.columns[0]
        candidate_cols = [c for c in df_tier.columns if c != ts_col]

        if len(candidate_cols) < samples_per_tier:
            raise ValueError(f"Nhóm {tier_name} có ít hơn {samples_per_tier} cột.")

        chosen_cols = rng.sample(candidate_cols, k=samples_per_tier)

        for idx, col_name in enumerate(chosen_cols, start=1):
            out_df, range_a, range_b = extract_slice(
                df_tier=df_tier,
                ts_col=ts_col,
                col_name=col_name,
                history_days=history_days,
                actual_days=actual_days,
                rng=rng,
            )

            out_name = f"user_{tier_name}_{idx}_{col_name}.csv"
            out_path = output_dir / out_name
            out_df.to_csv(out_path, index=False)

            mean_val = float(out_df["value"].mean())
            print(
                f"[{tier_name.upper():6s} #{idx}] {col_name} -> {out_name} "
                f"| rows={len(out_df)} (A={history_days*24}h + B={actual_days*24}h) "
                f"| mean={mean_val:.2f} kWh | A: ({range_a}) | B: ({range_b})"
            )

            manifest.append({
                "file": out_name,
                "tier": tier_name,
                "client_id": col_name,
                "rows": len(out_df),
                "history_hours": history_days * 24,
                "actual_hours": actual_days * 24,
                "mean_kwh": round(mean_val, 4),
                "part_A_history": range_a,
                "part_B_actual": range_b,
            })

    manifest_path = output_dir / "sample_users_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nĐã tạo xong 6 file user tại: {output_dir}")
    print(f"File mô tả danh sách: {manifest_path}")


if __name__ == "__main__":
    generate_6_user_files(
        tier_dir=Path("/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_1"),
        output_dir=Path("/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_1/sample_users"),
        samples_per_tier=2,
        history_days=7,
        actual_days=1,
        seed=42,
    )