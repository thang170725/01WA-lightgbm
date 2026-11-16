"""
Script chạy nhanh tất cả 8 biểu đồ EDA và lưu vào backend/dataset/images/
Chạy từ thư mục backend/:  python run_eda.py
"""
import matplotlib
matplotlib.use("Agg")  # Backend không cần GUI — chỉ lưu file

from app.utils.visualizations import VisualDatasetTXTV1

print("=" * 60)
print("  ĐANG TẢI DATASET... (file ~680MB, có thể mất 1-2 phút)")
print("=" * 60)

visual = VisualDatasetTXTV1(dataset_path="dataset/LD2011_2014.txt")
print(f"  ✓ Đã tải xong: {visual.rows} dòng, {len(visual.columns)-1} users")
print()

SAVE_PATH = "dataset/images"

# --- 1. Line plot từng user ---
print("[1/8] Đang vẽ: Daily Load Chart (27 users, 3 pages)...")
visual.dayly_load_chart(
    images=3, size_rnd=27,
    space_chart=(0.5, 0.5), rc_chart=(3, 3),
    save=True, save_path=SAVE_PATH
)
print("  ✓ Lưu: users_page_1.png, users_page_2.png, users_page_3.png")

# --- 2. Aggregate load 24h ---
print("[2/8] Đang vẽ: Aggregate Load Chart (9 ngày ngẫu nhiên)...")
visual.plot_aggregate_load_chart(
    size_rnd=9, rc_chart=(3, 3),
    save=True, save_path=SAVE_PATH
)
print("  ✓ Lưu: aggregate_random_days.png")

# --- 3. Heatmap giờ × ngày ---
print("[3/8] Đang vẽ: Time-Pattern Heatmap...")
visual.plot_time_pattern_heatmap(
    figsize=(15, 8),
    save=True, save_path=SAVE_PATH
)
print("  ✓ Lưu: time_pattern_heatmap_original.png")

# --- 4. Histogram ---
print("[4/8] Đang vẽ: Histogram phân phối tổng tải...")
visual.plot_histogram(
    figsize=(12, 6),
    save=True, save_path=SAVE_PATH
)
print("  ✓ Lưu: aggregate_histogram.png")

# --- 5. Boxplot theo user ---
print("[5/8] Đang vẽ: Random Users Boxplot...")
visual.plot_random_users_boxplot(
    save=True, save_path=SAVE_PATH, show=False
)
print("  ✓ Lưu: users_boxplot_random.png")

# --- 6. ACF theo user ---
print("[6/8] Đang vẽ: Random Users ACF...")
visual.plot_random_users_acf(
    save=True, save_path=SAVE_PATH, show=False
)
print("  ✓ Lưu: users_acf_random.png")

# --- 7. Monthly Boxplot ---
print("[7/8] Đang vẽ: Monthly Boxplot...")
visual.plot_monthly_boxplot(
    save=True, save_path=SAVE_PATH, show=False
)
print("  ✓ Lưu: monthly_boxplot.png")

# --- 8. Monthly ACF ---
print("[8/8] Đang vẽ: Monthly ACF...")
visual.plot_monthly_acf(
    save=True, save_path=SAVE_PATH, show=False
)
print("  ✓ Lưu: monthly_acf.png")

print()
print("=" * 60)
print("  ✅ HOÀN TẤT! Tất cả biểu đồ đã lưu tại:")
print(f"     {SAVE_PATH}/")
print("=" * 60)

