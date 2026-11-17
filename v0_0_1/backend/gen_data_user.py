import pandas as pd
import matplotlib.pyplot as plt

# ==== helper debug ====

def debug_df(df, name):
    print(f"\n=== {name} ===")
    print(df.head())
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Describe:\n", df.describe())

# ==== load dữ liệu gốc ====

print("Loading raw data...")

df = pd.read_csv(
r"/home/thang/workspace/lightgbm/backend/dataset/LD2011_2014.txt",
sep=';',
decimal=',',
parse_dates=[0]
)

df.columns = ["time"] + list(df.columns[1:])

debug_df(df, "RAW DATA")

# ==== chọn 1 user ====

user_col = df.columns[316]

user_df = df[["time", user_col]].rename(
columns={user_col: "value_kw"}
)

user_df = user_df.sort_values("time").dropna()

debug_df(user_df, "USER DATA (RAW)")

# ==== lọc năm 2014 ====

user_df = user_df[user_df["time"].dt.year == 2014]

print("\n=== AFTER YEAR FILTER ===")
print("Time range:", user_df["time"].min(), "->", user_df["time"].max())
print("Rows:", len(user_df))

# ==== convert kW → kWh (15 phút) ====

user_df["value_kwh"] = user_df["value_kw"] / 4

print("\n=== AFTER kW -> kWh ===")
print(user_df[["value_kw", "value_kwh"]].head(10))

# ==== resample về 1 giờ (sum điện năng) ====

user_df = user_df.set_index("time")

user_df = user_df["value_kwh"].resample("1H").sum()

# ⚠️ lúc này là Series → convert lại DataFrame

user_df = user_df.to_frame(name="value")

user_df = user_df.reset_index()

debug_df(user_df, "AFTER RESAMPLE 1H")

# ==== kiểm tra time diff ====

print("\n=== TIME DIFF CHECK ===")
print(user_df["time"].diff().value_counts().head())

# ==== kiểm tra missing timestamp ====

full_range = pd.date_range(
start=user_df["time"].min(),
end=user_df["time"].max(),
freq="1H"
)

missing = set(full_range) - set(user_df["time"])

print("\nMissing timestamps:", len(missing))

# ==== kiểm tra mean < 500 kWh (khớp miền dữ liệu train < 500 kWh) ====

MAX_MEAN_KWH = 500.0
mean_kwh = user_df["value"].mean()
print(f"\nMean consumption of {user_col} in 2014: {mean_kwh:.2f} kWh")

if mean_kwh >= MAX_MEAN_KWH or mean_kwh <= 0:
    raise ValueError(
        f"User {user_col} có mean = {mean_kwh:.2f} kWh (cần 0 < mean < {MAX_MEAN_KWH} kWh)."
    )

# ==== lấy A + B = 14 ngày (336 giờ) ====
# - A = 7 ngày đầu (168 giờ): đưa vào AI để dự đoán ra C
# - B = 7 ngày tiếp theo (168 giờ): dữ liệu thực tế dùng để đối chiếu với C (hỗ trợ cả chế độ 24h và 7 ngày)
HISTORY_DAYS = 7   # A: 168 giờ lịch sử
ACTUAL_DAYS = 7    # B: 168 giờ thực tế để so sánh với dự đoán C

# Lấy giai đoạn tập Test (tháng 12/2014) hoặc 14 ngày liên tiếp có mean < 500 kWh
test_slice = user_df[user_df["time"] >= "2014-12-01"].head(24 * (HISTORY_DAYS + ACTUAL_DAYS))
if len(test_slice) == 24 * (HISTORY_DAYS + ACTUAL_DAYS) and test_slice["value"].mean() < MAX_MEAN_KWH:
    user_df = test_slice.reset_index(drop=True)
else:
    user_df = user_df.head(24 * (HISTORY_DAYS + ACTUAL_DAYS)).reset_index(drop=True)

debug_df(user_df, f"FINAL DATA (A={HISTORY_DAYS}d + B={ACTUAL_DAYS}d = {len(user_df)}h, mean={user_df['value'].mean():.2f} kWh)")

# ==== save ====

output_path = "/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_0/user_data_316.csv"

user_df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")
print(f"  - Phần A (lịch sử cho AI dự đoán): { HISTORY_DAYS * 24 } giờ ({user_df['time'].iloc[0]} -> {user_df['time'].iloc[HISTORY_DAYS * 24 - 1]})")
print(f"  - Phần B (thực tế để so với C)  : { ACTUAL_DAYS * 24 } giờ ({user_df['time'].iloc[HISTORY_DAYS * 24]} -> {user_df['time'].iloc[-1]})")

# ==== plot để nhìn trực quan ====

print("\nPlotting data...")

user_df.set_index("time")["value"].plot(figsize=(12, 4))
plt.title("Electricity Consumption (Hourly)")
plt.xlabel("Time")
plt.ylabel("kWh")
plt.show()
