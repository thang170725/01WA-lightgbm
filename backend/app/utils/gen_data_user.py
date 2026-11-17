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

# ==== lấy 8 ngày (đủ lag_7d) ====

user_df = user_df.head(24 * 8)

debug_df(user_df, "FINAL DATA (8 DAYS)")

# ==== save ====

output_path = "/home/thang/workspace/lightgbm/v0_0_1/data/0_0_1_0/user_data_316.csv"

user_df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")

# ==== plot để nhìn trực quan ====

print("\nPlotting data...")

user_df.set_index("time")["value"].plot(figsize=(12, 4))
plt.title("Electricity Consumption (Hourly)")
plt.xlabel("Time")
plt.ylabel("kWh")
plt.show()
