import pandas as pd

# === tạo dataset hourly và lưu ra csv ===
# vì dataset gốc quá lớn nên cần cắt giảm để tối ưu hơn
def build_hourly_dataset(
    input_path="backend/dataset/LD2011_2014.txt",                   # đường dẫn dataset gốc
    output_path="backend/dataset/hourly_electricity_filtered.csv",  # đường dẫn dataset sau khi xử lý (lưu ý đuôi file là .csv)
    top_clients=180,                                                # lấy 180 người tốt nhất thay vì 370 người của dataset ban đầu
    min_mean_load=1.0,
    keep_year="2014",                                                # chỉ giữ 1 năm cuối
    save: bool = False
):
    # ===== 1. Đọc file & đổi tên cột (để dễ thao tác) =====
    df = pd.read_csv(
        input_path,
        sep=";",
        decimal=",",
        parse_dates=[0]
    )
    df.rename(  # đổi tên cột
        columns={"Unnamed: 0": "timestamp"},
        inplace=True
    )
    df.set_index(
        "timestamp",
        inplace=True
    )

    # ===== 2. Giảm kích thước để tối ưu mô hình ======
    df = df.astype("float32")

    # ===== 3. Đổi dữ liệu từ 15 phút -> 1 giờ =====
    df = df/4
    df_hourly = df.resample("1h").sum()

    # ===== 4. Lọc (giảm dataset giúp chương trình train nhanh hơn)
    df_hourly = df_hourly.loc[keep_year]
    print("After year filter:", df_hourly.shape)

    # ===== 5. Chọn active clients (để chọn ra dữ liệu tốt nhất)====
    # ===== 5.1. tính phân trăm người dùng có mức tiêu thụ > 0 =====
    # client_1 → 0.95  (95% thời gian có điện)
    # client_2 → 0.10  (chủ yếu = 0 → rác)
    active_ratio = (df_hourly > 0).mean()

    # ===== 5.2. tính mức tiêu thụ trung bìn từng user ======
    # client_1 → 3.2 kW
    # client_2 → 0.1 kW
    mean_load = df_hourly.mean()

    # ===== 5.3. gom lại thành bảng stats =======
    client_stats = pd.DataFrame({
        "active_ratio": active_ratio,
        "mean_load": mean_load
    })

    # ===== 5.4. Chọn clients tốt nhất ======
    client_stats = client_stats[client_stats["mean_load"] >= min_mean_load]
    keep_clients = (            
        client_stats
        .sort_values(
            by="active_ratio",
            ascending=False
        )
        .head(top_clients)
        .index
        .tolist()
    )
    df_hourly = df_hourly[keep_clients]
    print("Rows, Cols:", df_hourly.shape)
    print("Clients kept:", len(keep_clients))

    # ===== 6. Save =====
    if save == True:
        df_hourly.to_csv(
            output_path,
            index_label="timestamp",
            float_format="%.4f"
        )
        print(f"Saved: {output_path}")
   
    return df_hourly

if __name__ == "__main__":
    build_hourly_dataset(
        input_path="backend/dataset/LD2011_2014.txt",
        output_path="backend/dataset/hourly_electricity_filtered_v2.csv",
        top_clients=180,
        min_mean_load=3.0,
        keep_year="2014",
        save=True
    )