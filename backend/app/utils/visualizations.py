"""
==========================================================================
FILE: visualizations.py
MỤC ĐÍCH: Khám phá dữ liệu (EDA — Exploratory Data Analysis) bằng biểu đồ
           trước khi tiến hành xử lý & train model.

DATASET GỐC: LD2011_2014.txt (UCI Electricity Load Diagrams 2011-2014)
  - File CSV dạng ";" phân cách, dấu thập phân là ","
  - Cột đầu: timestamp (mỗi 15 phút), các cột còn lại: 370 khách hàng
  - Mỗi ô = lượng điện năng tiêu thụ (kWh) trong khoảng 15 phút đó

CÁC BIỂU ĐỒ TRONG CLASS VisualDatasetTXTV1:
  ┌─────────────────────────────────┬─────────────────────────────────────────┐
  │ Hàm                             │ Mô tả                                   │
  ├─────────────────────────────────┼─────────────────────────────────────────┤
  │ dayly_load_chart()              │ Line plot: biến thiên điện từng user     │
  │ plot_aggregate_load_chart()     │ Line plot: tổng tải 370 user trong 24h   │
  │ plot_time_pattern_heatmap()     │ Heatmap: giờ × ngày trong tuần           │
  │ plot_histogram()                │ Histogram: phân phối tổng tải            │
  │ plot_random_users_boxplot()     │ Boxplot: so sánh phân phối giữa users    │
  │ plot_random_users_acf()         │ ACF: kiểm tra tính chu kỳ (seasonal)     │
  │ plot_monthly_boxplot()          │ Boxplot: phân phối tiêu thụ theo tháng   │
  │ plot_monthly_acf()              │ ACF: tính chu kỳ dài hạn theo tháng      │
  └─────────────────────────────────┴─────────────────────────────────────────┘

CÁCH CHẠY (từ thư mục backend/):
    python -m app.utils.visualizations

Ảnh sẽ được lưu tại backend/dataset/images/
==========================================================================
"""

import matplotlib
matplotlib.use("Agg")  # Dùng backend không cần GUI → chạy được trên server headless

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.plotting import autocorrelation_plot


# ===========================================================================
# CLASS: VisualDatasetTXTV1
# ---------------------------------------------------------------------------
# Class chuyên biệt để vẽ biểu đồ EDA từ dataset gốc dạng .txt (UCI).
# Khi khởi tạo, class sẽ đọc toàn bộ file vào self.df (DataFrame).
# Sau đó gọi từng hàm để vẽ biểu đồ tương ứng.
# ===========================================================================
class VisualDatasetTXTV1:
    """
    Class thực hiện Exploratory Data Analysis (EDA) cho dataset điện năng UCI.

    Parameters
    ----------
    dataset_path : str
        Đường dẫn tới file LD2011_2014.txt (dataset gốc từ UCI).
        Ví dụ: "dataset/LD2011_2014.txt" (khi chạy từ backend/)

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame chứa toàn bộ dữ liệu, cột đầu là "timestamp",
        các cột còn lại là tên khách hàng (MT_001, MT_002, ..., MT_370).
    columns : pd.Index
        Danh sách tên cột (bao gồm "timestamp" và tên 370 users).
    rows : int
        Tổng số dòng dữ liệu (≈ 140256 dòng cho 4 năm, mỗi 15 phút).
    """

    def __init__(self, dataset_path: str = None):
        # -------------------------------------------------------------------
        # ĐỌC DỮ LIỆU:
        # - sep=";"       → file dùng dấu chấm phẩy ngăn cách cột
        # - decimal=","   → số thập phân dùng dấu phẩy (kiểu châu Âu)
        #                   Ví dụ: "12,5" được đọc thành 12.5
        # - parse_dates=[0] → tự động parse cột đầu tiên thành datetime
        # -------------------------------------------------------------------
        self.df: pd.DataFrame = pd.read_csv(
            dataset_path,
            sep=";",
            decimal=",",
            parse_dates=[0]
        )

        # -------------------------------------------------------------------
        # ĐỔI TÊN CỘT: Cột đầu tiên mặc định tên "Unnamed: 0"
        #   → đổi thành "timestamp" cho dễ dùng
        # -------------------------------------------------------------------
        self.df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)

        # Lưu lại metadata cơ bản
        self.columns = self.df.columns   # ["timestamp", "MT_001", "MT_002", ...]
        self.rows = len(self.df)          # Tổng số hàng (≈ 140256)

    # =======================================================================
    # 1. BIỂU ĐỒ ĐƯỜNG — Mức tiêu thụ điện từng user (Daily Load Chart)
    # =======================================================================
    # MỤC ĐÍCH:
    #   Vẽ line plot cho từng khách hàng ngẫu nhiên để xem xu hướng tiêu thụ
    #   điện qua thời gian. Giúp nhận biết:
    #     - User nào có mức tiêu thụ cao/thấp
    #     - Xu hướng tăng/giảm theo mùa
    #     - User nào có pattern bất thường (đột biến, gián đoạn)
    #
    # CÁCH HOẠT ĐỘNG:
    #   1. Chọn ngẫu nhiên `size_rnd` users từ 370 khách hàng
    #   2. Resample dữ liệu theo 5 ngày (lấy trung bình) → đường mượt hơn
    #   3. Chia thành nhiều page, mỗi page có rc_chart[0] x rc_chart[1] biểu đồ
    #   4. Lưu hoặc hiển thị từng page
    # =======================================================================
    def dayly_load_chart(self,
        images: int,                              # Số lượng page (ảnh) muốn tạo
        size_rnd: int,                            # Tổng số user muốn vẽ (chia đều vào các page)
        space_chart: tuple,                       # Khoảng cách giữa các subplot (wspace, hspace)
        rc_chart: tuple,                          # Grid layout mỗi page: (số_hàng, số_cột)
        save: bool = False,                       # True → lưu file PNG thay vì hiển thị
        save_path: str = "dataset/images"         # Thư mục lưu ảnh (tính từ backend/)
    ):
        """
        Vẽ biểu đồ đường mô tả biến thiên mức sử dụng điện của từng user.

        Ví dụ gọi:
            visual.dayly_load_chart(
                images=3,           # tạo 3 ảnh
                size_rnd=27,        # chọn 27 user ngẫu nhiên
                space_chart=(0.5, 0.5),
                rc_chart=(3, 3),    # mỗi ảnh có 3x3 = 9 biểu đồ
                save=True
            )
            → Tạo 3 file: users_page_1.png, users_page_2.png, users_page_3.png
        """
        # Bước 1: Chọn ngẫu nhiên `size_rnd` user (không trùng lặp)
        # self.columns[1:] → bỏ cột "timestamp", chỉ lấy tên user
        users = np.random.choice(self.columns[1:], size=size_rnd, replace=False)

        # Bước 2: Resample theo 5 ngày rồi lấy trung bình
        # → Giảm nhiễu, đường biểu đồ mượt hơn, dễ nhận diện xu hướng
        # Ví dụ: thay vì 140K điểm (15 phút), chỉ còn ~292 điểm (5 ngày)
        daily_df = (
            self.df
            .set_index("timestamp")   # đặt timestamp làm index để resample
            .resample("5D")           # gộp mỗi 5 ngày
            .mean()                   # lấy giá trị trung bình
            .reset_index()            # đưa timestamp về thành cột bình thường
        )

        # Bước 3: Vẽ biểu đồ, chia thành nhiều page
        per_page = rc_chart[0] * rc_chart[1]  # Số subplot trên mỗi page
        for page in range(images):
            # Lấy nhóm user cho page hiện tại
            # page=0 → users[0:9], page=1 → users[9:18], ...
            subset = users[page * per_page : (page + 1) * per_page]

            # Tạo figure với grid subplot
            fig, axes = plt.subplots(rc_chart[0], rc_chart[1])
            axes = axes.flatten()  # Chuyển 2D array → 1D để dễ index

            for i, user in enumerate(subset):
                sns.lineplot(
                    x=daily_df["timestamp"],  # Trục X: thời gian
                    y=daily_df[user],          # Trục Y: điện năng trung bình 5 ngày
                    ax=axes[i],               # Vẽ lên subplot thứ i
                    linewidth=0.5,            # Đường mảnh để nhìn rõ trend
                    color="black"
                )

                x = daily_df["timestamp"]

                axes[i].set_title(user)          # Tiêu đề = tên user (VD: MT_004)
                axes[i].set_xlabel("")           # Ẩn label trục X (tiết kiệm chỗ)
                axes[i].set_ylabel("")           # Ẩn label trục Y
                # Chỉ hiện 2 tick: ngày đầu và ngày cuối
                axes[i].set_xticks([x.iloc[0], x.iloc[-1]])
                axes[i].tick_params(axis="both", labelsize=5)   # Font nhỏ
                axes[i].tick_params(axis="x", rotation=30)      # Xoay label X

            # Điều chỉnh khoảng cách giữa các subplot
            plt.subplots_adjust(wspace=space_chart[0], hspace=space_chart[1])

            if save:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(
                    f"{save_path}/users_page_{page+1}.png",
                    dpi=300,            # Độ phân giải cao
                    bbox_inches="tight" # Cắt viền thừa
                )
            else:
                plt.show()
            plt.close(fig)  # Đóng figure → giải phóng bộ nhớ

    # =======================================================================
    # 2. BIỂU ĐỒ ĐƯỜNG — Tổng tiêu thụ 370 user trong 24h
    #    (Aggregate Load Chart)
    # =======================================================================
    # MỤC ĐÍCH:
    #   Xem đường tổng lượng điện tiêu thụ của TẤT CẢ 370 khách hàng
    #   trong vòng 24 giờ của một ngày ngẫu nhiên. Giúp nhận biết:
    #     - Pattern trong ngày: thấp lúc đêm, cao lúc trưa/tối
    #     - Giờ cao điểm (peak hours) và giờ thấp điểm (off-peak)
    #     - Sự khác biệt giữa các ngày khác nhau
    #
    # CÁCH HOẠT ĐỘNG:
    #   1. Tính cột "aggregate" = tổng điện 370 user tại mỗi timestamp
    #   2. Chọn ngẫu nhiên `size_rnd` ngày
    #   3. Vẽ mỗi ngày thành 1 subplot: trục X = giờ (0-24), Y = tổng tải
    # =======================================================================
    def plot_aggregate_load_chart(
        self,
        size_rnd=9,                             # Số ngày ngẫu nhiên muốn vẽ
        rc_chart=(3, 3),                        # Grid layout: (hàng, cột)
        save=False,                             # True → lưu file
        save_path="dataset/images"              # Thư mục lưu ảnh (tính từ backend/)
    ):
        """
        Vẽ biểu đồ đường tổng lượng điện tiêu thụ 370 user trong 24h.

        Mỗi subplot là 1 ngày ngẫu nhiên, trục X = giờ (0→24),
        trục Y = tổng lượng điện (aggregate) của 370 user.

        → Nhận biết pattern intra-day: thấp ban đêm, cao ban ngày.
        """
        # Bước 1: Copy DataFrame → không sửa trực tiếp df gốc
        df = self.df.copy()

        # Bước 2: Đảm bảo timestamp là kiểu datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Bước 3: Tạo các feature phụ trợ
        # aggregate: tổng điện năng của 370 user tại mỗi thời điểm (trục Y)
        # df.iloc[:,1:] = bỏ cột "timestamp", lấy 370 cột user, rồi sum theo hàng (axis=1)
        df["aggregate"] = df.iloc[:, 1:].sum(axis=1)

        # date: chỉ lấy phần ngày (bỏ giờ/phút) → để nhóm theo ngày
        df["date"] = df["timestamp"].dt.date

        # hour: giờ dạng thập phân (VD: 14:30 → 14.5) → làm trục X
        df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60

        # Bước 4: Lấy danh sách tất cả ngày, chọn ngẫu nhiên `size_rnd` ngày
        all_days = df["date"].unique()
        random_days = np.random.choice(all_days, size=size_rnd, replace=False)

        # Bước 5: Tạo figure và vẽ từng ngày vào subplot
        fig, axes = plt.subplots(rc_chart[0], rc_chart[1])
        axes = axes.flatten()

        for i, day in enumerate(random_days):
            day_data = df[df["date"] == day]  # Lọc dữ liệu của ngày đó

            sns.lineplot(
                x=day_data["hour"],       # Trục X: giờ trong ngày (0.0 → 23.75)
                y=day_data["aggregate"],  # Trục Y: tổng điện năng 370 user
                ax=axes[i],
                color="black"
            )

            axes[i].set_title(str(day))              # Tiêu đề = ngày (VD: 2014-06-15)
            axes[i].set_xlim(0, 24)                  # Giới hạn trục X: 0h → 24h
            axes[i].set_xticks([0, 6, 12, 18, 24])  # Tick mỗi 6 giờ
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="both", labelsize=5)

        # Tắt các subplot thừa (nếu size_rnd < tổng số ô)
        for j in range(len(random_days), len(axes)):
            axes[j].axis("off")

        plt.suptitle("Aggregate Load Profiles of Random Days", fontsize=14)
        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/aggregate_random_days.png",
                dpi=300,
                bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close(fig)

    # =======================================================================
    # 3. HEATMAP — Mẫu thời gian tiêu thụ theo giờ × ngày trong tuần
    #    (Time-Pattern Heatmap)
    # =======================================================================
    # MỤC ĐÍCH:
    #   Vẽ heatmap biểu diễn TRUNG BÌNH lượng điện tiêu thụ tại mỗi
    #   GIỜ (0→23) cho từng NGÀY TRONG TUẦN (Thứ 2 → Chủ Nhật).
    #   Giúp nhận biết:
    #     - Giờ cao điểm / thấp điểm trong ngày
    #     - Sự khác biệt ngày thường vs cuối tuần
    #     - Pattern theo mùa làm việc (sáng, trưa, tối)
    #
    # CÁCH HOẠT ĐỘNG:
    #   1. Tính aggregate (tổng 370 user) cho mỗi timestamp
    #   2. Thêm cột hour, day_name, day_of_week
    #   3. GroupBy [day_name, hour] → tính mean → pivot thành bảng 7×24
    #   4. Vẽ heatmap bằng seaborn
    #
    # ĐỌC KẾT QUẢ:
    #   - Ô đỏ đậm → tiêu thụ rất cao (VD: 12h-18h ngày thường)
    #   - Ô xanh đậm → tiêu thụ rất thấp (VD: 2h-5h sáng)
    #   - So sánh hàng Saturday/Sunday vs Monday-Friday → cuối tuần thấp hơn
    # =======================================================================
    def plot_time_pattern_heatmap(self,
        figsize=(15, 8),                        # Kích thước figure
        save=False,
        save_path="dataset/images"              # Thư mục lưu ảnh (tính từ backend/)
    ):
        """
        Vẽ heatmap trung bình tiêu thụ theo giờ × ngày trong tuần.

        Trục X: Giờ (0-23)
        Trục Y: Ngày trong tuần (Monday → Sunday)
        Màu: Mức tiêu thụ trung bình (đỏ = cao, xanh = thấp)
        """
        # Bước 1: Copy data → tránh sửa đổi df gốc
        df: pd.DataFrame = self.df.copy()

        # Bước 2: Chuyển đổi kiểu dữ liệu timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Bước 3: Tạo các feature cần thiết
        # aggregate: tổng điện năng của 370 người tại mỗi thời điểm
        df["aggregate"] = df.iloc[:, 1:].sum(axis=1)

        # hour: giờ trong ngày (0→23)
        df["hour"] = df["timestamp"].dt.hour

        # day_name: tên ngày (Monday, Tuesday, ...)
        df["day_name"] = df["timestamp"].dt.day_name()

        # day_of_week: số thứ tự ngày (0=Monday, 6=Sunday) — dùng để sort
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Bước 4: Tính TRUNG BÌNH điện năng cho mỗi cặp (ngày, giờ)
        # GroupBy [day_name, day_of_week, hour] → mean(aggregate)
        # Ví dụ: trung bình Monday lúc 12h = 450.5 kWh
        pivot_df = (
            df.groupby(["day_name", "day_of_week", "hour"])["aggregate"]
            .mean()
            .reset_index()
        )

        # Bước 5: Xoay dữ liệu từ dạng LONG → WIDE (pivot)
        # Kết quả: bảng 7 hàng (ngày) × 24 cột (giờ)
        # ┌───────────┬────────┬────────┬─────┬────────┐
        # │ day_name  │   0    │   1    │ ... │   23   │
        # ├───────────┼────────┼────────┼─────┼────────┤
        # │ Monday    │ 120.5  │ 110.3  │ ... │ 140.2  │
        # │ Tuesday   │ 118.7  │ 108.9  │ ... │ 138.6  │
        # └───────────┴────────┴────────┴─────┴────────┘
        heatmap_data = pivot_df.pivot(
            index="day_name",
            columns="hour",
            values="aggregate"
        )

        # Bước 6: Sắp xếp lại thứ tự ngày (Monday → Sunday)
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
        # Lọc chỉ lấy ngày có trong dữ liệu (tránh lỗi nếu thiếu ngày nào)
        days_order = [day for day in days_order if day in heatmap_data.index]
        heatmap_data = heatmap_data.reindex(days_order)

        # Bước 7: Vẽ HEATMAP
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            heatmap_data,                # Bảng 7×24: trung bình điện theo ngày/giờ
            cmap="RdBu_r",               # Bảng màu: Đỏ (cao) ↔ Xanh (thấp)
            annot=False,                 # Không ghi số lên từng ô (24 cột × 7 hàng = nhiều)
            linewidths=0.5,              # Đường viền mỏng giữa các ô
            cbar_kws={"label": "Average Aggregate Load (kWh)"},  # Label thanh màu
            ax=ax
        )
        ax.set_title("Time-Pattern Heatmap: Average Load by Hour & Day of Week",
                     fontsize=16, pad=20)
        ax.set_xlabel("Hour of Day (0-23)", fontsize=12)
        ax.set_ylabel("Day of Week", fontsize=12)
        plt.yticks(rotation=0)          # Label ngày nằm ngang
        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/time_pattern_heatmap_original.png",
                dpi=300,
                bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close(fig)

    # =======================================================================
    # 4. HISTOGRAM — Phân phối tổng lượng điện năng tiêu thụ
    # =======================================================================
    # MỤC ĐÍCH:
    #   Vẽ histogram + KDE (đường ước lượng mật độ xác suất) của cột aggregate.
    #   Giúp nhận biết:
    #     - Dữ liệu phân phối đối xứng hay lệch phải (right-skewed)?
    #     - Giá trị trung bình (mean) và trung vị (median) cách xa nhau không?
    #       → Nếu cách xa → dữ liệu bị lệch, cần transform log1p trước khi train
    #     - Có outlier cực đoan không?
    #
    # Ý NGHĨA ĐƯỜNG THẲNG DỌC:
    #   - Đỏ nét đứt: MEAN (trung bình số học)
    #   - Xanh nét đứt: MEDIAN (trung vị — giá trị chính giữa)
    #   → Mean ≈ Median → phân phối đối xứng (chuẩn)
    #   → Mean > Median → phân phối lệch phải (có nhiều giá trị lớn bất thường)
    # =======================================================================
    def plot_histogram(self,
        figsize=(12, 6),                        # Kích thước figure
        save=False,
        save_path="dataset/images"              # Thư mục lưu ảnh (tính từ backend/)
    ):
        """
        Vẽ histogram phân phối tổng lượng điện năng tiêu thụ (aggregate).

        Bao gồm:
        - X: tổng lượng điện năng tiêu thụ
        - Y: số lần xuất hiện của các mức tiêu thụ đó trong dữ liệu
        - Histogram 50 bins + đường KDE (kernel density estimation)
        - Đường dọc mean (đỏ nét đứt) và median (xanh nét đứt)
        """
        # Bước 1: Copy DataFrame → không sửa df gốc
        df = self.df.copy()

        # Bước 2: Tính aggregate = tổng điện 370 user tại mỗi timestamp
        df["aggregate"] = df.iloc[:, 1:].sum(axis=1)

        # Bước 3: Vẽ histogram
        fig, ax = plt.subplots(figsize=figsize)

        sns.histplot(
            data=df,
            x="aggregate",
            bins=50,          # Chia thành 50 khoảng (bins) để vẽ cột
            kde=True,         # Vẽ thêm đường KDE (ước lượng mật độ xác suất liên tục)
            color="skyblue",
            ax=ax
        )

        # Bước 4: Tính & vẽ đường dọc mean và median
        mean_val = df["aggregate"].mean()
        median_val = df["aggregate"].median()

        # Đường dọc tại vị trí mean (màu đỏ, nét đứt)
        ax.axvline(mean_val, color="red", linestyle="dashed", linewidth=2,
                   label=f"Mean: {mean_val:.2f}")

        # Đường dọc tại vị trí median (màu xanh lá, nét đứt)
        ax.axvline(median_val, color="green", linestyle="dashed", linewidth=2,
                   label=f"Median: {median_val:.2f}")

        ax.set_title("Distribution of Aggregate Load", fontsize=16, pad=20)
        ax.set_xlabel("Aggregate Load (kWh)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()              # Hiện legend (mean, median)
        ax.grid(True, alpha=0.3) # Lưới mờ

        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/aggregate_histogram.png",
                dpi=300,
                bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close(fig)

    # =======================================================================
    # HELPER: Lấy dữ liệu users ngẫu nhiên
    # =======================================================================
    # Hàm nội bộ dùng chung cho plot_random_users_boxplot và plot_random_users_acf.
    # Trả về tuple: (DataFrame chứa timestamp + users, mảng tên users đã chọn)
    # =======================================================================
    def get_random_users_data(self, size_rnd=9):
        """
        Chọn ngẫu nhiên `size_rnd` users và trả về DataFrame tương ứng.

        Returns
        -------
        tuple : (pd.DataFrame, np.array)
            - DataFrame gồm cột "timestamp" và `size_rnd` cột user
            - Mảng tên các user đã chọn
        """
        df = self.df.copy()

        # Đảm bảo cột "timestamp" tồn tại (phòng trường hợp đọc lại từ file khác)
        if "timestamp" not in df.columns:
            df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Lấy danh sách tất cả user (tất cả cột trừ "timestamp")
        users = [col for col in df.columns if col != "timestamp"]

        # Chọn ngẫu nhiên `size_rnd` user (không trùng lặp)
        random_users = np.random.choice(users, size=size_rnd, replace=False)

        # Trả về DataFrame chỉ chứa timestamp + các user đã chọn
        cols = ["timestamp"] + list(random_users)
        return df[cols], random_users

    # =======================================================================
    # 5. BOXPLOT — So sánh phân phối điện năng giữa các user ngẫu nhiên
    # =======================================================================
    # MỤC ĐÍCH:
    #   Vẽ boxplot so sánh phân phối lượng điện tiêu thụ của nhiều user.
    #   Giúp nhận biết:
    #     - User nào tiêu thụ nhiều, user nào tiêu thụ ít
    #     - Mức biến động (IQR) của từng user
    #     - Outlier (điểm nằm ngoài râu boxplot)
    #     - Cơ sở để quyết định loại bỏ user tiêu thụ quá lớn trong data_checker.py
    #
    # ĐỌC KẾT QUẢ BOXPLOT:
    #   ┌─────┐  ← Q3 (percentile 75%)
    #   │     │
    #   │─────│  ← Median (Q2, percentile 50%)
    #   │     │
    #   └─────┘  ← Q1 (percentile 25%)
    #      │     ← Râu (whisker): Q1 - 1.5×IQR
    #      ●     ← Outlier (điểm nằm ngoài whisker)
    # =======================================================================
    def plot_random_users_boxplot(self,
        size_rnd=9,                             # Số user ngẫu nhiên
        figsize=(14, 8),                        # Kích thước figure
        save=False,
        save_path="dataset/images",             # Thư mục lưu ảnh (tính từ backend/)
        show=True                               # True → hiển thị, False → chỉ lưu
    ):
        """
        Vẽ boxplot so sánh phân phối điện năng của `size_rnd` users ngẫu nhiên.
        """
        # Lấy dữ liệu users ngẫu nhiên từ helper
        df_users, random_users = self.get_random_users_data(size_rnd)

        # Chuyển dữ liệu từ WIDE → LONG format (seaborn boxplot cần dạng long)
        #
        # WIDE format (mỗi user = 1 cột):
        #   timestamp   | MT_001 | MT_002
        #   2014-01-01  |  12.5  |   8.3
        #
        # LONG format (gộp tất cả user thành 2 cột):
        #   timestamp   | customer | load
        #   2014-01-01  |  MT_001  | 12.5
        #   2014-01-01  |  MT_002  |  8.3
        df_melted = pd.melt(
            df_users,
            id_vars=["timestamp"],      # Cột giữ nguyên (không "dọc hóa")
            value_vars=random_users,    # Các cột user sẽ được "dọc hóa"
            var_name="customer",        # Tên cột mới chứa tên user
            value_name="load"           # Tên cột mới chứa giá trị điện năng
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=df_melted,
            x="customer",           # Trục X: tên user (MT_001, MT_002, ...)
            y="load",               # Trục Y: giá trị điện năng tiêu thụ
            ax=ax,
            palette="Set3",         # Bảng màu pastel dễ nhìn
            hue="customer",         # Tô màu theo customer (tương ứng palette)
            legend=False
        )

        ax.set_title("Load Distribution by Random Users", fontsize=16, pad=20)
        ax.set_xlabel("")
        ax.set_ylabel("Điện năng tiêu thụ (kWh)")
        plt.xticks(rotation=45)     # Xoay tên user cho dễ đọc
        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/users_boxplot_random.png",
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        plt.close(fig)

    # =======================================================================
    # 6. ACF — Kiểm tra tính chu kỳ (Autocorrelation Function)
    # =======================================================================
    # MỤC ĐÍCH:
    #   Vẽ biểu đồ ACF cho từng user → xem dữ liệu có tính chu kỳ không.
    #   ACF (Autocorrelation Function) đo mức tương quan giữa chuỗi thời gian
    #   tại thời điểm t với chính nó lùi lại `lag` bước về trước.
    #
    #   Giúp nhận biết:
    #     - Nếu ACF có đỉnh tại lag = 7 → chu kỳ tuần (weekly seasonality)
    #     - Nếu ACF có đỉnh tại lag = 30 → chu kỳ tháng
    #     - Nếu ACF giảm dần đều → dữ liệu có trend nhưng không có chu kỳ rõ
    #     → Thông tin này giúp quyết định có cần tạo feature lag 7d, 30d không
    #       (preprocessor.py sẽ dùng lag 24h, 48h, 72h, 7d)
    #
    # ĐỌC KẾT QUẢ ACF:
    #   Trục X: Độ trễ (lag) — số bước lùi về quá khứ
    #   Trục Y: Hệ số tương quan (-1 → +1)
    #     +1 → tương quan thuận hoàn hảo
    #      0 → không có tương quan
    #     -1 → tương quan nghịch hoàn hảo
    #   Đường nằm ngang: ngưỡng tin cậy 95% (nếu vượt ngưỡng → tương quan có ý nghĩa)
    # =======================================================================
    def plot_random_users_acf(self,
        size_rnd=9,                             # Số user ngẫu nhiên
        figsize=(16, 12),                       # Kích thước figure
        rc_chart=(3, 3),                        # Grid layout subplot
        save=False,
        save_path="dataset/images",             # Thư mục lưu ảnh (tính từ backend/)
        show=True,
        resample_daily=True                     # True → gộp theo ngày trước khi tính ACF
    ):
        """
        Vẽ biểu đồ ACF cho `size_rnd` users ngẫu nhiên.

        Parameters
        ----------
        resample_daily : bool
            Nếu True → resample dữ liệu theo ngày (sum) trước khi vẽ ACF.
            Dữ liệu 15 phút quá dày → ACF sẽ rất nhiễu khó đọc.
            Gộp theo ngày → ~1460 điểm (4 năm) → ACF rõ hơn.
        """
        df_users, random_users = self.get_random_users_data(size_rnd)

        # Resample theo ngày để ACF dễ nhìn hơn
        if resample_daily:
            df_users.set_index("timestamp", inplace=True)
            df_plot = df_users.resample("D").sum()  # Tổng tiêu thụ mỗi ngày
        else:
            df_plot = df_users.set_index("timestamp")

        fig, axes = plt.subplots(rc_chart[0], rc_chart[1], figsize=figsize)
        axes = axes.flatten()

        for i, user in enumerate(random_users):
            if i < len(axes):
                # autocorrelation_plot: hàm tích hợp của pandas
                # Tự động tính hệ số ACF cho tất cả lag (0 đến N/2) và vẽ
                autocorrelation_plot(df_plot[user], ax=axes[i])
                axes[i].set_title(f"Khách hàng: {user}")
                axes[i].set_xlabel("Độ trễ (Lag - đơn vị: ngày)")
                axes[i].set_ylabel("Hệ số ACF")

        # Tắt subplot thừa (nếu có)
        for j in range(len(random_users), len(axes)):
            axes[j].axis("off")

        plt.suptitle("Biểu đồ ACF — Kiểm tra tính chu kỳ", fontsize=18)
        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/users_acf_random.png",
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        plt.close(fig)

    # =======================================================================
    # 7. MONTHLY BOXPLOT — Phân phối tiêu thụ điện theo tháng
    # =======================================================================
    # MỤC ĐÍCH:
    #   Vẽ boxplot theo tháng cho từng user ngẫu nhiên.
    #   Giúp nhận biết:
    #     - Monthly seasonality: tháng nào tiêu thụ nhiều, tháng nào ít
    #       (VD: mùa đông dùng sưởi nhiều, mùa hè dùng điều hòa nhiều)
    #     - User nào có seasonal pattern rõ ràng
    #     → Cung cấp insight cho feature "month" trong preprocessor.py
    #
    # CÁCH HOẠT ĐỘNG:
    #   1. Resample dữ liệu 15 phút → tổng theo ngày (96 dòng → 1 dòng/ngày)
    #   2. Thêm cột "month" (1→12)
    #   3. Vẽ boxplot: trục X = tháng, trục Y = tổng tiêu thụ ngày
    # =======================================================================
    def plot_monthly_boxplot(self,
        size_rnd=9,
        figsize=(16, 12),
        rc_chart=(3, 3),
        save=False,
        save_path="dataset/images",             # Thư mục lưu ảnh (tính từ backend/)
        show=True
    ):
        """
        Vẽ boxplot phân phối tiêu thụ theo tháng cho `size_rnd` users.

        Mỗi subplot = 1 user, trục X = tháng (1-12), trục Y = tổng tiêu thụ ngày.
        → Nhận biết monthly seasonality.
        """
        df = self.df.copy()

        if "timestamp" not in df.columns:
            df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Lấy tên các user (tất cả cột trừ "timestamp")
        users = [col for col in df.columns if col != "timestamp"]

        # Chọn ngẫu nhiên `size_rnd` user
        random_users = np.random.choice(users, size=size_rnd, replace=False)

        # Chỉ lấy các cột cần thiết để tránh xử lý 370 cột không cần thiết
        df_selected = df[["timestamp"] + list(random_users)].copy()

        # Resample theo ngày: gộp 96 dòng (15 phút × 24h) → 1 dòng tổng tiêu thụ ngày
        df_selected.set_index("timestamp", inplace=True)
        df_daily = (
            df_selected
            .resample("D")    # Gộp theo ngày
            .sum()             # Tổng lượng tiêu thụ trong ngày
            .reset_index()
        )

        # Thêm cột month (1-12) để dùng làm trục X boxplot
        df_daily["month"] = df_daily["timestamp"].dt.month

        fig, axes = plt.subplots(rc_chart[0], rc_chart[1], figsize=figsize)
        axes = axes.flatten()

        for i, user in enumerate(random_users):
            sns.boxplot(
                data=df_daily,
                x="month",          # Trục X: tháng (1→12)
                y=user,             # Trục Y: tổng tiêu thụ ngày của user đó
                ax=axes[i],
                palette="Set3",
                hue="month",        # Tô màu theo tháng
                legend=False
            )
            axes[i].set_title(user)
            axes[i].set_xlabel("Tháng")
            axes[i].set_ylabel("Tổng tiêu thụ (kWh/ngày)")

        # Tắt subplot thừa
        for j in range(len(random_users), len(axes)):
            axes[j].axis("off")

        plt.suptitle("Monthly Load Distribution by User", fontsize=16)
        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/monthly_boxplot.png",
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        plt.close(fig)

    # =======================================================================
    # 8. MONTHLY ACF — Kiểm tra tính chu kỳ dài hạn theo tháng
    # =======================================================================
    # MỤC ĐÍCH:
    #   Tương tự ACF ở hàm 6, nhưng dữ liệu được gộp theo THÁNG thay vì ngày.
    #   Giúp nhận biết:
    #     - Có chu kỳ năm (yearly seasonality) không?
    #     - Mức độ tương quan giữa các tháng cách nhau n tháng
    #
    # LƯU Ý:
    #   Dataset 4 năm (2011-2014) → chỉ có 48 điểm (48 tháng)
    #   → Số lượng mẫu nhỏ nên ACF ít lag, kết quả chỉ mang tính tham khảo
    # =======================================================================
    def plot_monthly_acf(self,
        size_rnd=9,
        figsize=(16, 12),
        rc_chart=(3, 3),
        save=False,
        save_path="dataset/images",             # Thư mục lưu ảnh (tính từ backend/)
        show=True
    ):
        """
        Vẽ biểu đồ ACF theo tháng cho `size_rnd` users.

        Dữ liệu được resample theo tháng (ME = Month End) trước khi tính ACF.
        → Kiểm tra yearly seasonality (chu kỳ 12 tháng).
        """
        df = self.df.copy()

        if "timestamp" not in df.columns:
            df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        users = [col for col in df.columns if col != "timestamp"]
        random_users = np.random.choice(users, size=size_rnd, replace=False)

        df_selected = df[["timestamp"] + list(random_users)].copy()

        # Resample theo tháng: gộp toàn bộ dữ liệu 15 phút → tổng tiêu thụ tháng
        # "ME" = Month End → mỗi tháng cho ra 1 điểm (timestamp = cuối tháng)
        df_selected.set_index("timestamp", inplace=True)
        df_monthly = (
            df_selected
            .resample("ME")   # Gộp theo cuối tháng
            .sum()             # Tổng tiêu thụ tháng
        )

        fig, axes = plt.subplots(rc_chart[0], rc_chart[1], figsize=figsize)
        axes = axes.flatten()

        for i, user in enumerate(random_users):
            # autocorrelation_plot: vẽ ACF cho chuỗi tháng (48 điểm)
            autocorrelation_plot(df_monthly[user], ax=axes[i])
            axes[i].set_title(user)
            axes[i].set_xlabel("Độ trễ (Lag - đơn vị: tháng)")
            axes[i].set_ylabel("Hệ số ACF")

        # Tắt subplot thừa
        for j in range(len(random_users), len(axes)):
            axes[j].axis("off")

        plt.suptitle("Monthly ACF — Kiểm tra chu kỳ năm", fontsize=16)
        plt.tight_layout()

        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/monthly_acf.png",
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        plt.close(fig)


# ===========================================================================
# BLOCK MAIN — Chạy trực tiếp file này để vẽ tất cả biểu đồ EDA
# ===========================================================================
# CÁCH CHẠY (từ thư mục backend/):
#     python -m app.utils.visualizations
#
# LƯU Ý:
# - Dataset gốc LD2011_2014.txt (~680MB) phải nằm trong backend/dataset/
# - Ảnh lưu tại backend/dataset/images/
# - Backend không có màn hình nên dùng Agg (đã set ở đầu file)
# ===========================================================================
if __name__ == "__main__":
    SAVE_PATH = "dataset/images1"

    print("=" * 60)
    print("  ĐANG TẢI DATASET... (file ~680MB, mất 1-2 phút)")
    print("=" * 60)

    # Khởi tạo class EDA với đường dẫn tới dataset gốc
    visual = VisualDatasetTXTV1(dataset_path="/home/thang/workspace/lightgbm/backend/dataset/hourly_electricity_2013_2014.csv")
    print(f"  ✓ Đã tải: {visual.rows} dòng, {len(visual.columns)-1} users\n")

    # --- 1. Line plot: biến thiên điện năng từng user ---
    # Tạo 3 ảnh, mỗi ảnh 3×3 = 9 biểu đồ → tổng 27 user ngẫu nhiên
    print("[1/8] Vẽ: Daily Load Chart...")
    visual.dayly_load_chart(
        images=3,
        size_rnd=27,
        space_chart=(0.5, 0.5),
        rc_chart=(3, 3),
        save=True,
        save_path=SAVE_PATH
    )
    print("  ✓ Lưu: users_page_1.png, users_page_2.png, users_page_3.png")

    # --- 2. Line plot: tổng tải 370 user trong 24h ---
    # Chọn 9 ngày ngẫu nhiên, xem profile tổng tải cả ngày
    print("[2/8] Vẽ: Aggregate Load Chart...")
    visual.plot_aggregate_load_chart(
        size_rnd=9,
        rc_chart=(3, 3),
        save=True,
        save_path=SAVE_PATH
    )
    print("  ✓ Lưu: aggregate_random_days.png")

    # --- 3. Heatmap: pattern theo giờ × ngày trong tuần ---
    # Phát hiện giờ cao/thấp điểm, khác biệt ngày thường vs cuối tuần
    print("[3/8] Vẽ: Time-Pattern Heatmap...")
    visual.plot_time_pattern_heatmap(
        figsize=(15, 8),
        save=True,
        save_path=SAVE_PATH
    )
    print("  ✓ Lưu: time_pattern_heatmap_original.png")

    # --- 4. Histogram: phân phối tổng tải ---
    # Xem dữ liệu có bị lệch (skewed) không → quyết định có log1p không
    print("[4/8] Vẽ: Histogram...")
    visual.plot_histogram(
        figsize=(12, 6),
        save=True,
        save_path=SAVE_PATH
    )
    print("  ✓ Lưu: aggregate_histogram.png")

    # --- 5. Boxplot: phân phối điện năng theo user ---
    # So sánh mức tiêu thụ và outlier giữa các user
    print("[5/8] Vẽ: Random Users Boxplot...")
    visual.plot_random_users_boxplot(
        save=True,
        save_path=SAVE_PATH,
        show=False
    )
    print("  ✓ Lưu: users_boxplot_random.png")

    # --- 6. ACF: kiểm tra tính chu kỳ từng user ---
    # Xác nhận weekly seasonality → cơ sở cho lag 7 ngày trong preprocessor
    print("[6/8] Vẽ: Random Users ACF...")
    visual.plot_random_users_acf(
        save=True,
        save_path=SAVE_PATH,
        show=False
    )
    print("  ✓ Lưu: users_acf_random.png")

    # --- 7. Boxplot theo tháng: monthly seasonality ---
    # Xem tháng nào tiêu thụ nhiều/ít → seasonal pattern
    print("[7/8] Vẽ: Monthly Boxplot...")
    visual.plot_monthly_boxplot(
        save=True,
        save_path=SAVE_PATH,
        show=False
    )
    print("  ✓ Lưu: monthly_boxplot.png")

    # --- 8. ACF theo tháng: chu kỳ năm ---
    # Kiểm tra yearly pattern (tham khảo, mẫu chỉ 48 điểm)
    print("[8/8] Vẽ: Monthly ACF...")
    visual.plot_monthly_acf(
        save=True,
        save_path=SAVE_PATH,
        show=False
    )
    print("  ✓ Lưu: monthly_acf.png")

    print()
    print("=" * 60)
    print(f"  ✅ HOÀN TẤT! Ảnh đã lưu tại: backend/{SAVE_PATH}/")
    print("=" * 60)
