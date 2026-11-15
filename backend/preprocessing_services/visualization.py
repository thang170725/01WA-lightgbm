import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.plotting import autocorrelation_plot

# ==================================================
# ==== Class chuyên biệt để vẽ biểu đồ ==========
# ==================================================
class VisualDatasetTXT:
    # Khởi tạo đối tượng trực quan hóa và nạp dữ liệu dataset.
    def __init__(self, dataset_path: str = None):
        self.df: pd.DataFrame = pd.read_csv(
            dataset_path,
            sep=",",
            decimal=".",
            parse_dates=[0]
        )
        # đổi tên cột
        self.df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)

        self.columns = self.df.columns
        self.rows = len(self.df)

    # ====== Line plot =====
    # ===== 1. biểu đồ đường mô tả sự biến thiên về mức sử dụng điện của các user =====
    # dùng để xem mức phân bổ sử dụng điện của user
    def dayly_load_chart(self,                                     
        images: int,                                # số lượng ảnh    
        size_rnd: int,                              # số lượng biểu đồ muốn tạo (18: muốn tạo 18 cái biểu đồ)                       
        space_chart: tuple,                         # khoảng cách của các biểu đồ trong cùng một page (w,h)        
        rc_chart: tuple,                            # số hàng cột trong 1 images ((3,3): tạo ra 1 ảnh bên trong có 9 buổi đồ 3x3)        
        save:bool = False,                          # cho phép lưu hay không
        save_path: str = "backend/dataset/images"   # đường dẫn lưu ảnh (tính từ thư mục chạy lệnh python)
    ):
        # 1. chọn user muốn vẽ biểu đồ
        users = np.random.choice(self.columns[1:], size=size_rnd, replace=False)                

        # 2. gom theo 5 ngày để vẽ biểu đồ cho đẹp (đủ thấy xu hướng)
        daily_df = (
            self.df
            .set_index("timestamp")
            .resample("5D")
            .mean()
            .reset_index()
        )

        # 3. sử dụng lineplot để vẽ biểu đồ
        per_page = rc_chart[0]*rc_chart[1]
        for page in range(images):
            # lấy ra nhóm user để vẽ vào page hiện tại
            subset = users[page*per_page:(page+1)*per_page]

            fig, axes = plt.subplots(
                rc_chart[0],rc_chart[1],
            )
            axes = axes.flatten()

            for i, user in enumerate(subset):
                sns.lineplot(
                    x=daily_df["timestamp"], 
                    y=daily_df[user],
                    ax=axes[i],
                    linewidth=0.5,
                    color="black"
                )
                x = daily_df["timestamp"]

                axes[i].set_title(user)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
                axes[i].set_xticks([x.iloc[0], x.iloc[-1]])
                axes[i].tick_params(
                    axis="both",
                    labelsize=5
                )
                axes[i].tick_params(
                    axis="x",
                    rotation=30
                )

            plt.subplots_adjust(
                wspace=space_chart[0],
                hspace=space_chart[1]
            )
            if save:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/users_page_{page+1}.png", dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close(fig)
    
    # ====== 2. biểu đồ đường xem sự biến thiên tổng lượng điện năng tiêu thụ trong 24h của 370 người ======
    def plot_aggregate_load_chart(
        self,
        size_rnd=9,                  
        rc_chart=(3,3),
        save=False,
        save_path="./backend/dataset/images"        
    ):
        # 1. copy DataFrame không sửa trực tiếp vào df gốc
        df = self.df.copy()

        # 2. chuyển đổi kiểu dữ liệu
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 3. tạo thêm cột mới
        df["aggregate"] = df.iloc[:,1:].sum(axis=1) # để làm trục Y
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60 # để làm trúc X

        all_days = df["date"].unique()

        # 4. chọn random 1 vài ngày để vẽ biểu đồ
        random_days = np.random.choice(
            all_days,
            size=size_rnd,
            replace=False
        )

        fig, axes = plt.subplots(
            rc_chart[0],rc_chart[1],
        )

        axes = axes.flatten()

        for i, day in enumerate(random_days):
            day_data = df[df["date"] == day]

            sns.lineplot(
                x=day_data["hour"],
                y=day_data["aggregate"],
                ax=axes[i],
                color="black"
            )

            axes[i].set_title(str(day))
            axes[i].set_xlim(0,24)

            axes[i].set_xticks(
                [0,6,12,18,24]
            )

            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].tick_params(
                axis="both",
                labelsize=5
            )

        for j in range(len(random_days), len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Aggregate Load Profiles of Random Days",
            fontsize=14
        )

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

    # ======= 3. Heatmap (biểu đồ độ tương quan)
    # heatmap trung bình lượng điện tiêu thụ tại mỗi giờ từng ngày trong tuần
    def plot_time_pattern_heatmap(self,
        figsize=(15, 8),                        # kích thước 1 biểu đồ trên window
        save=False,
        save_path="./backend/dataset/images"
    ):
        # 1. copy data tránh sửa trên df gốc
        df: pd.DataFrame = self.df.copy()
        print(df.head())

        # 2. đổi kiểu dữ liệu & thêm đặc trưng
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 3. thêm cột mới
        df["aggregate"] = df.iloc[:,1:].sum(axis=1) # tổng điện năng của 370 người mỗi một khung giờ
        df["hour"] = df["timestamp"].dt.hour
        df["day_name"] = df["timestamp"].dt.day_name()
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        print(df.head())

        # 3. tính trung bình lượng điện năng
        pivot_df = df.groupby(['day_name', 'day_of_week', 'hour'])['aggregate'].mean().reset_index()
        print(pivot_df.head())

        # 4. xoay dữ liệu từ dạng dài thành dạng rộng
        # bảng lưu trùng bình lượng điện năng tiêu thụ trong mỗi giờ từ thứ 2 -> cn
        heatmap_data = pivot_df.pivot(index='day_name', columns='hour', values='aggregate')
        print(heatmap_data.head()) 

        # 5. tính trung bình lượng điện năng của 0 -> 24
        diff = heatmap_data.mean()
        heatmap_data_diff = heatmap_data - diff # xem có cao hơn hay nhỏ hơn giá trị mốc (diff) - giá trị trung bình
        print(diff, heatmap_data_diff)

        # 6. cấu hình lại để chuẩn bị vẽ lên đồ thị
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_order = [day for day in days_order if day in heatmap_data.index]
        heatmap_data = heatmap_data.reindex(days_order)

        # 7. vẽ đồ thị
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            heatmap_data, 
            cmap='RdBu_r', 
            annot=False, 
            linewidths=.5, 
            cbar_kws={'label': 'Difference from Average Load'},
            ax=ax
        )
        ax.set_title('Time-Pattern Heatmap Plot of Aggregate Load', fontsize=16, pad=20)
        ax.set_xlabel('Hour of Day (0-23)', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)   
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 8. lưu đồ thị
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/time_pattern_heatmap_original.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

    # ====== 4. Histogram tổng lượng điện năng tiêu thụ từng khoảng =====
    def plot_histogram(self,
        figsize=(12, 6),
        save=False,
        save_path="./backend/dataset/images"
    ):
        # 1. copy tránh sửa vào df gốc
        df = self.df.copy()

        # 2. tạo feature
        df["aggregate"] = df.iloc[:,1:].sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(
            data=df, 
            x="aggregate", 
            bins=50, 
            kde=True, 
            color='skyblue',
            ax=ax
        )
        
        mean_val = df["aggregate"].mean()
        median_val = df["aggregate"].median()
        
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_title('Distribution of Aggregate Load', fontsize=16, pad=20)
        ax.set_xlabel('Aggregate Load', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                f"{save_path}/aggregate_histogram.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)
    
    # ===== helper random users =====
    # Chọn ngẫu nhiên một nhóm khách hàng và trả về dữ liệu của họ.
    def get_random_users_data(self, size_rnd=9):

        df = self.df.copy()

        if "timestamp" not in df.columns:
            df.rename(
                columns={"Unnamed: 0": "timestamp"},
                inplace=True
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        users = [
            col for col in df.columns
            if col != "timestamp"
        ]

        random_users = np.random.choice(
            users,
            size=size_rnd,
            replace=False
        )

        cols = ["timestamp"] + list(random_users)

        return df[cols], random_users

    # ====== 5. Boxplot random users ======
    # dùng để xem phân bố lượng điện tiêu thụ của các user
    def plot_random_users_boxplot(self,
        size_rnd=9,
        figsize=(14, 8),
        save=False,
        save_path="./backend/dataset/images",
        show=True
    ):

        df_users, random_users = self.get_random_users_data(size_rnd)

        # chuyển dữ liệu wide -> long
        df_melted = pd.melt(
            df_users,
            id_vars=["timestamp"],
            value_vars=random_users,
            var_name="customer",
            value_name="load"
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=df_melted,
            x="customer",
            y="load",
            ax=ax,
            palette="Set3",
            hue="customer",
            legend=False
        )

        ax.set_xlabel("")
        ax.set_ylabel("Điện năng tiêu thụ")

        plt.xticks(rotation=45)

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

    # ====== 6. ACF random users ======
    # dùng để xem tính chu kỳ / seasonal của dữ liệu điện năng
    def plot_random_users_acf(self,
        size_rnd=9,
        figsize=(16, 12),
        rc_chart=(3, 3),
        save=False,
        save_path="./backend/dataset/images",
        show=True,
        resample_daily=True
    ):

        df_users, random_users = self.get_random_users_data(size_rnd)

        # resample theo ngày để biểu đồ dễ nhìn hơn
        if resample_daily:

            df_users.set_index(
                "timestamp",
                inplace=True
            )

            df_plot = df_users.resample("D").sum()

        else:

            df_plot = df_users.set_index("timestamp")

        fig, axes = plt.subplots(
            rc_chart[0],
            rc_chart[1],
            figsize=figsize
        )

        axes = axes.flatten()

        for i, user in enumerate(random_users):

            if i < len(axes):

                autocorrelation_plot(
                    df_plot[user],
                    ax=axes[i]
                )

                axes[i].set_title(f"Khách hàng: {user}")

                axes[i].set_xlabel("Độ trễ (Lag)")
                axes[i].set_ylabel("Hệ số ACF")

        # tắt subplot thừa
        for j in range(len(random_users), len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Biểu đồ ACF",
            fontsize=18
        )

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

    # ====== 7. Monthly Boxplot ======
    # boxplot theo tháng của nhiều user
    def plot_monthly_boxplot(self,
        size_rnd=9,
        figsize=(16, 12),
        rc_chart=(3, 3),
        save=False,
        save_path="./backend/dataset/images",
        show=True
    ):

        df = self.df.copy()

        if "timestamp" not in df.columns:
            df.rename(
                columns={"Unnamed: 0": "timestamp"},
                inplace=True
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        users = [
            col for col in df.columns
            if col != "timestamp"
        ]

        random_users = np.random.choice(
            users,
            size=size_rnd,
            replace=False
        )

        # lấy timestamp + user
        df_selected = df[
            ["timestamp"] + list(random_users)
        ].copy()

        # resample theo ngày
        df_selected.set_index(
            "timestamp",
            inplace=True
        )

        df_daily = (
            df_selected
            .resample("D")
            .sum()
            .reset_index()
        )

        # thêm cột month
        df_daily["month"] = (
            df_daily["timestamp"]
            .dt.month
        )

        fig, axes = plt.subplots(
            rc_chart[0],
            rc_chart[1],
            figsize=figsize
        )

        axes = axes.flatten()

        for i, user in enumerate(random_users):

            sns.boxplot(
                data=df_daily,
                x="month",
                y=user,
                ax=axes[i],
                palette="Set3",
                hue="month",
                legend=False
            )

            axes[i].set_title(user)

            axes[i].set_xlabel("Tháng")
            axes[i].set_ylabel("Tổng tiêu thụ")

        # tắt subplot thừa
        for j in range(len(random_users), len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Monthly Boxplot",
            fontsize=16
        )

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

    # ====== 8. Monthly ACF ======
    # ACF theo tháng để xem tính chu kỳ dài hạn
    def plot_monthly_acf(self,
        size_rnd=9,
        figsize=(16, 12),
        rc_chart=(3, 3),
        save=False,
        save_path="./backend/dataset/images",
        show=True
    ):

        df = self.df.copy()

        if "timestamp" not in df.columns:
            df.rename(
                columns={"Unnamed: 0": "timestamp"},
                inplace=True
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        users = [
            col for col in df.columns
            if col != "timestamp"
        ]

        random_users = np.random.choice(
            users,
            size=size_rnd,
            replace=False
        )

        df_selected = df[
            ["timestamp"] + list(random_users)
        ].copy()

        # resample theo tháng
        df_selected.set_index(
            "timestamp",
            inplace=True
        )

        df_monthly = (
            df_selected
            .resample("ME")
            .sum()
        )

        fig, axes = plt.subplots(
            rc_chart[0],
            rc_chart[1],
            figsize=figsize
        )

        axes = axes.flatten()

        for i, user in enumerate(random_users):

            autocorrelation_plot(
                df_monthly[user],
                ax=axes[i]
            )

            axes[i].set_title(user)

            axes[i].set_xlabel("Độ trễ")
            axes[i].set_ylabel("Hệ số ACF")

        # tắt subplot thừa
        for j in range(len(random_users), len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Monthly ACF",
            fontsize=16
        )

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

    # Tạo báo cáo theo ngày về dữ liệu của một khách hàng trong tháng.
    def client_monthly_report(self, client_id="MT_288", month="2014-12"):
        """Return a daily audit table without changing the source data."""
        if client_id not in self.df.columns:
            raise KeyError(f"Client {client_id!r} is not present in this dataset")
        values = self.df.set_index("timestamp")[client_id].loc[month]
        hourly = values.resample("1h").mean()
        report = hourly.rename("value").to_frame()
        report["is_zero"] = report["value"].eq(0)
        report["is_missing"] = report["value"].isna()
        return report.groupby(report.index.date).agg(
            zero_count=("is_zero", "sum"),
            nonzero_count=("value", lambda s: int(s.gt(0).sum())),
            missing_count=("is_missing", "sum"),
            minimum=("value", "min"),
            maximum=("value", "max"),
            mean=("value", "mean"),
        ).rename_axis("date").reset_index()

    # Vẽ cấu trúc giá trị bằng không và thiếu theo giờ của một khách hàng.
    def plot_client_zero_structure(self, client_id="MT_288", month="2014-12", save=False, save_path="./backend/dataset/images"):
        """Visualize zeros and missing hourly values; no imputation is applied."""
        if client_id not in self.df.columns:
            raise KeyError(f"Client {client_id!r} is not present in this dataset")
        values = self.df.set_index("timestamp")[client_id].loc[month].resample("1h").mean()
        fig, ax = plt.subplots(figsize=(16, 4))
        colors = np.where(values.isna(), "grey", np.where(values.eq(0), "red", "steelblue"))
        ax.bar(values.index, values.fillna(0), color=colors, width=0.03)
        ax.set_title(f"{client_id}: hourly zeros and missing values ({month})")
        ax.set_ylabel("Load")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        if save:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f"{save_path}/{client_id}_{month.replace('-', '')}_zero_audit.png", dpi=200)
        return fig, ax

    # So sánh phân phối sử dụng điện theo giờ và ngày trong tuần.
    def plot_client_distribution_by_time(self, client_id="MT_288", save=False, save_path="./backend/dataset/images"):
        """Compare one client's distribution by hour and weekday."""
        if client_id not in self.df.columns:
            raise KeyError(f"Client {client_id!r} is not present in this dataset")
        values = self.df[["timestamp", client_id]].dropna().copy()
        values["hour"] = values["timestamp"].dt.hour
        values["weekday"] = values["timestamp"].dt.dayofweek
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.boxplot(data=values, x="hour", y=client_id, ax=axes[0], showfliers=False)
        sns.boxplot(data=values, x="weekday", y=client_id, ax=axes[1], showfliers=False)
        axes[0].set_title(f"{client_id}: distribution by hour")
        axes[1].set_title(f"{client_id}: distribution by weekday")
        fig.tight_layout()
        if save:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f"{save_path}/{client_id}_time_distributions.png", dpi=200)
        return fig, axes

if __name__ == "__main__":
    visual = VisualDatasetTXT(dataset_path="backend/dataset/hourly_electricity_2013_2014.csv")
    # visual.dayly_load_chart(
    #     images=3,
    #     size_rnd=27,
    #     space_chart=(0.5,0.5),
    #     rc_chart=(3,3),
    #     save=False
    # )

    # visual.plot_aggregate_load_chart(
    #     size_rnd=9,
    #     rc_chart=(3,3),
    #     save=False
    # )
    
    # 3. Gọi thử Time-pattern Heatmap
    # visual.plot_time_pattern_heatmap(
    #     figsize=(15, 8),
    #     save=False
    # )

    # # 2. Gọi thử Histogram
    # visual.plot_histogram(
    #     figsize=(12, 6),
    #     save=False
    # )

    visual.plot_client_distribution_by_time()

    visual.plot_random_users_boxplot(save=True)
    visual.plot_random_users_acf(save=True)
    visual.plot_monthly_boxplot(save=True)
    visual.plot_monthly_acf(save=True)
