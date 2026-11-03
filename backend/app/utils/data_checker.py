import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === class check data ===
# use this class before running class Preprocessor 
class DataCheckerTXT:
    """
    Check chất lượng dataset điện năng trước khi train model.
    Không modify data, chỉ phân tích.
    """
    def __init__(self, dataset_path:str="backend/dataset/LD2011_2014.txt"):
        self.df = pd.read_csv(
            dataset_path,
            sep=';',
            decimal=',',
            parse_dates=[0]
        )
    
    # 1. Tổng quan về dataset
    def overview(self):
        print("\n=== DATA OVERVIEW ===")
        print("Shape:", self.df.shape)
        print("Rows: ", self.df.shape[0])
        print("Columns:", self.df.shape[1])
    
    # 2. Missing values
    def check_missing(self):
        print("\n=== MISSING VALUES ===")

        missing = self.df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("No missing values.")
        else:
            print(missing)
    
       # -------------------------
    
    # 3. Zero ratio per client
    def check_zero_ratio(self):
        print("\n=== ZERO RATIO PER CLIENT ===")

        zero_ratio = (self.df == 0).mean().sort_values(ascending=False)

        print("Top 10 clients with most zeros:")
        print(zero_ratio.head(10))

        print("\nMost active clients:")
        print(zero_ratio.tail(10))
    
    # 4. kiểm tra chuỗi 0 liên tục
    def max_consecutive_zeros(self):
        '''
        Cái này quan trọng hơn count đơn thuần.
        Ví dụ:
            1-2 điểm zero lẻ tẻ → có thể bình thường
            500 bước liên tục zero → bất thường.
        '''
        print("\n=== Check for a string of consecutive zeros ===")
        for col in self.df.columns[:20]:
            groups = (self.df[col] != 0).cumsum()

            print(col, self.df[col].eq(0).groupby(groups).sum().max())

    # 5. Detect "inactive period"
    def detect_activation_time(self):
        print("\n=== FIRST NON-ZERO (CLIENT ACTIVATION TIME) ===")

        activation = {}

        for col in self.df.columns:
            series = self.df[col]

            if (series != 0).sum() == 0:
                activation[col] = None
                continue

            first_active = series.ne(0).idxmax()
            activation[col] = first_active

        activation_series = pd.Series(activation).dropna()

        print("Sample activation times:")
        print(activation_series.head(10))

    # 6. Sparsity analysis
    def sparsity_report(self):
        print("\n=== SPARSITY REPORT ===")

        total_values = self.df.size
        zero_values = (self.df == 0).sum().sum()

        sparsity = zero_values / total_values

        print(f"Total values: {total_values}")
        print(f"Zero values: {zero_values}")
        print(f"Sparsity: {sparsity:.4f}")

    # 7. Time frequency check
    def check_time_frequency(self):
        print("\n=== TIME FREQUENCY CHECK ===")

        diff = self.df.index.to_series().diff().value_counts()

        print(diff.head(10))

    # RUN ALL
    def run_all(self):
        self.overview()
        self.check_missing()
        self.check_zero_ratio()
        self.max_consecutive_zeros()
        self.detect_activation_time()
        self.sparsity_report()
        self.check_time_frequency()

class CheckerAndProcessorCSV:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    # ==== 1. kiểm tra chênh lệch mức tiêu thụ điện ====
    def analyze_monthly_consumption(
        self,
        df,
        timestamp_col="timestamp",
        plot=True,
        top_k_outliers=10
    ):    
        df = df.copy()
        # ===== 1. xử lý time =====
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)

        # ===== 2. tổng điện theo tháng =====
        monthly_usage = df.resample("1ME").sum()

        # ===== 3. tổng điện cả năm của từng user =====
        total_per_user = monthly_usage.sum(axis=0)

        # ===== 4. thống kê =====
        print("\n=== Tổng điện tiêu thụ của từng user ===")
        print(total_per_user.sort_values(ascending=False))

        print("\n=== Thống kê ===")
        print(total_per_user.describe())

        # ===== 5. coefficient of variation =====
        cv = total_per_user.std() / total_per_user.mean()

        print(f"\nCoefficient of Variation (CV): {cv:.4f}")

        if cv < 0.1:
            print("=> User tiêu thụ khá đồng đều")
        elif cv < 0.3:
            print("=> Có khác biệt vừa phải")
        else:
            print("=> Chênh lệch tiêu thụ lớn")

        # =========================================================
        # ===== 6. Detect outlier bằng IQR ========================
        # =========================================================

        q1 = total_per_user.quantile(0.25)
        q3 = total_per_user.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = total_per_user[
            (total_per_user < lower_bound) |
            (total_per_user > upper_bound)
        ]

        print("\n=== User gây lệch dữ liệu (Outliers) ===")

        if len(outliers) == 0:
            print("Không phát hiện outlier")
        else:
            outliers = outliers.sort_values(ascending=False)

            for user, value in outliers.head(top_k_outliers).items():
                ratio = value / total_per_user.median()

                print(
                    f"{user:10s} | "
                    f"total={value:,.0f} | "
                    f"gấp {ratio:.2f} lần median"
                )

        # =========================================================
        # ===== 7. histogram =====================================
        # =========================================================

        if plot:
            # histogram raw
            plt.figure(figsize=(10, 5))
            plt.hist(total_per_user, bins=30)
            plt.xlabel("Total yearly consumption")
            plt.ylabel("Number of users")
            plt.title("Distribution of electricity consumption")
            plt.grid(True)
            plt.show()

            # histogram log
            plt.figure(figsize=(10, 5))
            plt.hist(np.log1p(total_per_user), bins=30)
            plt.xlabel("log(total consumption)")
            plt.ylabel("Number of users")
            plt.title("Log Distribution")
            plt.grid(True)
            plt.show()

            # boxplot
            plt.figure(figsize=(12, 2))
            plt.boxplot(total_per_user, vert=False)
            plt.title("Boxplot of user consumption")
            plt.show()

        return monthly_usage, total_per_user

    # ==== 2. lọc user tiêu thụ quá lớn (outlier)====
    def remove_large_consumers(
        self,
        timestamp_col="timestamp",
        upper_quantile=0.95,
        save_path=None,
        verbose=True
    ):
        df = self.df.copy()

        # ===== 1. xử lý timestamp =====
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)

        # ===== 2. tổng điện theo tháng =====
        monthly_usage = df.resample("1ME").sum()

        # ===== 3. tổng điện cả năm từng user =====
        total_per_user = monthly_usage.sum(axis=0)

        # ===== 4. threshold =====
        threshold = total_per_user.quantile(upper_quantile)

        # ===== 5. user giữ lại =====
        keep_users = total_per_user[
            total_per_user <= threshold
        ].index.tolist()

        # ===== 6. user bị loại =====
        removed_users = total_per_user[
            total_per_user > threshold
        ].sort_values(ascending=False)

        # =====================================================
        # ===== 7. log ========================================
        # =====================================================

        if verbose:
            print("\n=== REMOVE LARGE CONSUMERS ===")

            print(f"\nQuantile threshold: {upper_quantile}")
            print(f"Consumption threshold: {threshold:,.0f}")

            print(f"\nUsers before: {len(total_per_user)}")
            print(f"Users after : {len(keep_users)}")
            print(f"Removed     : {len(removed_users)}")

            print("\n=== Removed Users ===")

            for user, value in removed_users.items():

                ratio = value / total_per_user.median()

                print(
                    f"{user:10s} | "
                    f"total={value:,.0f} | "
                    f"gấp {ratio:.2f} lần median"
                )

        # =====================================================
        # ===== 8. tạo cleaned dataframe ======================
        # =====================================================

        cleaned_df = df[keep_users].copy()

        # đưa timestamp trở lại column
        cleaned_df = cleaned_df.reset_index()

        # =====================================================
        # ===== 9. save =======================================
        # =====================================================

        if save_path is not None:
            cleaned_df.to_csv(save_path, index=False)

            if verbose:
                print(f"\nSaved cleaned dataset -> {save_path}")

        return cleaned_df, keep_users, removed_users

if __name__ == "__main__": 

    checker = CheckerAndProcessorCSV(csv_path="backend/dataset/hourly_electricity_filtered.csv")
    cleaned_df, keep_users, removed_users = (
        checker.remove_large_consumers(
            upper_quantile=0.95,
            save_path="backend/dataset/hourly_electricity_filtered_v2.csv"
        )
    )
    checker.analyze_monthly_consumption(df=cleaned_df)