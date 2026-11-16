import numpy as np
import pandas as pd

# Lớp tạo đặc trưng thời gian thực (Realtime / Autoregressive Feature Builder) khi dự báo từng bước tương lai
class FeatureBuilder:
    def __init__(self):
        # Input : Không có tham số
        # Output: None - Khởi tạo bộ tạo đặc trưng
        pass

    # ==== time features ====
    def build_time_features(
        self,
        time_str  # Input: chuỗi thời gian hoặc đối tượng Timestamp cần trích xuất đặc trưng
    ):
        # Input :
        #   - time_str (str | pd.Timestamp): Mốc thời gian cần dự đoán (vd: "2014-12-01 08:00:00")
        # Output:
        #   - dict: Dictionary chứa 4 đặc trưng thời gian {"hour_sin", "hour_cos", "day_of_week", "is_weekend"}
        dt = pd.to_datetime(time_str)

        hour = dt.hour                      # Giờ trong ngày (0 -> 23)

        day_of_week = dt.dayofweek          # Thứ trong tuần (0: Thứ 2 -> 6: Chủ Nhật)

        is_weekend = (
            1 if day_of_week >= 5 else 0    # Cờ đánh dấu cuối tuần (1: Thứ 7/CN, 0: Ngày thường)
        )

        # cyclical encoding
        # Mã hóa chu kỳ 24 giờ bằng hàm lượng giác sin/cos để giữ tính liên tục giữa 23h và 0h
        hour_sin = np.sin(
            2 * np.pi * hour / 24
        )

        hour_cos = np.cos(
            2 * np.pi * hour / 24
        )

        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }

    # ==== lag features ====
    def build_lag_features(
        self,
        history  # Input: danh sách (list) các giá trị tiêu thụ điện quá khứ (>= 168 giờ)
    ):
        """
        history:
            list giá trị theo thời gian
            (đã sort tăng dần)
        """
        # Input :
        #   - history (list[float]): Danh sách lịch sử tiêu thụ điện theo giờ đã sắp xếp tăng dần theo thời gian (tối thiểu 168 điểm = 7 ngày)
        # Output:
        #   - dict: Dictionary chứa 7 đặc trưng lịch sử gồm 4 biến trễ (lag_24h, lag_48h, lag_72h, lag_7d)
        #           và 3 biến thống kê trượt (rolling_mean_24h, rolling_mean_7d, rolling_std_24h)

        if len(history) < 168:
            raise ValueError(
                "History phải >= 168 điểm (7 ngày)"
            )

        # ===== lag =====
        lag_24h = history[-24]      # Giá trị tiêu thụ cách thời điểm dự đoán 24 giờ (1 ngày trước)

        lag_48h = history[-48]      # Giá trị tiêu thụ cách thời điểm dự đoán 48 giờ (2 ngày trước)

        lag_72h = history[-72]      # Giá trị tiêu thụ cách thời điểm dự đoán 72 giờ (3 ngày trước)

        lag_7d = history[-168]      # Giá trị tiêu thụ cách thời điểm dự đoán 168 giờ (7 ngày trước)

        # ===== rolling =====
        rolling_mean_24h = np.mean(
            history[-24:]           # Trung bình trượt của 24 giờ gần nhất
        )

        rolling_mean_7d = np.mean(
            history[-168:]          # Trung bình trượt của 168 giờ (7 ngày) gần nhất
        )

        rolling_std_24h = np.std(
            history[-24:]           # Độ lệch chuẩn trượt của 24 giờ gần nhất
        )

        return {
            # lag
            "lag_24h": np.log1p(lag_24h),
            "lag_48h": np.log1p(lag_48h),
            "lag_72h": np.log1p(lag_72h),
            "lag_7d": np.log1p(lag_7d),

            # rolling
            "rolling_mean_24h": np.log1p(
                rolling_mean_24h
            ),

            "rolling_mean_7d": np.log1p(
                rolling_mean_7d
            ),

            "rolling_std_24h": np.log1p(
                rolling_std_24h
            )
        }

    # ==== build all features ====
    def build_features(
        self,
        time_str,  # Input 1: mốc thời gian của bước dự đoán
        history    # Input 2: danh sách giá trị tiêu thụ lịch sử (>= 168 giờ)
    ):
        # Input :
        #   - time_str (str | pd.Timestamp): Mốc thời gian cần dự đoán
        #   - history (list[float])        : Chuỗi giá trị lịch sử tiêu thụ (>= 168 điểm)
        # Output:
        #   - features (dict)              : Dictionary tổng hợp đầy đủ 11 đặc trưng đầu vào cho model
        features = {}

        # time features
        features.update(
            self.build_time_features(
                time_str
            )
        )

        # lag + rolling features
        features.update(
            self.build_lag_features(
                history
            )
        )

        return features
