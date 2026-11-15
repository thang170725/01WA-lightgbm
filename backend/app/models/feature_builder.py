import numpy as np
import pandas as pd

# [Ghi chú mới] Lớp trích xuất 11 đặc trưng đầu vào khi dự báo tương lai (Realtime / Autoregressive Feature Builder)
class FeatureBuilder:
    def __init__(self):
        pass

    # ==== time features ====
    # [Input] :
    #   - time_str (str | pd.Timestamp): Mốc thời gian cần tạo đặc trưng (vd: "2014-12-01 08:00:00")
    # [Output]:
    #   - dict: Dictionary chứa 4 đặc trưng thời gian {"hour_sin", "hour_cos", "day_of_week", "is_weekend"}
    def build_time_features(
        self,
        time_str
    ):
        dt = pd.to_datetime(time_str)

        hour = dt.hour

        day_of_week = dt.dayofweek

        is_weekend = (
            1 if day_of_week >= 5 else 0
        )

        # cyclical encoding
        # [Ghi chú mới] Mã hóa giờ theo hình tròn lượng giác (Sin/Cos) để giữ tính liên tục giữa 23h và 0h
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
    # [Input] :
    #   - history (list[float]): Danh sách giá trị tiêu thụ quá khứ đã sắp xếp theo thời gian (độ dài >= 168 giờ)
    # [Output]:
    #   - dict: Dictionary chứa 7 đặc trưng lịch sử (4 biến trễ lag + 3 biến thống kê trượt rolling)
    def build_lag_features(
        self,
        history
    ):
        """
        history:
            list giá trị theo thời gian
            (đã sort tăng dần)
        """

        if len(history) < 168:
            raise ValueError(
                "History phải >= 168 điểm (7 ngày)"
            )

        # ===== lag =====
        # [Ghi chú mới] Lấy giá trị tiêu thụ tại các mốc cách hiện tại 24h (1 ngày), 48h (2 ngày), 72h (3 ngày) và 168h (7 ngày)
        lag_24h = history[-24]

        lag_48h = history[-48]

        lag_72h = history[-72]

        lag_7d = history[-168]

        # ===== rolling =====
        # [Ghi chú mới] Tính trung bình trượt 24h, trung bình trượt 7 ngày và độ lệch chuẩn 24h gần nhất
        rolling_mean_24h = np.mean(
            history[-24:]
        )

        rolling_mean_7d = np.mean(
            history[-168:]
        )

        rolling_std_24h = np.std(
            history[-24:]
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
    # [Input] :
    #   - time_str (str | pd.Timestamp): Mốc thời gian của bước dự đoán
    #   - history  (list[float])       : Danh sách giá trị lịch sử tiêu thụ (>= 168 điểm)
    # [Output]:
    #   - features (dict)              : Dictionary tổng hợp đầy đủ 11 đặc trưng đầu vào cho mô hình
    def build_features(
        self,
        time_str,
        history
    ):
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
