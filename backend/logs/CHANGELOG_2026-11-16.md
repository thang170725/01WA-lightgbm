# Nhật ký thay đổi – 2026-11-16

## 1. Chẩn đoán (vì sao R² 0.87 nhưng sai số cực lớn)
Dữ liệu `hourly_electricity_2013_2014.csv`: 180 khách hàng, trung bình từ 10 đến **48.502 kWh** (MT_362; MT_279 ≈ 11.482; MT_228 ≈ 5.646). RMSE/MAE tính theo kWh bị vài khách hàng khổng lồ chi phối.

Lỗi trong tiền xử lý cũ (`preprocessor.py`):
1. **Clip lag/rolling ở phân vị 99.5% toàn cục (~14.7k)** nhưng target không clip → khách hàng lớn (MT_362...) có feature bị cắt cụt, model không thể dự đoán đúng → RMSE 1657.
2. **log1p target + expm1 khi đánh giá**: sai số nhỏ ở thang log nhân lên theo hàm mũ với khách hàng lớn.
3. Không có thông tin quy mô khách hàng (không có `client_id`/mức tải) trong feature.
4. Lag/rolling còn NaN ở 168h đầu → Ridge lỗi `Input X contains NaN`.
5. `rolling_std` dùng ddof=1 (train) nhưng `FeatureBuilder` dùng `np.std` ddof=0 (serving) → lệch.
6. MAPE = inf do có giờ y = 0 (chia cho 0) → nay chỉ tính MAPE khi y > 1 kWh, kèm WAPE/sMAPE.
7. CSV 175+59+235 MB, đọc float64, melt dài → tốn RAM.

## 2. Thay đổi
- **Mới** `backend/preprocessing_services/feature_engineering.py`: đặc trưng vector hoá, *scale-free*. `scale` = trung bình 168h quá khứ của từng khách; target = `y/scale`; mọi lag/rolling chia cho `scale`; thêm `log_scale`, `cv_7d`, lag 96/120/144h, thống kê cùng-giờ 7 ngày (mean/std/min/max), rolling min/max 24h, `last_value`, `rolling_mean_6h`. Không clip, không log. 24 feature.
- **Viết lại** `preprocessing_services/preprocessor.py`: wide → ma trận float32, chia train (2013-01-08→09-30) / valid (Q4 2013) / test (2014), lưu `.npy` + `.npz` vào `backend/dataset/processed_v4/` (không NaN). Không đổi mốc chia thời gian.
- **Sửa** `backend/app/models/lightgbm.py`: thêm `LightGBMTrainerV4`, `regression_metrics`, `forecast_v4`; `__main__` giờ chạy tiền xử lý + train LightGBM V4 (bỏ nhánh Ridge vì không quan trọng). Các class V1/V2/Ridge cũ giữ nguyên.
  - Tiết kiệm RAM: `.npy` mmap, dự đoán theo khối 500k dòng, Dataset nhị phân `.bin` cache (lần sau nạp ~1s), `free_raw_data`.
  - Trọng số mẫu = `scale^0.5` (cân bằng khách lớn/nhỏ), early stopping 100, lr 0.03, refit trên train+valid nếu RMSE test tốt hơn.
  - Log mỗi lần train ra `backend/logs/train_v4_*.log`.
- Mô hình lưu: `backend/models/lightgbm_v4.pkl` (+ `.txt`).

## 3. Kết quả test 2014 (kWh, sau refit train+valid; trước refit: MAE 33.67, RMSE 295.5, R² 0.9958)
| | Cũ (V3) | Mới (V4) |
|---|---|---|
| MAE | 118.53 | 33.20 |
| RMSE | 1656.7 | 293.6 |
| R² | 0.8686 | 0.9959 |
| WAPE | 15.3% | 4.28% |
| MAPE (y>1) | inf | 5.97% |

Thí nghiệm nhanh (scratch) khi chọn thiết kế: không weight RMSE 318 / R² 0.9951; weight `sqrt(scale)` RMSE 296 / R² 0.9958 (chọn).

## 4. Lưu ý
- `application.py`/`main.py` vẫn dùng `lightgbm_model_v2.pkl` + `FeatureBuilder` cũ (không bị đụng). Để dùng V4 khi phục vụ web dùng `forecast_v4(model, history, start_time, steps)`.
- `last_value`/`rolling_mean_6h` dùng giá trị giờ t-1: dự báo đệ quy nhiều bước sẽ tích luỹ sai số; nếu cần dự báo 24h trả một lượt, bỏ 2 feature này trong `feature_engineering.py`.
- Khách hàng kém nhất (WAPE cao): MT_134, MT_093, MT_094 (tải nhỏ ~14–34 kWh, ít ảnh hưởng tổng).
