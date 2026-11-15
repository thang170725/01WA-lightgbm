# Tài liệu Module Huấn luyện & Dự báo Mô hình (`backend/app/models`)

Thư mục này chứa toàn bộ khung huấn luyện (Training Framework), quản lý mô hình (Model Serialization) và trích xuất đặc trưng thời gian thực (Realtime Feature Engineering) cho 2 mô hình dự báo phụ tải điện năng theo giờ:
1. **Mô hình 1 — LightGBM Regressor (`LightGBMTrainerV2`)**: Mô hình Gradient Boosted Decision Trees chính của dự án.
2. **Mô hình 2 — Ridge Regression (`LinearTrainerV2`)**: Mô hình hồi quy tuyến tính có điều chuẩn $L_2$ dùng làm Baseline đối sánh.

---

## 1. Cấu trúc các file trong `backend/app/models/`

| File | Lớp / Hàm chính | Vai trò |
|---|---|---|
| `lightgbm.py` | `LightGBMTrainer`, `LightGBMTrainerV2`, `LinearTrainerV2`, `run_training_pipeline()` | Khung nạp dữ liệu, xây dựng kiến trúc model, huấn luyện, đánh giá metrics trên tập Test và dự báo cuốn chiếu 24 giờ tới. |
| `model_manager.py` | `ModelManager` | Đóng gói (`save_model`) và tải (`load_model`) đối tượng model kèm bộ chỉ số `evaluate` dưới định dạng `.pkl`. |
| `feature_builder.py` | `FeatureBuilder` | Tính toán 11 đặc trưng đầu vào (4 đặc trưng thời gian + 4 biến trễ lag + 3 biến thống kê trượt rolling) cho từng bước dự báo tương lai. |

---

## 2. Đặc trưng đầu vào (11 Features) & Biến mục tiêu (Target)

Khi nhận dữ liệu đã tiền xử lý từ 3 file `train_ready3.csv`, `valid_ready3.csv`, `test_ready3.csv`, cả 2 mô hình đều sử dụng chung:

* **Nhóm đặc trưng thời gian (Time features - 4 cột)**:
  * `hour_sin`, `hour_cos`: Mã hóa chu kỳ giờ trong ngày bằng hàm lượng giác $\sin(2\pi \cdot \text{hour}/24)$ và $\cos(2\pi \cdot \text{hour}/24)$.
  * `day_of_week`: Thứ trong tuần (`0` = Thứ Hai $\to$ `6` = Chủ Nhật).
  * `is_weekend`: Cờ đánh dấu ngày cuối tuần (`1` nếu Thứ Bảy/Chủ Nhật, `0` nếu ngày thường).
* **Nhóm đặc trưng độ trễ (Lag features - 4 cột)**:
  * `lag_24h` (cách 1 ngày), `lag_48h` (cách 2 ngày), `lag_72h` (cách 3 ngày), `lag_7d` (cách 168 giờ = 7 ngày).
* **Nhóm đặc trưng thống kê trượt (Rolling features - 3 cột)**:
  * `rolling_mean_24h` (trung bình 24h gần nhất), `rolling_mean_7d` (trung bình 168h gần nhất), `rolling_std_24h` (độ lệch chuẩn 24h gần nhất).
* **Biến mục tiêu (`target_value`)**:
  * Giá trị tiêu thụ điện đã biến đổi $\log(1 + y)$ (`np.log1p`). Khi đánh giá (`evaluate`) và dự báo (`predict_new`), kết quả được biến đổi ngược về đơn vị thực tế (kWh) bằng `np.expm1`.

---

## 3. Giải thích chi tiết Input / Output của từng hàm trong `lightgbm.py`

### 3.1. Lớp `LightGBMTrainerV2` và `LinearTrainerV2`

| Tên phương thức | Input (Đầu vào) | Output (Đầu ra) | Giải thích hoạt động |
|---|---|---|---|
| `__init__(train_path, valid_path, test_path)` | • `train_path` (`str`)<br>• `valid_path` (`str`)<br>• `test_path` (`str`) | `None` | Lưu đường dẫn 3 tập dữ liệu, cấu hình danh sách 11 `features` và `target`. Hỗ trợ nạp trễ (Lazy Loading) để không báo lỗi khi khởi tạo khung trước khi có file CSV. |
| `get_data()` | Không có tham số | `tuple` 6 phần tử:<br>`(X_train, y_train, X_valid, y_valid, X_test, y_test)` | Kiểm tra sự tồn tại của 3 file CSV, đọc vào DataFrame và tách thành ma trận đặc trưng `X` $(N \times 11)$ cùng vector nhãn `y` $(N,)$. |
| `build_model()` | Không có tham số | • `lgb.LGBMRegressor` *(ở LightGBMTrainerV2)*<br>• `sklearn.linear_model.Ridge` *(ở LinearTrainerV2)* | Khởi tạo mô hình với các siêu tham số đã tinh chỉnh (`n_estimators=3000`, `learning_rate=0.005`, `num_leaves=127`, `max_depth=8` cho LightGBM; `alpha=1.0` cho Ridge). |
| `evaluate(y_true, preds)` | • `y_true` (`ndarray \| Series`): Nhãn thực tế (thang `log1p`)<br>• `preds` (`ndarray \| Series`): Nhãn dự báo (thang `log1p`) | `dict`:<br>`{"mae", "relative_mae", "mape", "rmse", "r2"}` | Dùng `np.expm1` chuyển về đơn vị kWh gốc, sau đó tính 5 chỉ số đánh giá độ chính xác trên tập Test. |
| `train_model(save=False)` | • `save` (`bool`, mặc định `False`) | `tuple`:<br>`(model, evaluate)` | Chạy toàn bộ quy trình huấn luyện: nạp dữ liệu $\to$ khớp mô hình (LightGBM dùng `early_stopping(50)` trên tập Valid) $\to$ dự đoán tập Test $\to$ tính metrics $\to$ in bảng xếp hạng độ quan trọng đặc trưng (`feature_importances_` / `coef_`). |
| `preprocess_input(input_dict)` | • `input_dict` (`dict`): Dictionary chứa 11 đặc trưng của 1 mốc thời gian | `pd.DataFrame` kích thước `(1, 11)` | Chuyển đổi dictionary đầu vào thành DataFrame 1 dòng theo đúng thứ tự cột `self.features`. |
| `predict_new(model, input_dict)` | • `model`: Mô hình đã train<br>• `input_dict` (`dict`): 11 đặc trưng | `float` | Dự đoán mức tiêu thụ điện (kWh) cho 1 thời điểm mới (đã qua `np.expm1`). |
| `forecast_future(model, history, start_time, steps)` | • `model`: Mô hình đã train<br>• `history` (`list[float]` $\ge 168$ giờ)<br>• `start_time` (`str \| Timestamp`)<br>• `steps` (`int`, vd: `24`) | `pd.DataFrame` gồm 2 cột `["time", "prediction"]` | Dự báo cuốn chiếu (Autoregressive): ở mỗi bước $t$, gọi `FeatureBuilder` tạo 11 features từ `history`, dự đoán giá trị mới và `append` vào `history` để dự báo bước $t+1$. |
| `predict_from_csv(model, csv_file, steps=24)` | • `model`: Mô hình đã train<br>• `csv_file`: File CSV (`time`, `value` $\ge 168$ dòng)<br>• `steps` (`int`, mặc định `24`) | `pd.DataFrame` gồm 2 cột `["time", "prediction"]` | Đọc file CSV lịch sử của user, trích chuỗi `history` và gọi `forecast_future` để dự báo 24 giờ tiếp theo. |

### 3.2. Hàm điều phối `run_training_pipeline()`
* **Input**: `train_path`, `valid_path`, `test_path` (`str`) và `model_dir` (`str`, mặc định `"backend/models"`).
* **Output**: Dictionary chứa đối tượng `model` và kết quả `evaluate` của cả 2 mô hình (`"lightgbm"` và `"ridge"`), đồng thời lưu tự động 2 file `backend/models/lightgbm_model_v2.pkl` và `backend/models/linear_v2.pkl`.

---

## 4. Hướng dẫn chạy khi đã có dữ liệu tiền xử lý

Khi đã có 3 file `train_ready3.csv`, `valid_ready3.csv`, `test_ready3.csv` trong thư mục `backend/dataset/`, chạy lệnh sau từ thư mục gốc dự án:

```powershell
python -c "import sys; sys.path.insert(0, 'backend'); from app.models.lightgbm import run_training_pipeline; run_training_pipeline()"
```
