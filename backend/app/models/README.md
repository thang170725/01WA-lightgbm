#  Tài Liệu Huấn Luyện & Đánh Giá Mô Hình (Nhánh `tuan_anh`)

> Tài liệu giải thích chi tiết những công việc đã thực hiện ở nhánh **`tuan_anh`**, bao gồm: hoàn thiện mã nguồn huấn luyện 2 mô hình (**LightGBM Regressor** và **Ridge Regression**), gỡ comment các luồng train/API, bổ sung chú thích `Input` / `Output` cho toàn bộ các hàm, đồng bộ thang đo `log1p` giữa lúc Train và lúc Predict, và tổng hợp kết quả thực nghiệm.

---

##  Mục Lục
1. [Tóm tắt các công việc đã thực hiện ở nhánh `tuan_anh`](#1-tóm-tắt-các-công-việc-đã-thực-hiện-ở-nhánh-tuan_anh)
2. [Kiến trúc & Đặc trưng đầu vào của 2 mô hình](#2-kiến-trúc--đặc-trưng-đầu-vào-của-2-mô-hình)
3. [Giải thích chi tiết Input / Output các hàm](#3-giải-thích-chi-tiết-input--output-các-hàm)
   - [3.1 File `backend/app/models/lightgbm.py`](#31-file-backendappmodelslightgbmpy)
   - [3.2 File `backend/app/models/model_manager.py`](#32-file-backendappmodelsmodel_managerpy)
   - [3.3 File `backend/app/models/feature_builder.py`](#33-file-backendappmodelsfeature_builderpy)
   - [3.4 File `backend/application.py`](#34-file-backendapplicationpy)
4. [Cải tiến kỹ thuật: Đồng bộ biến đổi `log1p` cho 7 biến Lag & Rolling](#4-cải-tiến-kỹ-thuật-đồng-bộ-biến-đổi-log1p-cho-7-biến-lag--rolling)
5. [Kết quả huấn luyện & Thực nghiệm trên tập Test](#5-kết-quả-huấn-luyện--thực-nghiệm-trên-tập-test)
6. [Hướng dẫn chạy lệnh huấn luyện & kiểm tra kết quả](#6-hướng-dẫn-chạy-lệnh-huấn-luyện--kiểm-tra-kết-quả)

---

## 1. Tóm tắt các công việc đã thực hiện ở nhánh `tuan_anh`

| File cập nhật | Nội dung thực hiện |
| :--- | :--- |
| [`backend/app/models/lightgbm.py`](lightgbm.py) | • Gỡ comment khối huấn luyện mô hình 2 (`LinearTrainerV2`) trong `__main__` để chạy 1 lệnh huấn luyện đồng thời cả **LightGBM** và **Ridge Regression**.<br>• Giữ nguyên các comment cũ và bổ sung đầy đủ chú thích `Input` / `Output` cho toàn bộ các hàm trong `LightGBMTrainer`, `LightGBMTrainerV2`, `LinearTrainerV2`.<br>• Thêm hàm `_sync_log1p_features()` đồng bộ phép biến đổi `np.log1p` cho 7 cột `lag` và `rolling`.<br>• Thêm hàm `_resolve_csv_path()` và xử lý `sys.path` để tránh xung đột tên file `lightgbm.py` với thư viện `lightgbm` cài đặt. |
| [`backend/app/models/model_manager.py`](model_manager.py) | • Bổ sung chú thích `Input` / `Output` cho `__init__`, `save_model`, `load_model`.<br>• Tự động nhận diện đường dẫn thư mục `models` dù chạy lệnh từ thư mục gốc hay từ `backend/`. |
| [`backend/app/models/feature_builder.py`](feature_builder.py) | • Bổ sung chú thích `Input` / `Output` và giải thích ý nghĩa vật lý của 4 đặc trưng thời gian (`build_time_features`), 7 đặc trưng lịch sử (`build_lag_features`) và hàm tổng hợp (`build_features`). |
| [`backend/app/utils/preprocessor.py`](../utils/preprocessor.py) | • Cập nhật hàm `transform_target()` để biến đổi `np.log1p` đồng bộ cho cả 7 cột `lag` & `rolling` cùng với `target_value`. |
| [`backend/application.py`](../../application.py) | • Gỡ comment phần khởi tạo `LinearTrainerV2`, nạp file `linear_v2.pkl` và mở lại endpoint `POST /predict_linear` phục vụ giao diện Web.<br>• Bổ sung chú thích `Input` / `Output` cho các hàm xử lý API. |
| [`backend/test.py`](../../test.py) | • Gỡ comment phần huấn luyện và kiểm thử lưu/nạp cả 2 model qua `ModelManager`. |

---

## 2. Kiến trúc & Đặc trưng đầu vào của 2 mô hình

Cả hai mô hình (**LightGBM Regressor** và **Ridge Regression**) được huấn luyện trên cùng một bộ dữ liệu (`train_ready3.csv`, `valid_ready3.csv`, `test_ready3.csv`) với **11 đặc trưng đầu vào (`features`)** và **1 biến mục tiêu (`target_value`)**:

* **4 đặc trưng thời gian (Time features)**:
  * `hour_sin` = $\sin(2\pi \cdot \text{hour}/24)$
  * `hour_cos` = $\cos(2\pi \cdot \text{hour}/24)$
  * `day_of_week` (`0`: Thứ Hai $\rightarrow$ `6`: Chủ Nhật)
  * `is_weekend` (`1`: Thứ Bảy/Chủ Nhật, `0`: Ngày thường)
* **4 đặc trưng độ trễ (Lag features — ở thang đo `log1p`)**:
  * `lag_24h` (cách 24 giờ — cùng giờ hôm trước)
  * `lag_48h` (cách 48 giờ — cùng giờ 2 ngày trước)
  * `lag_72h` (cách 72 giờ — cùng giờ 3 ngày trước)
  * `lag_7d` (cách 168 giờ — cùng giờ tuần trước)
* **3 đặc trưng thống kê trượt (Rolling features — ở thang đo `log1p`)**:
  * `rolling_mean_24h` (trung bình trượt 24 giờ gần nhất)
  * `rolling_mean_7d` (trung bình trượt 168 giờ gần nhất)
  * `rolling_std_24h` (độ lệch chuẩn trượt 24 giờ gần nhất)
* **Biến mục tiêu (`target_value`)**:
  * Lượng điện tiêu thụ theo giờ ở thang đo $\ln(1 + y)$ (`np.log1p`). Khi đánh giá (`evaluate`) và dự báo (`predict_new`), kết quả được đưa về đơn vị thực tế (**kWh**) bằng hàm ngược `np.expm1()`.

---

## 3. Giải thích chi tiết Input / Output các hàm

### 3.1. File `backend/app/models/lightgbm.py`

#### A. Hàm tiện ích
| Hàm | Input (Đầu vào) | Output (Đầu ra) | Chức năng |
| :--- | :--- | :--- | :--- |
| `_resolve_csv_path(path)` | `path` (`str`): Đường dẫn file CSV | `str`: Đường dẫn hợp lệ tồn tại trên ổ đĩa | Tự động tìm file CSV dù lệnh được gọi từ thư mục gốc dự án hay từ trong thư mục `backend/`. |
| `_sync_log1p_features(df)` | `df` (`pd.DataFrame`): Bảng dữ liệu Train/Valid/Test | `pd.DataFrame`: Bảng đã đồng bộ `np.log1p` | Kiểm tra nếu 7 cột `lag` & `rolling` đang ở thang kWh thô thì áp dụng `np.log1p` để đồng bộ với `target_value` và `FeatureBuilder`. |

#### B. Lớp `LightGBMTrainerV2` (Mô hình 1) & `LinearTrainerV2` (Mô hình 2)
| Phương thức | Input (Đầu vào) | Output (Đầu ra) | Giải thích hoạt động |
| :--- | :--- | :--- | :--- |
| `__init__(train_path, valid_path, test_path)` | • `train_path` (`str`)<br>• `valid_path` (`str`)<br>• `test_path` (`str`) | `None` | Đọc 3 tập dữ liệu CSV, đồng bộ `log1p` cho các cột đặc trưng trễ/trượt và khởi tạo danh sách 11 `features` cùng cột `target`. |
| `get_data()` | Không có tham số (dùng `self.train`, `self.valid`, `self.test`) | `tuple` 6 phần tử:<br>`(X_train, y_train, X_valid, y_valid, X_test, y_test)` | Tách dữ liệu thành ma trận đặc trưng `X` kích thước $(N \times 11)$ và vector mục tiêu `y` kích thước $(N,)$ cho cả 3 tập. |
| `build_model()` | Không có tham số | • `lgb.LGBMRegressor` *(LightGBMTrainerV2)*<br>• `sklearn.linear_model.Ridge` *(LinearTrainerV2)* | Khởi tạo mô hình:<br>• **LightGBM**: `n_estimators=3000`, `learning_rate=0.005`, `num_leaves=127`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=20`.<br>• **Ridge**: `alpha=1.0`, `random_state=42`. |
| `evaluate(y_true, preds)` | • `y_true` (`ndarray \| Series`): Nhãn thực tế (thang `log1p`)<br>• `preds` (`ndarray \| Series`): Giá trị dự đoán (thang `log1p`) | `dict`:<br>`{"mae", "relative_mae", "mape", "rmse", "r2"}` | Dùng `np.expm1()` chuyển về đơn vị kWh gốc, sau đó tính 5 chỉ số đánh giá sai số trên tập Test. |
| `train_model(save=False)` | • `save` (`bool`, mặc định `False`) | `tuple`:<br>`(model, evaluate)` | Huấn luyện mô hình trên tập Train (LightGBM có `early_stopping(50)` trên tập Valid), dự đoán trên tập Test, in bảng `evaluate` và xếp hạng độ quan trọng đặc trưng (`feature_importances_` / `coef_`). |
| `preprocess_input(input_dict)` | • `input_dict` (`dict`): Từ điển 11 đặc trưng của 1 mốc thời gian | `pd.DataFrame` kích thước `(1, 11)` | Chuyển dictionary đầu vào thành DataFrame 1 dòng sắp xếp đúng thứ tự cột `self.features`. |
| `predict_new(model, input_dict)` | • `model`: Mô hình đã train<br>• `input_dict` (`dict`): 11 đặc trưng | `float` (đơn vị kWh) | Dự đoán mức tiêu thụ điện cho 1 thời điểm mới và biến đổi ngược bằng `np.expm1()`. |
| `forecast_future(model, history, start_time, steps)` | • `model`: Mô hình đã train<br>• `history` (`list[float]` $\ge 168$ giờ)<br>• `start_time` (`str \| Timestamp`)<br>• `steps` (`int`, mặc định `24`) | `pd.DataFrame` gồm 2 cột `["time", "prediction"]` | Dự báo cuốn chiếu (Autoregressive): ở mỗi giờ tương lai, gọi `FeatureBuilder` tạo 11 features từ `history`, dự đoán giá trị mới rồi `history.append(pred)` để dùng cho giờ kế tiếp. |
| `predict_from_csv(model, csv_file, steps=24)` | • `model`: Mô hình đã train<br>• `csv_file`: File CSV (`time, value` $\ge 168$ dòng)<br>• `steps` (`int`, mặc định `24`) | `pd.DataFrame` gồm 2 cột `["time", "prediction"]` | Đọc file CSV lịch sử của người dùng và gọi `forecast_future()` để trả về dự báo 24 giờ tiếp theo. |

---

### 3.2. File `backend/app/models/model_manager.py`
| Phương thức | Input | Output | Chức năng |
| :--- | :--- | :--- | :--- |
| `__init__(model_dir="models")` | `model_dir` (`str`): Đường dẫn thư mục lưu model | `None` | Khởi tạo đường dẫn và tự động tạo thư mục lưu trữ nếu chưa tồn tại. |
| `save_model(model, evaluate, save_path)` | • `model` (`object`)<br>• `evaluate` (`dict`)<br>• `save_path` (`str`) | `path` (`str`): Đường dẫn file `.pkl` đã lưu | Đóng gói `{'model': model, 'evaluate': evaluate}` vào chung 1 file `.pkl` bằng `joblib.dump`. |
| `load_model(name)` | `name` (`str`): Tên file `.pkl` cần nạp | `dict`: `{'model': ..., 'evaluate': ...}` | Đọc và trả về đối tượng model kèm bộ chỉ số đánh giá đã lưu. |

---

### 3.3. File `backend/app/models/feature_builder.py`
| Phương thức | Input | Output | Chức năng |
| :--- | :--- | :--- | :--- |
| `build_time_features(time_str)` | `time_str` (`str \| Timestamp`) | `dict` (4 đặc trưng thời gian) | Tính `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`. |
| `build_lag_features(history)` | `history` (`list[float]` $\ge 168$ điểm) | `dict` (7 đặc trưng lịch sử ở thang `log1p`) | Trích xuất `lag_24h`, `lag_48h`, `lag_72h`, `lag_7d` và tính `rolling_mean_24h`, `rolling_mean_7d`, `rolling_std_24h` (đã qua `np.log1p`). |
| `build_features(time_str, history)` | `time_str`, `history` | `dict` (đầy đủ 11 đặc trưng) | Hợp nhất nhóm đặc trưng thời gian và lịch sử cho 1 bước dự báo. |

---

### 3.4. File `backend/application.py`
| Hàm / Endpoint | Input | Output | Chức năng |
| :--- | :--- | :--- | :--- |
| `_predict_from_upload(file, trainer, model, evaluate)` | `file` (`UploadFile`), `trainer`, `model`, `evaluate` (`dict`) | `dict`: `{"result": [...], "evaluate": {...}}` | Đọc CSV người dùng upload ($\ge 168$ dòng), chạy dự báo cuốn chiếu 24 giờ và trả về JSON cho Frontend. |
| `POST /predict_lightgbm` | `file` (`UploadFile` qua `multipart/form-data`) | JSON dự báo 24h + metrics của LightGBM | Endpoint dự báo bằng mô hình LightGBM (`lightgbm_model_v2.pkl`). |
| `POST /predict_linear` | `file` (`UploadFile` qua `multipart/form-data`) | JSON dự báo 24h + metrics của Ridge | Endpoint dự báo bằng mô hình Ridge Regression (`linear_v2.pkl`). |

---

## 4. Cải tiến kỹ thuật: Đồng bộ biến đổi `log1p` cho 7 biến Lag & Rolling

* **Vấn đề trước khi đồng bộ**:
  * Trong file CSV tiền xử lý ban đầu, chỉ có `target_value` được biến đổi $\ln(1 + y)$ (`log1p`), còn 7 cột `lag` và `rolling` vẫn giữ giá trị thô (kWh, lên tới hàng nghìn kWh).
  * Trong khi đó, lúc dự báo thực tế trên Web (`FeatureBuilder.build_lag_features`), cả 7 biến `lag` và `rolling` lại được biến đổi `np.log1p()`.
  * Sự lệch thang đo này khiến mô hình tuyến tính **Ridge Regression** bị bùng nổ hàm mũ khi lấy `np.expm1()` ($R^2 = -8.7911$, $\text{RMSE} = 889.28\text{ kWh}$), đồng thời gây lệch dữ liệu khi gọi API dự báo 24 giờ.
* **Giải pháp ở nhánh `tuan_anh`**:
  * Bổ sung bước đồng bộ `_sync_log1p_features()` trong `lightgbm.py` và cập nhật `transform_target()` trong `preprocessor.py` để cả 7 biến `lag` & `rolling` luôn ở thang đo `log1p` giống hệt `target_value` và `FeatureBuilder`.
  * Nhờ đó, **Ridge Regression** tăng $R^2$ từ `-8.7911` lên **`0.9778`**, giảm `MAE` từ `154.55 kWh` xuống **`18.67 kWh`**, và **LightGBM** vẫn giữ ngôi đầu trên toàn bộ các chỉ số評価 (`MAE = 16.87 kWh`, `MAPE = 10.00%`, `R² = 0.9780`).

---

## 5. Kết quả huấn luyện & Thực nghiệm trên tập Test

* **Quy mô dữ liệu**:
  * **Train**: `1,087,662` dòng × `14` cột
  * **Valid**: `250,344` dòng × `14` cột
  * **Test**: `131,157` dòng × `14` cột

### 5.1. Bảng so sánh Metrics trên tập Test

| Chỉ số đánh giá | Mô hình 1: LightGBM (`LightGBMTrainerV2`) | Mô hình 2: Ridge Regression (`LinearTrainerV2`) |
| :--- | :---: | :---: |
| **MAE** *(Sai số tuyệt đối trung bình)* | **16.8725 kWh**  | 18.6716 kWh |
| **Relative MAE** *(Tỷ lệ MAE / Trung bình)* | **7.61%**  | 8.43% |
| **RMSE** *(Sai số toàn phương trung bình)* | **42.1877 kWh**  | 42.2998 kWh |
| **MAPE** *(Sai số phần trăm tuyệt đối TB)* | **10.00%**  | 11.29% |
| **R² Score** *(Hệ số xác định)* | **0.9780 (97.80%)**  | 0.9778 (97.78%) |
| **Best Iteration** | `3000` (`valid l2: 0.012766`) | — |

### 5.2. Xếp hạng độ quan trọng đặc trưng (Feature Importance & Coefficients)

| Hạng | LightGBM (`feature_importances_`) | Số lần tách nhánh | Ridge Regression (`coef_`) | Hệ số hồi quy |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `lag_7d` | `53,406` | `lag_7d` | `+0.540181` |
| 2 | `rolling_mean_24h` | `51,120` | `rolling_mean_24h` | `+0.459309` |
| 3 | `lag_24h` | `50,513` | `rolling_mean_7d` | `-0.371981` |
| 4 | `rolling_std_24h` | `50,335` | `lag_24h` | `+0.181194` |
| 5 | `rolling_mean_7d` | `48,394` | `lag_72h` | `+0.144649` |
| 6 | `lag_72h` | `33,507` | `lag_48h` | `+0.055288` |
| 7 | `lag_48h` | `29,601` | `hour_sin` | `-0.021583` |
| 8 | `day_of_week` | `27,847` | `hour_cos` | `-0.019302` |
| 9 | `hour_cos` | `16,088` | `rolling_std_24h` | `-0.011984` |
| 10 | `hour_sin` | `13,441` | `day_of_week` | `-0.006617` |
| 11 | `is_weekend` | `2,045` | `is_weekend` | `-0.003327` |

---

## 6. Hướng dẫn chạy lệnh huấn luyện & kiểm tra kết quả

### Chạy huấn luyện đồng thời cả 2 model
Từ thư mục gốc dự án (`01WA-lightgbm/`), chạy:
```bash
python backend/app/models/lightgbm.py
```
Sau khi chạy xong, 2 file model sẽ được tạo tại:
* `backend/models/lightgbm_model_v2.pkl`
* `backend/models/linear_v2.pkl`

### Kiểm tra nhanh kết quả đã lưu trong `.pkl`
```bash
python -c "import sys; sys.path.insert(0, 'backend'); from app.models.model_manager import ModelManager; mm = ModelManager('backend/models'); print('LightGBM:', mm.load_model('lightgbm_model_v2.pkl')['evaluate']); print('Ridge:', mm.load_model('linear_v2.pkl')['evaluate'])"
```
