# LightGBM Electricity Forecast

Dự án dự báo tiêu thụ điện theo giờ bằng **LightGBM** và **Ridge Regression**, kèm giao diện web Next.js để upload CSV và xem kết quả dự đoán 24 giờ tiếp theo.

Repository: [https://github.com/thang170725/lightgbm](https://github.com/thang170725/lightgbm)

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Yêu cầu môi trường](#3-yêu-cầu-môi-trường)
4. [Clone & cài đặt từ đầu](#4-clone--cài-đặt-từ-đầu)
5. [Tải dataset gốc](#5-tải-dataset-gốc)
6. [Quy trình xử lý dữ liệu & train model (A → Z)](#6-quy-trình-xử-lý-dữ-liệu--train-model-a--z)
7. [Chạy backend API](#7-chạy-backend-api)
8. [Chạy frontend](#8-chạy-frontend)
9. [Chạy bằng Docker Compose](#9-chạy-bằng-docker-compose)
10. [Đẩy dự án lên GitHub](#10-đẩy-dự-án-lên-github)
11. [Xóa dự án trên máy để giải phóng tài nguyên](#11-xóa-dự-án-trên-máy-để-giải-phóng-tài-nguyên)
12. [Lỗi thường gặp](#12-lỗi-thường-gặp)

---
## 0. Preview
<img src="lightgbm.jpg" alt="preview" width=800 />

## 1. Tổng quan dự án

| Thành phần | Công nghệ |
|---|---|
| Backend API | Python 3.10, FastAPI, LightGBM, scikit-learn |
| Frontend | Next.js 16, React 19, Tailwind CSS, Recharts |
| Container | Docker Compose |

**Luồng làm việc chính:**

```
Dataset gốc (UCI)
    ↓
visualizations.py          → Khám phá & vẽ biểu đồ
    ↓
build_hourly_dataset.py    → Chuyển 15 phút → hourly, lọc user
    ↓
data_checker.py            → Kiểm tra chất lượng dữ liệu
    ↓
preprocessor.py            → Tiền xử lý, tạo feature, chia train/valid/test
    ↓
lightgbm.py                → Train model LightGBM / Ridge
    ↓
model_manager.py           → Lưu & load model (.pkl)
    ↓
feature_builder.py         → Tạo feature khi predict
    ↓
gen_data_user.py           → Sinh CSV mẫu để test
    ↓
Frontend                   → Upload CSV → nhận forecast 24h
```

---

## 2. Cấu trúc thư mục

```
lightgbm/
├── compose.yaml                          # Docker Compose (backend + frontend)
├── README.md
├── docs/
│   └── task.md
├── backend/
│   ├── application.py                    # FastAPI app (API predict)
│   ├── main.py                           # Script predict từ terminal
│   ├── test.py                           # Script test / debug
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── models/
│   │   │   ├── lightgbm.py               # Train & predict LightGBM / Ridge
│   │   │   ├── model_manager.py          # Lưu / load model
│   │   │   └── feature_builder.py        # Tạo feature khi predict realtime
│   │   └── utils/
│   │       ├── visualizations.py         # Vẽ biểu đồ EDA
│   │       ├── build_hourly_dataset.py   # Biến đổi dataset gốc → hourly
│   │       ├── data_checker.py           # Kiểm tra chất lượng dữ liệu
│   │       ├── preprocessor.py           # Tiền xử lý & chia tập train
│   │       └── gen_data_user.py          # Sinh dữ liệu test cho user
│   ├── dataset/                          # Dữ liệu (KHÔNG đẩy lên Git)
│   │   ├── LD2011_2014.txt               # Dataset gốc (~680 MB, tải từ UCI)
│   │   ├── hourly_electricity_filtered_v2.csv
│   │   ├── train_ready3.csv
│   │   ├── valid_ready3.csv
│   │   ├── test_ready3.csv
│   │   └── user_data.csv
│   └── models/                           # Model đã train (.pkl)
│       ├── lightgbm_model_v2.pkl
│       └── linear_v2.pkl
└── frontend/
    ├── app/page.tsx                      # Trang chính
    ├── routes/UploadApi.ts               # Gọi API backend
    └── src/components/                   # Upload, Chart, Table, Metrics
```

---

## 3. Yêu cầu môi trường

| Công cụ | Phiên bản khuyến nghị |
|---|---|
| Git | Bất kỳ bản mới |
| Python | 3.10 hoặc 3.11 |
| pip | Mới nhất |
| Node.js | 18+ (khuyến nghị 20) |
| npm | Đi kèm Node.js |
| Docker & Docker Compose | Tùy chọn, dùng khi muốn chạy container |

**Dung lượng ước tính sau khi setup đầy đủ:**

- Dataset gốc: ~680 MB
- Các file CSV đã xử lý: ~300–500 MB
- Model `.pkl`: ~35 MB mỗi file LightGBM
- `node_modules` frontend: ~300 MB

---

## 4. Clone & cài đặt từ đầu

### Bước 4.1 — Clone repository

```bash
git clone https://github.com/thang170725/lightgbm.git
cd lightgbm
```

### Bước 4.2 — Tạo virtual environment Python (khuyến nghị)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### Bước 4.3 — Cài dependencies backend

```bash
pip install -r backend/requirements.txt
```

### Bước 4.4 — Cài dependencies frontend

```bash
cd frontend
npm install
cd ..
```

### Bước 4.5 — Tạo thư mục dữ liệu (nếu chưa có)

```bash
mkdir -p backend/dataset/images
mkdir -p backend/models
```

---

## 5. Tải dataset gốc

Dataset **không** nằm trong Git (file quá lớn). Bạn cần tải thủ công:

1. Truy cập: [UCI Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
2. Tải file `LD2011_2014.txt`
3. Đặt vào đúng đường dẫn:

```bash
backend/dataset/LD2011_2014.txt
```

Kiểm tra:

```bash
ls -lh backend/dataset/LD2011_2014.txt
# Kết quả mong đợi: file ~680 MB
```

---

## 6. Quy trình xử lý dữ liệu & train model (A → Z)

> **Lưu ý:** Tất cả lệnh dưới đây chạy từ **thư mục gốc** `lightgbm/` (đã activate `.venv`).

---

### Bước 1 — Khám phá dữ liệu bằng biểu đồ (`visualizations.py`)

File: `backend/app/utils/visualizations.py`

Mục đích: vẽ biểu đồ EDA trước khi xử lý — line plot, heatmap theo giờ/ngày, histogram, boxplot, ACF.

```bash
cd backend
python -m app.utils.visualizations
cd ..
```

Các biểu đồ có sẵn trong class `VisualDatasetTXTV1`:

| Hàm | Mô tả |
|---|---|
| `dayly_load_chart()` | Đường tiêu thụ điện từng user |
| `plot_aggregate_load_chart()` | Tổng tiêu thụ 370 user trong 24h |
| `plot_time_pattern_heatmap()` | Heatmap theo giờ × ngày trong tuần |
| `plot_histogram()` | Phân phối tổng tải |
| `plot_random_users_boxplot()` | Boxplot ngẫu nhiên theo user |
| `plot_random_users_acf()` | ACF — kiểm tra tính chu kỳ |
| `plot_monthly_boxplot()` | Boxplot theo tháng |
| `plot_monthly_acf()` | ACF theo tháng |

Ảnh lưu tại `backend/dataset/images/` nếu gọi với `save=True`.

---

### Bước 2 — Biến đổi dataset gốc (`build_hourly_dataset.py`)

File: `backend/app/utils/build_hourly_dataset.py`

Mục đích:
- Đọc file 15 phút → resample về **1 giờ**
- Chỉ giữ năm **2014**
- Lọc **180 user** tiêu thụ tích cực nhất

Chỉnh tham số trong block `if __name__ == "__main__"` rồi chạy:

```bash
cd backend
python -m app.utils.build_hourly_dataset
cd ..
```

Hoặc gọi trực tiếp từ Python:

```python
from app.utils.build_hourly_dataset import build_hourly_dataset

build_hourly_dataset(
    input_path="backend/dataset/LD2011_2014.txt",
    output_path="backend/dataset/hourly_electricity_filtered_v2.csv",
    top_clients=180,
    min_mean_load=3.0,
    keep_year="2014",
    save=True,          # đặt True để lưu file
)
```

**Kết quả mong đợi:** `backend/dataset/hourly_electricity_filtered_v2.csv`

---

### Bước 3 — Kiểm tra chất lượng dữ liệu (`data_checker.py`)

File: `backend/app/utils/data_checker.py`

Hai class chính:

| Class | Dùng khi |
|---|---|
| `DataCheckerTXT` | Kiểm tra file gốc `.txt` (missing, zero ratio, sparsity…) |
| `CheckerAndProcessorCSV` | Kiểm tra CSV hourly, lọc outlier user tiêu thụ quá lớn |

**Kiểm tra file gốc:**

```python
from app.utils.data_checker import DataCheckerTXT

checker = DataCheckerTXT("backend/dataset/LD2011_2014.txt")
checker.run_all()
```

**Lọc user outlier trên CSV hourly:**

```bash
cd backend
python -m app.utils.data_checker
cd ..
```

Hoặc:

```python
from app.utils.data_checker import CheckerAndProcessorCSV

checker = CheckerAndProcessorCSV("backend/dataset/hourly_electricity_filtered_v2.csv")
cleaned_df, keep_users, removed_users = checker.remove_large_consumers(
    upper_quantile=0.95,
    save_path="backend/dataset/hourly_electricity_filtered_v2.csv",
)
checker.analyze_monthly_consumption(df=cleaned_df)
```

---

### Bước 4 — Tiền xử lý dữ liệu (`preprocessor.py`)

File: `backend/app/utils/preprocessor.py`

Pipeline trong class `Preprocessor`:

1. `from_wide_to_long()` — wide → long format
2. `build_time_features()` — `hour_sin`, `hour_cos`, `day_of_week`
3. `build_lag_features()` — lag 24h/48h/72h/7d, rolling mean/std
4. `clip_outliers()` — clip theo P99.5 của tập train
5. `transform_target()` — `log1p` target
6. `split_data()` — train đến 30/09/2014, valid đến 30/11/2014, còn lại là test
7. `save_splits()` — lưu `train_ready3.csv`, `valid_ready3.csv`, `test_ready3.csv`

**Chạy full pipeline:**

```python
from app.utils.preprocessor import Preprocessor

preprocessor = Preprocessor(dataset_path="backend/dataset/hourly_electricity_filtered_v2.csv")
train, valid, test = preprocessor.run_pipeline()
```

**Kết quả mong đợi:**

```
backend/dataset/train_ready3.csv
backend/dataset/valid_ready3.csv
backend/dataset/test_ready3.csv
```

---

### Bước 5 — Train model (`lightgbm.py`)

File: `backend/app/models/lightgbm.py`

| Class | Mô tả |
|---|---|
| `LightGBMTrainerV2` | Train LightGBM với early stopping |
| `LinearTrainerV2` | Train Ridge Regression (baseline) |

**Train LightGBM:**

```python
from app.models.lightgbm import LightGBMTrainerV2
from app.models.model_manager import ModelManager

trainer = LightGBMTrainerV2(
    train_path="backend/dataset/train_ready3.csv",
    valid_path="backend/dataset/valid_ready3.csv",
    test_path="backend/dataset/test_ready3.csv",
)
model, evaluate = trainer.train_model()

manager = ModelManager(model_dir="backend/models")
manager.save_model(model, evaluate, save_path="lightgbm_model_v2.pkl")
```

**Train Ridge (tùy chọn):**

```python
from app.models.lightgbm import LinearTrainerV2

trainer = LinearTrainerV2(
    train_path="backend/dataset/train_ready3.csv",
    valid_path="backend/dataset/valid_ready3.csv",
    test_path="backend/dataset/test_ready3.csv",
)
model, evaluate = trainer.train_model(save=False)

manager = ModelManager(model_dir="backend/models")
manager.save_model(model, evaluate, save_path="linear_v2.pkl")
```

**Kết quả mong đợi:**

```
backend/models/lightgbm_model_v2.pkl
backend/models/linear_v2.pkl
```

---

### Bước 5b — Quản lý model (`model_manager.py`)

File: `backend/app/models/model_manager.py`

```python
from app.models.model_manager import ModelManager

manager = ModelManager(model_dir="backend/models")

# Lưu
manager.save_model(model, evaluate, save_path="lightgbm_model_v2.pkl")

# Load
loaded = manager.load_model(name="lightgbm_model_v2.pkl")
model = loaded["model"]
metrics = loaded["evaluate"]
```

---

### Bước 5c — Feature khi predict (`feature_builder.py`)

File: `backend/app/models/feature_builder.py`

Dùng khi dự đoán từng giờ (không có sẵn lag trong CSV):

```python
from app.models.feature_builder import FeatureBuilder

fb = FeatureBuilder()
features = fb.build_features(
    time_str="2014-12-01 08:00:00",
    history=[...],   # list >= 168 giá trị quá khứ
)
```

---

### Bước 6 — Sinh dữ liệu test (`gen_data_user.py`)

File: `backend/app/utils/gen_data_user.py`

Tạo file CSV mẫu (8 ngày = 192 giờ, đủ điều kiện >= 168 rows) để test predict:

```bash
cd backend
python -m app.utils.gen_data_user
cd ..
```

**Kết quả:** `backend/dataset/user_data_3.csv` (hoặc chỉnh `output_path` trong file)

Format CSV bắt buộc khi upload lên frontend/API:

```csv
time,value
2014-01-01 00:00:00,12.5
2014-01-01 01:00:00,11.3
...
```

---

### Bước 7 — Test predict từ terminal (`main.py`)

```bash
cd backend
python main.py
cd ..
```

In ra timeline gồm dữ liệu lịch sử + 24 giờ dự đoán.

---

### Tóm tắt lệnh chạy nhanh (từ thư mục gốc)

```bash
# 1. EDA (tùy chọn)
cd backend && python -m app.utils.visualizations && cd ..

# 2. Build hourly — nhớ set save=True trong file hoặc gọi qua Python
cd backend && python -m app.utils.build_hourly_dataset && cd ..

# 3. Check & lọc outlier
cd backend && python -m app.utils.data_checker && cd ..

# 4–5. Preprocess + Train — chạy qua Python interactive hoặc script riêng
python -c "
from app.utils.preprocessor import Preprocessor
from app.models.lightgbm import LightGBMTrainerV2, LinearTrainerV2
from app.models.model_manager import ModelManager
import sys; sys.path.insert(0, 'backend')

Preprocessor('backend/dataset/hourly_electricity_filtered_v2.csv').run_pipeline()

for Cls, name in [(LightGBMTrainerV2, 'lightgbm_model_v2.pkl'), (LinearTrainerV2, 'linear_v2.pkl')]:
    t = Cls('backend/dataset/train_ready3.csv', 'backend/dataset/valid_ready3.csv', 'backend/dataset/test_ready3.csv')
    m, e = t.train_model()
    ModelManager('backend/models').save_model(m, e, save_path=name)
"

# 6. Sinh dữ liệu test
cd backend && python -m app.utils.gen_data_user && cd ..
```

---

## 7. Chạy backend API

API cần có sẵn model trong `backend/models/` trước khi khởi động.

```bash
cd backend
uvicorn application:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Mô tả |
|---|---|---|
| `http://127.0.0.1:8000/predict_lightgbm` | POST | Upload CSV → forecast LightGBM |
| `http://127.0.0.1:8000/predict_linear` | POST | Upload CSV → forecast Ridge |

Body: `multipart/form-data`, field `file` = file CSV (`time`, `value`, tối thiểu **168 dòng**).

Test nhanh bằng curl:

```bash
curl -X POST http://127.0.0.1:8000/predict_lightgbm \
  -F "file=@backend/dataset/user_data.csv"
```

---

## 8. Chạy frontend

Mở terminal mới (backend vẫn đang chạy):

```bash
cd frontend
npm run dev
```

Truy cập: [http://localhost:3000](http://localhost:3000)

**Cách test:**
1. Chọn model (LightGBM hoặc Ridge)
2. Upload file CSV (`time`, `value`, >= 168 rows)
3. Xem biểu đồ forecast 24h, bảng số liệu và metrics (MAE, RMSE, MAPE, R²)

---

## 9. Chạy bằng Docker Compose

Yêu cầu: đã train xong model và có file trong `backend/models/`.

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Backend | http://localhost:8000 |
| Frontend | http://localhost:3000 |

Dừng container:

```bash
docker compose down
```

---

## 10. Đẩy dự án lên GitHub

### 10.1 — Kiểm tra `.gitignore`

Đảm bảo **không** commit file quá lớn:

```gitignore
# Dữ liệu (tải lại từ UCI sau khi clone)
backend/dataset/*.txt
backend/dataset/*.csv
backend/dataset/images/

# Model (train lại sau khi clone, hoặc dùng Git LFS nếu muốn giữ)
backend/models/*.pkl

# Python
.venv/
**/__pycache__/
*.pyc

# Node
frontend/node_modules/
frontend/.next/
```

> **Quan trọng:** File `LD2011_2014.txt` (~680 MB) và các CSV processed **không nên** đẩy lên GitHub. Người clone sẽ tải dataset và chạy lại pipeline theo mục 5–6.

### 10.2 — Commit & push

```bash
git add .
git status          # kiểm tra không có file .txt/.csv/.pkl lớn
git commit -m "Cập nhật README hướng dẫn setup đầy đủ"
git push origin main   # hoặc tên branch của bạn (vd: thang)
```

### 10.3 — Nếu muốn giữ model trên Git (tùy chọn)

Model LightGBM ~35 MB — có thể dùng [Git LFS](https://git-lfs.github.com/):

```bash
git lfs install
git lfs track "backend/models/*.pkl"
git add .gitattributes
git add backend/models/
git commit -m "Thêm model qua Git LFS"
git push
```

---

## 11. Xóa dự án trên máy để giải phóng tài nguyên

**Chỉ làm sau khi đã push thành công lên GitHub và xác nhận code trên remote đầy đủ.**

```bash
# 1. Kiểm tra đã push hết chưa
cd /đường/dẫn/tới/lightgbm
git status
git log origin/main..HEAD    # không còn commit local chưa push

# 2. Dừng container nếu đang chạy
docker compose down

# 3. Deactivate virtualenv
deactivate

# 4. Xóa thư mục dự án
cd ..
rm -rf lightgbm
```

**Khôi phục sau này:**

```bash
git clone https://github.com/thang170725/lightgbm.git
cd lightgbm
# Làm lại từ Mục 4 → 8
```

---

## 12. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError: LD2011_2014.txt` | Chưa tải dataset UCI | Làm lại [Mục 5](#5-tải-dataset-gốc) |
| `Model not found: lightgbm_model_v2.pkl` | Chưa train model | Làm lại [Bước 5](#bước-5--train-model-lightgbmpy) |
| `Need at least 168 rows` | CSV upload quá ít dữ liệu | Cần tối thiểu 7 ngày × 24 giờ |
| Frontend không gọi được API | Backend chưa chạy hoặc sai port | Kiểm tra `uvicorn` đang chạy port `8000` |
| `ModuleNotFoundError: app` | Chạy sai thư mục | API/train script chạy từ `backend/`; hoặc thêm `backend` vào `PYTHONPATH` |
| Docker build lỗi thiếu model | Chưa train trước khi `docker compose up` | Train model trước, đảm bảo file `.pkl` trong `backend/models/` |
| Biểu đồ không hiện (headless server) | Không có display | Gọi hàm vẽ với `save=True` thay vì `plt.show()` |

---

## Tham khảo thêm

- Dataset: [UCI Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- Ghi chú visualization: [thang170725/Notes](https://github.com/thang170725/Notes.git) → Stack → Python → Seaborn
