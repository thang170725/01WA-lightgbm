# 📊 Hướng Dẫn Khám Phá Dữ Liệu (EDA) — LightGBM Electricity Forecast

> Tài liệu kỹ thuật chi tiết dành cho các thành viên trong nhóm về phần **Khám phá & Trực quan hóa dữ liệu (EDA)**, giải thích toàn bộ mã nguồn trong [`visualizations.py`](../backend/app/utils/visualizations.py) và script thực thi [`run_eda.py`](../backend/run_eda.py).

---

## 📑 Mục Lục
1. [Phân công nhiệm vụ nhóm (Theo task.md)](#1-phân-công-nhiệm-vụ-nhóm-theo-taskmd)
2. [Vai trò của EDA trong bài toán dự báo phụ tải điện](#2-vai-trò-của-eda-trong-bài-toán-dự-báo-phụ-tải-điện)
3. [Đặc điểm tập dữ liệu đầu vào (UCI LD2011_2014.txt)](#3-đặc-điểm-tập-dữ-liệu-đầu-vào-uci-ld2011_2014txt)
4. [Giải thích chi tiết mã nguồn `visualizations.py`](#4-giải-thích-chi-tiết-mã-nguồn-visualizationspy)
   - [4.0 Khởi tạo & nạp dữ liệu: `__init__`](#40-khởi-tạo--nạp-dữ-liệu-__init__)
   - [4.1 Hàm 1: `dayly_load_chart()` — Đường tiêu thụ từng khách hàng](#41-hàm-1-dayly_load_chart--đường-tiêu-thụ-từng-khách-hàng)
   - [4.2 Hàm 2: `plot_aggregate_load_chart()` — Tổng tải 24h trong ngày](#42-hàm-2-plot_aggregate_load_chart--tổng-tải-24h-trong-ngày)
   - [4.3 Hàm 3: `plot_time_pattern_heatmap()` — Heatmap Giờ × Ngày trong tuần](#43-hàm-3-plot_time_pattern_heatmap--heatmap-giờ--ngày-trong-tuần)
   - [4.4 Hàm 4: `plot_histogram()` — Phân phối tổng tải & kiểm tra độ lệch](#44-hàm-4-plot_histogram--phân-phối-tổng-tải--kiểm-tra-độ-lệch)
   - [4.5 Hàm 5: `plot_random_users_boxplot()` — So sánh biến thiên giữa các user](#45-hàm-5-plot_random_users_boxplot--so-sánh-biến-thiên-giữa-các-user)
   - [4.6 Hàm 6: `plot_random_users_acf()` — Kiểm tra chu kỳ tuần (Weekly Seasonality)](#46-hàm-6-plot_random_users_acf--kiểm-tra-chu-kỳ-tuần-weekly-seasonality)
   - [4.7 Hàm 7: `plot_monthly_boxplot()` — Mùa vụ theo 12 tháng](#47-hàm-7-plot_monthly_boxplot--mùa-vụ-theo-12-tháng)
   - [4.8 Hàm 8: `plot_monthly_acf()` — Kiểm tra tương quan theo tháng](#48-hàm-8-plot_monthly_acf--kiểm-tra-tương-quan-theo-tháng)
5. [Giải thích chi tiết mã nguồn script `run_eda.py`](#5-giải-thích-chi-tiết-mã-nguồn-script-run_edapy)
6. [Bảng ánh xạ từ EDA sang Tiền xử lý & Xây dựng mô hình](#6-bảng-ánh-xạ-từ-eda-sang-tiền-xử-lý--xây-dựng-mô-hình)
7. [Hướng dẫn chạy lệnh cho cả nhóm](#7-hướng-dẫn-chạy-lệnh-cho-cả-nhóm)

---

## 1. Phân công nhiệm vụ nhóm (Theo task.md)

Theo phân công trong [`docs/task.md`](task.md), nhiệm vụ EDA được chia thành các phần:

| Thành viên | Biểu đồ phụ trách | Hàm tương ứng trong code |
| :--- | :--- | :--- |
| **Thắng** | Biểu đồ biến thiên tiêu thụ (Line Plot) | `dayly_load_chart()`, `plot_aggregate_load_chart()` |
| **Minh** | Biểu đồ Time-pattern Heatmap & Histogram | `plot_time_pattern_heatmap()`, `plot_histogram()` |
| **Long** | Biểu đồ Boxplot & ACF (Autocorrelation) | `plot_random_users_boxplot()`, `plot_random_users_acf()` |

---

## 2. Vai trò của EDA trong bài toán dự báo phụ tải điện

Trong các bài toán dự báo chuỗi thời gian (Time Series Forecasting), **EDA (Exploratory Data Analysis)** giúp nhóm trả lời các câu hỏi nền tảng trước khi xây dựng mô hình:
1. Chuỗi phụ tải có **tính chu kỳ (seasonality)** theo giờ trong ngày, ngày trong tuần, hay tháng trong năm không?
2. Phân phối dữ liệu có bị **lệch (skewed)** không? Có cần biến đổi hàm toán học (như `log1p`) không?
3. Có hiện tượng **outlier** (người dùng tiêu thụ đột biến hàng nghìn kWh so với mặt bằng chung) không?
4. Cần tạo những đặc trưng độ trễ (**lag features**) ở các mốc thời gian nào (1 ngày trước, 2 ngày trước, hay 1 tuần trước)?

```
Dataset thô (LD2011_2014.txt)
          │
          ▼
   [visualizations.py / run_eda.py]
          │
          ├──> 1. Heatmap: Giờ cao điểm / ngày làm việc vs cuối tuần
          ├──> 2. Histogram: Phân phối lệch phải -> cần log1p target
          ├──> 3. ACF: Chu kỳ tuần rõ rệt -> cần Lag 168h (7 ngày)
          └──> 4. Boxplot: Nhận diện outlier khách hàng -> lọc bỏ ở data_checker
          │
          ▼
 [Quyết định Preprocessing & Feature Engineering chính xác]
```

---

## 3. Đặc điểm tập dữ liệu đầu vào (UCI LD2011_2014.txt)

* **Nguồn**: [UCI Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).
* **Định dạng file**: Text phân cách bởi dấu chấm phẩy (`;`), dấu thập phân là dấu phẩy (`,`).
* **Chu kỳ đo**: 15 phút một lần (tương ứng 96 điểm đo mỗi ngày).
* **Quy mô**:
  * 140,256 dòng (từ 2011 đến cuối 2014).
  * 371 cột: Cột đầu là timestamp, 370 cột còn lại là mã khách hàng từ `MT_001` đến `MT_370`.
  * Dung lượng: ~700 MB.
* **Lưu ý quan trọng**: Nhiều khách hàng mới bắt đầu dùng điện vào năm 2012 hoặc 2013, do đó các mốc thời gian đầu năm 2011 có rất nhiều giá trị 0.0.

---

## 4. Giải thích chi tiết mã nguồn `visualizations.py`

File nằm tại: [`backend/app/utils/visualizations.py`](../backend/app/utils/visualizations.py).

### 4.0 Khởi tạo & nạp dữ liệu: `__init__`

```python
class VisualDatasetTXTV1:
    def __init__(self, dataset_path: str = None):
        self.df: pd.DataFrame = pd.read_csv(
            dataset_path,
            sep=";",
            decimal=",",
            parse_dates=[0]
        )
        self.df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
        self.columns = self.df.columns
        self.rows = len(self.df)
```

* `sep=";"`: File gốc dùng dấu chấm phẩy để phân tách cột.
* `decimal=","`: Chuyển định dạng số thực châu Âu (ví dụ `12,5` thành số float `12.5`).
* `parse_dates=[0]`: Tự động nhận diện cột chỉ số 0 thành kiểu đối tượng thời gian `datetime64[ns]` của pandas.
* `rename(columns={"Unnamed: 0": "timestamp"})`: Đặt lại tên cột đầu tiên cho dễ gọi và xử lý các hàm time-series.

---

### 4.1 Hàm 1: `dayly_load_chart()` — Đường tiêu thụ từng khách hàng

*(Lưu ý: Tên hàm trong code đặt là `dayly_load_chart`)*

```python
def dayly_load_chart(self, images, size_rnd, space_chart, rc_chart, save=False, save_path="dataset/images"):
```

* **Mục đích**: Khảo sát mức tiêu thụ điện của các khách hàng ngẫu nhiên xuyên suốt 4 năm.
* **Kỹ thuật xử lý**:
  1. `np.random.choice(...)`: Lấy mẫu ngẫu nhiên `size_rnd` khách hàng (bỏ qua cột `timestamp`).
  2. `self.df.set_index("timestamp").resample("5D").mean()`: **Resample** dữ liệu thô (15 phút) thành trung bình mỗi 5 ngày. Kỹ thuật này giúp loại bỏ nhiễu tần số cao và giảm số điểm vẽ từ ~140,000 xuống còn ~292 điểm, giúp đồ thị mượt mà và thấy rõ xu hướng dài hạn.
  3. `plt.subplots(rc_chart[0], rc_chart[1])`: Tạo lưới đồ thị con (mặc định 3 hàng × 3 cột = 9 ô/trang).
  4. Lặp qua từng trang (`images`) để lưu thành các file: `users_page_1.png`, `users_page_2.png`, `users_page_3.png`.
* **Ý nghĩa thực tế**:
  * Rất nhiều khách hàng có giá trị 0.0 kéo dài trong suốt 2011-2012 và chỉ bắt đầu có dữ liệu từ 2013-2014.
  * $\rightarrow$ **Quyết định cho pipeline**: Trong [`build_hourly_dataset.py`](../backend/app/utils/build_hourly_dataset.py), nhóm quyết định chỉ lấy dữ liệu năm **2014** (năm có dữ liệu đầy đủ và ổn định nhất).

---

### 4.2 Hàm 2: `plot_aggregate_load_chart()` — Tổng tải 24h trong ngày

```python
def plot_aggregate_load_chart(self, size_rnd=9, rc_chart=(3, 3), save=False, save_path="dataset/images"):
```

* **Mục đích**: Khám phá **đường phụ tải ngày (daily load profile)** của toàn bộ 370 khách hàng gộp lại.
* **Kỹ thuật xử lý**:
  1. `df["aggregate"] = df.iloc[:, 1:].sum(axis=1)`: Tính tổng mức tiêu thụ của toàn bộ 370 khách hàng tại từng thời điểm.
  2. `df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60`: Chuyển đổi giờ và phút thành một số thực liên tục từ 0.0 đến 23.75 để làm trục hoành (trục X).
  3. Chọn ngẫu nhiên `size_rnd=9` ngày bất kỳ trong lịch sử để vẽ vào 9 subplot.
* **Ý nghĩa thực tế**:
  * Các ngày đều tuân theo quy luật sinh hoạt:
    * **0h - 6h**: Mức tiêu thụ thấp nhất (thời gian ngủ / off-peak).
    * **6h - 12h**: Tải tăng nhanh khi bắt đầu giờ làm việc và sinh hoạt.
    * **12h - 18h**: Đạt đỉnh cao điểm (peak load).
    * **20h - 24h**: Giảm dần.
  * $\rightarrow$ **Quyết định cho pipeline**: Mã hóa đặc trưng giờ bằng hàm lượng giác tuần hoàn: `hour_sin = sin(2*pi*hour/24)` và `hour_cos = cos(2*pi*hour/24)` trong [`preprocessor.py`](../backend/app/utils/preprocessor.py).

---

### 4.3 Hàm 3: `plot_time_pattern_heatmap()` — Heatmap Giờ × Ngày trong tuần

```python
def plot_time_pattern_heatmap(self, figsize=(15, 8), save=False, save_path="dataset/images"):
```

* **Mục đích**: Nhận diện tương quan phụ tải đồng thời giữa **Giờ trong ngày (0–23h)** và **Ngày trong tuần (Thứ 2 – Chủ Nhật)**.
* **Kỹ thuật xử lý**:
  1. Trích xuất đặc trưng thời gian: `hour`, `day_name`, `day_of_week`.
  2. `pivot_df = df.groupby(["day_name", "day_of_week", "hour"])["aggregate"].mean()`: Tính trung bình tiêu thụ điện của từng ô (ví dụ: trung bình lúc 14h chiều thứ Ba).
  3. `heatmap_data = pivot_df.pivot(...)`: Chuyển bảng dạng dài (long) thành ma trận kích thước 7 hàng × 24 cột (wide).
  4. Sắp xếp thứ tự các hàng theo đúng tuần từ `Monday` đến `Sunday`.
  5. Vẽ heatmap bằng `sns.heatmap(..., cmap="RdBu_r")`: Bảng màu tương phản Đỏ (cao) - Xanh (thấp).
* **Ý nghĩa thực tế**:
  * Các ngày làm việc (Thứ 2 đến Thứ 6) có cường độ tiêu thụ ban ngày rất cao (ô đỏ sẫm).
  * Thứ 7 và Chủ Nhật (cuối tuần) mức tiêu thụ giảm rõ rệt (màu nhạt và xanh hơn).
  * $\rightarrow$ **Quyết định cho pipeline**: Tạo đặc trưng `day_of_week` (0 đến 6) và đặc trưng phân biệt ngày làm việc / ngày nghỉ (`is_weekend`).

---

### 4.4 Hàm 4: `plot_histogram()` — Phân phối tổng tải & kiểm tra độ lệch

```python
def plot_histogram(self, figsize=(12, 6), save=False, save_path="dataset/images"):
```

* **Mục đích**: Kiểm tra dạng phân phối xác suất của biến phụ tải tổng, phát hiện hiện tượng lệch (skewness) và outlier.
* **Kỹ thuật xử lý**:
  1. `sns.histplot(data=df, x="aggregate", bins=50, kde=True)`: Biểu đồ cột tần suất 50 khoảng kèm đường mật độ liên tục (KDE).
  2. `mean_val = df["aggregate"].mean()`, `median_val = df["aggregate"].median()`.
  3. `ax.axvline(...)`: Vẽ hai đường chỉ báo trục đứng:
     * **Màu đỏ nét đứt**: Giá trị trung bình (Mean).
     * **Màu xanh nét đứt**: Giá trị trung vị (Median).
* **Ý nghĩa thực tế**:
  * Phân phối có **Mean > Median** và đuôi kéo dài sang phải $\rightarrow$ Dữ liệu bị **lệch phải nghiêm trọng (right-skewed)**.
  * Các mô hình hồi quy (LightGBM / Ridge) khi huấn luyện trên dữ liệu lệch phải sẽ dễ bị chi phối quá mức bởi các giá trị cực lớn.
  * $\rightarrow$ **Quyết định cho pipeline**: Áp dụng phép biến đổi logarit $y_{\text{train}} = \log(1 + y)$ (`np.log1p`) trước khi fit mô hình trong [`preprocessor.py`](../backend/app/utils/preprocessor.py).

---

### 4.5 Hàm 5: `plot_random_users_boxplot()` — So sánh biến thiên giữa các user

```python
def plot_random_users_boxplot(self, size_rnd=9, figsize=(14, 8), save=False, save_path="dataset/images", show=True):
```

* **Mục đích**: So sánh thang đo tiêu thụ điện và biên độ dao động (IQR) giữa các khách hàng khác nhau.
* **Kỹ thuật xử lý**:
  1. Lấy dữ liệu ngẫu nhiên 9 khách hàng qua hàm trợ giúp `get_random_users_data()`.
  2. `pd.melt(...)`: Chuyển bảng dạng WIDE sang LONG để `sns.boxplot()` vẽ nhiều nhóm cạnh nhau.
  3. Cấu trúc hộp boxplot:
     * Đáy hộp: Phân vị 25% (Q1).
     * Vạch giữa: Trung vị (Median / Q2).
     * Đỉnh hộp: Phân vị 75% (Q3).
     * Điểm ngoài râu (ngoài $1.5 \times \text{IQR}$): Các điểm ngoại lai (Outliers).
* **Ý nghĩa thực tế**:
  * Sự phân hóa rất lớn giữa các khách hàng: Hộ dân chỉ tiêu thụ vài chục kWh, trong khi cơ sở sản xuất tiêu thụ hàng nghìn kWh.
  * $\rightarrow$ **Quyết định cho pipeline**: Bước [`data_checker.py`](../backend/app/utils/data_checker.py) dùng ngưỡng quantile 0.95 để lọc bỏ bớt các siêu người dùng (heavy consumers) nhằm giúp mô hình học mẫu số chung ổn định hơn.

---

### 4.6 Hàm 6: `plot_random_users_acf()` — Kiểm tra chu kỳ tuần (Weekly Seasonality)

```python
def plot_random_users_acf(self, size_rnd=9, figsize=(16, 12), rc_chart=(3, 3), save=False, save_path="dataset/images", show=True, resample_daily=True):
```

* **Mục đích**: Phân tích hàm tự tương quan **ACF (Autocorrelation Function)** để xác định độ trễ (lags) nào có liên kết mạnh nhất với hiện tại.
* **Kỹ thuật xử lý**:
  1. `resample_daily=True`: Gộp dữ liệu thành tổng theo ngày (`resample("D").sum()`) trước khi tính ACF để loại bỏ nhiễu tần số cao.
  2. `autocorrelation_plot(df_plot[user])`: Hàm chuẩn của `pandas.plotting` tính tương quan giữa $x_t$ và $x_{t - k}$ với các độ trễ $k$.
  3. Dải nằm ngang nét đứt thể hiện khoảng tin cậy 95%. Hệ số ACF vượt ngoài khoảng này chứng tỏ tương quan có ý nghĩa thống kê.
* **Ý nghĩa thực tế**:
  * Đồ thị ACF có **các đỉnh cực đại lặp lại định kỳ tại các mốc bội số của 7 ngày** (Lag = 7, 14, 21, 28 ngày).
  * $\rightarrow$ **Quyết định quan trọng cho pipeline**: Chứng minh sự tồn tại của chu kỳ tuần (Weekly Seasonality). Khi làm việc ở dữ liệu hourly, bắt buộc phải tạo:
    * `lag_24`: Cùng giờ ngày hôm trước (24 giờ).
    * `lag_48`: Cùng giờ 2 ngày trước (48 giờ).
    * `lag_168`: Cùng giờ tuần trước ($7 \times 24 = 168$ giờ).

---

### 4.7 Hàm 7: `plot_monthly_boxplot()` — Mùa vụ theo 12 tháng

```python
def plot_monthly_boxplot(self, size_rnd=9, figsize=(16, 12), rc_chart=(3, 3), save=False, save_path="dataset/images", show=True):
```

* **Mục đích**: Đánh giá biến thiên phụ tải qua 12 tháng trong năm (Monthly Seasonality).
* **Kỹ thuật xử lý**:
  1. Gộp phụ tải theo ngày (`resample("D").sum()`).
  2. Trích xuất tháng `df_daily["month"] = df_daily["timestamp"].dt.month`.
  3. Vẽ boxplot phân bố lượng điện của 12 tháng cho từng user.
* **Ý nghĩa thực tế**:
  * Phụ tải tăng mạnh vào mùa đông (sưởi ấm) và mùa hè (điều hòa) ở nhiều khách hàng.
  * $\rightarrow$ **Quyết định cho pipeline**: Bổ sung đặc trưng `month` vào tập feature.

---

### 4.8 Hàm 8: `plot_monthly_acf()` — Kiểm tra tương quan theo tháng

```python
def plot_monthly_acf(self, size_rnd=9, figsize=(16, 12), rc_chart=(3, 3), save=False, save_path="dataset/images", show=True):
```

* **Mục đích**: Khảo sát tương quan tự thân dài hạn ở cấp độ tháng để tìm chu kỳ năm (12 tháng).
* **Lưu ý**: Tập dữ liệu có 4 năm (2011-2014) $\rightarrow$ chỉ có 48 điểm tháng. Cỡ mẫu nhỏ nên kết quả đồ thị này chỉ mang tính tham khảo định tính, không đủ dữ liệu tin cậy để tạo lag theo năm.

---

## 5. Giải thích chi tiết mã nguồn script `run_eda.py`

File nằm tại: [`backend/run_eda.py`](../backend/run_eda.py).

### Tại sao lại cần script `run_eda.py` độc lập?

1. **Khử hoàn toàn phụ thuộc vào giao diện hiển thị (`matplotlib.use("Agg")`)**:
   * Khi chạy trên terminal, máy chủ không màn hình (headless) hoặc Docker, Matplotlib mặc định sẽ tìm kiếm cửa sổ hiển thị đồ họa (GUI qua Tkinter/Qt). Nếu không có, lệnh vẽ sẽ báo lỗi hoặc bị treo vô hạn chờ người dùng đóng cửa sổ.
   * `matplotlib.use("Agg")` ép Matplotlib chỉ render ảnh ra bộ nhớ và lưu thành file PNG tĩnh, an toàn 100% trong mọi môi trường.
2. **Kịch bản thực thi một chạm (One-click execution)**:
   * Script gọi tuần tự toàn bộ 8 hàm, in tiến trình rõ ràng từ `[1/8]` đến `[8/8]` và tự động lưu trọn bộ 10 file ảnh vào `backend/dataset/images/`.
3. **Quản lý đường dẫn tập trung**:
   * Toàn bộ ảnh được quản lý qua biến `SAVE_PATH = "dataset/images"`, đảm bảo tính nhất quán khi chạy từ thư mục `backend/`.

---

## 6. Bảng ánh xạ từ EDA sang Tiền xử lý & Xây dựng mô hình

| Biểu đồ EDA | Hiện tượng quan sát | Quyết định trong Pipeline | File code tương ứng |
| :--- | :--- | :--- | :--- |
| **1. Daily Load Chart** | Giai đoạn 2011-2012 nhiều user chưa dùng điện (toàn số 0). | Chỉ lấy dữ liệu năm **2014** để dữ liệu liên tục. | [`build_hourly_dataset.py`](../backend/app/utils/build_hourly_dataset.py) |
| **2. Aggregate Load 24h** | Phụ tải biến thiên tuần hoàn theo 24 giờ trong ngày. | Tạo đặc trưng vòng lặp góc: `hour_sin`, `hour_cos`. | [`preprocessor.py`](../backend/app/utils/preprocessor.py) |
| **3. Time-Pattern Heatmap** | Ngày trong tuần có mức tiêu thụ cao hơn hẳn thứ 7 & CN. | Tạo biến phân loại ngày: `day_of_week`. | [`preprocessor.py`](../backend/app/utils/preprocessor.py) |
| **4. Histogram** | Phân phối điện năng lệch phải cực mạnh. | Biến đổi hàm mục tiêu: $y = \log(1 + y)$ (`log1p`). | [`preprocessor.py`](../backend/app/utils/preprocessor.py) |
| **5. Users Boxplot** | Một số user tiêu thụ gấp hàng chục lần mức trung bình. | Lọc bỏ khách hàng vượt phân vị 95% (`upper_quantile=0.95`). | [`data_checker.py`](../backend/app/utils/data_checker.py) |
| **6. Users ACF** | Hệ số tự tương quan tăng vọt tại các chu kỳ 7 ngày. | Tạo biến trễ: `lag_24`, `lag_48`, `lag_72`, và `lag_168` (7 ngày). | [`preprocessor.py`](../backend/app/utils/preprocessor.py) |
| **7. Monthly Boxplot** | Phụ tải khác nhau giữa các tháng hè / đông. | Bổ sung đặc trưng thời gian: `month`. | [`preprocessor.py`](../backend/app/utils/preprocessor.py) |

---

## 7. Hướng dẫn chạy lệnh cho cả nhóm

### Bước 1: Chuẩn bị môi trường & dữ liệu
1. Đảm bảo đã tải file dataset gốc `LD2011_2014.txt` (~680MB) và đặt vào:
   ```text
   backend/dataset/LD2011_2014.txt
   ```
   *(File này nằm trong `.gitignore` nên mỗi người cần tải riêng về máy cá nhân).*

### Bước 2: Chạy sinh toàn bộ ảnh EDA

Mở terminal tại thư mục gốc của dự án và chạy:

```bash
# 1. Đi vào thư mục backend
cd backend

# 2. Chạy script EDA tự động
python run_eda.py
```

*Hoặc chạy trực tiếp module:*
```bash
python -m app.utils.visualizations
```

> **Thời gian chạy**: Khoảng 1 - 2 phút (tùy theo tốc độ đọc ổ cứng đối với file 700MB).

### Bước 3: Xem các ảnh kết quả

Toàn bộ 10 file ảnh sẽ được lưu tại thư mục: `backend/dataset/images/`
* `users_page_1.png`, `users_page_2.png`, `users_page_3.png` (Line plot từng khách hàng)
* `aggregate_random_days.png` (Tổng tải 24h)
* `time_pattern_heatmap_original.png` (Heatmap giờ × ngày)
* `aggregate_histogram.png` (Histogram & KDE)
* `users_boxplot_random.png` (Boxplot users)
* `users_acf_random.png` (ACF kiểm tra chu kỳ tuần)
* `monthly_boxplot.png` (Boxplot theo tháng)
* `monthly_acf.png` (ACF theo tháng)

**Lệnh mở nhanh thư mục ảnh:**
* **Windows**: `explorer backend\dataset\images` (hoặc `explorer dataset\images` nếu đang ở thư mục `backend`)
* **macOS**: `open backend/dataset/images`
* **Linux**: `xdg-open backend/dataset/images`
