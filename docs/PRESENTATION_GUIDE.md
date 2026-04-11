# 🎓 Hướng Dẫn Trình Bày Project Trước Giảng Viên

## Mục Lục
1. [Kiến Thức Cần Nắm Vững](#phần-1-kiến-thức-cần-nắm-vững)
2. [Dàn Ý Trình Bày](#phần-2-dàn-ý-trình-bày-15-20-phút)
3. [Câu Hỏi Giảng Viên Hay Hỏi](#phần-3-câu-hỏi-giảng-viên-hay-hỏi--cách-trả-lời)
4. [Thuật Ngữ Cheat Sheet](#phần-4-thuật-ngữ-cheat-sheet)

---

## Phần 1: Kiến Thức Cần Nắm Vững

### 1.1 Machine Learning là gì? (PHẢI HIỂU)

> **Giải thích đơn giản:** Cho máy tính học từ dữ liệu có sẵn, rồi dùng kiến thức đó để dự đoán dữ liệu mới.

**Ví dụ trong project:**
- Mình có **8,685 xe** đã biết giá (dữ liệu có sẵn)
- Máy học pattern: "Toyota Vios 2020 thường có giá ~400 triệu"
- Khi nhập 1 xe mới → máy dự đoán giá

**Phân loại:**
- **Supervised Learning** (Học có giám sát) ← Project dùng cái này
  - Có input (thông tin xe) VÀ output (giá xe) để học
- **Unsupervised Learning**: Không có output, máy tự tìm pattern
- **Regression**: Dự đoán **số liên tục** (giá = 420.5 triệu) ← Project dùng cái này
- **Classification**: Phân **loại** (spam / không spam)

---

### 1.2 Thư Viện Được Sử Dụng (PHẢI BIẾT TÊN + CHỨC NĂNG)

| Thư viện | Chức năng | Dùng ở đâu trong project |
|----------|-----------|--------------------------|
| **scikit-learn** | Thư viện ML chính của Python, cung cấp các thuật toán ML | `model.py` — train 6 models |
| **pandas** | Xử lý dữ liệu dạng bảng (DataFrame) | `data_loader.py` — đọc CSV, clean data |
| **numpy** | Tính toán số học (mảng, ma trận) | Tính metrics, xử lý số |
| **streamlit** | Tạo web app từ Python (không cần HTML/JS) | `app.py` — giao diện web |
| **plotly** | Tạo biểu đồ interactive (hover, zoom) | `visualizations.py` — 19 charts |
| **matplotlib/seaborn** | Tạo biểu đồ tĩnh (backup) | Có trong requirements |

**Câu để nói với giảng viên:**
> "Em sử dụng scikit-learn làm framework Machine Learning chính, pandas để xử lý dữ liệu, Streamlit để deploy thành web application, và Plotly để tạo biểu đồ phân tích interactive."

---

### 1.3 Quy Trình Machine Learning (PHẢI THUỘC)

Đây là luồng xử lý chính của project, giảng viên **chắc chắn sẽ hỏi**.
Mỗi bước bên dưới chỉ rõ **file nào, hàm nào, dòng nào** để bạn có thể mở code chỉ trực tiếp.

---

#### Bước 1: Thu thập dữ liệu

| | Chi tiết |
|---|---|
| **File** | `data/data.csv` (5 MB) |
| **Nguồn** | Kaggle — Vietnam Car Prices |
| **Quy mô** | 15,442 bản ghi gốc |
| **Các cột gốc** | Name, Price, Năm sản xuất, Số Km đã đi, Nhiên liệu, Hộp số... (tiếng Việt) |

---

#### Bước 2: Load & Tiền xử lý dữ liệu (Data Preprocessing)

Toàn bộ nằm trong **`data_loader.py`**, class `CarPriceDataLoader`.
Pipeline chạy tuần tự qua hàm `get_full_pipeline()` (dòng 380-414):

| Sub-step | Hàm | Dòng | Làm gì |
|----------|------|------|--------|
| 2a. Load CSV | `load_data()` | 28-51 | Đọc file CSV, thử 3 encoding (utf-8, utf-8-sig, latin-1) |
| 2b. Chuẩn hóa tên cột | `clean_column_names()` | 53-111 | Đổi tên cột tiếng Việt → tiếng Anh (VD: `'Năm sản xuất:'` → `'Year'`). Xử lý cả trường hợp cột bị garbled encoding |
| 2c. Trích xuất Brand/Model | `extract_brand_model()` | 114-138 | Tách tên xe: `"Toyota Vios 2020"` → Brand=`"Toyota"`, Model=`"Vios"` |
| 2d. Parse giá | `clean_price()` | 140-204 | Chuyển `"1 Tỷ 200 Triệu"` → `1200` (triệu). Hàm con `parse_price()` xử lý cả format garbled |
| 2e. Parse KM, Year, Seats | `clean_numeric_features()` | 206-258 | Chuyển `"38,000 Km"` → `38000`. Loại data ngoài range hợp lệ |
| 2f. Loại outliers | `clean_data()` | 283-321 | Dùng **IQR × 3** loại giá cực đoan. Xóa duplicate và missing rows. Kết quả: **8,685 xe** |

**Code mấu chốt — parse giá tiếng Việt** (`data_loader.py` dòng 147-191):
```python
def parse_price(price_str):
    # "1 Tỷ 200 Triệu" → tìm Tỷ: 1×1000=1000, tìm Triệu: 200 → total = 1200
    ty_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:[Tt]ỷ|...)', price_str)
    trieu_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:[Tt]riệu|...)', price_str)
```

---

#### Bước 3: Feature Engineering (Tạo đặc trưng)

Vẫn trong **`data_loader.py`**:

| Sub-step | Hàm | Dòng | Làm gì |
|----------|------|------|--------|
| 3a. Tạo Age | `engineer_features()` | 260-281 | `Age = 2026 - Year` (tuổi xe) |
| 3b. Tạo KM_Negative | `engineer_features()` | 278 | `KM_Negative = -Kilometers` (đảo dấu để khấu hao đúng) |
| 3c. Label Encoding | `encode_categorical_features()` | 323-345 | `"Toyota"` → `45`, `"Vios"` → `123` dùng `sklearn.LabelEncoder` |
| 3d. Chọn features | `prepare_features()` | 347-378 | Chọn 5 features cuối cùng: `[Year, Age, KM_Negative, Brand_Encoded, Model_Encoded]` |

**Code mấu chốt — KM_Negative** (`data_loader.py` dòng 275-279):
```python
# KM âm để có tương quan ĐÚNG
# KM cao → Giá trị âm hơn → Giá thấp hơn
self.df['KM_Negative'] = -self.df['Kilometers']
```

**Code mấu chốt — Label Encoding** (`data_loader.py` dòng 338-342):
```python
le = LabelEncoder()
col_data = self.df[[col]].iloc[:, 0].astype(str)
self.df[f'{col}_Encoded'] = le.fit_transform(col_data)
```

---

#### Bước 4: Chia dữ liệu (Train/Test Split)

| | Chi tiết |
|---|---|
| **File** | `model.py`, hàm `train()` |
| **Dòng** | 86-90 |
| **Tỷ lệ** | 80% Training (6,948 xe), 20% Testing (1,737 xe) |
| **Config** | `test_size=0.2`, `random_state=42` (từ `config.py` dòng 15-16) |

**Code chính** (`model.py` dòng 88-90):
```python
self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
)
```

---

#### Bước 5: Huấn luyện 6 Models

| | Chi tiết |
|---|---|
| **File** | `model.py`, hàm `train()` |
| **Dòng** | 96-165 |
| **Class** | `CarPricePredictor` |

**6 models được khởi tạo** (`model.py` dòng 96-110):

| Model | Code | Dòng |
|-------|------|------|
| Linear Regression | `LinearRegression()` | 97 |
| Ridge Regression | `Ridge(alpha=1.0)` | 98 |
| Lasso Regression | `Lasso(alpha=1.0)` | 99 |
| SVR | `SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)` | 100 |
| **Random Forest** ⭐ | `RandomForestRegressor(n_estimators=100, n_jobs=-1)` | 101-105 |
| Gradient Boosting | `GradientBoostingRegressor(n_estimators=100)` | 106-109 |

**Vòng lặp train từng model** (`model.py` dòng 115-155):
```python
for model_key in model_types:       # Duyệt qua 6 models
    model.fit(self.X_train, self.y_train)     # Train   (dòng 123)
    y_pred_test = model.predict(self.X_test)  # Predict (dòng 127)
    r2_test = r2_score(self.y_test, y_pred_test)  # Đánh giá (dòng 131)
    if r2_test > self.best_score:             # Chọn best (dòng 150)
        self.best_model = model
```

---

#### Bước 6: Đánh giá & Chọn model tốt nhất

| | Chi tiết |
|---|---|
| **File** | `model.py` |
| **Tính metrics** | Hàm `get_metrics()` (dòng 195-222) |
| **So sánh models** | Hàm `compare_all_models()` (dòng 280-298) |
| **Feature importance** | Hàm `get_feature_importance()` (dòng 224-261) |

**3 metrics được tính** (`model.py` dòng 129-133):
```python
r2_test  = r2_score(self.y_test, y_pred_test)           # R² Score
mae_test = mean_absolute_error(self.y_test, y_pred_test) # MAE
rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))  # RMSE
```

**Kết quả: Random Forest thắng** (`model.py` dòng 150-153):
```python
if r2_test > self.best_score:
    self.best_score = r2_test          # 0.8618
    self.best_model = model            # RandomForestRegressor
    self.best_model_name = model_name  # "Random Forest"
```

---

#### Bước 7: Lưu model & Deploy

Chia thành 2 phần:

**7a. Lưu model vào file pickle (chạy 1 lần)**

| | Chi tiết |
|---|---|
| **File** | `train_model.py`, hàm `main()` |
| **Dòng** | 38-53 |
| **Output** | `models/best_model.pkl` (57 MB) |

```python
# train_model.py dòng 45-53
pickle.dump({
    'predictor': predictor,  # Toàn bộ 6 models đã train
    'df': df,                # Dataset đã clean
    'X': X, 'y': y,          # Features & target
    'features': features,    # Tên features
    'loader': loader         # Data loader
}, f)
```

**7b. Streamlit Web App (load model đã lưu)**

| | Chi tiết |
|---|---|
| **File** | `app.py` |
| **Load model** | Hàm `load_data_and_train_model()` (dòng 73-116), dùng `@st.cache_resource` |
| **Giao diện** | Hàm `main()` (dòng 120-658), gồm 5 tabs |
| **Biểu đồ** | `visualizations.py` — 19 hàm `create_*()` |
| **Config** | `config.py` — constants, settings |

**Load model từ pickle** (`app.py` dòng 88-102):
```python
if os.path.exists(model_path):          # Nếu có file .pkl
    data = pickle.load(f)               # Load (nhanh ~1 giây)
    predictor = data['predictor']       # Lấy model đã train
else:
    predictor.train(...)                # Train mới (chậm ~2 phút)
```

**Dự đoán giá xe** (`app.py` dòng 174-197):
```python
if st.sidebar.button("DỰ ĐOÁN GIÁ XE"):
    car_features = {
        'Year': selected_year,
        'Age': age,
        'KM_Negative': -selected_km,        # Đảo dấu KM
        'Brand_Encoded': brand_encoded,
        'Model_Encoded': model_encoded
    }
    predicted_price = predictor.predict(car_features)  # Gọi model predict
```

**5 Tabs giao diện** (`app.py`):

| Tab | Dòng | Nội dung |
|-----|------|----------|
| Tab 1 — Dự Đoán | 219-295 | Kết quả dự đoán + xe tương tự |
| Tab 2 — Biểu Đồ | 298-350 | 7 charts phân tích (scatter, box, line...) |
| Tab 3 — Hiệu Suất | 354-473 | Metrics, model comparison, residuals |
| Tab 4 — Dữ Liệu | 476-509 | Dataset explorer + download CSV |
| Tab 5 — Features | 512-654 | Feature importance + model giải thích |

---

#### Tóm tắt: Bản đồ file ↔ bước

```
data/data.csv             ← Bước 1: Dữ liệu gốc
        │
data_loader.py            ← Bước 2 + 3: Preprocessing + Feature Engineering
        │                    get_full_pipeline() gọi 9 hàm liên tiếp
        │
model.py                  ← Bước 4 + 5 + 6: Split → Train → Evaluate
        │                    class CarPricePredictor
        │
train_model.py            ← Bước 7a: Lưu model → models/best_model.pkl
        │
app.py                    ← Bước 7b: Load model + Web UI (5 tabs)
  ├── config.py            ← Constants, settings
  └── visualizations.py    ← 19 chart functions
```

---

### 1.4 Các Model — Giải Thích Đơn Giản (PHẢI HIỂU Ý TƯỞNG)

#### 🔵 Linear Regression — R² = 19.67%
- **Ý tưởng:** Vẽ một đường thẳng fit với data
- **Công thức:** `Giá = a×Year + b×Brand + c×KM + ...`
- **Tại sao kém:** Giá xe KHÔNG theo đường thẳng (Mercedes 2015 ≠ Toyota 2015)
- **Khi nào dùng:** Khi dữ liệu có quan hệ tuyến tính đơn giản

#### 🔵 Ridge Regression — R² = 19.67%
- **Ý tưởng:** Giống Linear + thêm **phạt** nếu hệ số quá lớn (L2 Regularization)
- **Mục đích:** Tránh overfitting
- **Tại sao kém:** Vẫn là đường thẳng, không giúp gì khi model quá đơn giản

#### 🔵 Lasso Regression — R² = 19.65%
- **Ý tưởng:** Giống Ridge nhưng dùng **L1 Regularization** → có thể loại bỏ features không quan trọng
- **Điểm khác Ridge:** Lasso có thể đưa hệ số về = 0 (feature selection)
- **Tại sao kém:** Cùng lý do — dữ liệu xe phi tuyến

#### 🔵 SVR (Support Vector Regression) — R² = 5.18%
- **Ý tưởng:** Tạo một "ống" (tube) bao quanh data, dùng kernel trick để xử lý phi tuyến
- **Tại sao kém nhất:** Chưa scale features (KM: -500k, Year: 2020 → scale rất khác)
- **Cách fix:** Cần `StandardScaler()` trước khi train

#### 🟢 Random Forest — R² = 86.18% ⭐ TỐT NHẤT
- **Ý tưởng:** Tạo **100 cây quyết định** (decision trees), mỗi cây vote, lấy trung bình
- **Tại sao tốt nhất:** 
  - Tự xử lý phi tuyến
  - Không cần scale features
  - Robust với outliers
  - Cho biết feature nào quan trọng nhất
- **Hyperparameters:** `n_estimators=100`, `n_jobs=-1` (dùng tất cả CPU)

#### 🟡 Gradient Boosting — R² = 73.72%
- **Ý tưởng:** Tạo cây **tuần tự**, mỗi cây sửa lỗi của cây trước
- **Khác RF:** RF = parallel (song song), GB = sequential (tuần tự)
- **Tại sao kém hơn RF:** Dễ overfit hơn, cần tuning nhiều hơn

---

### 1.5 Metrics Đánh Giá (PHẢI THUỘC CÔNG THỨC + Ý NGHĨA)

#### R² Score (Coefficient of Determination)
- **Công thức:** `R² = 1 - (Tổng sai số model / Tổng biến thiên thực tế)`
- **Ý nghĩa:** Model giải thích được **bao nhiêu %** sự biến thiên của giá xe
- **R² = 0.86** → Model giải thích 86% sự thay đổi giá
- **R² = 1.0** → Hoàn hảo | **R² = 0** → Tệ như đoán trung bình

#### MAE (Mean Absolute Error)
- **Công thức:** `MAE = Trung bình(|Giá thật - Giá dự đoán|)`
- **Ý nghĩa:** Trung bình mỗi lần dự đoán sai **bao nhiêu tiền**
- **MAE = 87 triệu** → Mỗi lần dự đoán sai trung bình 87 triệu

#### RMSE (Root Mean Squared Error)
- **Công thức:** `RMSE = √(Trung bình((Giá thật - Giá dự đoán)²))`
- **Ý nghĩa:** Giống MAE nhưng **phạt nặng hơn** các dự đoán sai nhiều
- **RMSE > MAE** → Có một số xe model dự đoán sai khá nhiều

---

### 1.6 Feature Engineering — Điểm Đặc Biệt Của Project (ĐIỂM CỘNG)

**Đây là phần giảng viên sẽ ấn tượng nếu bạn giải thích tốt:**

#### KM_Negative — Tại sao phải đảo dấu?
```
VẤN ĐỀ: KM cao → Giá thấp (nghịch đảo)
  Nhưng model học: KM tăng → feature value tăng → giá tăng (SAI!)

GIẢI PHÁP: KM_Negative = -KM
  KM cao (200k) → KM_Negative = -200000 → value nhỏ → giá thấp ✅
  KM thấp (30k) → KM_Negative = -30000 → value lớn hơn → giá cao ✅
```

#### Feature Importance — Yếu tố nào quyết định giá?
```
Model xe (Vios, Camry...): 44.79% ← QUAN TRỌNG NHẤT
Hãng xe (Toyota, Honda...): 24.26%
Tuổi xe:                    10.84%
Số KM:                      10.46%
Năm sản xuất:                9.64%
```
→ **Insight:** "Dòng xe" quyết định giá nhiều hơn "Hãng xe"!

---

## Phần 2: Dàn Ý Trình Bày (15-20 phút)

### Slide 1: Giới thiệu (1 phút)
> "Đồ án của em là hệ thống **Dự đoán giá xe ô tô Việt Nam** sử dụng Machine Learning. Hệ thống nhận đầu vào là thông tin xe (hãng, dòng, năm, số KM) và dự đoán giá bán với độ chính xác 86%."

### Slide 2: Bài toán & Dataset (2 phút)
- Bài toán: Supervised Learning → Regression
- Dataset: 15,442 xe từ Kaggle, sau khi clean còn 8,685 xe
- Features: Brand, Model, Year, Kilometers
- Target: Price (triệu VNĐ)

### Slide 3: Quy trình xử lý dữ liệu (3 phút)
- **Demo trực tiếp:** Mở `data_loader.py` giải thích pipeline
- Nhấn mạnh: Parse tiếng Việt ("1 Tỷ 200 Triệu" → 1200)
- Nhấn mạnh: KM_Negative trick (điểm sáng tạo)

### Slide 4: Models & So sánh (4 phút)
- Liệt kê 6 models đã thử
- **Bảng so sánh** R², MAE, RMSE
- Giải thích tại sao Random Forest tốt nhất
- Show Feature Importance chart

### Slide 5: Demo Web App (4 phút)
- **Chạy app live:** `python -m streamlit run app.py`
- Demo Tab 1: Dự đoán 1 xe cụ thể
- Demo Tab 2: Show biểu đồ phân tích
- Demo Tab 3: Show hiệu suất model
- Demo Tab 5: Show feature importance

### Slide 6: Kết luận (2 phút)
- Random Forest đạt R² = 86.18%, MAE = 87 triệu
- Ứng dụng thực tế: Người mua/bán xe tham khảo giá
- Hạn chế: Chưa có features như tình trạng xe, màu sắc, phụ kiện
- Hướng phát triển: Thêm features, hyperparameter tuning, deploy cloud

---

## Phần 3: Câu Hỏi Giảng Viên Hay Hỏi & Cách Trả Lời

### ❓ "Tại sao chọn Random Forest mà không dùng model khác?"

> "Em đã train 6 models khác nhau và so sánh trên test set. Random Forest đạt R² cao nhất (86%) vì dữ liệu giá xe có quan hệ **phi tuyến** phức tạp — ví dụ cùng năm 2020 nhưng Mercedes giá gấp 3 Toyota. Linear Regression chỉ vẽ được đường thẳng nên R² chỉ 19%. Random Forest tạo 100 cây quyết định, mỗi cây học một pattern khác nhau rồi vote kết quả, nên capture được các quan hệ phức tạp này."

### ❓ "Giải thích R² = 0.86 nghĩa là gì?"

> "R² = 0.86 nghĩa là model của em giải thích được **86%** sự biến thiên của giá xe. Còn 14% là do các yếu tố em chưa có trong dataset như tình trạng xe, có tai nạn không, có phụ kiện gì... R² = 1.0 là hoàn hảo, R² = 0 là tệ như đoán trung bình."

### ❓ "MAE = 87 triệu nghĩa là gì? Có chấp nhận được không?"

> "MAE = 87 triệu nghĩa là trung bình mỗi lần dự đoán sai khoảng 87 triệu VNĐ. So với giá xe trung bình khoảng 700 triệu, đó là sai số khoảng 12% — **chấp nhận được** cho mức tham khảo. Người mua xe có thể dùng giá dự đoán để biết xe đó đắt hay rẻ hơn thị trường."

### ❓ "Overfitting thì sao? Làm sao biết model không bị overfit?"

> "Em đã kiểm tra: R² Train = 96%, R² Test = 86%. Gap khoảng 10% — cho thấy có **slight overfit** nhưng chấp nhận được. Random Forest có cơ chế chống overfit tự nhiên: mỗi cây chỉ thấy một phần dữ liệu (bootstrap sampling) và chỉ xét một phần features mỗi lần split. Nếu muốn giảm overfit thêm, em có thể set `max_depth` hoặc dùng cross-validation."

### ❓ "Tại sao dùng KM_Negative thay vì để KM bình thường?"

> "Vì KM càng cao thì xe càng cũ → giá càng thấp. Nhưng nếu để KM dương, feature value tăng khi KM tăng, model có thể học sai là 'KM cao → giá cao'. Bằng cách đảo dấu, KM_Negative = -200000 cho xe đi 200K km, giá trị này nhỏ hơn → model học đúng là giá thấp hơn."

### ❓ "Tại sao SVR chỉ đạt 5%? Có cách fix không?"

> "SVR cần **feature scaling** (StandardScaler) vì nó tính khoảng cách giữa data points. Hiện tại Year có range 24, nhưng KM_Negative có range 500,000 — scale rất khác biệt nên SVR bị dominated bởi features lớn. Nếu thêm StandardScaler trước khi train SVR, kết quả sẽ cải thiện đáng kể."

### ❓ "Train/Test split 80/20 — có dùng Cross-Validation không?"

> "Hiện tại em dùng hold-out 80/20 với `random_state=42` để kết quả reproducible. Chưa dùng K-Fold Cross-Validation. Nếu muốn đánh giá chặt hơn, em có thể dùng `cross_val_score` với K=5 để chắc chắn hơn rằng R² = 86% không phải do may mắn khi chia data."

### ❓ "Label Encoding có vấn đề gì không? Tại sao không dùng One-Hot?"

> "Label Encoding gán số tùy ý cho các category, ví dụ Toyota=45, Honda=20 — model có thể nghĩ Toyota 'lớn hơn' Honda. Tuy nhiên Random Forest xử lý tốt vấn đề này vì nó split theo điều kiện `if Brand == 45` chứ không so sánh thứ tự. Nếu dùng One-Hot Encoding, sẽ có 100+ cột cho dòng xe → quá nhiều features. Với Random Forest + Label Encoding, kết quả đã rất tốt (86%)."

### ❓ "Hạn chế của project là gì?"

> "Có một số hạn chế: (1) Dataset chỉ có 8,685 xe, chưa đủ lớn; (2) Thiếu features quan trọng như tình trạng xe, lịch sử bảo dưỡng, có tai nạn không; (3) Label Encoding không tối ưu cho Linear models; (4) Chưa hyperparameter tuning sâu; (5) Giá xe thay đổi theo thời gian nhưng model train một lần."

---

## Phần 4: Thuật Ngữ Cheat Sheet

| Thuật ngữ | Tiếng Việt | Giải thích 1 câu |
|-----------|-----------|-------------------|
| **Machine Learning** | Học máy | Cho máy học pattern từ data |
| **Supervised Learning** | Học có giám sát | Có cả input lẫn output để học |
| **Regression** | Hồi quy | Dự đoán số liên tục (giá xe) |
| **Feature** | Đặc trưng | Input của model (Year, Brand, KM...) |
| **Target** | Biến mục tiêu | Output cần dự đoán (Price) |
| **Training Set** | Tập huấn luyện | 80% data để model học |
| **Test Set** | Tập kiểm tra | 20% data để đánh giá model |
| **Overfitting** | Quá khớp | Model "học thuộc lòng" train data, predict test kém |
| **Underfitting** | Chưa khớp | Model quá đơn giản, chưa học đủ pattern |
| **R² Score** | Hệ số xác định | % biến thiên được model giải thích (0→1) |
| **MAE** | Sai số tuyệt đối TB | Trung bình sai bao nhiêu tiền |
| **RMSE** | Căn bình phương sai số | Như MAE nhưng phạt nặng dự đoán sai nhiều |
| **Feature Engineering** | Tạo đặc trưng | Tạo features mới từ data thô |
| **Label Encoding** | Mã hóa nhãn | Chuyển text → số (Toyota → 45) |
| **Regularization** | Điều chuẩn | Phạt model quá phức tạp (Ridge/Lasso) |
| **Ensemble** | Kết hợp | Dùng nhiều models vote (RF, GB) |
| **Decision Tree** | Cây quyết định | Model chia data bằng if/else liên tục |
| **Bootstrap** | Lấy mẫu lặp | Lấy random samples từ data gốc |
| **Hyperparameter** | Siêu tham số | Setting model tự chọn (n_estimators=100) |
| **Cross-Validation** | Kiểm chứng chéo | Chia data nhiều lần để đánh giá chính xác |
| **Pickle** | Serialization | Lưu model đã train vào file để load lại |
| **Cache** | Bộ nhớ đệm | Streamlit lưu kết quả để không chạy lại |
| **Deploy** | Triển khai | Đưa model ra sử dụng (web app) |

---

## 💡 Tips Trình Bày

### ✅ NÊN:
1. **Demo live** — Chạy app trước mặt giảng viên, dự đoán 2-3 xe
2. **So sánh kết quả** — "Toyota Vios 2020 → model dự đoán 420 triệu, thực tế khoảng 400-450 triệu"
3. **Thể hiện hiểu biết** — Giải thích TẠI SAO chọn RF, không chỉ nói "vì R² cao"
4. **Nói hạn chế** — Giảng viên đánh giá cao sinh viên biết mình thiếu gì
5. **Show biểu đồ** — Feature Importance, Actual vs Predicted là 2 biểu đồ quan trọng nhất

### ❌ KHÔNG NÊN:
1. ~~Đọc code từ đầu đến cuối~~ → Chỉ giải thích ý tưởng chính
2. ~~Nói "em không hiểu phần này"~~ → Nắm vững ít nhất phần 1 ở trên
3. ~~Chỉ nói kết quả~~ → Phải giải thích **quy trình** đạt được kết quả
4. ~~Bỏ qua preprocessing~~ → Giảng viên thường đánh giá cao phần xử lý data
