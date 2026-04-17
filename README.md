# Vietnam Car Price Predictor

Hệ thống định giá ô tô cũ tại thị trường Việt Nam sử dụng Machine Learning (Random Forest). Dự án đạt độ chính xác **R² = 86.54%** và được triển khai dưới dạng Web Application (Streamlit).


## Tính năng nổi bật

- **Độ chính xác cao**: Chỉ số R² đạt 86.54%, MAE (Sai số trung bình) ~112 triệu VNĐ.
- **Xử lý dữ liệu thông minh**: Kỹ thuật `KM_Negative` giúp mô hình hiểu đúng quy luật hao mòn: Đi càng nhiều - Giá càng giảm.
- **Tương tác thời gian thực**: Dự báo ngay lập tức sau khi nhập thông số xe.
- **Thư viện AI mạnh mẽ**: So sánh 6 thuật toán và tự động chọn Random Forest là mô hình tối ưu nhất.
- **Hỗ trợ tiếng Việt**: Xử lý triệt để lỗi font tiếng Việt trong tên xe và định dạng tiền tệ "Tỷ", "Triệu".

## Hiệu suất mô hình

| Thuật toán | R² (Test) | MAE | Verdict |
| :--- | :--- | :--- | :--- |
| **Random Forest** ⭐ | **86.54%** | **112.9M VNĐ** | 🏆 **BEST** |
| Gradient Boosting | 71.70% | 203.7M VNĐ | Good |
| Lasso Regression | 10.65% | 403.5M VNĐ | Poor (Underfit) |
| SVR (RBF Kernel) | -4.61% | 397.1M VNĐ | Fail (Needs Scaling) |

### Tầm quan trọng của các yếu tố (Feature Importance)
1. **Dòng Xe (Model)**: 40.68% (Yếu tố quan trọng nhất)
2. **Hãng Xe (Brand)**: 29.11%
3. **Số KM (Kilometers)**: 14.41%
4. **Năm SX (Year)**: 7.91%
5. **Tuổi Xe (Age)**: 7.89%

## Cài đặt & Chạy ứng dụng

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình (Chỉ chạy 1 lần đầu)
```bash
python train_model.py
```

### 3. Khởi chạy Web App
```bash
python -m streamlit run app.py
```

## Cấu trúc dự án

```
car-price-predictor/
├── app.py                  # Giao diện Web (Streamlit)
├── model.py                # Định nghĩa các mô hình AI
├── data_loader.py          # Tiền xử lý dữ liệu & Feature Engineering
├── visualizations.py       # Module vẽ 16 biểu đồ Plotly
├── data/
│   └── data.csv           # Tập dữ liệu 15,000+ xe
├── docs/                   # Tài liệu hướng dẫn & Phân tích
│   ├── Q&A_PRESENTATION.md # 10 câu hỏi phản biện
│   ├── INSTALLATION.md     # Hướng dẫn cài đặt chi tiết
│   └── ALGORITHM_DETAILS.md # Chi tiết 6 thuật toán
└── models/
    └── best_model.pkl      # Bộ não AI đã được huấn luyện
```
