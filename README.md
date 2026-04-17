# 🚗 Car Price Predictor - Vietnam Market

Hệ thống dự báo giá xe ô tô cũ tại Việt Nam dựa trên Machine Learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](http://localhost:8501)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## 📖 Giới thiệu dự án
Dự án sử dụng mô hình **Random Forest** để phân tích dữ liệu từ hơn 9,000 giao dịch xe thực tế, giúp người dùng định giá nhanh một chiếc xe dựa trên Hãng, Dòng, Năm sản xuất và Số KM đã đi.

- **Độ chính xác (R²):** 86.54%
- **Sai số MAE:** ~113 Triệu VNĐ
- **Công nghệ:** Python, Scikit-learn, Streamlit, Plotly.

## 🚀 Hướng dẫn nhanh
1. **Cài đặt:** `pip install -r requirements.txt`
2. **Train AI:** `python train_model.py`
3. **Chạy Web:** `python -m streamlit run app.py`

## 📚 Tài liệu chi tiết
Để phục vụ cho buổi thuyết trình và báo cáo, vui lòng xem các tài liệu chuyên sâu trong thư mục `docs/`:

1.  **[Hướng dẫn Cài đặt](docs/INSTALLATION.md)**: Chi tiết cách setup.
2.  **[Slide Format & Kịch bản](docs/SLIDE_FORMAT_GUIDE.md)**: Gợi ý nội dung slide.
3.  **[Q&A - Câu hỏi Phản biện](docs/Q&A_PRESENTATION.md)**: 10 câu hỏi giảng viên thường hỏi.
4.  **[Chi tiết Thuật toán](docs/ALGORITHM_DETAILS.md)**: So sánh 6 mô hình AI.
5.  **[Phân tích luồng dự án](docs/PROJECT_FLOW.md)**: Kiến trúc hệ thống.

---
*Built for educational purposes and market analysis.*
**Lead Developer:** [Your Name]
