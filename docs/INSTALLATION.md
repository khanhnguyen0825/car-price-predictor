# 🛠️ Hướng Dẫn Cài Đặt Hệ Thống

Tài liệu này hướng dẫn bạn cách thiết lập môi trường và vận hành dự án **Car Price Predictor** từ lúc tải code cho đến khi chạy Web thành công.

---

## 📋 1. Yêu Cầu Hệ Thống
- **Hệ điều hành**: Windows 10/11, macOS hoặc Linux.
- **Python**: Phiên bản **3.9** trở lên (Khuyến nghị dùng 3.12+).
- **Phần mềm hỗ trợ**: VS Code, PyCharm hoặc bất kỳ trình soạn thảo mã nguồn nào.

---

## 🚀 2. Quy Trình Cài Đặt (3 Bước)

### **Bước 1: Tải mã nguồn & Môi trường**
Mở Terminal hoặc Command Prompt tại thư mục dự án và cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

*Các thư viện chính sẽ được cài bao gồm: Streamlit, Scikit-learn, Pandas, Plotly.*

### **Bước 2: Huấn luyện bộ não AI (Training)**
Dự án được thiết kế để tách biệt phần huấn luyện. Bạn cần chạy lệnh này để máy quét qua tập dữ liệu 15,000 xe và tạo ra file mô hình:

```bash
python train_model.py
```

*Kết quả: File `models/best_model.pkl` sẽ được tạo ra. Nếu file này đã có sẵn, bạn có thể bỏ qua bước này.*

### **Bước 3: Khởi chạy Giao diện Web**
Sử dụng lệnh Streamlit để bật ứng dụng lên trình duyệt:

```bash
python -m streamlit run app.py
```

*Sau khi chạy, ứng dụng sẽ mở tại địa chỉ: **http://localhost:8501***

---

## 📂 3. Cấu Trúc Thư Mục Quan Trọng

- `app.py`: Trung tâm điều khiển giao diện người dùng.
- `data_loader.py`: Nơi chứa toàn bộ logic làm sạch dữ liệu (như IQR, Regex).
- `model.py`: Chứa các thuật toán AI và logic huấn luyện.
- `visualizations.py`: Nơi định nghĩa các biểu đồ Plotly.
- `data/data.csv`: "Thức ăn" cho AI - Tập dữ liệu gốc.
- `models/best_model.pkl`: "Bộ não" AI sau khi đã học xong.

---
