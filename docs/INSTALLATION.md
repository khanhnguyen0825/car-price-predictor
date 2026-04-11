# Hướng Dẫn Cài Đặt Car Price Predictor

## 📋 Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 4GB
- **Dung lượng**: ~100MB cho project + dependencies

---

## 🚀 Cài Đặt Nhanh (3 Bước)

### **Bước 1: Tải Source Code**

**Option A - Clone từ Git:**
```bash
git clone <repository-url>
cd car-price-predictor
```

**Option B - Tải ZIP:**
- Tải file ZIP từ repository
- Giải nén vào thư mục mong muốn
- Mở terminal/cmd tại thư mục đó

### **Bước 2: Cài Dependencies**

```bash
pip install -r requirements.txt
```

**Dependencies chính:**
- `streamlit` - Web framework
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Machine learning
- `plotly` - Interactive charts
- `scipy` - Statistical functions

### **Bước 3: Train Model**

```bash
python train_model.py
```

Quá trình này mất ~1-2 phút, chỉ chạy **1 lần duy nhất** hoặc khi muốn retrain.

---

## 🎯 Chạy Ứng Dụng

```bash
python -m streamlit run app.py
```

App sẽ mở tại: **http://localhost:8501**

---

## 📂 Cấu Trúc Project

```
car-price-predictor/
├── data/
│   └── data.csv              # Dataset xe
├── models/
│   └── best_model.pkl        # Pre-trained model (tự động tạo)
├── docs/                     # Documentation
├── app.py                    # Streamlit web app
├── model.py                  # ML models
├── data_loader.py            # Data preprocessing
├── visualizations.py         # Plotly charts
├── train_model.py            # Training script
├── config.py                 # Configurations
└── requirements.txt          # Dependencies
```

---

## 🔧 Troubleshooting

### **Lỗi: Module not found**
```bash
pip install -r requirements.txt
```

### **Lỗi: File không tồn tại**
Đảm bảo file `data/data.csv` có trong project.

### **App chạy chậm**
Chạy `python train_model.py` trước để tạo pre-trained model.

### **Clear cache Streamlit**
```bash
python -m streamlit cache clear
```

---

## 📝 Workflow Thông Thường

**Lần đầu setup:**
```bash
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

**Các lần sau:**
```bash
python -m streamlit run app.py
```

**Khi có data mới:**
```bash
python train_model.py          # Retrain model
python -m streamlit run app.py # Chạy app
```

---

## 💡 Tips

- **Model file**: `models/best_model.pkl` được tạo sau khi train, giúp app load nhanh
- **Port bận**: Đổi port với flag `--server.port 8502`
- **Auto-reload**: Streamlit tự reload khi sửa code
- **Data mới**: Thay file `data/data.csv` rồi chạy `train_model.py`

---

## 📞 Support

Nếu gặp vấn đề, check:
1. Python version: `python --version`
2. Pip version: `pip --version`
3. Dependencies: `pip list`
