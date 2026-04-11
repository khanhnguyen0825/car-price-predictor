# 🚗 Vietnam Car Price Predictor

Machine Learning-powered car price prediction system using real Vietnam market data. Achieved **R² = 0.8618** with Random Forest model.

## 🎯 Features

- **Accurate Predictions**: R² score of 86.18% with MAE of only 87M VNĐ
- **Multiple ML Models**: Linear Regression, Ridge, Random Forest, Gradient Boosting
- **Interactive Web UI**: Beautiful Streamlit interface with 5 tabs
- **Real Vietnam Data**: 8,685 cars from Vietnamese market
- **Vietnamese Language Support**: Full UTF-8 encoding
- **Rich Visualizations**: 8+ Plotly interactive charts
- **Correct Depreciation Logic**: Higher KM → Lower price ✅

## 📊 Model Performance

| Model | R² (Test) | MAE | RMSE |
|-------|-----------|-----|------|
| **Random Forest** ⭐ | **0.8618** | **87M VNĐ** | **160M VNĐ** |
| Gradient Boosting | 0.7372 | 149M VNĐ | 221M VNĐ |
| Ridge Regression | 0.1967 | 288M VNĐ | 386M VNĐ |
| Linear Regression | 0.1967 | 288M VNĐ | 386M VNĐ |

### Feature Importance
1. **Dòng Xe (Model)**: 44.79%
2. **Hãng Xe (Brand)**: 24.26%
3. **Tuổi Xe (Age)**: 10.84%
4. **KM_Negative**: 10.46%
5. **Năm SX (Year)**: 9.64%

> **Note**: `KM_Negative` uses inverted kilometers (`-km`) to ensure correct depreciation logic: higher kilometers = lower price.

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to project
cd car-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Usage

1. **Open browser** at `http://localhost:8501`
2. **Select car details** in sidebar:
   - Hãng xe (Brand): Toyota, Honda, etc.
   - Dòng xe (Model): Vios, Civic, etc.
   - Năm sản xuất (Year): 2000-2024
   - Số KM đã đi (Kilometers): 0-500,000
3. **Click "DỰ ĐOÁN GIÁ XE"** button
4. **View prediction** and analysis

## 📁 Project Structure

```
car-price-predictor/
├── app.py                  # Streamlit web application
├── config.py               # Configuration and constants
├── data_loader.py          # Data loading and preprocessing
├── model.py                # ML models
├── visualizations.py       # Plotly charts
├── requirements.txt        # Python dependencies
├── download_dataset.py     # Kaggle dataset downloader
├── .streamlit/
│   └── config.toml        # Streamlit theme
├── data/
│   └── data.csv           # Car price dataset (15K+ cars)
└── models/
    └── best_model.pkl     # Trained Random Forest model
```

## 📊 Dataset

- **Source**: Kaggle - Vietnam Car Prices
- **Raw Records**: 15,442 listings
- **After Cleaning**: 8,685 cars (valid KM data)
- **Features**: Brand, Model, Year, Kilometers, Age
- **Price Range**: 29M - 2,130M VNĐ
- **Top Brands**: Toyota, Mercedes, Hyundai, Kia, Ford

### Data Processing Pipeline

1. **Parse Vietnamese format**: "38,000 Km" → 38000
2. **Feature engineering**: Create `Age` and `KM_Negative`
3. **Label encoding**: Brand & Model → numeric codes
4. **Outlier removal**: Clean extreme values
5. **Train/test split**: 80/20

## 🎨 Web Application Tabs

1. **🚗 Dự Đoán**: Prediction results and similar cars
2. **📊 Biểu Đồ**: Price trends and brand comparisons
3. **📈 Hiệu Suất Model**: Performance metrics and residuals
4. **🗃️ Dữ Liệu**: Dataset explorer and download
5. **🎯 Features**: Feature importance analysis

## 🛠️ Technology Stack

- **ML Framework**: scikit-learn 1.5+
- **Web Framework**: Streamlit 1.40+
- **Visualization**: Plotly 5.24+
- **Data Processing**: Pandas, NumPy
- **Language**: Python 3.9+

## 🔧 Key Implementation Details

### Feature Engineering

**KM_Negative**: Ensures correct depreciation
```python
# In training
df['KM_Negative'] = -df['Kilometers']

# In prediction
km_negative = -selected_km
```

**Why negative?** High KM (200K) → `-200000` → lower feature value → lower predicted price ✅

### Model Selection

Random Forest chosen for:
- Best R² score (0.8618)
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance

## 💡 Model Insights

### Why 86% Accuracy?

Cars have **standardized features** that strongly predict price:
- **Brand + Model**: 69% combined importance
- **Age**: Natural depreciation indicator
- **Kilometers**: Usage/wear indicator

Unlike houses where location (often missing) accounts for 40-50% of value.

### Prediction Examples

| Brand | Model | Year | KM | Predicted Price |
|-------|-------|------|-----|-----------------|
| Toyota | Vios | 2020 | 30K | ~430M VNĐ |
| Toyota | Vios | 2020 | 100K | ~396M VNĐ |
| Honda | Civic | 2018 | 50K | ~520M VNĐ |

## 📝 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 🐛 Known Issues & Solutions

### Issue: KM doesn't affect price
**Solution**: Restart Streamlit after model retraining to clear `@st.cache_resource` cache

### Issue: Vietnamese characters not displaying
**Solution**: Ensure UTF-8 encoding in all files and terminal

## 📧 Contact

Built with ❤️ using Python & Machine Learning

---

**Last Updated**: 2026-02-01  
**Model Version**: Random Forest v2.0 (with KM_Negative fix)
