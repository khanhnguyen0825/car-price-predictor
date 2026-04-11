"""
Configuration file for Car Price Predictor
Contains constants, settings, and feature descriptions
"""

import datetime

# App Settings
APP_TITLE = "Dự Đoán Giá Xe"
APP_SUBTITLE = "Machine Learning"
APP_ICON = ""


# Model Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# UI Settings - Car Theme (Orange/Blue)
PRIMARY_COLOR = "#FF6B35"
SECONDARY_COLOR = "#004E89"
BACKGROUND_COLOR = "#0f0c29"
ACCENT_COLOR = "#F77F00"

# Dataset Path
DATASET_PATH = "data/data.csv"

# Current year for age calculation
CURRENT_YEAR = datetime.datetime.now().year

# Feature Descriptions (Vietnamese + English)
FEATURE_DESCRIPTIONS = {
    'Brand': 'Hãng xe - Car Brand (Toyota, Honda, Mazda...)',
    'Model': 'Dòng xe - Car Model (Vios, Civic, CX-5...)',
    'Year': 'Năm sản xuất - Manufacturing Year',
    'Kilometers': 'Số km đã đi - Kilometers Driven',
    'Fuel Type': 'Loại nhiên liệu - Fuel Type (Xăng/Dầu/Điện)',
    'Transmission': 'Hộp số - Transmission (Tự động/Số sàn)',
    'Seats': 'Số chỗ ngồi - Number of Seats',
    'Price': 'Giá (triệu VNĐ) - Price in Million VND'
}

# Default Values for UI
DEFAULT_VALUES = {
    'Brand': 'Toyota',
    'Model': 'Vios',
    'Year': 2020,
    'Kilometers': 50000,
    'Fuel': 'Xăng',
    'Transmission': 'Tự động',
    'Seats': 5
}

# Popular Vietnamese Car Brands
POPULAR_BRANDS = [
    'Toyota', 'Honda', 'Mazda', 'Ford', 'Hyundai',
    'Kia', 'Mitsubishi', 'Nissan', 'Chevrolet', 'Suzuki',
    'Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Vinfast'
]

# Fuel Types
FUEL_TYPES = ['Xăng', 'Dầu', 'Điện', 'Hybrid']

# Transmission Types
TRANSMISSION_TYPES = ['Tự động', 'Số sàn']

# Year Range
MIN_YEAR = 2000
MAX_YEAR = CURRENT_YEAR

# Kilometer Range
MIN_KM = 0
MAX_KM = 500000

# Seats Options
SEATS_OPTIONS = [2, 4, 5, 7, 9, 16]

# Price Range (millions VND)
MIN_PRICE = 50
MAX_PRICE = 5000
