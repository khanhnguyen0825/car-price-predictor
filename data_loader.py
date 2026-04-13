"""
Module tải và tiền xử lý dữ liệu cho bộ dữ liệu giá xe Việt Nam

Các bước xử lý chính (theo thứ tự trong get_full_pipeline):
  Bước 1 - load_data:               Đọc file CSV, xử lý encoding tiếng Việt
  Bước 2 - clean_column_names:      Chuẩn hóa tên cột từ tiếng Việt sang tiếng Anh
  Bước 3 - extract_brand_model:     Trích xuất Hãng xe và Dòng xe từ tên xe
  Bước 4 - clean_price:             Làm sạch và chuyển đổi giá sang triệu VNĐ
  Bước 5 - clean_numeric_features:  Làm sạch Năm SX, Số KM, Số chỗ ngồi
  Bước 6 - engineer_features:       Tạo thêm feature mới (Age, KM_Negative,...)
  Bước 7 - clean_data:              Xóa outliers và dữ liệu thiếu
  Bước 8 - encode_categorical:      Mã hóa Brand/Model từ text sang số
  Bước 9 - prepare_features:        Chọn features cuối cùng đưa vào model
"""

# ── Thư viện xử lý dữ liệu ──────────────────────────────────
import pandas as pd          # Đọc CSV, xử lý DataFrame
import numpy as np           # Tính toán số học, xử lý NaN
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Mã hóa categorical
import os                    # Thao tác hệ thống file
from pathlib import Path     # Xử lý đường dẫn file đa nền tảng
import re                    # Regular expression để parse chuỗi
import config                # Hằng số cấu hình của project

class CarPriceDataLoader:
    """
    Class tải và tiền xử lý bộ dữ liệu giá xe Việt Nam
    
    Cách dùng:
        loader = CarPriceDataLoader()
        X, y, features, df = loader.get_full_pipeline()
    """
    
    def __init__(self, csv_path=None):
        # Tự động xác định đường dẫn file CSV từ thư mục project
        # Nếu không truyền csv_path, dùng đường dẫn mặc định trong config.py
        if csv_path is None:
            current_file = Path(__file__).resolve()   # Đường dẫn tuyệt đối của file này
            project_root = current_file.parent         # Thư mục chứa file này (project root)
            csv_path = project_root / config.DATASET_PATH  # Ghép với đường dẫn data trong config
        
        self.csv_path = str(csv_path)   # Lưu đường dẫn CSV
        self.df = None                   # DataFrame sẽ được tải sau
        self.label_encoders = {}         # Lưu các LabelEncoder để decode sau này
        
    def load_data(self):
        """
        Tải dữ liệu từ file CSV với encoding UTF-8
        
        Thử nhiều loại encoding khác nhau để tránh lỗi UnicodeDecodeError:
        - utf-8: Encoding chuẩn Unicode
        - utf-8-sig: UTF-8 với BOM (Byte Order Mark)
        - latin-1: Encoding cho ký tự Tây Âu
        """
        print(f"[LOAD] Đang tải dữ liệu từ {self.csv_path}...")
        
        # Thử các encoding khác nhau cho tới khi thành công
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"[SUCCESS] Đã tải thành công với {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        print(f"[INFO] Đã tải {len(self.df):,} bản ghi")
        print(f"[INFO] Các cột: {list(self.df.columns)}")
        
        return self.df
    
    def clean_column_names(self):
        """
        BƯỚC 2: Chuẩn hóa tên cột
        
        Vấn đề: File CSV có tên cột tiếng Việt (ví dụ: 'Năm sản xuất:')
        → Cần đổi thành tên tiếng Anh ngắn gọn (ví dụ: 'Year') để dễ xử lý
        
        Xử lý 2 trường hợp:
        - column_mapping: Tên cột UTF-8 đúng (đọc trên macOS/Linux)
        - garbled_mapping: Tên cột bị lỗi ký tự (đọc trên Windows với encoding sai)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Xóa cột 'Condition' cũ (trùng với 'Tình trạng:') để tránh conflict
        if 'Condition' in self.df.columns:
            self.df = self.df.drop(columns=['Condition'])
            print("Dropped duplicate 'Condition' column")
        
        # In ra một số cột đầu tiên để debug
        print(f"Current columns: {list(self.df.columns)[:5]}...")  # Show first 5
        
        # Bảng ánh xạ: tên cột tiếng Việt → tên cột tiếng Anh
        # Trường hợp encoding đúng (UTF-8 chuẩn)
        column_mapping = {
            'Năm sản xuất:': 'Year',           # Năm sản xuất xe
            'Tình trạng:': 'Condition_VN',      # Tình trạng xe (mới/cũ)
            'Số Km đã đi:': 'Kilometers',       # Số km đã đi
            'Xuất xứ:': 'Origin',               # Xuất xứ (nhập khẩu/trong nước)
            'Kiểu dáng:': 'Style',              # Kiểu dáng (sedan, SUV,...)
            'Số chỗ:': 'Seats',                 # Số chỗ ngồi (viết tắt)
            'Màu ngoại thất:': 'Exterior_Color',# Màu sơn ngoài
            'Màu nội thất:': 'Interior_Color',  # Màu nội thất
            'Nhiên liệu:': 'Fuel',              # Loại nhiên liệu
            'Hộp số:': 'Transmission',          # Hộp số (tự động/số sàn)
            'Dẫn động:': 'Drivetrain',          # Dẫn động (2WD/4WD)
            'Số chỗ ngồi:': 'Seats',            # Số chỗ ngồi (đầy đủ)
            'Số cửa:': 'Doors',                 # Số cửa xe
            'Động cơ:': 'Engine'                # Thông số động cơ
        }
        
        # Bảng ánh xạ dự phòng: khi đọc file bằng latin-1 trên Windows
        # → Ký tự tiếng Việt bị mã hóa sai, tạo ra các ký tự lạ
        garbled_mapping = {
            'NÄ\x83m sáº£n xuáº¥t:': 'Year',
            'TÃ¬nh tráº¡ng:': 'Condition_VN',
            'Sá»\x91 Km Ä\x91Ã£ Ä\x91i:': 'Kilometers',
            'Xuáº¥t xá»©:': 'Origin',
            'Kiá»\x83u dÃ¡ng:': 'Style',
            'Sá»\x91 chá»\x97:': 'Seats',
            'MÃ\xa0u ngoáº¡i tháº¥t:': 'Exterior_Color',
            'MÃ\xa0u ná»\x99i tháº¥t:': 'Interior_Color',
            'Nhiên liệu:': 'Fuel',
            'Há»\x99p sá»\x91:': 'Transmission',
            'Dáº«n Ä\x91á»\x99ng:': 'Drivetrain',
            'Sá»\x91 chá»\x97 ngá»\x93i:': 'Seats',
            'Sá»\x91 cá»\xada:': 'Doors',
            'Ä\x90á»\x99ng cÆ¡:': 'Engine'
        }
        
        # Gộp cả 2 bảng lại, áp dụng đổi tên cột
        full_mapping = {**column_mapping, **garbled_mapping}
        self.df = self.df.rename(columns=full_mapping)
        
        print(f"[SUCCESS] Đã chuẩn hóa tên cột")
        print(f"[INFO] Các cột mới: {list(self.df.columns)[:10]}...")  # Show first 10
        
        return self.df

    
    def extract_brand_model(self):
        """
        BƯỚC 3: Trích xuất Hãng xe (Brand) và Dòng xe (Model) từ tên xe
        
        Cột 'Name' chứa tên đầy đủ như: 'Toyota Vios 1.5E CVT 2020'
        → Brand = 'Toyota'  (từ đầu tiên)
        → Model = 'Vios'    (từ thứ hai)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n[EXTRACT] Đang trích xuất Hãng xe và Dòng xe...")
        
        # Lấy từ ĐẦU TIÊN của tên xe làm Brand
        # vd: 'Toyota Vios 1.5E' → 'Toyota'
        self.df['Brand'] = self.df['Name'].str.split().str[0]
        
        # Hàm nội bộ: lấy từ THỨ HAI của tên xe làm Model
        def extract_model(name):
            words = name.split()
            if len(words) >= 2:
                # Lấy từ thứ 2, xóa dấu phẩy và chấm nếu có
                # vd: 'Toyota Vios, 1.5E' → 'Vios'
                model = words[1].replace(',', '').replace('.', '')
                return model
            return 'Unknown'  # Nếu tên xe chỉ có 1 từ
        
        # Áp dụng hàm extract_model cho từng dòng
        self.df['Model'] = self.df['Name'].apply(extract_model)
        
        print(f"[INFO] Số hãng xe: {self.df['Brand'].nunique()}")
        print(f"[INFO] Số dòng xe: {self.df['Model'].nunique()}")
        
        return self.df
    
    def clean_price(self):
        """
        BƯỚC 4: Làm sạch và chuyển đổi giá xe sang triệu VNĐ
        
        Vấn đề: Cột 'Price' lưu giá dạng chuỗi tiếng Việt:
          - '1 Tỷ 118 Triệu'  → 1118 (triệu VNĐ)
          - '646 Triệu'       → 646  (triệu VNĐ)
          - '2.5 Tỷ'         → 2500 (triệu VNĐ)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n[CLEAN] Đang làm sạch dữ liệu giá xe...")
        
        def parse_price(price_str):
            """Chuyển đổi chuỗi giá tiếng Việt sang số triệu VNĐ"""
            if pd.isna(price_str):
                return np.nan  # Giá trị rỗng → bỏ qua
            
            price_str = str(price_str).strip()
            total_millions = 0  # Tổng giá tính bằng triệu VNĐ
            
            # ── Xử lý phần TỶ ─────────────────────────────────
            # Tìm số trước từ 'Tỷ' (cả 2 dạng: đúng và bị lỗi encoding)
            # Ví dụ: '1 Tỷ 118 Triệu' → group(1) = '1' → 1 × 1000 = 1000 triệu
            if any(x in price_str for x in ['Tỷ', 'tỷ', 'Tá»·', 'tá»·']):
                ty_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:[Tt]ỷ|[Tt]á»·)', price_str)
                if ty_match:
                    total_millions += float(ty_match.group(1)) * 1000  # 1 Tỷ = 1000 Triệu
            
            # ── Xử lý phần TRIỆU ──────────────────────────────
            # Tìm số trước từ 'Triệu'
            # Ví dụ: '1 Tỷ 118 Triệu' → group(1) = '118' → +118 triệu
            if any(x in price_str for x in ['Triệu', 'triệu', 'Triá»u', 'triá»u']):
                trieu_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:[Tt]riệu|[Tt]riá»u)', price_str)
                if trieu_match:
                    total_millions += float(trieu_match.group(1))
            
            # ── Fallback: Nếu không có từ tiếng Việt ─────────
            # Thử parse trực tiếp như số thuần túy
            if total_millions == 0:
                cleaned = re.sub(r'[^\d.]', '', price_str)  # Xóa mọi ký tự không phải số
                if cleaned:
                    try:
                        price = float(cleaned)
                        # Nếu > 1000, khả năng đơn vị là nghìn VNĐ → chia 1000
                        total_millions = price / 1000 if price >= 1000 else price
                    except:
                        return np.nan
            
            return total_millions if total_millions > 0 else np.nan
        
        # Áp dụng hàm parse_price cho toàn bộ cột 'Price'
        # Kết quả lưu vào cột mới 'Price_Million'
        self.df['Price_Million'] = self.df['Price'].apply(parse_price)
        
        # Xóa các dòng có giá không hợp lệ (NaN hoặc <= 0)
        initial_count = len(self.df)
        self.df = self.df[self.df['Price_Million'].notna()]  # Xóa NaN
        self.df = self.df[self.df['Price_Million'] > 0]       # Xóa giá âm/bằng 0
        
        removed = initial_count - len(self.df)
        print(f"[INFO] Đã xóa {removed} dòng có giá không hợp lệ")
        print(f"[INFO] Khoảng giá: {self.df['Price_Million'].min():.1f} - {self.df['Price_Million'].max():.1f} triệu VNĐ")
        
        return self.df
    
    def clean_numeric_features(self):
        """
        BƯỚC 5: Làm sạch các thuộc tính số (Năm SX, Số KM, Số chỗ ngồi)
        
        Mục tiêu: Chuyển dữ liệu từ dạng chữ về dạng số thực và loại bỏ các giá trị phi lý.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n[CLEAN] Đang làm sạch các thuộc tính số (Năm SX, Số KM, Số chỗ ngồi)...")
        
        # ── Xử lý Năm Sản Xuất ──────────────────────────────
        if 'Year' in self.df.columns:
            # Chuyển về dạng số, nếu lỗi (ví dụ chữ) thì biến thành NaN (Not a Number)
            self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
            # Chỉ giữ xe trong khoảng năm cho phép (vd: 1980 - 2026) định nghĩa trong config.py
            self.df = self.df[(self.df['Year'] >= config.MIN_YEAR) & (self.df['Year'] <= config.MAX_YEAR)]
            print(f"[INFO] Khoảng năm: {self.df['Year'].min():.0f} - {self.df['Year'].max():.0f}")
        
        # ── Xử lý Số KM đã đi ──────────────────────────────
        if 'Kilometers' in self.df.columns:
            def parse_km(km_str):
                """Hàm phụ: bóc tách số từ chuỗi KM (vd: '38.000 km' -> 38000)"""
                if pd.isna(km_str) or km_str == '':
                    return np.nan
                
                # Chuyển về chữ thường và xóa các ký tự gây nhiễu
                km_str = str(km_str).lower()
                km_str = km_str.replace(',', '').replace('.', '').replace(' ', '').replace('km', '').strip()
                try:
                    value = float(km_str)
                    return value if value > 0 else np.nan
                except:
                    return np.nan
            
            self.df['Kilometers'] = self.df['Kilometers'].apply(parse_km)
            
            # Chỉ lọc theo KM nếu có đủ lượng dữ liệu hợp lệ (tránh mất sạch dữ liệu)
            valid_km = self.df['Kilometers'].notna().sum()
            if valid_km > 100:
                # Chỉ lấy xe đi dưới mức KM tối đa trong config (vd: 500k KM)
                self.df = self.df[(self.df['Kilometers'] >= 0) & (self.df['Kilometers'] <= config.MAX_KM)]
                print(f"[INFO] Khoảng KM: {self.df['Kilometers'].min():.0f} - {self.df['Kilometers'].max():.0f}")
                print(f"[INFO] Số bản ghi KM hợp lệ: {valid_km}/{len(self.df)}")
            else:
                print(f"[WARNING] Dữ liệu KM bị lỗi quá nhiều - tạm thời lấy giá trị trung bình (50k KM)")
                # Nếu dữ liệu KM quá ít, ta điền tạm 50k KM để các thuật toán không bị lỗi
                self.df['Kilometers'] = self.df['Kilometers'].fillna(50000)
        
        # ── Xử lý Số ghế (Seats) ────────────────────────────
        if 'Seats' in self.df.columns:
            self.df['Seats'] = pd.to_numeric(self.df['Seats'], errors='coerce')
            valid_seats = self.df['Seats'].notna().sum()
            if valid_seats > 100:
                # Giới hạn số chỗ từ 2 đến 16 chỗ (phổ biến)
                self.df = self.df[(self.df['Seats'] >= 2) & (self.df['Seats'] <= 16)]
                print(f"[INFO] Khoảng số chỗ: {self.df['Seats'].min():.0f} - {self.df['Seats'].max():.0f}")
            else:
                print(f"[WARNING] Dữ liệu số chỗ không khả dụng (bỏ qua lọc chỗ ngồi)")
        
        return self.df
    
    def engineer_features(self):
        """
        BƯỚC 6: Tạo các đặc trưng mới (Feature Engineering)
        
        Mục tiêu: Tạo ra các cột dữ liệu thông minh hơn từ dữ liệu thô để tăng độ chính xác của model.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n[FEATURE] Đang tạo các feature mới từ dữ liệu hiện tại...")
        
        # 1. Tuổi xe (Age): Càng cũ giá thường càng giảm
        self.df['Age'] = config.CURRENT_YEAR - self.df['Year']
        print(f"[INFO] Đã tạo feature Age (Tuổi xe = {config.CURRENT_YEAR} - Năm SX)")
        
        # 2. Giá trung bình mỗi năm: Để phân tích xem xe có giữ giá hay không
        self.df['Price_Per_Year'] = self.df['Price_Million'] / (self.df['Age'] + 1)  # +1 để tránh chia 0 cho xe đời mới nhất
        print(f"[INFO] Đã tạo feature Price_Per_Year (Chỉ số khấu hao)")
        
        # 3. KM_Negative: ĐẢO DẤU SỐ KM
        # Lý do: Trong Machine Learning, ta muốn KM tăng -> giá giảm. 
        # Nhưng máy thường học KM tăng -> giá tăng. 
        # Bằng cách đảo thành số âm: KM càng nhiều (vd: -200.000) thì giá trị càng NHỎ 
        # -> Model học đúng quy luật hơn.
        self.df['KM_Negative'] = -self.df['Kilometers']
        print(f"[INFO] Đã tạo feature KM_Negative (mẹo quan trọng nhất để tăng xác suất học đúng)")
        
        return self.df
    
    def clean_data(self):
        """
        BƯỚC 7: Làm sạch tổng thể - xóa outliers và dữ liệu thiếu
        
        3 thao tác chính:
        1. Xóa dòng có quá nhiều giá trị thiếu (< 30% cột có dữ liệu)
        2. Xóa dòng trùng lặp
        3. Loại bỏ outliers bằng phương pháp IQR
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n[CLEAN] Đang làm sạch dữ liệu (loại outliers và dữ liệu thiếu)...")
        initial_count = len(self.df)
        
        # ── Xóa dòng có quá nhiều giá trị thiếu ─────────────
        # threshold = 30% tổng số cột → dòng phải có ít nhất 30% cột có dữ liệu
        # Ví dụ: 20 cột → cần ít nhất 6 cột có dữ liệu mới giữ lại
        threshold = len(self.df.columns) * 0.3
        self.df = self.df.dropna(thresh=threshold)
        
        # ── Xóa dòng trùng lặp ────────────────────────────
        # Cùng xe, cùng giá, cùng thông số → chỉ giữ 1 dòng
        self.df = self.df.drop_duplicates()
        
        # ── Loại bỏ outliers bằng phương pháp IQR ─────────
        # IQR (Interquartile Range) = Q3 - Q1
        # Ngưỡng thông thường: Q1 - 1.5*IQR đến Q3 + 1.5*IQR
        # Ở đây dùng 3.0*IQR (ít nghiêm ngặt hơn) để giữ nhiều dữ liệu hơn
        Q1 = self.df['Price_Million'].quantile(0.25)  # Phân vị 25%
        Q3 = self.df['Price_Million'].quantile(0.75)  # Phân vị 75%
        IQR = Q3 - Q1                                 # Khoảng tứ phân vị
        
        lower_bound = Q1 - 3.0 * IQR  # Ngưỡng dưới (giá quá rẻ bất thường)
        upper_bound = Q3 + 3.0 * IQR  # Ngưỡng trên (giá quá đắt bất thường)
        
        # Chỉ giữ lại các dòng có giá nằm trong ngưỡng
        before_outlier_removal = len(self.df)
        self.df = self.df[
            (self.df['Price_Million'] >= lower_bound) &
            (self.df['Price_Million'] <= upper_bound)
        ]
        
        final_count = len(self.df)
        removed = initial_count - final_count
        if initial_count > 0:
            print(f"[INFO] Đã xóa {removed} dòng ({removed/initial_count*100:.1f}%)")
        else:
            print(f"[INFO] Đã xóa {removed} dòng")
        print(f"[SUCCESS] Dataset cuối cùng: {final_count:,} bản ghi")
        
        return self.df
    
    def encode_categorical_features(self):
        """
        BƯỚC 8: Mã hóa dữ liệu dạng chữ thành số (Label Encoding)
        
        Mục tiêu: Các thuật toán Machine Learning chỉ hiểu số. 
        Ta cần biến "Toyota" -> 45, "Vios" -> 123.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n[ENCODE] Đang mã hóa hãng/dòng xe thành các con số...")
        
        # Các cột cần mã hóa: Hãng, Dòng, Nhiên liệu, Hộp số
        categorical_columns = ['Brand', 'Model', 'Fuel', 'Transmission']
        
        for col in categorical_columns:
            if col in self.df.columns:
                # Nếu dữ liệu trống, thay bằng chữ 'Unknown'
                self.df[col] = self.df[col].fillna('Unknown')
                
                # Khởi tạo bộ mã hóa
                le = LabelEncoder()
                # Chuyển đổi dữ liệu sang dạng chuỗi trước khi đánh mã số
                col_data = self.df[[col]].iloc[:, 0].astype(str)
                # Lưu mã số vào cột mới ví dụ: 'Brand_Encoded'
                self.df[f'{col}_Encoded'] = le.fit_transform(col_data)
                # Lưu lại bộ mã hóa le để sau này dùng 'dịch ngược' (Decode)
                self.label_encoders[col] = le
                print(f"[INFO] Mã hóa {col}: Đã ánh xạ {len(le.classes_)} nhóm khác nhau thành số")
        
        return self.df
    
    def prepare_features(self):
        """
        BƯỚC 9: Chọn features cuối cùng để đưa vào train model
        
        Features được chọn:
          - Year:          Năm sản xuất
          - Age:           Tuổi xe (tính theo năm)
          - KM_Negative:   Số KM (đảo dấu âm)
          - Brand_Encoded: Mã số hãng xe
          - Model_Encoded: Mã số dòng xe
        
        Target (nhãn dự đoán):
          - Price_Million: Giá xe (triệu VNĐ)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n[PREPARE] Đang chuẩn bị features cho model...")
        
        # Features số học (continuous)
        # Dùng KM_Negative thay vì Kilometers để có tương quan chiều đúng:
        # KM tăng → giá giảm → KM_Negative càng âm → model hiểu đúng hơn
        numeric_features = ['Year', 'Age', 'KM_Negative']
        
        # Features đã mã hóa (từ encode_categorical_features)
        # Tự động tìm tất cả cột kết thúc bằng '_Encoded'
        # Ví dụ: Brand_Encoded, Model_Encoded
        encoded_features = [col for col in self.df.columns if col.endswith('_Encoded')]
        
        # Ghép tất cả features lại
        feature_columns = numeric_features + encoded_features
        
        # Chỉ giữ lại các cột thực sự tồn tại trong DataFrame
        feature_columns = [col for col in feature_columns if col in self.df.columns]
        
        # Xóa các dòng còn thiếu giá trị ở features hoặc target
        required_cols = feature_columns + ['Price_Million']
        self.df = self.df.dropna(subset=required_cols)
        
        # X = ma trận features (đầu vào cho model)
        # y = vector target (giá xe cần dự đoán)
        X = self.df[feature_columns]
        y = self.df['Price_Million']
        
        print(f"[INFO] Đã chọn {len(feature_columns)} features")
        print(f"   {feature_columns}")
        print(f"[INFO] Target: Price_Million (triệu VNĐ)")
        print(f"[NOTE] Sử dụng KM_Negative (feature duy nhất) để khấu hao chính xác")
        
        return X, y, feature_columns
    
    def get_full_pipeline(self):
        """
        THỰC THI TOÀN BỘ PIPELINE: 
        Chạy lần lượt tất cả 9 bước từ đọc dữ liệu đến chuẩn bị sẵn sàng cho Model.
        """
        print("[START] Bắt đầu quy trình tiền xử lý dữ liệu xe từ A-Z...\n")
        
        # Bước 1: Đọc file
        self.load_data()
        
        # Bước 2: Tên cột (Việt -> Anh)
        self.clean_column_names()
        
        # Bước 3: Tách Hãng & Dòng xe
        self.extract_brand_model()
        
        # Bước 4: Chuyển giá về triệu đồng
        self.clean_price()
        
        # Bước 5: Làm sạch số (Năm, KM, Chỗ)
        self.clean_numeric_features()
        
        # Bước 6: Tạo đặc trưng thông minh (Age, Neg_KM)
        self.engineer_features()
        
        # Bước 7: Loại bỏ dữ liệu ảo/sai (Outliers/Missing)
        self.clean_data()
        
        # Bước 8: Đánh mã số cho Hãng/Dòng
        self.encode_categorical_features()
        
        # Bước 9: Trích xuất X (đầu vào) và y (đầu ra)
        X, y, feature_names = self.prepare_features()
        
        print("\n[SUCCESS] Chúc mừng! Dữ liệu đã sạch và sẵn sàng để huấn luyện!")
        print(f"[THỐNG KÊ] Số mẫu học: {X.shape[0]:,} xe")
        print(f"[THỐNG KÊ] Số đặc trưng: {X.shape[1]} (Year, Age, KM_Neg, Brand_Enc, Model_Enc)")
        
        return X, y, feature_names, self.df


# ─────────────────────────────────────────────────────────────
# Utility functions - Hàm tiện ích dùng chung
# ─────────────────────────────────────────────────────────────

def format_price(price_in_millions):
    """
    Định dạng giá tiền từ triệu VNĐ sang chuỗi dễ đọc
    
    Ví dụ:
        format_price(500)  → '500 triệu VNĐ'
        format_price(1500) → '1.50 tỷ VNĐ'
    """
    if price_in_millions >= 1000:
        return f"{price_in_millions/1000:.2f} tỷ VNĐ"   # ≥ 1000 triệu → đổi sang tỷ
    else:
        return f"{price_in_millions:.0f} triệu VNĐ"      # < 1000 triệu → giữ nguyên


if __name__ == "__main__":
    # Test the data loader
    loader = CarPriceDataLoader()
    X, y, features, df = loader.get_full_pipeline()
    
    print(f"\nKhoảng giá: {format_price(y.min())} - {format_price(y.max())}")
    print(f"Giá trung bình: {format_price(y.mean())}")
    print(f"\nTop 5 hãng xe:")
    print(df['Brand'].value_counts().head())
