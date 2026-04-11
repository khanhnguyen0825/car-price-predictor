"""
Module Machine Learning để Dự Đoán Giá Xe

Module này triển khai 6 thuật toán regression và so sánh hiệu suất:
- Linear Regression: Hồi quy tuyến tính cơ bản, đơn giản nhất
- Ridge Regression:  Tương tự Linear nhưng thêm phạt L2 để tránh overfitting
- Lasso Regression:  Tương tự Linear nhưng thêm phạt L1, có thể đưa hệ số về 0
- SVR:               Support Vector Regression với kernel RBF (phi tuyến)
- Random Forest:     Tập hợp 100 decision trees, bầu chọn kết quả
- Gradient Boosting: Xây dựng models nối tiếp, mỗi model sửa lỗi model trước

Kết quả: Random Forest cho R² = 86.54% - tốt nhất cho bài toán giá xe
"""

# ── Thư viện Machine Learning ──────────────────────────────────────
import pandas as pd       # Xử lý DataFrame
import numpy as np        # Tính toán số học
from sklearn.model_selection import train_test_split  # Chia dữ liệu train/test

# Các thuật toán Linear
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Các thuật toán Ensemble (kết hợp nhiều models)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Support Vector Machine
from sklearn.svm import SVR

# Tiền xử lý dữ liệu
from sklearn.preprocessing import StandardScaler

# Các công thức tính độ chính xác của model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config                                          # Hằng số cấu hình project
from data_loader import CarPriceDataLoader, format_price  # Module tải dữ liệu

class CarPricePredictor:
    """
    Class Dự Đoán Giá Xe sử dụng Machine Learning
    
    Chức năng chính:
    - Train nhiều models cùng lúc
    - Tự động chọn model tốt nhất dựa trên R² score
    - Dự đoán giá xe mới
    - So sánh performance giữa các models
    """
    
    def __init__(self):
        # Model tốt nhất sau khi train
        self.model = None
        # Dictionary lưu tất cả models đã train
        self.models = {}
        
        # Dữ liệu training và testing
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Predictions để đánh giá
        self.y_pred_train = None
        self.y_pred_test = None
        
        # Thông tin về features
        self.feature_names = None
        
        # Thông tin model tốt nhất
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def train(self, X, y, feature_names=None, model_types=['rf', 'gb', 'ridge', 'linear']):
        """
        Train nhiều models và tự động chọn model tốt nhất
        
        Quy trình:
        1. Chia dữ liệu thành train/test (80/20)
        2. Train tất cả models được chỉ định
        3. Đánh giá performance trên test set
        4. Chọn model có R² score cao nhất
        
        Args:
            X: DataFrame chứa features (Year, Age, KM_Negative, Brand, Model)
            y: Target variable - giá xe (đơn vị: triệu VNĐ)
            feature_names: Tên các features (optional)
            model_types: Danh sách models muốn train
                       - 'linear': Linear Regression
                       - 'ridge': Ridge Regression  
                       - 'lasso': Lasso Regression
                       - 'svr': Support Vector Regression
                       - 'rf': Random Forest (recommended)
                       - 'gb': Gradient Boosting
        """
        print("\n Training Car Price Prediction Models...")
        print("=" * 60)
        
        self.feature_names = feature_names if feature_names is not None else list(X.columns)
        
        # Chia dữ liệu: 80% training, 20% testing
        # random_state để kết quả reproducible (giống nhau mỗi lần chạy)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        print(f" Training set: {len(self.X_train)} samples")
        print(f" Test set: {len(self.X_test)} samples")
        
        # Khởi tạo các models với hyperparameters
        model_dict = {
            'linear': ('Linear Regression', LinearRegression()),
            'ridge': ('Ridge Regression', Ridge(alpha=1.0)),  # alpha: độ mạnh regularization
            'lasso': ('Lasso Regression', Lasso(alpha=1.0)),  # alpha: độ mạnh regularization + feature selection
            'svr': ('SVR (RBF Kernel)', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)),
            'rf': ('Random Forest', RandomForestRegressor(
                n_estimators=100,  # Số cây decision trees
                random_state=config.RANDOM_STATE,
                n_jobs=-1  # Sử dụng tất cả CPU cores
            )),
            'gb': ('Gradient Boosting', GradientBoostingRegressor(
                n_estimators=100,  # Số boosting stages
                random_state=config.RANDOM_STATE
            ))
        }
        
        print(f"\n Training {len(model_types)} models...\n")
        
        # ── Vòng lặp train từng model ─────────────────────────────
        # Mỗi vòng: train → predict → tính metrics → so sánh với best
        for model_key in model_types:
            if model_key not in model_dict:
                continue  # Bỏ qua nếu key không hợp lệ
                
            model_name, model = model_dict[model_key]
            print(f"  Training {model_name}...")
            
            # BƯỚC 1: Train model trên tập training
            # fit() = "học" từ dữ liệu X_train/y_train
            model.fit(self.X_train, self.y_train)
            
            # BƯỚC 2: Dự đoán trên cả 2 tập để so sánh
            # Nếu train tốt nhưng test kém → overfitting
            y_pred_train = model.predict(self.X_train)  # Dự đoán trên tập train
            y_pred_test = model.predict(self.X_test)    # Dự đoán trên tập test
            
            # BƯỚC 3: Tính các chỉ số đánh giá
            # R²:   1.0 = hoàn hảo | 0.0 = không tốt hơn đoán trung bình
            # MAE:  Sai số tuyệt đối trung bình (đơn vị: triệu VNĐ)
            # RMSE: Phạt nặng hơn cho sai số lớn (root mean square error)
            r2_train = r2_score(self.y_train, y_pred_train)
            r2_test  = r2_score(self.y_test,  y_pred_test)
            mae_test  = mean_absolute_error(self.y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            # BƯỚC 4: Lưu kết quả vào dictionary
            self.models[model_name] = {
                'model':    model,
                'r2_train': r2_train,
                'r2_test':  r2_test,
                'mae':      mae_test,
                'rmse':     rmse_test
            }
            
            print(f"    R\u00b2 (Train): {r2_train:.4f}")
            print(f"    R\u00b2 (Test):  {r2_test:.4f}")
            print(f"    MAE:        {mae_test:.0f} tri\u1ec7u VN\u0110")
            print(f"    RMSE:       {rmse_test:.0f} tri\u1ec7u VN\u0110\n")
            
            # BƯỚC 5: Cập nhật best model nếu R² test cao hơn hiện tại
            if r2_test > self.best_score:
                self.best_score     = r2_test
                self.best_model     = model
                self.best_model_name = model_name
                self.y_pred_train   = y_pred_train  # Lưu lại predictions
                self.y_pred_test    = y_pred_test   # để hiển thị biểu đồ sau
        
        # Gán model tốt nhất làm model mặc định
        # → Tất cả lệnh gọi predict() sau đây đều dùng model này
        self.model = self.best_model
        
        print("=" * 60)
        print(f" Best Model: {self.best_model_name}")
        print(f" R\u00b2 Score: {self.best_score:.4f}")
        print(f" MAE: {self.models[self.best_model_name]['mae']:.0f} tri\u1ec7u VN\u0110")
        print("=" * 60)
        
        return self  # Trả về self để có thể chain methods
    
    def predict(self, features_dict):
        """
        Dự đoán giá cho một chiếc xe dựa trên features
        
        Args:
            features_dict: Dictionary chứa thông tin xe
                         Ví dụ: {'Year': 2020, 'Brand_Encoded': 5, 'Model_Encoded': 42, ...}
        
        Returns:
            Giá dự đoán (đơn vị: triệu VNĐ)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Tạo DataFrame với đúng thứ tự features như lúc train
        # Nếu feature không có trong input → dùng giá trị mặc định = 0
        feature_values = []
        for feature in self.feature_names:
            if feature in features_dict:
                feature_values.append(features_dict[feature])
            else:
                feature_values.append(0)  # Giá trị mặc định khi thiếu feature
        
        # Chuyển thành DataFrame 1 hàng → predict
        X_new = pd.DataFrame([feature_values], columns=self.feature_names)
        prediction = self.model.predict(X_new)[0]  # [0] vì kết quả là array
        
        return prediction  # Đơn vị: triệu VNĐ
    
    def get_metrics(self):
        """
        Lấy các chỉ số đánh giá performance của model
        
        Returns:
            Dictionary chứa:
            - r2_train: R² score trên training set (% variance giải thích được)
            - r2_test: R² score trên test set 
            - mae: Mean Absolute Error (sai số trung bình)
            - rmse: Root Mean Squared Error (sai số bình phương trung bình)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Tính metrics cho cả tập train và test để so sánh
        # Train metrics: đánh giá model học tốt không
        # Test  metrics: đánh giá model tổng quát tốt không (quan trọng hơn)
        metrics = {
            'train': {
                'r2':   r2_score(self.y_train, self.y_pred_train),
                'mae':  mean_absolute_error(self.y_train, self.y_pred_train),
                'rmse': np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
            },
            'test': {
                'r2':   r2_score(self.y_test, self.y_pred_test),      # R² trên test set
                'mae':  mean_absolute_error(self.y_test, self.y_pred_test),   # Sai số TB
                'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))  # RMSE
            }
        }
        
        # Dấu hiệu Overfitting:
        # R²_train >> R²_test → model học vẹt, không tổng quát hóa tốt
        return metrics
    
    def get_feature_importance(self):
        """
        Lấy độ quan trọng của các features (chỉ cho tree-based models)
        
        Chỉ hoạt động với:
        - Random Forest
        - Gradient Boosting
        
        Không hoạt động với Linear/Ridge/Lasso/SVR
        
        Returns:
            Dictionary {feature_name: importance_score}
            Sắp xếp theo thứ tự giảm dần
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_dict = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting)
            # feature_importances_ có sẵn, là tỷ lệ đóng góp của mỗi feature
            # Tất cả importances cộng lại = 1.0
            importances = self.model.feature_importances_
            for i, name in enumerate(self.feature_names):
                importance_dict[name] = importances[i]
                
        elif hasattr(self.model, 'coef_'):
            # Linear models (Linear, Ridge, Lasso)
            # Không có feature_importances_, dùng |coef_| làm proxy
            # Normalize về tổng = 1 để so sánh với tree models
            coefficients = np.abs(self.model.coef_)
            total = coefficients.sum()
            if total > 0:
                coefficients = coefficients / total  # Chuẩn hóa về [0, 1]
            for i, name in enumerate(self.feature_names):
                importance_dict[name] = coefficients[i]
        
        # Sắp xếp giảm dần theo mức độ quan trọng
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def get_formula(self):
        """
        Lấy công thức hồi quy đưới dạng chuỗi (chỉ cho Linear models)
        
        Ví dụ: Price = 500.25 + 3.4 × Year - 0.005 × KM_Negative + ...
        
        Không dùng được với Random Forest, Gradient Boosting, SVR
        vì các model này phân tích phi tuyến
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Kiểm tra xem model có thuộc tính coef_ không (chỉ linear models)
        if not hasattr(self.model, 'coef_'):
            return f"{self.best_model_name} (formula not available for tree models)"
        
        # Xây dựng công thức: Price = intercept + coef1*f1 + coef2*f2 + ...
        formula = f"Price = {self.model.intercept_:.2f}"
        for i, name in enumerate(self.feature_names):
            coef = self.model.coef_[i]
            sign = '+' if coef >= 0 else '-'  # Dấu cộng/trừ
            formula += f" {sign} {abs(coef):.4f} × {name}"
        
        return formula
    
    def compare_all_models(self):
        """
        Tạo DataFrame so sánh hiệu suất tất cả models đã train
        
        Kết quả sắp xếp theo R² (Test) giảm dần → model tốt nhất lên đầu
        
        Returns:
            DataFrame với các cột: Model, R² (Train), R² (Test), MAE, RMSE
        """
        if not self.models:
            raise ValueError("No models trained. Call train() first.")
        
        # Xây dựng danh sách kết quả của từng model
        comparison = []
        for model_name, metrics in self.models.items():
            comparison.append({
                'Model':        model_name,
                'R² (Train)':   metrics['r2_train'],
                'R² (Test)':    metrics['r2_test'],
                'MAE (triệu)': metrics['mae'],
                'RMSE (triệu)': metrics['rmse']
            })
        
        # Sắp xếp theo R² (Test) giảm dần → dễ thấy model nào tốt hơn
        df = pd.DataFrame(comparison)
        df = df.sort_values('R² (Test)', ascending=False).reset_index(drop=True)
        
        return df


# ─────────────────────────────────────────────────────────────
# Chựy trực tiếp: python model.py
# Dùng để test model độc lập không cần chạy Streamlit app
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(" Testing Car Price Predictor...\n")
    
    # Bước 1: Tải và xử lý dữ liệu
    loader = CarPriceDataLoader()
    X, y, features, df = loader.get_full_pipeline()
    
    # Bước 2: Train tất cả 6 models
    predictor = CarPricePredictor()
    predictor.train(X, y, feature_names=features,
                   model_types=['linear', 'ridge', 'lasso', 'svr', 'rf', 'gb'])
    
    # Bước 3: In kết quả đánh giá
    print("\n Model Performance:")
    metrics = predictor.get_metrics()
    print(f"  Train R\u00b2: {metrics['train']['r2']:.4f}")
    print(f"  Test  R\u00b2: {metrics['test']['r2']:.4f}")
    print(f"  Test MAE: {format_price(metrics['test']['mae'])}")
    
    # Bước 4: In độ quan trọng của từng feature
    print("\n Feature Importance (Top 5):")
    importance = predictor.get_feature_importance()
    for i, (feature, imp) in enumerate(list(importance.items())[:5], 1):
        print(f"  {i}. {feature:30s}: {imp*100:6.2f}%")
    
    # Bước 5: So sánh tất cả models
    print("\nSo sánh các Models:")
    comparison = predictor.compare_all_models()
    print(comparison.to_string(index=False))
    
    # Bước 6: Test dự đoán một xe cụ thể
    print("\n Sample Prediction:")
    print(" Toyota Vios 2020, 50K KM")
    
    # Lấy mã số Brand/Model từ dữ liệu thực tế để đảm bảo đúng mã
    toyota_encoded = (
        df[df['Brand'] == 'Toyota']['Brand_Encoded'].iloc[0]
        if 'Brand_Encoded' in df.columns else 0
    )
    vios_encoded = (
        df[df['Model'] == 'Vios']['Model_Encoded'].iloc[0]
        if 'Model_Encoded' in df.columns and len(df[df['Model'] == 'Vios']) > 0 else 0
    )
    
    # Đối tượng xe cần dự đoán
    test_car = {
        'Year':          2020,
        'Age':           2026 - 2020,   # Tuổi xe = Năm hiện tại - Năm SX
        'KM_Negative':   -50000,        # 50,000 km → âm (-50000)
        'Brand_Encoded': toyota_encoded,
        'Model_Encoded': vios_encoded
    }
    
    predicted_price = predictor.predict(test_car)
    print(f"   Predicted Price: {format_price(predicted_price)}")
