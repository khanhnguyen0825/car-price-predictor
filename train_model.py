"""
Script để train model và lưu vào file pickle

Chạy script này TRƯỚC KHI chạy Streamlit app để:
1. Train model một lần duy nhất
2. Lưu model vào models/best_model.pkl
3. App sẽ load model từ file này (nhanh hơn 100x)

Cách chạy:
    python train_model.py
"""

import pickle
import os
from data_loader import CarPriceDataLoader
from model import CarPricePredictor

def main():
    print("="*60)
    print(" SCRIPT TRAIN MODEL CHO CAR PRICE PREDICTOR")
    print("="*60)
    
    # Step 1: Load và preprocess data
    print("\n[1/3] Đang load và preprocess dữ liệu...")
    loader = CarPriceDataLoader()
    X, y, features, df = loader.get_full_pipeline()
    
    # Step 2: Train models
    print("\n[2/3] Đang train models...")
    predictor = CarPricePredictor()
    predictor.train(
        X, y,
        feature_names=features,
        model_types=['linear', 'ridge', 'lasso', 'svr', 'rf', 'gb']
    )
    
    # Step 3: Save model
    print("\n[3/3] Đang lưu model...")
    model_path = 'models/best_model.pkl'
    
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs('models', exist_ok=True)
    
    # Lưu toàn bộ predictor + loader + data
    with open(model_path, 'wb') as f:
        pickle.dump({
            'predictor': predictor,
            'loader': loader,
            'df': df,
            'X': X,
            'y': y,
            'features': features
        }, f)
    
    print(f"\n[SUCCESS] Model đã được lưu vào: {model_path}")
    print(f"[INFO] Best model: {predictor.best_model_name}")
    print(f"[INFO] R² Score: {predictor.best_score:.4f}")
    
    # Display metrics
    print("\n" + "="*60)
    print(" KẾT QUẢ TRAINING")
    print("="*60)
    
    metrics = predictor.get_metrics()
    print(f"\nTrain R²: {metrics['train']['r2']:.4f}")
    print(f"Test R²:  {metrics['test']['r2']:.4f}")
    print(f"Test MAE: {metrics['test']['mae']:.0f} triệu VNĐ")
    
    # Feature importance
    print("\nFeature Importance (Top 5):")
    importance = predictor.get_feature_importance()
    for i, (feature, imp) in enumerate(list(importance.items())[:5], 1):
        print(f"  {i}. {feature:30s}: {imp*100:6.2f}%")
    
    # Model comparison
    print("\nSo sánh các Models:")
    comparison = predictor.compare_all_models()
    print(comparison.to_string(index=False))
    
    print("\n" + "="*60)
    print(" HOÀN TẤT!")
    print("="*60)
    print("\nBạn có thể chạy Streamlit app bằng lệnh:")
    print("  python -m streamlit run app.py")
    print("\nApp sẽ load model từ file (rất nhanh, không train lại)")
    print("="*60)

if __name__ == "__main__":
    main()
