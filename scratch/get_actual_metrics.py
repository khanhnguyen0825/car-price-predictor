import pickle
import os
import sys
import io

# Set stdout to utf-8 for Vietnamese characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add root directory to sys.path to find 'model' and 'data_loader'
sys.path.append(os.getcwd())

model_path = r'models\best_model.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        
    predictor = data['predictor']
    df = data['df']
    
    print("--- MODEL SUMMARY ---")
    print(f"Best Model Name: {predictor.best_model_name}")
    print(f"R2 Score (Test): {predictor.best_score}")
    
    print("\n--- ALL MODELS PERFORMANCE ---")
    comparison_df = predictor.compare_all_models()
    print(comparison_df.to_string(index=False))
    
    print("\n--- DATASET STATS ---")
    print(f"Total records in clean df: {len(df)}")
    print(f"Price Range: {df['Price_Million'].min()} - {df['Price_Million'].max()}")
    print(f"Price Mean: {df['Price_Million'].mean()}")
    
    print("\n--- FEATURE IMPORTANCE ---")
    importance = predictor.get_feature_importance()
    for k, v in importance.items():
        print(f"{k}: {v}")
else:
    print(f"Error: Model file not found at {model_path}")
