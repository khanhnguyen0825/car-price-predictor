"""
Streamlit Web Application để Dự Đoán Giá Xe

Cấu trúc application:
- Tab 1: Dự đoán giá - Input features và nhận kết quả dự đoán
- Tab 2: Biểu đồ phân tích - Visualizations của dữ liệu
- Tab 3: Hiệu suất model - Performance metrics và feature importance

Công nghệ:
- Streamlit: Web framework
- Plotly: Interactive charts
- Scikit-learn: Machine learning models
"""

import streamlit as st
import pandas as pd
import config
from data_loader import CarPriceDataLoader, format_price
from model import CarPricePredictor
import visualizations as viz

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #4F4F4F;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #FF6B35 0%, #F77F00 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.2);
    }
    .prediction-price {
        font-size: 3.5rem;
        font-weight: bold;
        color: white;
        margin: 0;
    }
    .prediction-label {
        font-size: 1.2rem;
        color: white;
        opacity: 0.9;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_train_model():
    """
    Load data và model (cached for performance)
    
    Thứ tự ưu tiên:
    1. Thử load pre-trained model từ file pickle (NHANH - 1 giây)
    2. Nếu không có file, train model mới (CHẬM - 1-2 phút)
    """
    import pickle
    import os
    
    model_path = 'models/best_model.pkl'
    
    # Kiểm tra xem có pre-trained model không
    if os.path.exists(model_path):
        with st.spinner('Đang load pre-trained model...'):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                predictor = data['predictor']
                df = data['df']
                X = data['X']
                y = data['y']
                features = data['features']
                loader = data['loader']
                
                st.success(f'Đã load model: {predictor.best_model_name} (R² = {predictor.best_score:.2%})')
                return predictor, df, X, y, features, loader
            except Exception as e:
                st.warning(f'Lỗi load model: {e}. Training model mới...')
    
    # Nếu không có file hoặc load lỗi, train model mới
    with st.spinner('Đang train model... (1-2 phút)'):
        loader = CarPriceDataLoader()
        X, y, features, df = loader.get_full_pipeline()
        
        predictor = CarPricePredictor()
        predictor.train(X, y, feature_names=features, model_types=['linear', 'ridge', 'lasso', 'svr', 'rf', 'gb'])
        
        st.info('Chạy `python train_model.py` lần sau để load nhanh hơn!')
        
    return predictor, df, X, y, features, loader


# Main app
def main():
    # Header
    st.markdown(f'<div class="main-header">{config.APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{config.APP_SUBTITLE}</div>', unsafe_allow_html=True)
    
    # Load data and model
    predictor, df, X, y, features, loader = load_data_and_train_model()
    
    # Sidebar - Input Controls
    st.sidebar.header("Thông Tin Xe")
    
    # Brand selection
    available_brands = sorted(df['Brand'].unique())
    selected_brand = st.sidebar.selectbox(
        "Hãng Xe",
        options=available_brands,
        index=available_brands.index('Toyota') if 'Toyota' in available_brands else 0
    )
    
    # Model selection (filtered by brand)
    available_models = sorted(df[df['Brand'] == selected_brand]['Model'].unique())
    selected_model = st.sidebar.selectbox(
        "Dòng Xe",
        options=available_models if len(available_models) > 0 else ['Unknown'],
        index=0
    )
    
    # Year slider
    selected_year = st.sidebar.slider(
        "Năm Sản Xuất",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=2020,
        step=1
    )
    
    # Kilometers input
    selected_km = st.sidebar.number_input(
        "Số KM Đã Đi",
        min_value=0,
        max_value=500000,
        value=50000,
        step=5000
    )
    
    # Calculate derived features
    age = config.CURRENT_YEAR - selected_year
    km_per_year = selected_km / (age + 1) if age >= 0 else 0
    
    # Get encodings
    brand_encoded = df[df['Brand'] == selected_brand]['Brand_Encoded'].iloc[0] if len(df[df['Brand'] == selected_brand]) > 0 else 0
    model_encoded = df[(df['Brand'] == selected_brand) & (df['Model'] == selected_model)]['Model_Encoded'].iloc[0] if len(df[(df['Brand'] == selected_brand) & (df['Model'] == selected_model)]) > 0 else 0
    
    # Predict button
    if st.sidebar.button("DỰ ĐOÁN GIÁ XE", type="primary", width='stretch'):
        # Use negative KM (matching training transformation)
        km_negative = -selected_km
        
        # Prepare features with NEGATIVE KM
        car_features = {
            'Year': selected_year,
            'Age': age,
            'KM_Negative': km_negative,
            'Brand_Encoded': brand_encoded,
            'Model_Encoded': model_encoded
        }
        
        # Make prediction
        predicted_price = predictor.predict(car_features)
        
        # Store in session state
        st.session_state['predicted_price'] = predicted_price
        st.session_state['car_info'] = {
            'Brand': selected_brand,
            'Model': selected_model,
            'Year': selected_year,
            'KM': selected_km
        }
    
    # Display info
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Thông Tin Dataset:**
    - Tổng số xe: {len(df):,}
    - Số hãng: {df['Brand'].nunique()}
    - Số dòng xe: {df['Model'].nunique()}
    - R² Score: **{predictor.get_metrics()['test']['r2']:.2%}**
    """)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dự Đoán", 
        "Biểu Đồ", 
        "Hiệu Suất Model", 
        "Dữ Liệu",
        "Features"
    ])
    
    # Tab 1: Prediction Results
    with tab1:
        st.header("Kết Quả Dự Đoán Giá Xe")
        
        if 'predicted_price' in st.session_state:
            predicted_price = st.session_state['predicted_price']
            car_info = st.session_state['car_info']
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Giá Dự Đoán</div>
                <div class="prediction-price">{format_price(predicted_price)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Car info summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Hãng Xe", car_info['Brand'])
            with col2:
                st.metric("Dòng Xe", car_info['Model'])
            with col3:
                st.metric("Năm SX", car_info['Year'])
            with col4:
                st.metric("Số KM", f"{car_info['KM']:,}")
            
            # Similar cars comparison
            st.subheader("Xe Tương Tự")
            
            # Filter by BOTH Brand AND Model (plus similar year)
            similar_cars = df[
                (df['Brand'] == car_info['Brand']) & 
                (df['Model'] == car_info['Model']) &  # FIX: Added Model filter
                (df['Year'] >= car_info['Year'] - 2) & 
                (df['Year'] <= car_info['Year'] + 2)
            ].sort_values('Price_Million')
            
            # Select columns to display (including Link and Kilometers if available)
            display_cols = ['Name', 'Year', 'Price_Million']
            if 'Kilometers' in similar_cars.columns:
                display_cols.append('Kilometers')
            if 'Link' in similar_cars.columns:
                display_cols.append('Link')
            
            similar_cars = similar_cars[display_cols].head(5)
            
            if len(similar_cars) > 0:
                # Format price
                similar_cars['Giá'] = similar_cars['Price_Million'].apply(format_price)
                
                # Display as cards with links (more interactive)
                for idx, row in similar_cars.iterrows():
                    # Simple title: just Name (already contains year)
                    with st.expander(f"{row['Name']}"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Năm sản xuất**: {int(row['Year']) if pd.notna(row['Year']) else 'N/A'}")
                            st.write(f"**Giá**: {row['Giá']}")
                            
                            # Add KM info
                            if 'Kilometers' in row and pd.notna(row['Kilometers']):
                                st.write(f"**Số km đã đi**: {int(row['Kilometers']):,} km")
                            
                            # Add product link if available
                            if 'Link' in row and pd.notna(row['Link']) and row['Link'].strip():
                                st.markdown(f"[Xem chi tiết sản phẩm]({row['Link']})")
                        
                        with col2:
                            # Try to display image from link (if it's an image URL or has og:image)
                            # For now just show placeholder, can enhance later with web scraping
                            pass
            else:
                st.info(f"Không tìm thấy xe {car_info['Brand']} {car_info['Model']} tương tự trong dataset.")

                
        else:
            st.info("Vui lòng nhập thông tin xe ở sidebar và nhấn nút **DỰ ĐOÁN GIÁ XE**")
    
    # Tab 2: Visualizations
    with tab2:
        st.header("Biểu Đồ Phân Tích")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price vs Year
            predicted_price = st.session_state.get('predicted_price')
            predicted_year = st.session_state.get('car_info', {}).get('Year')
            
            fig1 = viz.create_price_vs_year_scatter(
                df.sample(min(1000, len(df))),  # Sample for performance
                predicted_year, 
                predicted_price
            )
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            # Brand comparison
            fig2 = viz.create_brand_comparison_box(df, top_n=10)
            st.plotly_chart(fig2, width='stretch')
        
        # Price distribution
        fig3 = viz.create_price_distribution(df, predicted_price)
        st.plotly_chart(fig3, width='stretch')
        
        # ===== NEW ADVANCED CHARTS =====
        st.subheader("Phân Tích Nâng Cao")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # KM vs Price
            car_info = st.session_state.get('car_info', {})
            fig4 = viz.create_km_vs_price_scatter(
                df,
                car_info.get('KM') if predicted_price else None,
                predicted_price
            )
            st.plotly_chart(fig4, width='stretch')
        
        with col4:
            # Top Models chart
            fig5 = viz.create_top_models_chart(df, top_n=10)
            st.plotly_chart(fig5, width='stretch')
        
        # Price Trend by Year
        fig6 = viz.create_price_trend_by_year(df)
        st.plotly_chart(fig6, width='stretch')
        
        # Age Depreciation
        fig7 = viz.create_age_depreciation_chart(df)
        st.plotly_chart(fig7, width='stretch')

    
    # Tab 3: Model Performance
    with tab3:
        st.header("Hiệu Suất Model")
        
        metrics = predictor.get_metrics()
        
        # Metrics cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "R² Score (Test)",
                f"{metrics['test']['r2']:.2%}",
                delta=f"{(metrics['test']['r2'] - metrics['train']['r2'])*100:.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "MAE (Test)",
                format_price(metrics['test']['mae']),
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "RMSE (Test)",
                format_price(metrics['test']['rmse']),
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("So Sánh Models")
            comparison_df = predictor.compare_all_models()
            st.dataframe(comparison_df, width='stretch', hide_index=True)
        
        with col2:
            st.subheader("Biểu Đồ So Sánh")
            fig = viz.create_model_comparison_chart(comparison_df)
            st.plotly_chart(fig, width='stretch')
        
        # Actual vs Predicted
        st.subheader("Actual vs Predicted")
        fig_scatter = viz.create_actual_vs_predicted_scatter(predictor.y_test, predictor.y_pred_test)
        st.plotly_chart(fig_scatter, width='stretch')
        
        # Residual plot
        st.subheader("Residual Analysis")
        fig_residual = viz.create_residual_plot(predictor.y_test, predictor.y_pred_test)
        st.plotly_chart(fig_residual, width='stretch')
        
        st.markdown("---")
        
        # ===== NEW ADVANCED CHARTS =====
        st.subheader(" Phân Tích Nâng Cao")
        
        # Row 1: Error Distribution + Model Comparison Bar
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Phân Bố Sai Số**")
            fig_error_dist = viz.create_error_distribution_chart(predictor.y_test, predictor.y_pred_test)
            st.plotly_chart(fig_error_dist, width='stretch')
            
            # Explanation
            st.info("""
             **Cách đọc biểu đồ:**
            - Phân bố tập trung ở 0 = model chính xác
            - Lệch trái/phải = model có bias (over/under predict)
            - Đỉnh nhọn = predictions nhất quán
            """)
        
        with col2:
            st.markdown("**So Sánh Models (Bar Chart)**")
            comparison_df = predictor.compare_all_models()
            fig_model_bar = viz.create_model_comparison_bar_chart(comparison_df)
            st.plotly_chart(fig_model_bar, width='stretch')
            
            # Show best model
            best = comparison_df.iloc[0]
            st.success(f"🏆 **Best Model**: {best['Model']} (R² = {best['R² (Test)']:.2%})")
        
        # Row 2: Correlation Heatmap + Prediction Intervals
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Ma Trận Tương Quan Features**")
            fig_corr = viz.create_correlation_heatmap(df, features)
            st.plotly_chart(fig_corr, width='stretch')
            
            # Explanation
            st.info("""
             **Cách đọc:**
            - Đỏ (+1): Tương quan dương mạnh
            - Xanh (-1): Tương quan âm mạnh  
            - Trắng (0): Không tương quan
            """)
        
        with col4:
            st.markdown("**Độ Chính Xác Từng Dự Đoán**")
            fig_intervals = viz.create_prediction_intervals_chart(
                predictor.y_test, 
                predictor.y_pred_test,
                sample_size=100
            )
            st.plotly_chart(fig_intervals, width='stretch')
            
            # Stats
            import numpy as np
            errors = np.abs(predictor.y_pred_test - predictor.y_test)
            st.metric("Sai số trung bình (MAE)", f"{np.mean(errors):.0f} triệu VNĐ")
    
    # Tab 4: Data Explorer
    with tab4:
        st.header("Khám Phá Dữ Liệu")
        
        # Statistics
        st.subheader("Thống Kê Tổng Quan")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng Số Xe", f"{len(df):,}")
        with col2:
            st.metric("Giá Trung Bình", format_price(df['Price_Million'].mean()))
        with col3:
            st.metric("Giá Thấp Nhất", format_price(df['Price_Million'].min()))
        with col4:
            st.metric("Giá Cao Nhất", format_price(df['Price_Million'].max()))
        
        # Data table
        st.subheader("Dữ Liệu Mẫu")
        display_df = df[['Name', 'Brand', 'Year', 'Price_Million']].copy()
        display_df['Price'] = display_df['Price_Million'].apply(format_price)
        st.dataframe(
            display_df[['Name', 'Brand', 'Year', 'Price']].head(100),
            width='stretch',
            hide_index=True
        )
        
        # Download option
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download Dataset (CSV)",
            data=csv,
            file_name="car_price_data.csv",
            mime="text/csv",
        )
    
    # Tab 5: Feature Analysis
    with tab5:
        st.header("Phân Tích Features & Models")
        
        # Model Performance Summary (NEW!)
        st.subheader("So Sánh Performance 6 Models")
        
        # Get all model metrics
        comparison_df = predictor.compare_all_models()
        
        # Style the dataframe with conditional formatting
        def highlight_best(row):
            if row['Model'] == predictor.best_model_name:
                return ['background-color: #90EE90; font-weight: bold'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = comparison_df.style.apply(highlight_best, axis=1)
        
        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            column_config={
                "R² (Train)": st.column_config.ProgressColumn(
                    "R² (Train)",
                    format="%.4f",
                    min_value=0,
                    max_value=1,
                ),
                "R² (Test)": st.column_config.ProgressColumn(
                    "R² (Test)",
                    format="%.4f",
                    min_value=0,
                    max_value=1,
                ),
            }
        )
        
        # Performance insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best Model",
                predictor.best_model_name,
                f"R² = {predictor.best_score:.4f}"
            )
        
        with col2:
            best_mae = predictor.models[predictor.best_model_name]['mae']
            st.metric(
                "Best MAE", 
                f"{best_mae:.0f}M VNĐ",
                delta=f"~{(best_mae/df['Price_Million'].mean())*100:.1f}% error"
            )
        
        with col3:
            st.metric(
                "Models Trained",
                len(predictor.models),
                "algorithms"
            )
        
        st.markdown("---")
        
        # Feature importance
        importance = predictor.get_feature_importance()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Tầm Quan Trọng Features")
            fig = viz.create_feature_importance_chart(importance, top_n=6)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Chi Tiết")
            importance_df = pd.DataFrame([
                {'Feature': k, 'Importance': f"{v*100:.2f}%"}
                for k, v in importance.items()
            ])
            st.dataframe(importance_df, width='stretch', hide_index=True)
        
        # Feature descriptions
        st.subheader("Giải Thích Features")
        
        # Current features used
        st.markdown("""
        ### Features Hiện Tại (5 features)
        
        1. **Model_Encoded** (44.79%) - Dòng xe cụ thể
           - Ví dụ: Vios, Civic, CX-5, Fortuner...
           - Feature quan trọng nhất! Mỗi model có giá riêng
        
        2. **Brand_Encoded** (24.26%) - Hãng xe  
           - Ví dụ: Toyota, Honda, Mercedes, Mazda...
           - Premium brands (Mercedes, BMW) → giá cao hơn
        
        3. **Age** (10.84%) - Tuổi xe (năm)
           - Tính: `Năm hiện tại - Năm sản xuất`
           - Xe càng cũ → giá càng thấp
        
        4. **KM_Negative** (10.46%) - Số KM đã đi (đảo ngược)
           - Tính: `-Kilometers` để đảm bảo correlation đúng
           - KM cao → giá trị âm lớn → giá thấp
        
        5. **Year** (9.64%) - Năm sản xuất
           - Range: 2000-2024
           - Xe mới hơn thường giá cao hơn
        """)
        
        # Model comparison
        st.markdown("---")
        st.subheader("Thuật Toán Đang Dùng")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Models Đã Train (6 models):**
            - Linear Regression
            - Ridge Regression  
            - Lasso Regression
            - SVR (Support Vector Regression)
            - Random Forest (Best: R² = 0.86)
            - Gradient Boosting
            """)
        
        with col2:
            st.markdown("""
            **Tại Sao Random Forest Tốt Nhất?**
            - Xử lý non-linear relationships
            - Không cần feature scaling
            - Robust với outliers
            - Cung cấp feature importance
            - R² = 0.8618 (86% accuracy!)
            """)
        
        # Additional info
        st.info("""
        **Lưu ý**: Feature importance chỉ có với Random Forest/Gradient Boosting. 
        Linear models (Linear, Ridge, Lasso) dùng coefficients thay vì importance.
        """)

if __name__ == "__main__":
    main()
