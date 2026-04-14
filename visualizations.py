"""
Module Visualization cho Car Price Predictor

Chứa 13 hàm tạo biểu đồ Plotly, chia thành 2 nhóm:

[NHÓM 1] Phân tích dữ liệu (dùng trong Tab "Biểu Đồ Phân Tích"):
  - create_price_vs_year_scatter:  Giá xe theo Năm sản xuất
  - create_brand_comparison_box:   So sánh giá theo Hãng xe (box plot)
  - create_price_distribution:     Phân bố giá toàn thị trường
  - create_km_vs_price_scatter:    Giá xe theo Số KM đã đi
  - create_top_models_chart:       Top dòng xe phổ biến
  - create_price_trend_by_year:    Xu hướng giá theo năm
  - create_age_depreciation_chart: Khấu hao theo tuổi xe

[NHÓM 2] Đánh giá model (dùng trong Tab "Hiệu Suất Model"):
  - create_model_comparison_chart:       So sánh R² các models (1 cột)
  - create_feature_importance_chart:     Độ quan trọng của features
  - create_residual_plot:                Phân tích sai số
  - create_actual_vs_predicted_scatter:  Thực tế vs Dự đoán
  - create_metrics_gauge:                R² dạng đồng hồ
  - create_error_distribution_chart:     Phân bố sai số + đường chuẩn
  - create_model_comparison_bar_chart:   So sánh R² Train vs Test
  - create_correlation_heatmap:          Ma trận tương quan features
  - create_prediction_intervals_chart:   Độ chính xác từng dự đoán
"""

# ── Thư viện tạo biểu đồ tương tác ──────────────────────────────
import plotly.graph_objects as go  # Tạo biểu đồ tùy chỉnh chi tiết
import plotly.express as px         # Tạo biểu đồ nhanh từ DataFrame
import pandas as pd                 # Xử lý dữ liệu dạng bảng
import numpy as np                  # Tính toán số học

def create_price_vs_year_scatter(df, predicted_year=None, predicted_price=None):
    """
    Tạo scatter plot giá xe theo năm sản xuất
    
    Mỗi điểm = 1 xe, màu sắc theo hãng
    Nếu có dự đoán, hiển thị dưới dạng ngôi sao đỏ
    """
    
    fig = px.scatter(
        df, 
        x='Year', 
        y='Price_Million',
        color='Brand',
        title='Giá Xe Theo Năm Sản Xuất',
        labels={'Year': 'Năm Sản Xuất', 'Price_Million': 'Giá (triệu VNĐ)'},
        opacity=0.6,
        hover_data=['Name']
    )
    
    # Add user's prediction point if provided
    if predicted_year is not None and predicted_price is not None:
        fig.add_trace(go.Scatter(
            x=[predicted_year],
            y=[predicted_price],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            name='Dự Đoán Của Bạn'
        ))
    
    fig.update_layout(
        height=500,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def create_brand_comparison_box(df, top_n=10):
    """
    Tạo box plot so sánh phân bố giá theo hãng xe

    Box plot cho thấy:
    - Đường giữa hộp: giá trung vị (median)
    - Hộp: khoảng giá từ Q1 đến Q3 (50% dữ liệu)
    - Râu: khoảng giá tổng thể
    - Chấm ngoài: outliers (xe giá bất thường)

    Chỉ hiển thị top_n hãng có nhiều xe nhất để biểu đồ gọn
    """
    # Lọc top N hãng xe theo số lượng xe trong dataset
    top_brands = df['Brand'].value_counts().head(top_n).index
    df_filtered = df[df['Brand'].isin(top_brands)]

    fig = px.box(
        df_filtered,
        x='Brand',
        y='Price_Million',
        title=f'So Sánh Giá Xe Theo Hãng (Top {top_n})',
        labels={'Brand': 'Hãng Xe', 'Price_Million': 'Giá (triệu VNĐ)'},
        color='Brand'
    )

    fig.update_layout(
        height=500,
        showlegend=False,       # Ẩn legend vì màu = hãng xe đã rõ
        xaxis_tickangle=-45,     # Nghiêng nhãn trục X để dễ đọc
        template='plotly_white'
    )

    return fig


def create_model_comparison_chart(comparison_df):
    """
    Tạo bar chart đơn giản so sánh R² (Test) của các ML models

    Dùng trong Tab "Hiệu Suất Model" - hiển thị bên cạnh bảng so sánh
    Chỉ hiển thị R² Test (không phân biệt Train/Test)
    → Xem create_model_comparison_bar_chart() để so sánh cả Train và Test
    """
    fig = go.Figure()

    # Cột duy nhất: R² Score trên Test set
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['R² (Test)'],
        name='R² Score',
        marker_color='#FF6B35'  # Màu cam chủ đạo của project
    ))

    fig.update_layout(
        title='So Sánh Hiệu Suất Models',
        xaxis_title='Model',
        yaxis_title='R² Score',
        height=400,
        showlegend=True,
        template='plotly_white'
    )

    return fig


def create_feature_importance_chart(importance_dict, top_n=6):
    """
    Tạo horizontal bar chart hiển thị độ quan trọng của features

    Giá trị importance = % đóng góp của feature vào quyết định của model
    Tổng tất cả features = 100%

    Kết quả project: Model_Encoded (40%) > Brand_Encoded (29%) > KM (14%)
    → Dòng xe + Hãng xe quyết định ~70% giá
    """
    # Lấy top N features quan trọng nhất
    items = list(importance_dict.items())[:top_n]
    features = [x[0] for x in items]
    importances = [x[1] * 100 for x in items]  # Chuyển từ [0,1] sang %

    # Đổi tên kỹ thuật sang tên dễ đọc
    feature_map = {
        'Model_Encoded': 'Dòng Xe',
        'Brand_Encoded': 'Hãng Xe',
        'Year':          'Năm SX',
        'Age':           'Tuổi Xe',
        'KM_Negative':   'Số KM',
        'Kilometers':    'Số KM',
        'KM_Per_Year':   'KM/Năm'
    }
    readable_features = [feature_map.get(f, f) for f in features]

    # Horizontal bar chart (nằm ngang) để dễ đọc tên feature
    fig = go.Figure(go.Bar(
        x=importances,
        y=readable_features,
        orientation='h',                               # Nằm ngang
        marker_color='#FF6B35',
        text=[f'{x:.1f}%' for x in importances],       # Hiển thị % trên cột
        textposition='outside'
    ))

    fig.update_layout(
        title='Tầm Quan Trọng Của Features',
        xaxis_title='Tầm Quan Trọng (%)',
        yaxis_title='Features',
        height=400,
        template='plotly_white'
    )

    return fig


def create_residual_plot(y_true, y_pred):
    """
    Tạo Residual Plot - biểu đồ phân tích sai số

    Residual = Giá thực tế - Giá dự đoán

    Cách đọc:
    - Điểm gần đường y=0: dự đoán chính xác
    - Điểm xa đường y=0:  sai số lớn
    - Màu đỏ: sai số lớn | Màu xanh: sai số nhỏ
    - Phân bố đều 2 phía: model không có bias
    - Phân bố lệch 1 phía: model có xu hướng over/under predict
    """
    # Tính residuals: sai lệch giữa thực tế và dự đoán
    residuals = y_true - y_pred

    fig = go.Figure()

    # Scatter plot: trục X = giá dự đoán, trục Y = sai số
    # Màu điểm theo |residual|: đỏ = sai nhiều, xanh = sai ít
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            color=np.abs(residuals),    # Màu theo độ lớn sai số
            colorscale='RdYlGn_r',      # Đỏ-Vàng-Xanh ngược (đỏ = lớn)
            showscale=True,
            colorbar=dict(title='Abs Error')
        ),
        name='Residuals'
    ))

    # Đường ngang y=0: biểu thị dự đoán hoàn hảo
    # Điểm nằm trên: model dự đoán thấp hơn thực tế
    # Điểm nằm dưới: model dự đoán cao hơn thực tế
    fig.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Zero Error')

    fig.update_layout(
        title='Residual Plot - Phân Tích Sai Số',
        xaxis_title='Giá Dự Đoán (triệu VNĐ)',
        yaxis_title='Residual (Thực Tế - Dự Đoán)',
        height=500,
        template='plotly_white'
    )

    return fig


def create_price_distribution(df, predicted_price=None):
    """
    Tạo histogram phân bố giá xe trên toàn thị trường

    Giúp user biết:
    - Phần lớn xe tập trung ở mức giá nào
    - Xe dự đoán đắt hay rẻ so với thị trường
    - Có nhiều xe giá cao bất thường không

    Nếu có predicted_price: vẽ thêm đường đứng đỏ để so sánh
    """
    fig = px.histogram(
        df,
        x='Price_Million',
        nbins=50,                           # Chia thành 50 khoảng giá
        title='Phân Bố Giá Xe Trên Thị Trường',
        labels={'Price_Million': 'Giá (triệu VNĐ)', 'count': 'Số Lượng'},
        color_discrete_sequence=['#FF6B35']
    )

    # Nếu có giá dự đoán, vẽ đường đứng để user biết xe của họ ở đâu
    if predicted_price is not None:
        fig.add_vline(
            x=predicted_price,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Dự Đoán: {predicted_price:.0f}M',
            annotation_position='top'
        )

    fig.update_layout(height=400, template='plotly_white')
    return fig


def create_actual_vs_predicted_scatter(y_true, y_pred):
    """
    Tạo scatter plot so sánh Giá Thực Tế vs Giá Dự Đoán

    Cách đọc:
    - Điểm nằm trên đường đỏ: model dự đoán cao hơn thực tế
    - Điểm nằm dưới đường đỏ: model dự đoán thấp hơn thực tế
    - Điểm nằm trên đường đỏ = model hoàn hảo
    - Các điểm càng gần đường đỏ → model càng chính xác
    """
    fig = go.Figure()

    # Vẽ các điểm predictions
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(size=6, color='#FF6B35', opacity=0.6),
        name='Predictions'
    ))

    # Đường Perfect Prediction: y = x (dự đoán = thực tế)
    # Model tốt sẽ có các điểm bám sát đường này
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))

    fig.update_layout(
        title='Giá Thực Tế vs Dự Đoán',
        xaxis_title='Giá Thực Tế (triệu VNĐ)',
        yaxis_title='Giá Dự Đoán (triệu VNĐ)',
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    return fig


def create_metrics_gauge(r2_score, title="R\u00b2 Score"):
    """
    Tạo đồng hồ đo (gauge chart) hiển thị R\u00b2 Score

    Dải đo:
      0-50%:  Xấu (màu xám nhạt)
      50-80%: Khá (màu xám)
      80%+:   Tốt (màu cam - model của project ở đây)
      90%+:   Đường ngưỡng đỏ (mục tiêu hướng đến)

    delta: so sánh với mục chuẩn 80% (hiển thị +6.54% với model này)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",    # Hiển thị đồng hồ + số + độ lệch
        value=r2_score * 100,         # Chuyển [0,1] sang %
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 80},      # So sánh với mục chuẩn 80%
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#FF6B35"},  # Màu thanh của đồng hồ
            'steps': [
                {'range': [0, 50],  'color': "lightgray"},  # Xấu
                {'range': [50, 80], 'color': "gray"}         # Khá
            ],
            'threshold': {  # Đường ngưỡng màu đỏ ở 90%
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300, template='plotly_white')
    return fig


# ─────────────────────────────────────────────────────────────
# NHOM 1 BỔ SUNG: Biểu đồ phân tích dữ liệu nâng cao
# ─────────────────────────────────────────────────────────────

def create_km_vs_price_scatter(df, predicted_km=None, predicted_price=None):
    """
    Tạo scatter plot Giá theo Số KM đã đi

    Thể hiện mối quan hệ: KM tăng → giá giảm (tương quan âm)
    Màu sắc theo hãng để phân biệt thị phần giá của từng hãng
    Sample 2000 xe để biểu đồ không bị chận với 9000+ điểm
    """
    # Lấy mẫu ngẫu nhiên tối đa 2000 xe để tăng tốc độ render
    df_sample = df.sample(min(2000, len(df)))

    fig = px.scatter(
        df_sample,
        x='Kilometers',
        y='Price_Million',
        color='Brand',
        title='Giá Xe Theo Số Km Đã Đi',
        labels={'Kilometers': 'Số Km Đã Đi', 'Price_Million': 'Giá (triệu VNĐ)'},
        opacity=0.6,
        hover_data=['Name', 'Year']
    )

    # Thêm điểm dự đoán của user nếu có
    if predicted_km is not None and predicted_price is not None:
        fig.add_trace(go.Scatter(
            x=[predicted_km],
            y=[predicted_price],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            name='Dự Đoán Của Bạn'
        ))

    fig.update_layout(height=500, hovermode='closest', template='plotly_white')
    return fig


def create_top_models_chart(df, top_n=10):
    """
    Tạo bar chart hiển thị top dòng xe theo giá trung bình

    Chọn top_n dòng xe có NHIỀU XE NHẤT trong dataset
    (không phải đắt nhất, mà là phổ biến nhất)
    → Dễ thấy dòng xe nào đang được rao bán nhiều nhất trên thị trường
    """
    # Lấy top N dòng xe theo số lượng (count)
    top_models = df['Model'].value_counts().head(top_n).index
    df_filtered = df[df['Model'].isin(top_models)]

    # Tính giá trung bình và số lượng xe mỗi dòng
    model_stats = df_filtered.groupby('Model').agg({
        'Price_Million': ['mean', 'count']
    }).reset_index()
    model_stats.columns = ['Model', 'Avg_Price', 'Count']
    model_stats = model_stats.sort_values('Avg_Price', ascending=False)  # Sắp xếp giảm dần

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_stats['Model'],
        y=model_stats['Avg_Price'],
        marker_color='#FF6B35',
        # Hiển thị cả giá TB và số xe trên mỗi cột
        text=[f'{x:.0f}M<br>({c} xe)' for x, c in zip(model_stats['Avg_Price'], model_stats['Count'])],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Giá TB: %{y:.0f} triệu VNĐ<extra></extra>'
    ))

    fig.update_layout(
        title=f'Top {top_n} Dòng Xe Phổ Biến - Giá Trung Bình',
        xaxis_title='Dòng Xe',
        yaxis_title='Giá Trung Bình (triệu VNĐ)',
        height=500,
        xaxis_tickangle=-45,
        template='plotly_white'
    )
    return fig


def create_price_trend_by_year(df):
    """
    Tạo line chart xu hướng giá xe theo năm sản xuất

    Vẽ 2 đường:
    - Giá Trung Bình (mean):  bị ảnh hưởng bởi xe giá cao bất thường
    - Giá Trung Vị (median): ổn định hơn, ít bị outlier ảnh hưởng

    Chỉ hiển thị năm có ít nhất 10 xe (loại năm ít dữ liệu)
    """
    # Tính giá TB, trung vị, và số xe theo từng năm
    year_stats = df.groupby('Year').agg({
        'Price_Million': ['mean', 'median', 'count']
    }).reset_index()
    year_stats.columns = ['Year', 'Mean', 'Median', 'Count']

    # Loại các năm ít dữ liệu (< 10 xe) để tránh đường bị nhp nhô
    year_stats = year_stats[year_stats['Count'] >= 10]

    fig = go.Figure()

    # Đường giá trung bình (cam, nét đặc)
    fig.add_trace(go.Scatter(
        x=year_stats['Year'],
        y=year_stats['Mean'],
        mode='lines+markers',
        name='Giá Trung Bình',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=8)
    ))

    # Đường giá trung vị (xanh, nét đứt)
    fig.add_trace(go.Scatter(
        x=year_stats['Year'],
        y=year_stats['Median'],
        mode='lines+markers',
        name='Giá Trung Vị',
        line=dict(color='#004E89', width=3, dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Xu Hướng Giá Xe Theo Năm Sản Xuất',
        xaxis_title='Năm Sản Xuất',
        yaxis_title='Giá (triệu VNĐ)',
        height=450,
        hovermode='x unified',  # Hover hiển thị cả 2 đường cùng lúc
        showlegend=True,
        template='plotly_white'
    )
    return fig


def create_age_depreciation_chart(df):
    """
    Tạo scatter plot khấu hao giá xe theo tuổi xe

    Thể hiện mối quan hệ: Tuổi tăng → giá giảm
    (giá trị thực tế phục thuộc nhiều vào hãng - màu sắc)

    Sample 1500 xe để biểu đồ không bị chầm
    """
    df_sample = df.sample(min(1500, len(df)))  # Lấy mẫu ngẫu nhiên

    fig = px.scatter(
        df_sample,
        x='Age',
        y='Price_Million',
        color='Brand',
        title='Khấu Hao Giá Xe Theo Tuổi',
        labels={'Age': 'Tuổi Xe (năm)', 'Price_Million': 'Giá (triệu VNĐ)'},
        opacity=0.5,
        hover_data=['Name']
    )

    fig.update_layout(height=500, hovermode='closest', template='plotly_white')
    return fig


# ============ MODEL PERFORMANCE CHARTS ============

def create_error_distribution_chart(y_true, y_pred):
    """
    Tạo histogram phân bố sai số dự đoán
    
    Giúp xác định:
    - Model có bias không (phân bố lệch trái/phải)
    - Có outliers không (errors cực lớn)
    - Phân bố có chuẩn (normal) không
    """
    # ── Bước 1: Tính errors = y_pred - y_true ────────────
    # Dương: model predict CAO hơn thực tế
    # Âm:     model predict THẤP hơn thực tế
    errors = y_pred - y_true

    fig = go.Figure()

    # ── Bước 2: Vẽ histogram tần suất sai số ───────────
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Sai số',
        marker_color='rgb(55, 83, 109)',
        opacity=0.7
    ))

    # ── Bước 3: Vẽ đường cong phân phối chuẩn để so sánh ──
    # Nếu histogram bám sát đường đỏ → sai số phân bố chuẩn (tốt)
    import numpy as np
    from scipy import stats
    mean = np.mean(errors)
    std  = np.std(errors)
    x_range = np.linspace(errors.min(), errors.max(), 100)
    # Scale đường cong theo số mẫu và độ rộng của mỗi bin
    y_range = stats.norm.pdf(x_range, mean, std) * len(errors) * (errors.max() - errors.min()) / 50

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name='Phân phối chuẩn',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Phân Bố Sai Số Dự Đoán',
        xaxis_title='Sai số (triệu VNĐ)',
        yaxis_title='Tần suất',
        showlegend=True,
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_model_comparison_bar_chart(comparison_df):
    """
    Tạo bar chart so sánh trực quan giữa các models
    
    Hiển thị R² score của tất cả models để dễ so sánh
    """
    fig = go.Figure()
    
    # R² Train bars
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['R² (Train)'],
        name='R² Train',
        marker_color='lightblue',
        text=comparison_df['R² (Train)'].apply(lambda x: f'{x:.3f}'),
        textposition='auto'
    ))
    
    # R² Test bars
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['R² (Test)'],
        name='R² Test',
        marker_color='darkblue',
        text=comparison_df['R² (Test)'].apply(lambda x: f'{x:.3f}'),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='So Sánh R² Score Giữa Các Models',
        xaxis_title='Model',
        yaxis_title='R² Score',
        barmode='group',
        height=450,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_correlation_heatmap(df, features):
    """
    Tạo Heatmap ma trận tương quan giữa các features

    Cách đọc màu sắc:
    - Đỏ (→ +1): Tương quan dương mạnh (cùng tăng)
    - Xanh (→ -1): Tương quan âm mạnh (ngược chiều)
    - Trắng (→ 0): Không có tương quan

    Ví dụ kết quả: Year và Age có tương quan -1.0
    (vì Age = 2026 - Year, chúng hoàn toàn ngược nhau)
    """
    # Chỉ lấy các cột là số trong nhóm features được chọn
    numeric_features = df[features].select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_features].corr()  # Tính Pearson correlation

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,       # Ma trận giá trị tương quan
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',          # Đỏ-Trắng-Xanh
        zmid=0,                     # Trung tâm màu tại 0
        text=corr_matrix.values,
        texttemplate='%{text:.2f}', # Hiển thị số trong ô
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title='Ma Trận Tương Quan Giữa Các Features',
        height=500,
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'},
        template='plotly_white'
    )
    return fig


def create_prediction_intervals_chart(y_true, y_pred, sample_size=100):
    """
    Tạo chart so sánh giá thực tế vs dự đoán với error bars

    Error bars = ðộ lớn sai số tại mỗi điểm
    - Error bar ngắn: dự đoán chính xác
    - Error bar dài:  dự đoán sai nhiều

    Lấy mẫu ngẫu nhiên để tránh biểu đồ quá đông điểm
    """
    import numpy as np

    # Lấy sample_size chỉ số ngẫu nhiên và sắp xếp tăng dần
    indices = np.random.choice(len(y_true), min(sample_size, len(y_true)), replace=False)
    indices = sorted(indices)

    # Lấy dữ liệu tại các vị trí đã chọn
    y_true_sample = y_true.iloc[indices].values if hasattr(y_true, 'iloc') else y_true[indices]
    y_pred_sample = y_pred[indices]
    errors = np.abs(y_pred_sample - y_true_sample)  # Độ lớn sai số (luôn dương)

    fig = go.Figure()

    # Điểm xanh: giá thực tế
    fig.add_trace(go.Scatter(
        x=list(range(len(indices))),
        y=y_true_sample,
        mode='markers',
        name='Giá thực tế',
        marker=dict(color='green', size=8)
    ))

    # Điểm đỏ: giá dự đoán + error bar biểu thị mức độ sai
    fig.add_trace(go.Scatter(
        x=list(range(len(indices))),
        y=y_pred_sample,
        mode='markers',
        name='Giá dự đoán',
        marker=dict(color='red', size=8),
        error_y=dict(
            type='data',
            array=errors,         # Chiều cao error bar = độ lớn sai số
            visible=True,
            color='rgba(255, 0, 0, 0.3)'
        )
    ))

    fig.update_layout(
        title=f'Độ Chính Xác Dự Đoán ({sample_size} mẫu ngẫu nhiên)',
        xaxis_title='Mẫu',
        yaxis_title='Giá (triệu VNĐ)',
        height=450,
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    return fig

