import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import models
from models.fast_directional_forecaster import FastDirectionalForecaster

@st.cache_data
def load_and_process_data():
    """Load and process the COE data"""
    try:
        # Try primary data source
        data = pd.read_csv("data/COE_Extended_2002_2025.csv")
        # Convert date column if needed
        if 'date' not in data.columns and 'month' in data.columns:
            data['date'] = pd.to_datetime(data['month'], format='%Y-%m')
        else:
            data['date'] = pd.to_datetime(data['date'])
        return data, None
    except Exception as e:
        try:
            # Try alternative data source
            data = pd.read_csv("attached_assets/Results of COE Bidding Exercise - Results_1749485110740.csv")
            
            # Process the raw data
            data_processed = []
            for _, row in data.iterrows():
                try:
                    exercise = str(row.get('Exercise', ''))
                    category = str(row.get('Category', ''))
                    premium = str(row.get('Quota Premium', ''))
                    quota = str(row.get('Quota', ''))
                    
                    # Extract date info
                    month, year, bidding_no = extract_date_info(exercise)
                    
                    # Clean premium
                    premium_clean = clean_price(premium)
                    quota_clean = clean_numeric(quota)
                    
                    # Standardize category
                    category_clean = standardize_category(category)
                    
                    if premium_clean > 0 and category_clean:
                        data_processed.append({
                            'month': f"{year}-{month:02d}",
                            'date': pd.to_datetime(f"{year}-{month:02d}"),
                            'bidding_no': bidding_no,
                            'vehicle_class': category_clean,
                            'premium': premium_clean,
                            'quota': quota_clean
                        })
                except:
                    continue
            
            return pd.DataFrame(data_processed), None
        except Exception as e2:
            return None, None

def clean_price(price_str):
    """Clean price string to numeric value"""
    if pd.isna(price_str): 
        return 0
    import re
    cleaned = re.sub(r'[\$,"]', '', str(price_str))
    try:
        return int(cleaned)
    except:
        return 0

def clean_numeric(val):
    """Clean numeric string"""
    if pd.isna(val):
        return 0
    import re
    cleaned = re.sub(r'[,"]', '', str(val))
    try:
        return int(cleaned)
    except:
        return 0

def extract_date_info(exercise):
    """Extract date information from exercise string"""
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    parts = exercise.lower().split()
    
    # Find month
    month = 1
    for part in parts:
        if part in months:
            month = months[part]
            break
    
    # Find year
    year = 2000
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = int(part)
            break
    
    # Find bidding number
    bidding_no = 1
    for part in parts:
        if part.isdigit() and len(part) == 1:
            bidding_no = int(part)
            break
    
    return month, year, bidding_no

def standardize_category(cat):
    """Standardize category names"""
    cat_str = str(cat).upper()
    if 'A' in cat_str and ('CARS' in cat_str or 'SMALL' in cat_str):
        return 'Category A'
    elif 'B' in cat_str:
        return 'Category B'
    elif 'C' in cat_str and ('GOODS' in cat_str or 'LIGHT' in cat_str):
        return 'Category C'
    elif 'D' in cat_str:
        return 'Category D'
    elif 'E' in cat_str and ('OPEN' in cat_str or 'BIG' in cat_str):
        return 'Category E'
    return None

@st.cache_resource
def initialize_model():
    """Initialize and train the directional forecasting model"""
    data, processor = load_and_process_data()
    if data is not None:
        try:
            model = FastDirectionalForecaster()
            model.fit(data)
            return model
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return None
    return None

def main():
    # Configure page
    st.set_page_config(
        page_title="Singapore COE Price Predictor",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for modern UI
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    
    .metric-change.positive {
        color: #059669;
    }
    
    .metric-change.negative {
        color: #dc2626;
    }
    
    .info-banner {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
    }
    
    .controls-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .stMultiSelect > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide streamlit style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    data, processor = load_and_process_data()
    
    if data is None:
        st.error("Failed to load data. Please check your data file.")
        return
    
    # Initialize model
    model = initialize_model()
    
    if model is None:
        st.error("Failed to initialize prediction model.")
        return
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>Singapore COE Price Predictor</h1>
        <p>AI-powered predictions for Certificate of Entitlement bidding cycles</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data info banner
    total_records = len(data)
    date_range = f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}"
    
    st.markdown(f"""
    <div class="info-banner">
        <strong>📊 Dataset:</strong> {total_records:,} historical COE bidding records ({date_range})
    </div>
    """, unsafe_allow_html=True)
    
    # Controls section
    st.markdown('<div class="controls-section">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        available_categories = sorted(data['vehicle_class'].unique())
        selected_categories = st.multiselect(
            "Select COE Categories",
            available_categories,
            default=available_categories[:3],
            help="Choose which COE categories to analyze"
        )
    
    with col2:
        prediction_cycles = st.slider(
            "Prediction Cycles Ahead",
            min_value=1,
            max_value=6,
            value=3,
            help="Number of bidding cycles to predict"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_predictions = st.button("🔄 Refresh", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_categories:
        st.warning("Please select at least one COE category to view predictions.")
        return
    
    # Generate predictions
    predictions = {}
    try:
        predictions = model.predict(prediction_cycles)
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return
    
    # Current Predictions Section
    st.markdown("## 🔮 Current Predictions")
    
    if predictions:
        cols = st.columns(len(selected_categories))
        for i, category in enumerate(selected_categories):
            if category in predictions:
                with cols[i]:
                    latest_actual = data[data['vehicle_class'] == category]['premium'].iloc[-1]
                    
                    # Handle prediction format
                    if isinstance(predictions[category], dict) and 'mean' in predictions[category]:
                        pred_value = float(predictions[category]['mean'][0])
                        confidence_lower = float(predictions[category].get('lower', [pred_value * 0.9])[0])
                        confidence_upper = float(predictions[category].get('upper', [pred_value * 1.1])[0])
                    elif isinstance(predictions[category], list):
                        pred_raw = predictions[category][0]
                        pred_value = float(pred_raw.item() if hasattr(pred_raw, 'item') else pred_raw)
                        confidence_lower = pred_value * 0.9
                        confidence_upper = pred_value * 1.1
                    else:
                        continue
                    
                    change = ((pred_value - latest_actual) / latest_actual) * 100
                    change_class = "positive" if change >= 0 else "negative"
                    change_symbol = "▲" if change >= 0 else "▼"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="metric-label">{category}</div>
                        <div class="metric-value">${pred_value:,.0f}</div>
                        <div class="metric-change {change_class}">{change_symbol} {abs(change):.1f}%</div>
                        <small style="color: #6b7280;">Range: ${confidence_lower:,.0f} - ${confidence_upper:,.0f}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Extended Predictions Table
    st.markdown("## 📅 Extended Price Predictions (Next 6 Bidding Cycles)")
    
    if predictions:
        # Generate extended predictions for 6 cycles (3 months)
        extended_predictions = {}
        try:
            extended_predictions = model.predict(6)
        except:
            extended_predictions = predictions
        
        # Create date range for next 6 bidding cycles
        latest_date = data['date'].max()
        prediction_dates = []
        for i in range(6):
            # Assuming 2 cycles per month, alternate between 1st and 2nd cycle
            cycle_num = (i % 2) + 1
            month_offset = i // 2
            pred_date = latest_date + pd.DateOffset(months=month_offset+1)
            prediction_dates.append({
                'cycle': i + 1,
                'date': pred_date,
                'month': pred_date.strftime('%Y-%m'),
                'cycle_label': f"{pred_date.strftime('%Y-%m')} Cycle {cycle_num}"
            })
        
        # Create tabs for each category
        category_tabs = st.tabs(selected_categories)
        
        for tab_idx, category in enumerate(selected_categories):
            with category_tabs[tab_idx]:
                if category in extended_predictions:
                    # Get prediction values
                    if isinstance(extended_predictions[category], dict) and 'mean' in extended_predictions[category]:
                        pred_values = extended_predictions[category]['mean']
                    elif isinstance(extended_predictions[category], list):
                        pred_values = [float(p.item() if hasattr(p, 'item') else p) for p in extended_predictions[category]]
                    else:
                        continue
                    
                    # Create prediction table
                    prediction_data = []
                    latest_price = data[data['vehicle_class'] == category]['premium'].iloc[-1]
                    prev_price = latest_price
                    
                    for i, date_info in enumerate(prediction_dates[:len(pred_values)]):
                        pred_price = pred_values[i]
                        change_amount = pred_price - prev_price
                        change_percent = (change_amount / prev_price) * 100
                        
                        # Determine confidence level based on cycle distance
                        if i < 2:
                            confidence = "High"
                        elif i < 4:
                            confidence = "Medium"
                        else:
                            confidence = "Low"
                        
                        prediction_data.append({
                            'Cycle': date_info['cycle_label'],
                            'Predicted Premium': f"${pred_price:,.0f}",
                            'Change from Previous': f"${change_amount:+,.0f} ({change_percent:+.1f}%)",
                            'Confidence': confidence
                        })
                        
                        prev_price = pred_price
                    
                    # Display as styled dataframe
                    pred_df = pd.DataFrame(prediction_data)
                    
                    st.markdown("""
                    <style>
                    .prediction-table {
                        background: white;
                        border-radius: 8px;
                        overflow: hidden;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        pred_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Cycle": st.column_config.TextColumn("Bidding Cycle", width="medium"),
                            "Predicted Premium": st.column_config.TextColumn("Predicted Premium", width="medium"),
                            "Change from Previous": st.column_config.TextColumn("Change from Previous", width="medium"),
                            "Confidence": st.column_config.TextColumn("Confidence", width="small")
                        }
                    )
    
    # Model Performance Section
    st.markdown("## 🎯 Model Performance")
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### Performance Metrics")
    
    # Create performance metrics table
    performance_data = []
    for category in selected_categories:
        if hasattr(model, 'get_performance_metrics'):
            metrics = model.get_performance_metrics(category)
            if metrics:
                # Calculate additional metrics
                mape = metrics.get('mape', 0)
                rmse = metrics.get('rmse', 0)
                mae = metrics.get('mae', 0)
                r2 = metrics.get('r2', 0)
                direction_accuracy = metrics.get('direction_accuracy', 50)
                vol_correlation = metrics.get('volatility_correlation', 0)
                
                performance_data.append({
                    'Category': category,
                    'MAPE': f"{mape:.2f}%",
                    'RMSE': f"${rmse:,.0f}",
                    'MAE': f"${mae:,.0f}",
                    'R²': f"{r2:.3f}",
                    'Vol Correlation': f"{vol_correlation:.3f}",
                    'Direction Accuracy': f"{direction_accuracy:.1f}%"
                })
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        
        st.dataframe(
            perf_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="small"),
                "MAPE": st.column_config.TextColumn("MAPE", width="small", help="Mean Absolute Percentage Error"),
                "RMSE": st.column_config.TextColumn("RMSE", width="small", help="Root Mean Square Error"),
                "MAE": st.column_config.TextColumn("MAE", width="small", help="Mean Absolute Error"),
                "R²": st.column_config.TextColumn("R²", width="small", help="Coefficient of Determination"),
                "Vol Correlation": st.column_config.TextColumn("Vol Correlation", width="small", help="Volatility Correlation"),
                "Direction Accuracy": st.column_config.TextColumn("Direction Accuracy", width="small", help="Directional Prediction Accuracy")
            }
        )
        
        # Performance insights
        avg_direction_accuracy = sum([float(row['Direction Accuracy'].replace('%', '')) for row in performance_data]) / len(performance_data)
        best_category = max(performance_data, key=lambda x: float(x['Direction Accuracy'].replace('%', '')))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Direction Accuracy", f"{avg_direction_accuracy:.1f}%")
        with col2:
            st.metric("Best Performing Category", best_category['Category'])
        with col3:
            st.metric("Model Type", "Fast Directional Forecaster")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Historical Analysis and Charts
    st.markdown("## 📈 Historical Analysis & Trends")
    
    for category in selected_categories:
        st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
        
        # Filter data for category
        category_data = data[data['vehicle_class'] == category].copy()
        category_data = category_data.sort_values('date')
        
        # Create comprehensive chart
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=category_data['date'],
            y=category_data['premium'],
            mode='lines+markers',
            name='Historical Prices',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<extra></extra>'
        ))
        
        # Add predictions if available
        if category in predictions:
            last_date = category_data['date'].iloc[-1]
            pred_dates = [last_date + pd.DateOffset(months=2*i) for i in range(1, prediction_cycles + 1)]
            
            if isinstance(predictions[category], dict) and 'mean' in predictions[category]:
                pred_values = predictions[category]['mean']
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_values,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='#f59e0b', width=3, dash='dash'),
                    marker=dict(size=6, color='#f59e0b'),
                    hovertemplate='<b>Predicted: $%{y:,.0f}</b><br>%{x}<extra></extra>'
                ))
                
                # Add confidence intervals if available
                if 'lower' in predictions[category] and 'upper' in predictions[category]:
                    fig.add_trace(go.Scatter(
                        x=pred_dates + pred_dates[::-1],
                        y=predictions[category]['upper'] + predictions[category]['lower'][::-1],
                        fill='toself',
                        fillcolor='rgba(245, 158, 11, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            elif isinstance(predictions[category], list):
                pred_values = [float(p.item() if hasattr(p, 'item') else p) for p in predictions[category]]
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_values,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='#f59e0b', width=3, dash='dash'),
                    marker=dict(size=6, color='#f59e0b'),
                    hovertemplate='<b>Predicted: $%{y:,.0f}</b><br>%{x}<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=f'{category} - Price Trends & Predictions',
            title_font_size=18,
            title_font_weight='bold',
            xaxis_title='Date',
            yaxis_title='Price (SGD)',
            template='plotly_white',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key statistics
        col1, col2, col3, col4 = st.columns(4)
        
        latest_price = category_data['premium'].iloc[-1]
        avg_price = category_data['premium'].tail(6).mean()
        volatility = category_data['premium'].tail(6).std()
        
        # Get model performance metrics
        if hasattr(model, 'get_performance_metrics'):
            metrics = model.get_performance_metrics(category)
            if metrics:
                success_rate = metrics.get('direction_accuracy', 50)
            else:
                success_rate = 50
        else:
            success_rate = 50
        
        with col1:
            st.metric("Latest Price", f"${latest_price:,.0f}")
        
        with col2:
            st.metric("6-Cycle Average", f"${avg_price:,.0f}")
        
        with col3:
            st.metric("Volatility", f"${volatility:,.0f}")
        
        with col4:
            st.metric("Model Accuracy", f"{success_rate:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Information
    st.markdown("## ℹ️ Model Information")
    st.markdown("""
    <div class="info-banner">
        <strong>Prediction Model:</strong> Fast Directional Forecaster<br>
        <strong>Method:</strong> Technical momentum analysis with exponential smoothing<br>
        <strong>Features:</strong> Price trends, volatility patterns, and directional indicators<br>
        <strong>Update Frequency:</strong> Real-time with each new bidding cycle
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()