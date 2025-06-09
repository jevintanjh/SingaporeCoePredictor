import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from models.simple_ensemble import SimpleCOEModel
from utils.data_processor import DataProcessor
from utils.visualizations import create_historical_chart, create_prediction_chart, create_performance_chart
from utils.metrics import calculate_metrics, format_metrics

# Page configuration
st.set_page_config(
    page_title="Singapore COE Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🚗 Singapore COE Price Prediction Dashboard")
st.markdown("Advanced ML ensemble model for predicting Certificate of Entitlement prices across all vehicle categories")

@st.cache_data
def load_and_process_data():
    """Load and process the COE data"""
    try:
        processor = DataProcessor()
        data = processor.load_data('data/COEBiddingResultsPrices_1749430265007.csv')
        processed_data = processor.preprocess_data(data)
        return processed_data, processor
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_resource
def initialize_model():
    """Initialize and train the ensemble model"""
    data, processor = load_and_process_data()
    if data is not None:
        try:
            model = SimpleCOEModel()
            model.fit(data)
            return model
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return None
    return None

def main():
    # Load data and initialize model
    data, processor = load_and_process_data()
    model = initialize_model()
    
    if data is None or model is None:
        st.error("Failed to load data or initialize model. Please check your data file.")
        return
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Category selection
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    selected_categories = st.sidebar.multiselect(
        "Select COE Categories",
        categories,
        default=categories
    )
    
    # Time range selection for historical data
    min_date = data['date'].min()
    max_date = data['date'].max()
    
    st.sidebar.subheader("Historical Data Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=max_date - pd.DateOffset(months=12),
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Prediction horizon
    prediction_cycles = st.sidebar.slider(
        "Prediction Cycles Ahead",
        min_value=1,
        max_value=6,
        value=3,
        help="Number of bidding cycles to predict"
    )
    
    # Filter data based on selection
    filtered_data = data[
        (data['date'] >= pd.to_datetime(start_date)) &
        (data['date'] <= pd.to_datetime(end_date)) &
        (data['vehicle_class'].isin(selected_categories))
    ]
    
    # Main dashboard layout
    if len(selected_categories) > 0:
        # Current predictions section
        st.header("🔮 Current Predictions")
        
        # Generate predictions
        predictions = {}
        try:
            predictions = model.predict(prediction_cycles)
            
            if predictions:
                # Display predictions in cards
                cols = st.columns(len(selected_categories))
                for i, category in enumerate(selected_categories):
                    if category in predictions:
                        with cols[i]:
                            latest_actual = data[data['vehicle_class'] == category]['premium'].iloc[-1]
                            pred_value = predictions[category]['mean'][0]
                            confidence_lower = predictions[category]['lower'][0]
                            confidence_upper = predictions[category]['upper'][0]
                            
                            change = ((pred_value - latest_actual) / latest_actual) * 100
                            
                            st.metric(
                                label=category,
                                value=f"${pred_value:,.0f}",
                                delta=f"{change:+.1f}%"
                            )
                            
                            st.caption(f"95% CI: ${confidence_lower:,.0f} - ${confidence_upper:,.0f}")
                    else:
                        with cols[i]:
                            st.metric(
                                label=category,
                                value="N/A",
                                help="Insufficient data for prediction"
                            )
            else:
                st.warning("Model predictions are being generated. Please wait...")
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            st.info("Showing historical data only")
        
        # Historical trends and predictions
        st.header("📈 Historical Trends & Predictions")
        
        for category in selected_categories:
            st.subheader(f"{category} Analysis")
            
            # Get category-specific data
            category_data = filtered_data[filtered_data['vehicle_class'] == category].copy()
            
            if len(category_data) > 0:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Historical and prediction chart
                    try:
                        if predictions and category in predictions and len(category_data) > 0:
                            # Ensure we have at least 6 months of historical data
                            if len(category_data) >= 12:  # At least 12 bidding cycles (6 months)
                                fig = create_prediction_chart(
                                    category_data, 
                                    predictions[category], 
                                    category,
                                    prediction_cycles
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"Insufficient historical data for {category} predictions. Showing available data.")
                                fig = create_historical_chart(category_data, category, show_volume=False)
                                st.plotly_chart(fig, use_container_width=True)
                        elif len(category_data) > 0:
                            fig = create_historical_chart(category_data, category, show_volume=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No data available for {category}")
                    except Exception as e:
                        st.error(f"Chart error for {category}: {str(e)}")
                        # Show basic data table as fallback
                        if len(category_data) > 0:
                            st.dataframe(category_data[['date', 'premium', 'quota', 'bids_received']].tail(10))
                
                with col2:
                    # Key statistics
                    st.markdown("**Key Statistics**")
                    
                    recent_data = category_data.tail(6)  # Last 6 cycles
                    
                    avg_premium = recent_data['premium'].mean()
                    volatility = recent_data['premium'].std()
                    avg_success_rate = (recent_data['bids_success'] / recent_data['bids_received']).mean() * 100
                    avg_bid_quota_ratio = (recent_data['bids_received'] / recent_data['quota']).mean()
                    
                    st.metric("Avg Premium (6 cycles)", f"${avg_premium:,.0f}")
                    st.metric("Volatility (6 cycles)", f"${volatility:,.0f}")
                    st.metric("Success Rate", f"{avg_success_rate:.1f}%")
                    st.metric("Bid-to-Quota Ratio", f"{avg_bid_quota_ratio:.2f}")
        
        # Model performance section
        st.header("🎯 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            try:
                # Calculate performance metrics for each category
                performance_data = []
                for category in categories:
                    category_data = data[data['vehicle_class'] == category].copy()
                    if len(category_data) > 20:  # Ensure sufficient data
                        # Get model performance for this category
                        metrics = model.get_performance_metrics(category)
                        if metrics:
                            performance_data.append({
                                'Category': category,
                                'MAPE': f"{metrics.get('mape', 0):.2f}%",
                                'RMSE': f"${metrics.get('rmse', 0):,.0f}",
                                'MAE': f"${metrics.get('mae', 0):,.0f}",
                                'R²': f"{metrics.get('r2', 0):.3f}"
                            })
                
                if performance_data:
                    perf_df = pd.DataFrame(performance_data)
                    st.dataframe(perf_df, use_container_width=True)
                else:
                    st.info("Performance metrics are being calculated...")
            
            except Exception as e:
                st.error(f"Error calculating performance metrics: {str(e)}")
        
        with col2:
            st.subheader("Market Insights")
            
            # Calculate market insights
            try:
                total_quota = filtered_data.groupby('date')['quota'].sum()
                total_bids = filtered_data.groupby('date')['bids_received'].sum()
                avg_premium = filtered_data.groupby('date')['premium'].mean()
                
                recent_quota_trend = ((total_quota.iloc[-1] - total_quota.iloc[-6]) / total_quota.iloc[-6]) * 100 if len(total_quota) >= 6 else 0
                recent_demand_trend = ((total_bids.iloc[-1] - total_bids.iloc[-6]) / total_bids.iloc[-6]) * 100 if len(total_bids) >= 6 else 0
                recent_price_trend = ((avg_premium.iloc[-1] - avg_premium.iloc[-6]) / avg_premium.iloc[-6]) * 100 if len(avg_premium) >= 6 else 0
                
                st.metric("Quota Trend (6 cycles)", f"{recent_quota_trend:+.1f}%")
                st.metric("Demand Trend (6 cycles)", f"{recent_demand_trend:+.1f}%")
                st.metric("Price Trend (6 cycles)", f"{recent_price_trend:+.1f}%")
                
                # Market insight text
                if recent_price_trend > 5:
                    trend_insight = "📈 Prices are trending upward significantly"
                elif recent_price_trend > 2:
                    trend_insight = "📊 Prices are moderately increasing"
                elif recent_price_trend < -5:
                    trend_insight = "📉 Prices are trending downward significantly"
                elif recent_price_trend < -2:
                    trend_insight = "📊 Prices are moderately decreasing"
                else:
                    trend_insight = "📊 Prices are relatively stable"
                
                st.info(trend_insight)
                
            except Exception as e:
                st.error(f"Error calculating market insights: {str(e)}")
        
        # Seasonal patterns
        st.header("🗓️ Seasonal Analysis")
        
        try:
            # Create seasonal analysis chart
            seasonal_data = data.copy()
            seasonal_data['month'] = seasonal_data['date'].dt.month
            seasonal_data['year'] = seasonal_data['date'].dt.year
            
            # Calculate monthly averages
            monthly_avg = seasonal_data.groupby(['month', 'vehicle_class'])['premium'].mean().reset_index()
            
            fig = px.line(
                monthly_avg[monthly_avg['vehicle_class'].isin(selected_categories)],
                x='month',
                y='premium',
                color='vehicle_class',
                title="Seasonal Price Patterns by Month",
                labels={'month': 'Month', 'premium': 'Average Premium ($)', 'vehicle_class': 'Category'}
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), 
                          ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating seasonal analysis: {str(e)}")
        
        # Data export section
        st.header("📁 Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Historical Data"):
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"coe_historical_data_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download Predictions"):
                try:
                    pred_data = []
                    if predictions:
                        for category in selected_categories:
                            if category in predictions:
                                for i in range(prediction_cycles):
                                    pred_data.append({
                                        'category': category,
                                        'cycle': i + 1,
                                        'predicted_premium': predictions[category]['mean'][i],
                                        'confidence_lower': predictions[category]['lower'][i],
                                        'confidence_upper': predictions[category]['upper'][i]
                                    })
                    
                    if pred_data:
                        pred_df = pd.DataFrame(pred_data)
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv,
                            file_name=f"coe_predictions_{prediction_cycles}cycles.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No prediction data available for download")
                except Exception as e:
                    st.error(f"Error preparing predictions for download: {str(e)}")
    
    else:
        st.warning("Please select at least one COE category to display data.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Data Source:** Singapore Land Transport Authority (LTA) via data.gov.sg | "
        "**Model:** Ensemble approach combining LSTM, Prophet, and XGBoost algorithms"
    )

if __name__ == "__main__":
    main()
