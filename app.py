import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from models.enhanced_directional_forecaster import EnhancedDirectionalForecaster
from utils.data_processor import DataProcessor
from utils.visualizations import create_historical_chart, create_prediction_chart, create_performance_chart
from utils.metrics import calculate_metrics, format_metrics
from utils.fast_validation import FastModelValidation
from utils.advanced_validation import AdvancedModelValidation
from utils.validation_visualizations import (
    create_walk_forward_chart, create_backtest_chart, create_direction_accuracy_chart,
    create_volatility_tracking_chart, create_validation_metrics_table, create_prediction_error_distribution
)

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
        st.success(f"Loaded {len(processed_data)} records from {processed_data['date'].min()} to {processed_data['date'].max()}")
        return processed_data, processor
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Try alternative path
        try:
            processor = DataProcessor()
            data = processor.load_data('attached_assets/COEBiddingResultsPrices_1749430265007.csv')
            processed_data = processor.preprocess_data(data)
            st.success(f"Loaded {len(processed_data)} records from alternative path")
            return processed_data, processor
        except Exception as e2:
            st.error(f"Alternative path also failed: {str(e2)}")
            return None, None

@st.cache_resource
def initialize_model():
    """Initialize and train the ensemble model"""
    data, processor = load_and_process_data()
    if data is not None:
        try:
            model = EnhancedDirectionalForecaster()
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
                            
                            # Add directional prediction insights
                            if hasattr(model, 'get_performance_metrics') and category in predictions:
                                pred_data = predictions[category]
                                if 'direction_confidence' in pred_data:
                                    avg_confidence = sum(pred_data['direction_confidence']) / len(pred_data['direction_confidence'])
                                    if avg_confidence > 0.7:
                                        confidence_status = "High Confidence"
                                    elif avg_confidence > 0.6:
                                        confidence_status = "Medium Confidence"
                                    else:
                                        confidence_status = "Low Confidence"
                                    
                                    st.caption(f"Direction: {confidence_status} ({avg_confidence:.1%})")
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
                    
                    # Add directional analysis insights
                    if hasattr(model, 'get_performance_metrics'):
                        metrics = model.get_performance_metrics(category)
                        if metrics:
                            st.markdown("**Directional Analysis**")
                            
                            if 'top_features' in metrics:
                                top_features = metrics['top_features']
                                if top_features:
                                    st.caption("Key Direction Indicators:")
                                    for feat, importance in top_features[:3]:
                                        st.caption(f"• {feat}: {importance:.3f}")
                            
                            if 'model_type' in metrics:
                                st.info(f"Using {metrics['model_type']} with {metrics.get('features_used', 0)} indicators")
        
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
                                'R²': f"{metrics.get('r2', 0):.3f}",
                                'Vol Correlation': f"{metrics.get('volatility_correlation', 0):.3f}"
                            })
                
                if performance_data:
                    perf_df = pd.DataFrame(performance_data)
                    st.dataframe(perf_df, use_container_width=True)
                else:
                    st.info("Performance metrics are being calculated...")
            except Exception as e:
                st.error(f"Error calculating performance metrics: {str(e)}")
        
        # Model Validation Section
        st.header("🔬 Model Stability & Validation")
        st.markdown("Walk-forward validation and backtesting results")
        
        validation_col1, validation_col2 = st.columns([3, 1])
        
        with validation_col2:
            st.subheader("Validation Controls")
            
            # Category selection for validation
            validation_category = st.selectbox(
                "Select Category for Validation",
                categories,
                key="validation_category"
            )
            
            # Validation parameters
            validation_cycles = st.slider(
                "Backtest Cycles",
                min_value=6,
                max_value=24,
                value=12,
                help="Number of past cycles to backtest"
            )
            
            # Run validation button
            run_validation = st.button("Run Validation", type="primary")
        
        with validation_col1:
            if run_validation:
                with st.spinner(f"Running fast validation for {validation_category}..."):
                    try:
                        validator = FastModelValidation()
                        
                        # Run fast validation
                        validation_result = validator.run_fast_validation(
                            EnhancedDirectionalForecaster, data, validation_category
                        )
                        
                        if validation_result:
                            validation_results = {validation_category: validation_result}
                            
                            # Display validation summary
                            summary = validator.format_fast_summary(validation_results)
                            st.text_area("Validation Summary", summary, height=200)
                            
                            # Create visualizations
                            if validation_result['walk_forward']:
                                wf_chart = create_walk_forward_chart(validation_results, validation_category)
                                if wf_chart:
                                    st.plotly_chart(wf_chart, use_container_width=True)
                            
                            if validation_result['backtest']:
                                bt_chart = create_backtest_chart(validation_results, validation_category)
                                if bt_chart:
                                    st.plotly_chart(bt_chart, use_container_width=True)
                                
                                # Volatility tracking
                                vol_chart = create_volatility_tracking_chart(validation_results, validation_category)
                                if vol_chart:
                                    st.plotly_chart(vol_chart, use_container_width=True)
                            
                            # Store results for comparison
                            st.session_state[f'validation_{validation_category}'] = validation_results
                            st.success("Fast validation completed successfully!")
                        else:
                            st.warning(f"Insufficient data for validation of {validation_category}")
                        
                    except Exception as e:
                        st.error(f"Validation error: {str(e)}")
            
            # Display stored validation results if available
            elif f'validation_{validation_category}' in st.session_state:
                stored_results = st.session_state[f'validation_{validation_category}']
                
                st.info("Previous validation results (click 'Run Validation' for fresh results)")
                
                # Show charts from stored results
                if stored_results[validation_category]['walk_forward']:
                    wf_chart = create_walk_forward_chart(stored_results, validation_category)
                    if wf_chart:
                        st.plotly_chart(wf_chart, use_container_width=True)
                
                if stored_results[validation_category]['backtest']:
                    bt_chart = create_backtest_chart(stored_results, validation_category)
                    if bt_chart:
                        st.plotly_chart(bt_chart, use_container_width=True)
            else:
                st.info("Click 'Run Validation' to perform comprehensive model testing")
        
        # Streamlined comprehensive validation
        st.subheader("Comprehensive Model Validation")
        
        if st.button("Run Complete Validation Suite", type="primary", help="Runs all validation tests in one optimized process"):
            with st.spinner("Running comprehensive validation suite..."):
                try:
                    # Run fast validation first
                    validator = FastModelValidation()
                    fast_results = validator.run_fast_validation(
                        EnhancedDirectionalForecaster, data, validation_category
                    )
                    
                    if fast_results:
                        st.success("✓ Comprehensive validation completed!")
                        
                        # Core performance metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            direction_acc = fast_results.get('direction_accuracy', 0)
                            status = "✓" if direction_acc > 50 else "✗"
                            st.metric("Direction Accuracy", f"{direction_acc:.1f}% {status}")
                        
                        with col2:
                            mape = fast_results.get('mape', 0)
                            quality = "Excellent" if mape < 5 else "Good" if mape < 10 else "Fair" if mape < 20 else "Poor"
                            st.metric("MAPE", f"{mape:.1f}%", delta=quality)
                        
                        with col3:
                            r2 = fast_results.get('r2', 0)
                            r2_quality = "Good" if r2 > 0.3 else "Fair" if r2 > 0 else "Poor"
                            st.metric("R² Score", f"{r2:.3f}", delta=r2_quality)
                        
                        with col4:
                            rmse = fast_results.get('rmse', 0)
                            st.metric("RMSE", f"${rmse:.0f}")
                        
                        # Enhanced model information
                        st.subheader("Model Analysis")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("**Model Features:**")
                            has_direction_model = fast_results.get('has_direction_model', False)
                            if has_direction_model:
                                st.success("✓ Enhanced directional prediction enabled")
                                st.info("• Advanced technical indicators (RSI, Bollinger Bands)")
                                st.info("• Ensemble classification (Random Forest + Gradient Boosting)")
                                st.info("• Multi-timeframe momentum analysis")
                            else:
                                st.warning("⚠ Basic directional prediction (insufficient data for enhanced model)")
                                st.info("• Simple trend-based forecasting")
                                st.info("• Exponential smoothing")
                        
                        with col_b:
                            st.markdown("**Validation Insights:**")
                            
                            # Direction accuracy assessment
                            if direction_acc > 60:
                                st.success("🎯 Excellent directional prediction")
                            elif direction_acc > 50:
                                st.success("✓ Above-random directional accuracy")
                            else:
                                st.error("✗ Below-random directional accuracy")
                            
                            # Data quality assessment
                            n_test = fast_results.get('n_test_points', 0)
                            if n_test > 20:
                                st.success(f"✓ Robust validation ({n_test} test points)")
                            elif n_test > 10:
                                st.warning(f"⚠ Moderate validation ({n_test} test points)")
                            else:
                                st.error(f"✗ Limited validation ({n_test} test points)")
                        
                        # Performance comparison
                        st.subheader("Performance Benchmark")
                        
                        # Create performance comparison chart
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        
                        metrics = ['Direction Accuracy', 'MAPE Quality', 'R² Score']
                        values = [
                            direction_acc,
                            max(0, 100 - mape),  # Inverse MAPE for better visualization
                            max(0, r2 * 100)     # R² as percentage
                        ]
                        benchmarks = [50, 80, 30]  # Benchmark thresholds
                        
                        fig.add_trace(go.Bar(
                            x=metrics,
                            y=values,
                            name='Current Model',
                            marker_color=['green' if v > b else 'orange' if v > b*0.7 else 'red' 
                                        for v, b in zip(values, benchmarks)]
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=metrics,
                            y=benchmarks,
                            mode='markers',
                            name='Benchmark',
                            marker=dict(color='blue', size=10, symbol='diamond')
                        ))
                        
                        fig.update_layout(
                            title=f'{validation_category} - Performance vs Benchmarks',
                            yaxis_title='Performance Score',
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed validation summary
                        st.subheader("Validation Summary")
                        
                        summary_text = f"""
**{validation_category} Validation Report**

**Core Metrics:**
• Direction Accuracy: {direction_acc:.1f}% ({'Above random' if direction_acc > 50 else 'Below random'})
• Mean Absolute Percentage Error: {mape:.1f}%
• R² Score: {r2:.3f} ({'Positive explanatory power' if r2 > 0 else 'No explanatory power'})
• Root Mean Square Error: ${rmse:.0f}

**Model Type:** {'Enhanced Directional' if has_direction_model else 'Basic Trend-based'}

**Data Quality:** {n_test} test points for validation

**Key Insights:**
• Model {'successfully' if direction_acc > 50 else 'struggles to'} predict price directions above random chance
• Price magnitude predictions show {'good' if mape < 15 else 'moderate' if mape < 25 else 'poor'} accuracy
• {'Sufficient' if n_test > 15 else 'Limited'} data available for robust validation

**Recommendations:**
{'• Model performs well for directional prediction' if direction_acc > 55 else '• Consider additional feature engineering for better directional accuracy'}
• {'Price forecasts are reliable for short-term planning' if mape < 20 else 'Use price forecasts with caution due to high error rates'}
"""
                        
                        st.text_area("Detailed Report", summary_text, height=300)
                        
                        # Store results
                        st.session_state[f'validation_{validation_category}'] = fast_results
                    
                    else:
                        st.error("Validation failed - insufficient data for analysis")
                        
                except Exception as e:
                    st.error(f"Validation error: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
        
        # Cross-category validation comparison
        if st.button("Compare All Categories", key="compare_all"):
            with st.spinner("Running fast validation across all categories..."):
                try:
                    validator = FastModelValidation()
                    all_results = validator.run_all_categories_fast(
                        EnhancedDirectionalForecaster, data
                    )
                    
                    # Direction accuracy comparison
                    dir_accuracy_chart = create_direction_accuracy_chart(all_results)
                    if dir_accuracy_chart:
                        st.plotly_chart(dir_accuracy_chart, use_container_width=True)
                    
                    # Validation metrics table
                    metrics_table = create_validation_metrics_table(all_results)
                    if not metrics_table.empty:
                        st.subheader("Fast Validation Metrics")
                        st.dataframe(metrics_table, use_container_width=True)
                    
                    # Summary text
                    summary = validator.format_fast_summary(all_results)
                    st.text_area("All Categories Summary", summary, height=300)
                    
                    st.session_state.all_validation_results = all_results
                    st.success("Fast validation completed for all categories!")
                    
                except Exception as e:
                    st.error(f"Cross-category validation error: {str(e)}")
        
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
