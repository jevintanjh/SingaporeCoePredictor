import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from models.improved_forecaster import ImprovedCOEForecaster
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
            model = ImprovedCOEForecaster()
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
                            ImprovedCOEForecaster, data, validation_category
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
        
        # Advanced validation techniques
        st.subheader("Advanced Model Stability Tests")
        
        advanced_col1, advanced_col2 = st.columns(2)
        
        with advanced_col1:
            st.markdown("**Robustness Testing**")
            
            if st.button("Bootstrap Validation", help="Test model stability with 100 resampled datasets"):
                with st.spinner("Running bootstrap validation..."):
                    try:
                        advanced_validator = AdvancedModelValidation()
                        bootstrap_results = advanced_validator.bootstrap_validation(
                            ImprovedCOEForecaster, data, validation_category
                        )
                        
                        if bootstrap_results:
                            st.success("Bootstrap validation completed!")
                            
                            # Display confidence intervals
                            for metric, stats in bootstrap_results.items():
                                if isinstance(stats, dict):
                                    metric_name = metric.replace('_', ' ').title()
                                    st.metric(
                                        metric_name,
                                        f"{stats['mean']:.2f} ± {stats['std']:.2f}",
                                        delta=f"95% CI: [{stats['ci_5']:.2f}, {stats['ci_95']:.2f}]"
                                    )
                        else:
                            st.warning("Insufficient data for bootstrap validation")
                    except Exception as e:
                        st.error(f"Bootstrap validation error: {str(e)}")
            
            if st.button("Regime Analysis", help="Test performance across high/low volatility periods"):
                with st.spinner("Analyzing regime performance..."):
                    try:
                        advanced_validator = AdvancedModelValidation()
                        regime_results = advanced_validator.regime_change_validation(
                            ImprovedCOEForecaster, data, validation_category
                        )
                        
                        if regime_results:
                            st.success("Regime analysis completed!")
                            
                            for regime_type, metrics in regime_results.items():
                                if metrics:
                                    st.subheader(f"{regime_type.replace('_', ' ').title()} Regime")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("MAPE", f"{metrics['mape']:.1f}%")
                                    with col2:
                                        st.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.1f}%")
                                    with col3:
                                        st.metric("Test Periods", f"{metrics['n_periods']}")
                        else:
                            st.warning("Unable to identify distinct volatility regimes")
                    except Exception as e:
                        st.error(f"Regime analysis error: {str(e)}")
        
        with advanced_col2:
            st.markdown("**Stress Testing**")
            
            if st.button("Comprehensive Stress Test", help="Test model with missing data, extreme volatility, and trend breaks"):
                with st.spinner("Running stress tests..."):
                    try:
                        advanced_validator = AdvancedModelValidation()
                        stress_results = advanced_validator.stress_testing(
                            ImprovedCOEForecaster, data, validation_category
                        )
                        
                        if stress_results:
                            st.success("Stress testing completed!")
                            
                            # Missing data test
                            if 'missing_data' in stress_results and stress_results['missing_data']:
                                md = stress_results['missing_data']
                                status = "Robust ✓" if md.get('can_handle_missing', False) else "Sensitive ✗"
                                st.metric("Missing Data Handling", status)
                            
                            # Extreme volatility test
                            if 'extreme_volatility' in stress_results and stress_results['extreme_volatility']:
                                ev = stress_results['extreme_volatility']
                                status = "Stable ✓" if ev.get('handles_extreme_volatility', False) else "Unstable ✗"
                                st.metric("High Volatility Handling", status)
                            
                            # Trend break test
                            if 'trend_breaks' in stress_results and stress_results['trend_breaks']:
                                tb = stress_results['trend_breaks']
                                status = "Adaptive ✓" if tb.get('handles_trend_breaks', False) else "Rigid ✗"
                                st.metric("Trend Break Adaptation", status)
                        else:
                            st.warning("Stress testing could not be completed")
                    except Exception as e:
                        st.error(f"Stress testing error: {str(e)}")
            
            if st.button("Rolling Origin Test", help="Test how performance changes with increasing training data"):
                with st.spinner("Running rolling origin validation..."):
                    try:
                        advanced_validator = AdvancedModelValidation()
                        rolling_results = advanced_validator.rolling_origin_validation(
                            ImprovedCOEForecaster, data, validation_category
                        )
                        
                        if rolling_results:
                            st.success("Rolling origin validation completed!")
                            
                            trend_desc = "Improving" if rolling_results['performance_trend'] < 0 else "Degrading"
                            st.metric("Performance Trend", trend_desc)
                            st.metric("Final MAPE", f"{rolling_results['final_performance']:.1f}%")
                            
                            # Show trend chart
                            if 'results' in rolling_results:
                                results = rolling_results['results']
                                
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=results['training_sizes'],
                                    y=results['test_performance'],
                                    mode='lines+markers',
                                    name='MAPE (%)',
                                    line=dict(color='blue')
                                ))
                                
                                fig.update_layout(
                                    title=f'{validation_category} - Performance vs Training Size',
                                    xaxis_title='Training Data Size',
                                    yaxis_title='MAPE (%)',
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Insufficient data for rolling origin validation")
                    except Exception as e:
                        st.error(f"Rolling origin validation error: {str(e)}")
        
        # Comprehensive validation report
        if st.button("Generate Comprehensive Report", type="primary"):
            with st.spinner("Generating comprehensive validation report..."):
                try:
                    advanced_validator = AdvancedModelValidation()
                    comprehensive_results = advanced_validator.run_comprehensive_validation(
                        ImprovedCOEForecaster, data, validation_category
                    )
                    
                    # Format and display report
                    report = advanced_validator.format_comprehensive_report(comprehensive_results)
                    st.text_area("Comprehensive Validation Report", report, height=400)
                    
                    # Store results
                    st.session_state[f'comprehensive_{validation_category}'] = comprehensive_results
                    st.success("Comprehensive validation report generated!")
                    
                except Exception as e:
                    st.error(f"Comprehensive validation error: {str(e)}")
        
        # Cross-category validation comparison
        if st.button("Compare All Categories", key="compare_all"):
            with st.spinner("Running fast validation across all categories..."):
                try:
                    validator = FastModelValidation()
                    all_results = validator.run_all_categories_fast(
                        ImprovedCOEForecaster, data
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
