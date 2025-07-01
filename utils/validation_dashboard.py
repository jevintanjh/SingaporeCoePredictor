
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from .model_testing_validation import COEModelTester, run_comprehensive_testing

def display_model_validation_dashboard(data):
    """Display comprehensive model validation dashboard in Streamlit"""
    
    st.header("🧪 Model Testing & Validation Results")
    st.markdown("Comprehensive analysis of all three COE prediction models with performance metrics, confusion matrices, and feature importance analysis.")
    
    # Run testing
    with st.spinner("Running comprehensive model testing..."):
        tester = run_comprehensive_testing(data)
    
    if not tester.results:
        st.error("No testing results available. Please check your data and models.")
        return
    
    # Model selection
    model_names = list(tester.results.keys())
    selected_model = st.selectbox("Select Model for Detailed Analysis", model_names)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Overview", "🎯 Detailed Analysis", "📈 Feature Importance", "📋 Model Comparison"])
    
    with tab1:
        st.subheader("Performance Metrics Overview")
        
        # Create performance comparison chart
        metrics_data = []
        for model_name, results in tester.results.items():
            overall = results['overall']
            metrics_data.append({
                'Model': model_name,
                'R² Score': overall['r2'],
                'Direction Accuracy': overall['direction_accuracy'],
                'MAPE': overall['mape'],
                'RMSE': overall['rmse'],
                'MAE': overall['mae']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Performance metrics bar chart
        fig_metrics = make_subplots(
            rows=2, cols=3,
            subplot_titles=['R² Score', 'Direction Accuracy (%)', 'MAPE (%)', 'RMSE ($)', 'MAE ($)', 'Overall Performance'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # R² Score
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df['R² Score'], 
                   name='R² Score', marker_color=colors),
            row=1, col=1
        )
        
        # Direction Accuracy
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df['Direction Accuracy']*100,
                   name='Direction Accuracy', marker_color=colors),
            row=1, col=2
        )
        
        # MAPE
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df['MAPE'],
                   name='MAPE', marker_color=colors),
            row=1, col=3
        )
        
        # RMSE
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'],
                   name='RMSE', marker_color=colors),
            row=2, col=1
        )
        
        # MAE
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'],
                   name='MAE', marker_color=colors),
            row=2, col=2
        )
        
        # Overall Performance Score (combination of metrics)
        overall_scores = []
        for _, row in metrics_df.iterrows():
            score = (row['R² Score'] * 0.3 + 
                    row['Direction Accuracy'] * 0.4 + 
                    (1 - min(row['MAPE'], 20) / 20) * 0.3) * 100
            overall_scores.append(score)
        
        fig_metrics.add_trace(
            go.Bar(x=metrics_df['Model'], y=overall_scores,
                   name='Overall Score', marker_color=colors),
            row=2, col=3
        )
        
        fig_metrics.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with tab2:
        st.subheader(f"Detailed Analysis: {selected_model}")
        
        if selected_model in tester.results:
            results = tester.results[selected_model]['overall']
            
            # Performance metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{results['r2']:.3f}", 
                         delta=f"{(results['r2'] - 0.8):.3f}" if results['r2'] > 0.8 else None)
            
            with col2:
                st.metric("Direction Accuracy", f"{results['direction_accuracy']:.1%}",
                         delta=f"{(results['direction_accuracy'] - 0.7):.1%}" if results['direction_accuracy'] > 0.7 else None)
            
            with col3:
                st.metric("MAPE", f"{results['mape']:.1f}%",
                         delta=f"{(10 - results['mape']):.1f}%" if results['mape'] < 10 else None)
            
            with col4:
                st.metric("RMSE", f"${results['rmse']:,.0f}")
            
            # Confusion Matrix
            st.subheader("Direction Prediction Confusion Matrix")
            cm = results['confusion_matrix']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Down', 'Predicted Up'],
                y=['Actual Up', 'Actual Down'],
                colorscale='RdYlBu_r',
                text=[[f"{cm[i,j]:.1%}" for j in range(cm.shape[1])] for i in range(cm.shape[0])],
                texttemplate="%{text}",
                textfont={"size": 16},
                hoverongaps=False
            ))
            
            fig_cm.update_layout(
                title="Confusion Matrix for Price Direction Predictions",
                xaxis_title="Predicted Direction",
                yaxis_title="Actual Direction"
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Model-specific insights
            st.subheader("Model Insights")
            report = tester.generate_model_report(selected_model)
            st.markdown(report)
    
    with tab3:
        st.subheader(f"Feature Importance: {selected_model}")
        
        if selected_model in tester.feature_importance:
            feature_imp = tester.feature_importance[selected_model]
            
            # Sort features by importance
            sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
            
            features = [f[0] for f in sorted_features]
            importances = [f[1] for f in sorted_features]
            
            # Feature importance chart
            fig_features = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig_features.update_layout(
                title=f"Top Feature Importances for {selected_model}",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(features) * 30)
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Feature descriptions
            st.subheader("Feature Descriptions")
            feature_descriptions = {
                'momentum_3_period': 'Short-term price momentum over 3 periods',
                'momentum_6_period': 'Medium-term price momentum over 6 periods',
                'volatility': 'Price volatility measure',
                'moving_average_ratio': 'Current price relative to moving average',
                'trend_indicator': 'Overall price trend direction',
                'rsi_simple': 'Relative Strength Index for overbought/oversold conditions',
                'price_history': 'Historical price patterns',
                'quota': 'Certificate quota (supply constraint)',
                'bids_received': 'Number of bids (demand indicator)',
                'success_rate': 'Bidding success rate (market efficiency)',
                'seasonality': 'Seasonal patterns in pricing',
                'trend_component': 'Long-term trend analysis'
            }
            
            for feature, importance in sorted_features[:5]:
                with st.expander(f"{feature} (Importance: {importance:.3f})"):
                    desc = feature_descriptions.get(feature, "Feature importance in model predictions")
                    st.write(desc)
        
        else:
            st.info("Feature importance not available for this model.")
    
    with tab4:
        st.subheader("Model Comparison Summary")
        
        # Comparison table
        comparison_data = []
        for model_name, results in tester.results.items():
            overall = results['overall']
            comparison_data.append({
                'Model': model_name,
                'R² Score': f"{overall['r2']:.3f}",
                'Direction Accuracy': f"{overall['direction_accuracy']:.1%}",
                'MAPE': f"{overall['mape']:.1f}%",
                'RMSE': f"${overall['rmse']:,.0f}",
                'MAE': f"${overall['mae']:,.0f}",
                'Performance Rating': 'Excellent' if overall['direction_accuracy'] > 0.75 else 'Good' if overall['direction_accuracy'] > 0.65 else 'Fair'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Recommendations
        st.subheader("Model Recommendations")
        
        best_model = max(tester.results.items(), key=lambda x: x[1]['overall']['direction_accuracy'])
        best_model_name = best_model[0]
        best_accuracy = best_model[1]['overall']['direction_accuracy']
        
        st.success(f"🏆 **Best Performing Model**: {best_model_name} with {best_accuracy:.1%} directional accuracy")
        
        # Model-specific recommendations
        recommendations = {
            'Fast Directional Forecaster': "Ideal for real-time trading decisions with fast execution requirements",
            'Interpretable N-BEATS': "Best for stakeholder presentations requiring explainable predictions",
            'N-BEATSx': "Optimal for comprehensive analysis incorporating market fundamentals"
        }
        
        for model_name in model_names:
            if model_name in recommendations:
                st.info(f"**{model_name}**: {recommendations[model_name]}")
        
        # Export results
        if st.button("Export Testing Results"):
            results_summary = pd.DataFrame(comparison_data)
            csv = results_summary.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="coe_model_testing_results.csv",
                mime="text/csv"
            )
