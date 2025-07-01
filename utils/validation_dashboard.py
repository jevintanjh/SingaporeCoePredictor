import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from .model_testing_validation import COEModelTester, run_comprehensive_testing

def display_model_validation_dashboard(data):
    """Display comprehensive model validation dashboard"""
    st.header("🧪 Model Testing & Validation Results")
    st.markdown("Comprehensive analysis of all three COE prediction models with performance metrics, confusion matrices, and feature importance analysis.")

    try:
        # Run comprehensive testing
        tester = run_comprehensive_testing(data)

        if not tester or not hasattr(tester, 'results') or not tester.results:
            st.warning("No validation results available. Please check if models are properly configured.")
            return

        # Ensure results is a proper dictionary
        if not isinstance(tester.results, dict):
            st.error("Invalid results format. Please regenerate the testing results.")
            return

        # Create visualization dashboard
        try:
            dashboard_fig = tester.create_visualization_dashboard(data)
            st.plotly_chart(dashboard_fig, use_container_width=True)
        except Exception as viz_error:
            st.warning(f"Visualization error: {str(viz_error)}")

        # Model comparison table
        st.subheader("📊 Model Performance Comparison")

        model_names = list(tester.results.keys())
        comparison_data = []

        for model_name in model_names:
            try:
                if model_name in tester.results and isinstance(tester.results[model_name], dict):
                    result_data = tester.results[model_name]
                    if 'overall' in result_data and isinstance(result_data['overall'], dict):
                        overall = result_data['overall']
                        comparison_data.append({
                            'Model': str(model_name),
                            'Direction Accuracy': f"{float(overall.get('direction_accuracy', 0)):.1%}",
                            'R² Score': f"{float(overall.get('r2', 0)):.3f}",
                            'MAPE': f"{float(overall.get('mape', 0)):.1f}%",
                            'RMSE': f"${float(overall.get('rmse', 0)):,.0f}",
                            'MAE': f"${float(overall.get('mae', 0)):,.0f}",
                            'Performance Rating': 'Excellent' if float(overall.get('direction_accuracy', 0)) > 0.75 else 'Good' if float(overall.get('direction_accuracy', 0)) > 0.65 else 'Fair'
                        })
            except (TypeError, ValueError, KeyError) as e:
                st.warning(f"Error processing results for {model_name}: {str(e)}")
                continue

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.warning("No valid comparison data available.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")