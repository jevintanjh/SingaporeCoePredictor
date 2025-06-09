import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_walk_forward_chart(validation_results, category):
    """Create walk-forward validation visualization"""
    if not validation_results or category not in validation_results:
        return None
    
    wf_data = validation_results[category]['walk_forward']
    if not wf_data:
        return None
    
    dates = pd.to_datetime(wf_data['dates'])
    
    fig = go.Figure()
    
    # Add actual vs predicted prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=wf_data['actuals'],
        mode='lines+markers',
        name='Actual Prices',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=wf_data['predictions'],
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f'{category} - Walk-Forward Validation Results',
        xaxis_title='Date',
        yaxis_title='Price (S$)',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_backtest_chart(validation_results, category):
    """Create backtesting visualization"""
    if not validation_results or category not in validation_results:
        return None
    
    bt_data = validation_results[category]['backtest']
    if not bt_data:
        return None
    
    dates = pd.to_datetime(bt_data['dates'])
    
    fig = go.Figure()
    
    # Add actual vs predicted prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=bt_data['actuals'],
        mode='lines+markers',
        name='Actual Prices',
        line=dict(color='green', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=bt_data['predictions'],
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'{category} - Backtesting Results (Last {bt_data["n_cycles"]} Cycles)',
        xaxis_title='Date',
        yaxis_title='Price (S$)',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_direction_accuracy_chart(validation_results):
    """Create direction accuracy comparison chart"""
    categories = []
    wf_accuracy = []
    bt_accuracy = []
    
    for category, results in validation_results.items():
        if results['walk_forward'] or results['backtest']:
            categories.append(category)
            
            wf_acc = results['walk_forward']['direction_accuracy'] if results['walk_forward'] else 0
            bt_acc = results['backtest']['direction_accuracy'] if results['backtest'] else 0
            
            wf_accuracy.append(wf_acc)
            bt_accuracy.append(bt_acc)
    
    if not categories:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=wf_accuracy,
        name='Walk-Forward',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=categories,
        y=bt_accuracy,
        name='Backtesting',
        marker_color='lightcoral'
    ))
    
    # Add 50% line (random chance)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                  annotation_text="Random Chance (50%)")
    
    fig.update_layout(
        title='Direction Accuracy by Category (%)',
        xaxis_title='COE Category',
        yaxis_title='Direction Accuracy (%)',
        barmode='group',
        showlegend=True,
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_volatility_tracking_chart(validation_results, category):
    """Create volatility tracking visualization"""
    if not validation_results or category not in validation_results:
        return None
    
    bt_data = validation_results[category]['backtest']
    if not bt_data or 'volatility_tracking' not in bt_data:
        return None
    
    vt = bt_data['volatility_tracking']
    if 'actual_volatility' not in vt or 'predicted_volatility' not in vt:
        return None
    
    fig = go.Figure()
    
    x_vals = list(range(len(vt['actual_volatility'])))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=vt['actual_volatility'],
        mode='lines+markers',
        name='Actual Volatility',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=vt['predicted_volatility'],
        mode='lines+markers',
        name='Predicted Volatility',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f'{category} - Volatility Tracking (Correlation: {vt["correlation"]:.3f})',
        xaxis_title='Time Period',
        yaxis_title='Price Volatility (S$)',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_validation_metrics_table(validation_results):
    """Create comprehensive validation metrics table"""
    table_data = []
    
    for category, results in validation_results.items():
        row = {'Category': category}
        
        # Walk-forward metrics
        if results['walk_forward']:
            wf = results['walk_forward']
            row.update({
                'WF_MAPE': f"{wf['mape']:.1f}%",
                'WF_Direction': f"{wf['direction_accuracy']:.1f}%",
                'WF_R²': f"{wf['r2']:.3f}",
                'WF_Predictions': wf['n_predictions']
            })
        else:
            row.update({
                'WF_MAPE': 'N/A',
                'WF_Direction': 'N/A',
                'WF_R²': 'N/A',
                'WF_Predictions': 0
            })
        
        # Backtest metrics
        if results['backtest']:
            bt = results['backtest']
            row.update({
                'BT_MAPE': f"{bt['mape']:.1f}%",
                'BT_Direction': f"{bt['direction_accuracy']:.1f}%",
                'BT_Vol_Corr': f"{bt['volatility_tracking']['correlation']:.3f}",
                'BT_Vol_Ratio': f"{bt['volatility_tracking']['ratio']:.3f}",
                'BT_Cycles': bt['n_cycles']
            })
        else:
            row.update({
                'BT_MAPE': 'N/A',
                'BT_Direction': 'N/A',
                'BT_Vol_Corr': 'N/A',
                'BT_Vol_Ratio': 'N/A',
                'BT_Cycles': 0
            })
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def create_prediction_error_distribution(validation_results, category):
    """Create prediction error distribution chart"""
    if not validation_results or category not in validation_results:
        return None
    
    wf_data = validation_results[category]['walk_forward']
    bt_data = validation_results[category]['backtest']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Walk-Forward Errors', 'Backtesting Errors'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Walk-forward errors
    if wf_data:
        wf_errors = np.array(wf_data['predictions']) - np.array(wf_data['actuals'])
        fig.add_trace(
            go.Histogram(x=wf_errors, name='WF Errors', nbinsx=20, opacity=0.7),
            row=1, col=1
        )
    
    # Backtest errors
    if bt_data:
        bt_errors = np.array(bt_data['predictions']) - np.array(bt_data['actuals'])
        fig.add_trace(
            go.Histogram(x=bt_errors, name='BT Errors', nbinsx=20, opacity=0.7),
            row=1, col=2
        )
    
    fig.update_layout(
        title=f'{category} - Prediction Error Distributions',
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(title_text="Prediction Error (S$)")
    fig.update_yaxes(title_text="Frequency")
    
    return fig