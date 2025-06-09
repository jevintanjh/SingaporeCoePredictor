import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_historical_chart(data, category, show_volume=True):
    """Create interactive historical chart for a COE category"""
    
    # Create subplot with secondary y-axis
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[f'{category} Premium Trend', 'Bidding Volume'],
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    # Main premium line
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['premium'],
            mode='lines+markers',
            name='Premium',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Premium:</b> $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    if show_volume:
        # Quota bars
        fig.add_trace(
            go.Bar(
                x=data['date'],
                y=data['quota'],
                name='Quota',
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Quota:</b> %{y}<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Bids received line
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['bids_received'],
                mode='lines+markers',
                name='Bids Received',
                line=dict(color='red', width=2),
                marker=dict(size=3),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                             '<b>Bids Received:</b> %{y}<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{category} COE Historical Analysis',
        height=500 if show_volume else 400,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Premium ($)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Count", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2 if show_volume else 1, col=1)
    
    return fig

def create_prediction_chart(historical_data, predictions, category, prediction_cycles):
    """Create chart showing historical data and future predictions"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['premium'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Premium:</b> $%{y:,.0f}<br>' +
                         '<extra></extra>'
        )
    )
    
    # Future predictions
    last_date = historical_data['date'].max()
    future_dates = [last_date + pd.DateOffset(days=15 * (i + 1)) for i in range(prediction_cycles)]
    
    # Prediction line
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions['mean'],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Predicted Premium:</b> $%{y:,.0f}<br>' +
                         '<extra></extra>'
        )
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=predictions['upper'] + predictions['lower'][::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            hoverinfo='skip',
            showlegend=True
        )
    )
    
    # Connect historical and prediction
    connect_x = [historical_data['date'].iloc[-1], future_dates[0]]
    connect_y = [historical_data['premium'].iloc[-1], predictions['mean'][0]]
    
    fig.add_trace(
        go.Scatter(
            x=connect_x,
            y=connect_y,
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name='Connection',
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'{category} COE Price Prediction',
        xaxis_title='Date',
        yaxis_title='Premium ($)',
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_performance_chart(performance_metrics):
    """Create performance metrics visualization"""
    
    categories = list(performance_metrics.keys())
    metrics = ['mape', 'rmse', 'mae', 'r2']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['MAPE (%)', 'RMSE ($)', 'MAE ($)', 'R² Score'],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1[:len(categories)]
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        values = [performance_metrics[cat].get(metric, 0) for cat in categories]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                name=metric.upper(),
                marker_color=colors,
                showlegend=False,
                text=[f'{v:.2f}' for v in values],
                textposition='auto'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Model Performance Metrics by Category',
        height=500,
        showlegend=False
    )
    
    return fig

def create_seasonal_analysis_chart(data, categories=None):
    """Create seasonal analysis chart"""
    
    if categories is None:
        categories = data['vehicle_class'].unique()
    
    # Filter data
    filtered_data = data[data['vehicle_class'].isin(categories)].copy()
    
    # Extract month and calculate monthly averages
    filtered_data['month'] = filtered_data['date'].dt.month
    monthly_avg = filtered_data.groupby(['month', 'vehicle_class'])['premium'].mean().reset_index()
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(categories)]
    
    for i, category in enumerate(categories):
        cat_data = monthly_avg[monthly_avg['vehicle_class'] == category]
        
        fig.add_trace(
            go.Scatter(
                x=cat_data['month'],
                y=cat_data['premium'],
                mode='lines+markers',
                name=category,
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                hovertemplate='<b>Category:</b> %{fullData.name}<br>' +
                             '<b>Month:</b> %{x}<br>' +
                             '<b>Avg Premium:</b> $%{y:,.0f}<br>' +
                             '<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Seasonal Premium Patterns by Month',
        xaxis_title='Month',
        yaxis_title='Average Premium ($)',
        height=400,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap for numerical features"""
    
    # Select numerical columns
    numerical_cols = ['premium', 'quota', 'bids_received', 'bids_success']
    correlation_data = data[numerical_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_data.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=400,
        width=400
    )
    
    return fig

def create_bid_quota_analysis(data, category):
    """Create bid-to-quota ratio analysis chart"""
    
    category_data = data[data['vehicle_class'] == category].copy()
    category_data['bid_quota_ratio'] = category_data['bids_received'] / category_data['quota']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Premium vs Bid-Quota Ratio', 'Bid-Quota Ratio Trend'],
        horizontal_spacing=0.1
    )
    
    # Scatter plot: Premium vs Bid-Quota Ratio
    fig.add_trace(
        go.Scatter(
            x=category_data['bid_quota_ratio'],
            y=category_data['premium'],
            mode='markers',
            name='Data Points',
            marker=dict(
                size=8,
                color=category_data['premium'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Premium ($)")
            ),
            hovertemplate='<b>Bid-Quota Ratio:</b> %{x:.2f}<br>' +
                         '<b>Premium:</b> $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Time series: Bid-Quota Ratio
    fig.add_trace(
        go.Scatter(
            x=category_data['date'],
            y=category_data['bid_quota_ratio'],
            mode='lines+markers',
            name='Bid-Quota Ratio',
            line=dict(color='orange', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Bid-Quota Ratio:</b> %{y:.2f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'{category} Bid-Quota Analysis',
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Bid-Quota Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Premium ($)", row=1, col=1)
    fig.update_yaxes(title_text="Bid-Quota Ratio", row=1, col=2)
    
    return fig

def create_volume_analysis_chart(data, categories=None):
    """Create bidding volume analysis chart"""
    
    if categories is None:
        categories = data['vehicle_class'].unique()
    
    filtered_data = data[data['vehicle_class'].isin(categories)].copy()
    
    fig = make_subplots(
        rows=len(categories), cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{cat} Bidding Volume' for cat in categories],
        vertical_spacing=0.02
    )
    
    colors = px.colors.qualitative.Set1[:len(categories)]
    
    for i, category in enumerate(categories):
        cat_data = filtered_data[filtered_data['vehicle_class'] == category]
        
        # Quota bars
        fig.add_trace(
            go.Bar(
                x=cat_data['date'],
                y=cat_data['quota'],
                name=f'{category} Quota',
                marker_color=colors[i],
                opacity=0.7,
                showlegend=(i == 0)
            ),
            row=i+1, col=1
        )
        
        # Bids received line
        fig.add_trace(
            go.Scatter(
                x=cat_data['date'],
                y=cat_data['bids_received'],
                mode='lines',
                name=f'{category} Bids',
                line=dict(color='red', width=2),
                showlegend=(i == 0)
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title='Bidding Volume Analysis Across Categories',
        height=200 * len(categories),
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=len(categories), col=1)
    
    return fig
