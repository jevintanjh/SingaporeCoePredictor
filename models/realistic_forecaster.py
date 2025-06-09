import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RealisticCOEForecaster:
    """
    Realistic COE price forecaster using proven financial time series techniques
    """
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def exponential_smoothing(self, series, alpha=0.3):
        """Simple exponential smoothing"""
        if len(series) == 0:
            return 0
        
        smoothed = series[0]
        for value in series[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        return smoothed
    
    def calculate_trend(self, series, periods=6):
        """Calculate recent trend from price changes"""
        if len(series) < periods:
            return 0
        
        recent_prices = series[-periods:]
        changes = np.diff(recent_prices)
        return np.mean(changes)
    
    def fit(self, data):
        """Fit models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 10:
                continue
            
            # Sort by date
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            # Store historical data
            self.models[category] = {
                'prices': prices,
                'dates': category_data['date'].values,
                'recent_volatility': np.std(prices[-12:]) if len(prices) >= 12 else np.std(prices[-6:]) if len(prices) >= 6 else prices[-1] * 0.05
            }
            
            # Calculate simple validation metrics if enough data
            if len(prices) > 20:
                split_point = int(len(prices) * 0.8)
                train_prices = prices[:split_point]
                test_prices = prices[split_point:]
                
                # Simple prediction: exponential smoothing + trend
                last_smoothed = self.exponential_smoothing(train_prices)
                trend = self.calculate_trend(train_prices)
                
                predictions = []
                for i in range(len(test_prices)):
                    pred = last_smoothed + (i + 1) * trend * 0.5  # Damped trend
                    predictions.append(pred)
                
                # Calculate metrics
                mae = mean_absolute_error(test_prices, predictions)
                rmse = np.sqrt(mean_squared_error(test_prices, predictions))
                mape = np.mean(np.abs((test_prices - predictions) / test_prices)) * 100
                
                self.performance_metrics[category] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2_score(test_prices, predictions)
                }
    
    def predict(self, steps=3):
        """Generate realistic predictions for all categories"""
        predictions = {}
        
        for category in self.categories:
            if category not in self.models:
                continue
            
            prices = self.models[category]['prices']
            if len(prices) == 0:
                continue
            
            # Use last 12 periods for prediction base
            recent_prices = prices[-12:] if len(prices) >= 12 else prices
            
            # Calculate base prediction using exponential smoothing
            base_forecast = self.exponential_smoothing(recent_prices)
            
            # Calculate trend (dampened for stability)
            trend = self.calculate_trend(recent_prices) * 0.3
            
            # Get volatility for confidence intervals
            volatility = self.models[category]['recent_volatility']
            
            # Generate step-wise predictions
            forecasts = []
            last_price = prices[-1]
            
            for step in range(1, steps + 1):
                # Combine smoothed forecast with dampened trend
                raw_forecast = base_forecast + step * trend
                
                # Apply conservative constraints
                max_change_per_step = min(volatility * 0.5, last_price * 0.05)  # Max 5% or 0.5x volatility per step
                
                if step == 1:
                    # First step can deviate more from current price
                    constrained_forecast = np.clip(raw_forecast, 
                                                 last_price - max_change_per_step * 2,
                                                 last_price + max_change_per_step * 2)
                else:
                    # Subsequent steps are more constrained
                    prev_forecast = forecasts[step - 2]
                    constrained_forecast = np.clip(raw_forecast,
                                                 prev_forecast - max_change_per_step,
                                                 prev_forecast + max_change_per_step)
                
                forecasts.append(constrained_forecast)
            
            # Calculate confidence intervals
            confidence_width = volatility * 1.5  # Narrower confidence intervals
            
            predictions[category] = {
                'mean': forecasts,
                'lower': [f - confidence_width for f in forecasts],
                'upper': [f + confidence_width for f in forecasts]
            }
        
        return predictions
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a category"""
        return self.performance_metrics.get(category, {})