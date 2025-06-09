import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class FastDirectionalForecaster:
    """
    Fast and reliable directional forecaster optimized for performance
    """
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def exponential_smoothing(self, series, alpha=0.3):
        """Proper exponential smoothing implementation"""
        if len(series) == 0:
            return 0
        
        smoothed = [series[0]]
        for i in range(1, len(series)):
            new_value = alpha * series[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(new_value)
        
        return smoothed[-1]
    
    def calculate_momentum_indicators(self, prices):
        """Calculate fast momentum indicators for direction prediction"""
        if len(prices) < 6:
            return {
                'momentum_3': 0, 'momentum_6': 0, 'volatility': 0.02,
                'ma_ratio': 1, 'trend': 0, 'rsi_simple': 50
            }
        
        indicators = {}
        
        # Short-term momentum
        if len(prices) >= 4:
            indicators['momentum_3'] = (prices[-1] - prices[-4]) / prices[-4]
        else:
            indicators['momentum_3'] = 0
            
        # Medium-term momentum
        if len(prices) >= 7:
            indicators['momentum_6'] = (prices[-1] - prices[-7]) / prices[-7]
        else:
            indicators['momentum_6'] = indicators['momentum_3']
        
        # Volatility
        recent_prices = prices[-min(6, len(prices)):]
        indicators['volatility'] = np.std(recent_prices) / np.mean(recent_prices)
        
        # Moving average ratio
        ma = np.mean(recent_prices)
        indicators['ma_ratio'] = prices[-1] / ma
        
        # Simple trend
        if len(prices) >= 3:
            indicators['trend'] = (prices[-1] - prices[-3]) / prices[-3]
        else:
            indicators['trend'] = 0
        
        # Simplified RSI
        if len(prices) >= 6:
            changes = np.diff(prices[-6:])
            gains = np.mean(changes[changes > 0]) if np.any(changes > 0) else 0
            losses = -np.mean(changes[changes < 0]) if np.any(changes < 0) else 0.01
            rs = gains / losses
            indicators['rsi_simple'] = 100 - (100 / (1 + rs))
        else:
            indicators['rsi_simple'] = 50
        
        return indicators
    
    def predict_direction_simple(self, prices):
        """Simple but effective direction prediction"""
        if len(prices) < 3:
            return 0.5  # No clear direction
        
        indicators = self.calculate_momentum_indicators(prices)
        
        # Direction scoring based on multiple factors
        direction_score = 0
        
        # Momentum signals (40% weight)
        if indicators['momentum_3'] > 0.02:  # 2% threshold
            direction_score += 0.2
        elif indicators['momentum_3'] < -0.02:
            direction_score -= 0.2
            
        if indicators['momentum_6'] > 0.05:  # 5% threshold
            direction_score += 0.2
        elif indicators['momentum_6'] < -0.05:
            direction_score -= 0.2
        
        # Trend signal (30% weight)
        if indicators['trend'] > 0.01:
            direction_score += 0.15
        elif indicators['trend'] < -0.01:
            direction_score -= 0.15
        
        # RSI signal (20% weight)
        if indicators['rsi_simple'] > 60:
            direction_score += 0.1
        elif indicators['rsi_simple'] < 40:
            direction_score -= 0.1
        
        # MA position signal (10% weight)
        if indicators['ma_ratio'] > 1.02:
            direction_score += 0.05
        elif indicators['ma_ratio'] < 0.98:
            direction_score -= 0.05
        
        # Convert to probability (0.5 = neutral, 1.0 = strong up, 0.0 = strong down)
        direction_prob = 0.5 + direction_score
        return np.clip(direction_prob, 0.1, 0.9)
    
    def fit(self, data):
        """Fit models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 10:
                continue
            
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            # Store model data
            self.models[category] = {
                'prices': prices,
                'dates': category_data['date'].values
            }
            
            # Calculate validation metrics
            self.calculate_validation_metrics(prices, category)
    
    def calculate_validation_metrics(self, prices, category):
        """Calculate validation metrics using walk-forward approach"""
        if len(prices) < 8:
            return
        
        try:
            # Use last 30% for validation, minimum 3 points
            split_idx = max(len(prices) - max(3, int(len(prices) * 0.3)), 5)
            train_prices = prices[:split_idx]
            test_prices = prices[split_idx:]
            
            if len(test_prices) < 2:
                return
            
            # Generate forecasts
            price_predictions = []
            direction_predictions = []
            
            for i in range(len(test_prices)):
                current_train = np.concatenate([train_prices, test_prices[:i]]) if i > 0 else train_prices
                
                # Price forecast using exponential smoothing + trend
                smoothed = self.exponential_smoothing(current_train)
                
                # Simple trend component
                if len(current_train) >= 3:
                    trend = (current_train[-1] - current_train[-3]) / 2
                    price_pred = smoothed + trend * 0.3
                else:
                    price_pred = smoothed
                
                price_predictions.append(price_pred)
                
                # Direction prediction
                direction_prob = self.predict_direction_simple(current_train)
                direction_predictions.append(direction_prob > 0.5)
            
            # Calculate metrics
            price_predictions = np.array(price_predictions)
            test_prices = np.array(test_prices)
            
            # Price accuracy metrics
            mae = mean_absolute_error(test_prices, price_predictions)
            rmse = np.sqrt(mean_squared_error(test_prices, price_predictions))
            mape = np.mean(np.abs((test_prices - price_predictions) / test_prices)) * 100
            r2 = r2_score(test_prices, price_predictions)
            
            # Ensure reasonable bounds
            r2 = np.clip(r2, -5, 1)
            mape = np.clip(mape, 0, 100)
            
            # Direction accuracy
            if len(test_prices) > 1:
                actual_directions = np.diff(test_prices) > 0
                pred_directions = np.array(direction_predictions[1:])  # Skip first prediction
                
                # Ensure same length
                min_length = min(len(actual_directions), len(pred_directions))
                if min_length > 0:
                    actual_directions = actual_directions[:min_length]
                    pred_directions = pred_directions[:min_length]
                    direction_accuracy = np.mean(actual_directions == pred_directions) * 100
                else:
                    direction_accuracy = 50
            else:
                direction_accuracy = 50
            
            self.performance_metrics[category] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'n_test_points': len(test_prices),
                'has_direction_model': True,
                'model_type': 'fast_directional'
            }
            
        except Exception as e:
            # Fallback metrics
            self.performance_metrics[category] = {
                'mae': np.std(prices) if len(prices) > 1 else 1000,
                'rmse': np.std(prices) if len(prices) > 1 else 1000,
                'mape': 15.0,
                'r2': 0.0,
                'direction_accuracy': 50.0,
                'n_test_points': len(prices),
                'has_direction_model': False,
                'error': str(e)
            }
    
    def predict(self, steps=3):
        """Generate predictions for all categories"""
        predictions = {}
        
        for category in self.categories:
            if category not in self.models:
                continue
            
            prices = self.models[category]['prices']
            if len(prices) < 3:
                continue
            
            # Generate forecasts
            forecasts = []
            current_prices = prices.copy()
            
            for step in range(steps):
                # Base forecast using exponential smoothing
                smoothed = self.exponential_smoothing(current_prices)
                
                # Trend component
                if len(current_prices) >= 3:
                    trend = (current_prices[-1] - current_prices[-3]) / 2
                else:
                    trend = 0
                
                # Direction adjustment
                direction_prob = self.predict_direction_simple(current_prices)
                direction_adjustment = 1.0
                
                if direction_prob > 0.6:  # Strong up signal
                    direction_adjustment = 1.01 + (direction_prob - 0.6) * 0.05
                elif direction_prob < 0.4:  # Strong down signal
                    direction_adjustment = 0.99 - (0.4 - direction_prob) * 0.05
                
                # Combined forecast
                forecast = (smoothed + trend * 0.3) * direction_adjustment
                
                # Apply volatility constraints
                if len(current_prices) >= 6:
                    recent_vol = np.std(current_prices[-6:])
                    max_change = recent_vol * 1.5
                    forecast = np.clip(forecast, 
                                     current_prices[-1] - max_change,
                                     current_prices[-1] + max_change)
                
                forecasts.append(forecast)
                current_prices = np.append(current_prices, forecast)
            
            # Calculate confidence intervals
            if len(prices) >= 6:
                volatility = np.std(prices[-6:])
            else:
                volatility = np.std(prices) if len(prices) > 1 else prices[-1] * 0.02
            
            lower_bounds = []
            upper_bounds = []
            
            for i, forecast in enumerate(forecasts):
                uncertainty = volatility * (1.1 + i * 0.2)
                lower_bounds.append(forecast - uncertainty)
                upper_bounds.append(forecast + uncertainty)
            
            predictions[category] = {
                'mean': forecasts,
                'lower': lower_bounds,
                'upper': upper_bounds
            }
        
        return predictions
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a category"""
        return self.performance_metrics.get(category, {
            'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0, 
            'direction_accuracy': 0, 'n_test_points': 0,
            'has_direction_model': False, 'model_type': 'fast_directional'
        })