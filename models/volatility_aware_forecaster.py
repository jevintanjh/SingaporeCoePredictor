import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class VolatilityAwareCOEForecaster:
    """
    COE price forecaster that explicitly models and incorporates volatility into predictions
    """
    
    def __init__(self):
        self.models = {}
        self.volatility_models = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def calculate_rolling_volatility(self, prices, window=6):
        """Calculate rolling volatility using different methods"""
        if len(prices) < window:
            return [np.std(prices)] * len(prices)
        
        volatilities = []
        for i in range(len(prices)):
            if i < window - 1:
                # Use available data for early periods
                vol = np.std(prices[:i+1])
            else:
                # Use rolling window
                vol = np.std(prices[i-window+1:i+1])
            volatilities.append(vol)
        
        return volatilities
    
    def calculate_garch_volatility(self, returns, window=12):
        """Simple GARCH-like volatility estimation"""
        if len(returns) < 3:
            return [np.std(returns)] * len(returns)
        
        volatilities = []
        alpha = 0.1  # Weight for recent returns
        beta = 0.85  # Weight for previous volatility
        
        # Initialize with sample volatility
        initial_vol = np.std(returns[:min(6, len(returns))])
        volatilities.append(initial_vol)
        
        for i in range(1, len(returns)):
            if i == 1:
                prev_vol = initial_vol
            else:
                prev_vol = volatilities[i-1]
            
            # GARCH(1,1) style update
            new_vol = np.sqrt(alpha * returns[i-1]**2 + beta * prev_vol**2)
            volatilities.append(new_vol)
        
        return volatilities
    
    def detect_volatility_regime(self, volatilities):
        """Detect high/low volatility regimes"""
        if len(volatilities) < 6:
            return 'normal'
        
        recent_vol = np.mean(volatilities[-3:])  # Last 3 periods
        historical_vol = np.mean(volatilities[:-3])  # Earlier periods
        
        if recent_vol > historical_vol * 1.5:
            return 'high'
        elif recent_vol < historical_vol * 0.7:
            return 'low'
        else:
            return 'normal'
    
    def volatility_adjusted_prediction(self, base_forecast, current_vol, historical_vols, regime):
        """Adjust base forecast based on volatility considerations"""
        if len(historical_vols) < 3:
            return base_forecast
        
        avg_historical_vol = np.mean(historical_vols[-12:])  # Last 12 periods
        
        # Volatility ratio
        vol_ratio = current_vol / avg_historical_vol if avg_historical_vol > 0 else 1
        
        # Regime-based adjustments
        if regime == 'high':
            # In high volatility periods, predictions should be more uncertain
            # and potentially mean-reverting
            vol_adjustment = 0.7  # Dampen extreme movements
        elif regime == 'low':
            # In low volatility periods, trends are more persistent
            vol_adjustment = 1.1  # Slightly amplify trends
        else:
            vol_adjustment = 1.0
        
        # Apply volatility adjustment
        adjusted_forecast = base_forecast * vol_adjustment
        
        return adjusted_forecast
    
    def exponential_smoothing_with_volatility(self, series, alpha=0.3):
        """Enhanced exponential smoothing considering volatility"""
        if len(series) == 0:
            return 0, 0
        
        # Calculate returns and volatility
        returns = np.diff(series) / series[:-1]
        volatilities = self.calculate_rolling_volatility(series)
        
        # Adaptive alpha based on volatility
        current_vol = volatilities[-1] if volatilities else 0
        avg_vol = np.mean(volatilities[-6:]) if len(volatilities) >= 6 else current_vol
        
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
            # Higher volatility -> lower alpha (more smoothing)
            adaptive_alpha = alpha / (1 + vol_ratio * 0.5)
        else:
            adaptive_alpha = alpha
        
        # Apply exponential smoothing
        smoothed = series[0]
        for value in series[1:]:
            smoothed = adaptive_alpha * value + (1 - adaptive_alpha) * smoothed
        
        return smoothed, current_vol
    
    def fit(self, data):
        """Fit volatility-aware models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 10:
                continue
            
            # Sort by date
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            # Calculate returns and volatilities
            returns = np.diff(prices) / prices[:-1]
            rolling_volatilities = self.calculate_rolling_volatility(prices)
            garch_volatilities = self.calculate_garch_volatility(returns)
            
            # Store model components
            self.models[category] = {
                'prices': prices,
                'returns': returns,
                'dates': category_data['date'].values,
                'rolling_volatilities': rolling_volatilities,
                'garch_volatilities': garch_volatilities,
                'current_regime': self.detect_volatility_regime(rolling_volatilities)
            }
            
            # Fit volatility model
            self.volatility_models[category] = {
                'avg_volatility': np.mean(rolling_volatilities[-12:]) if len(rolling_volatilities) >= 12 else np.mean(rolling_volatilities),
                'volatility_trend': np.mean(np.diff(rolling_volatilities[-6:])) if len(rolling_volatilities) >= 6 else 0
            }
            
            # Calculate validation metrics if enough data
            if len(prices) > 20:
                split_point = int(len(prices) * 0.8)
                train_prices = prices[:split_point]
                test_prices = prices[split_point:]
                
                # Generate predictions with volatility awareness
                predictions = []
                for i in range(len(test_prices)):
                    # Base prediction using exponential smoothing
                    base_pred, current_vol = self.exponential_smoothing_with_volatility(train_prices)
                    
                    # Get volatility context
                    train_vols = self.calculate_rolling_volatility(train_prices)
                    regime = self.detect_volatility_regime(train_vols)
                    
                    # Apply volatility adjustment
                    adjusted_pred = self.volatility_adjusted_prediction(
                        base_pred, current_vol, train_vols, regime
                    )
                    
                    predictions.append(adjusted_pred)
                    
                    # Update training data for next prediction
                    if i < len(test_prices) - 1:
                        train_prices = np.append(train_prices, test_prices[i])
                
                # Calculate metrics
                mae = mean_absolute_error(test_prices, predictions)
                rmse = np.sqrt(mean_squared_error(test_prices, predictions))
                mape = np.mean(np.abs((test_prices - predictions) / test_prices)) * 100
                
                # Calculate volatility tracking metrics
                test_vols = self.calculate_rolling_volatility(test_prices)
                pred_vols = self.calculate_rolling_volatility(predictions)
                vol_correlation = np.corrcoef(test_vols, pred_vols)[0, 1] if len(test_vols) > 1 else 0
                
                self.performance_metrics[category] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2_score(test_prices, predictions),
                    'volatility_correlation': vol_correlation
                }
    
    def predict(self, steps=3):
        """Generate volatility-aware predictions for all categories"""
        predictions = {}
        
        for category in self.categories:
            if category not in self.models:
                continue
            
            prices = self.models[category]['prices']
            volatilities = self.models[category]['rolling_volatilities']
            regime = self.models[category]['current_regime']
            
            if len(prices) == 0:
                continue
            
            # Get base forecast using volatility-aware exponential smoothing
            base_forecast, current_vol = self.exponential_smoothing_with_volatility(prices[-12:])
            
            # Calculate trend with volatility dampening
            recent_prices = prices[-6:] if len(prices) >= 6 else prices
            returns = np.diff(recent_prices) / recent_prices[:-1]
            trend = np.mean(returns) * recent_prices[-1]
            
            # Dampen trend based on volatility
            vol_model = self.volatility_models[category]
            if current_vol > vol_model['avg_volatility']:
                trend_damping = 0.5  # High volatility -> dampen trend
            else:
                trend_damping = 0.8  # Normal/low volatility -> moderate dampening
            
            dampened_trend = trend * trend_damping
            
            # Generate step-wise predictions with evolving volatility
            forecasts = []
            volatility_forecasts = []
            last_price = prices[-1]
            
            for step in range(1, steps + 1):
                # Base prediction
                raw_forecast = base_forecast + step * dampened_trend
                
                # Project volatility forward
                vol_trend = vol_model['volatility_trend']
                projected_vol_1 = current_vol + step * vol_trend * 0.5
                projected_vol_2 = current_vol * 0.3
                projected_vol = projected_vol_1 if projected_vol_1 > projected_vol_2 else projected_vol_2
                volatility_forecasts.append(projected_vol)
                
                # Apply volatility-based constraints
                if step == 1:
                    max_change = projected_vol * 1.5  # Allow volatility-based movement
                    constrained_forecast = np.clip(raw_forecast,
                                                 last_price - max_change,
                                                 last_price + max_change)
                else:
                    prev_forecast = forecasts[step - 2]
                    max_change = projected_vol * 1.0
                    constrained_forecast = np.clip(raw_forecast,
                                                 prev_forecast - max_change,
                                                 prev_forecast + max_change)
                
                # Apply regime-based adjustment
                final_forecast = self.volatility_adjusted_prediction(
                    constrained_forecast, projected_vol, volatilities, regime
                )
                
                forecasts.append(final_forecast)
            
            # Calculate dynamic confidence intervals based on projected volatility
            lower_bounds = []
            upper_bounds = []
            
            for i, (forecast, vol) in enumerate(zip(forecasts, volatility_forecasts)):
                # Confidence interval width based on projected volatility
                confidence_width = vol * (1.0 + i * 0.2)  # Expanding uncertainty
                lower_bounds.append(forecast - confidence_width)
                upper_bounds.append(forecast + confidence_width)
            
            predictions[category] = {
                'mean': forecasts,
                'lower': lower_bounds,
                'upper': upper_bounds,
                'volatility_forecast': volatility_forecasts,
                'regime': regime
            }
        
        return predictions
    
    def get_performance_metrics(self, category):
        """Get performance metrics including volatility tracking"""
        return self.performance_metrics.get(category, {})
    
    def get_volatility_insights(self, category):
        """Get volatility-specific insights"""
        if category not in self.models:
            return {}
        
        model = self.models[category]
        vol_model = self.volatility_models[category]
        
        current_vol = model['rolling_volatilities'][-1] if model['rolling_volatilities'] else 0
        avg_vol = vol_model['avg_volatility']
        
        return {
            'current_volatility': current_vol,
            'average_volatility': avg_vol,
            'volatility_ratio': current_vol / avg_vol if avg_vol > 0 else 1,
            'regime': model['current_regime'],
            'volatility_trend': vol_model['volatility_trend']
        }