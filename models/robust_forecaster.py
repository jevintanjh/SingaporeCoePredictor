import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class RobustCOEForecaster:
    """
    Robust COE forecaster with proper implementation and validation
    """
    
    def __init__(self):
        self.models = {}
        self.direction_models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def exponential_smoothing(self, series, alpha=0.3):
        """Proper exponential smoothing implementation"""
        if len(series) == 0:
            return 0
        
        # Initialize with first value
        smoothed = [series[0]]
        
        # Apply exponential smoothing
        for i in range(1, len(series)):
            new_value = alpha * series[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(new_value)
        
        return smoothed[-1]
    
    def simple_trend_forecast(self, series, periods=1):
        """Simple trend-based forecast"""
        if len(series) < 3:
            return [series[-1]] * periods if len(series) > 0 else [0] * periods
        
        # Calculate trend using linear regression on last 6 points
        n_points = min(6, len(series))
        y = series[-n_points:]
        x = np.arange(n_points)
        
        # Simple linear trend
        if n_points > 1:
            slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
            intercept = np.mean(y) - slope * np.mean(x)
            
            # Generate forecasts
            forecasts = []
            for i in range(1, periods + 1):
                forecast = intercept + slope * (n_points + i - 1)
                forecasts.append(forecast)
            
            return forecasts
        else:
            return [series[-1]] * periods
    
    def combined_forecast(self, series, periods=3):
        """Combine exponential smoothing with trend forecast"""
        if len(series) < 2:
            return [series[-1] if len(series) > 0 else 0] * periods
        
        # Exponential smoothing component
        smoothed_value = self.exponential_smoothing(series)
        
        # Trend component
        trend_forecasts = self.simple_trend_forecast(series, periods)
        
        # Combine with weights (60% smoothing, 40% trend)
        combined_forecasts = []
        for i, trend_val in enumerate(trend_forecasts):
            # Dampen trend for longer horizons
            trend_weight = 0.4 * (0.8 ** i)  # Reduce trend influence over time
            smooth_weight = 1 - trend_weight
            
            combined = smooth_weight * smoothed_value + trend_weight * trend_val
            combined_forecasts.append(combined)
            
            # Update smoothed value for next period
            smoothed_value = 0.7 * combined + 0.3 * smoothed_value
        
        return combined_forecasts
    
    def create_technical_features(self, prices):
        """Create robust technical features"""
        if len(prices) < 3:
            return {
                'momentum_3': 0, 'momentum_6': 0, 'volatility': 0.02,
                'ma_ratio': 1, 'trend': 0, 'rsi': 50
            }
        
        features = {}
        
        # Momentum
        if len(prices) >= 4:
            features['momentum_3'] = (prices[-1] - prices[-4]) / prices[-4]
        else:
            features['momentum_3'] = 0
            
        if len(prices) >= 7:
            features['momentum_6'] = (prices[-1] - prices[-7]) / prices[-7]
        else:
            features['momentum_6'] = features['momentum_3']
        
        # Volatility
        if len(prices) >= 6:
            recent_prices = prices[-6:]
            features['volatility'] = np.std(recent_prices) / np.mean(recent_prices)
        else:
            features['volatility'] = 0.02
        
        # Moving average ratio
        if len(prices) >= 6:
            ma = np.mean(prices[-6:])
            features['ma_ratio'] = prices[-1] / ma
        else:
            features['ma_ratio'] = 1
        
        # Simple trend
        if len(prices) >= 3:
            features['trend'] = (prices[-1] - prices[-3]) / prices[-3]
        else:
            features['trend'] = 0
        
        # RSI-like indicator
        if len(prices) >= 6:
            changes = np.diff(prices[-6:])
            gains = changes[changes > 0]
            losses = -changes[changes < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.01
            
            rs = avg_gain / avg_loss
            features['rsi'] = 100 - (100 / (1 + rs))
        else:
            features['rsi'] = 50
        
        return features
    
    def fit(self, data):
        """Fit models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 15:
                continue
            
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            # Store basic model info
            self.models[category] = {
                'prices': prices,
                'dates': category_data['date'].values
            }
            
            # Train direction model if enough data
            if len(prices) > 20:
                self.train_direction_model(data, category)
            
            # Calculate validation metrics
            self.calculate_robust_validation_metrics(prices, category)
    
    def train_direction_model(self, data, category):
        """Train direction prediction model"""
        category_data = data[data['vehicle_class'] == category].copy()
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        if len(prices) < 20:
            return
        
        X = []
        y = []
        
        # Create training samples
        for i in range(6, len(prices) - 1):
            features = self.create_technical_features(prices[:i+1])
            X.append(list(features.values()))
            
            # Direction: 1 if next price > current price
            y.append(1 if prices[i+1] > prices[i] else 0)
        
        if len(X) < 10:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        try:
            # Split for validation
            split_idx = max(10, int(len(X) * 0.8))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train classifier
            clf = RandomForestClassifier(
                n_estimators=50, 
                max_depth=4, 
                min_samples_split=5,
                random_state=42
            )
            clf.fit(X_train_scaled, y_train)
            
            # Validate
            if len(X_test) > 0:
                y_pred = clf.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Only keep if better than random
                if accuracy > 0.52:
                    self.direction_models[category] = clf
                    self.scalers[category] = scaler
        
        except Exception:
            pass
    
    def calculate_robust_validation_metrics(self, prices, category):
        """Calculate validation metrics with proper implementation"""
        if len(prices) < 10:
            return
        
        try:
            # Use last 30% for validation
            split_idx = max(5, int(len(prices) * 0.7))
            train_prices = prices[:split_idx]
            test_prices = prices[split_idx:]
            
            if len(test_prices) == 0:
                return
            
            # Generate forecasts using walk-forward validation
            predictions = []
            
            for i in range(len(test_prices)):
                # Use all data up to current point
                current_train = np.concatenate([train_prices, test_prices[:i]]) if i > 0 else train_prices
                
                # Make 1-step ahead forecast
                forecast = self.combined_forecast(current_train, 1)[0]
                predictions.append(forecast)
            
            # Calculate metrics only if we have valid predictions
            if len(predictions) == len(test_prices) and len(test_prices) > 0:
                # Ensure no invalid values
                predictions = np.array(predictions)
                test_prices = np.array(test_prices)
                
                # Remove any infinite or NaN values
                valid_mask = np.isfinite(predictions) & np.isfinite(test_prices) & (test_prices > 0)
                if np.sum(valid_mask) < len(test_prices) * 0.5:
                    return
                
                predictions = predictions[valid_mask]
                test_prices = test_prices[valid_mask]
                
                if len(predictions) < 2:
                    return
                
                # Calculate metrics
                mae = mean_absolute_error(test_prices, predictions)
                rmse = np.sqrt(mean_squared_error(test_prices, predictions))
                
                # MAPE with protection against division by zero
                mape_values = np.abs((test_prices - predictions) / test_prices) * 100
                mape = np.mean(mape_values[np.isfinite(mape_values)])
                
                # R² with bounds checking
                r2 = r2_score(test_prices, predictions)
                
                # Ensure R² is reasonable (clip extreme values)
                r2 = np.clip(r2, -10, 1)
                
                # Direction accuracy
                if len(test_prices) > 1:
                    actual_dirs = np.diff(test_prices) > 0
                    pred_dirs = np.diff(predictions) > 0
                    dir_acc = np.mean(actual_dirs == pred_dirs) * 100
                else:
                    dir_acc = 50
                
                self.performance_metrics[category] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'direction_accuracy': dir_acc,
                    'n_test_points': len(test_prices)
                }
                
        except Exception as e:
            # Fallback to simple metrics
            self.performance_metrics[category] = {
                'mae': np.std(prices),
                'rmse': np.std(prices),
                'mape': 10.0,
                'r2': 0.0,
                'direction_accuracy': 50.0,
                'n_test_points': 0,
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
            
            # Generate base forecast
            base_forecasts = self.combined_forecast(prices, steps)
            
            # Apply direction adjustment if available
            if category in self.direction_models:
                base_forecasts = self.apply_direction_adjustment(prices, base_forecasts, category)
            
            # Apply volatility constraints
            constrained_forecasts = self.apply_volatility_constraints(prices, base_forecasts)
            
            # Calculate confidence intervals
            if len(prices) >= 6:
                volatility = np.std(prices[-6:])
            else:
                volatility = np.std(prices) if len(prices) > 1 else prices[-1] * 0.02
            
            lower_bounds = []
            upper_bounds = []
            
            for i, forecast in enumerate(constrained_forecasts):
                # Expanding uncertainty
                uncertainty = volatility * (1.2 + i * 0.3)
                lower_bounds.append(forecast - uncertainty)
                upper_bounds.append(forecast + uncertainty)
            
            predictions[category] = {
                'mean': constrained_forecasts,
                'lower': lower_bounds,
                'upper': upper_bounds
            }
        
        return predictions
    
    def apply_direction_adjustment(self, prices, forecasts, category):
        """Apply direction model adjustment"""
        try:
            current_features = self.create_technical_features(prices)
            feature_vector = np.array([list(current_features.values())])
            feature_scaled = self.scalers[category].transform(feature_vector)
            
            direction_probs = self.direction_models[category].predict_proba(feature_scaled)[0]
            prob_up = direction_probs[1] if len(direction_probs) > 1 else 0.5
            
            # Adjust forecasts based on direction signal
            adjusted_forecasts = []
            for i, forecast in enumerate(forecasts):
                if prob_up > 0.6:  # Strong up signal
                    adjustment = 1 + (prob_up - 0.5) * 0.02  # Max 1% adjustment
                elif prob_up < 0.4:  # Strong down signal
                    adjustment = 1 - (0.5 - prob_up) * 0.02
                else:
                    adjustment = 1.0
                
                adjusted_forecasts.append(forecast * adjustment)
            
            return adjusted_forecasts
        
        except Exception:
            return forecasts
    
    def apply_volatility_constraints(self, prices, forecasts):
        """Apply volatility-based constraints"""
        if len(prices) < 3:
            return forecasts
        
        # Calculate recent volatility
        recent_vol = np.std(prices[-6:]) if len(prices) >= 6 else np.std(prices[-3:])
        
        # Maximum change per step
        max_change_per_step = recent_vol * 1.5
        
        constrained_forecasts = []
        last_price = prices[-1]
        
        for i, forecast in enumerate(forecasts):
            if i == 0:
                # First step: constrain from last actual price
                constrained = np.clip(forecast,
                                    last_price - max_change_per_step,
                                    last_price + max_change_per_step)
            else:
                # Subsequent steps: constrain from previous forecast
                prev_forecast = constrained_forecasts[i-1]
                step_constraint = max_change_per_step * 0.8
                constrained = np.clip(forecast,
                                    prev_forecast - step_constraint,
                                    prev_forecast + step_constraint)
            
            constrained_forecasts.append(constrained)
        
        return constrained_forecasts
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a category"""
        metrics = self.performance_metrics.get(category, {})
        
        if category in self.direction_models:
            metrics['has_direction_model'] = True
            metrics['model_type'] = 'robust_enhanced'
        else:
            metrics['has_direction_model'] = False
            metrics['model_type'] = 'robust_baseline'
        
        return metrics