import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedVolatilityForecaster:
    """
    Enhanced forecaster combining volatility awareness with directional prediction
    """
    
    def __init__(self):
        self.models = {}
        self.direction_models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def calculate_technical_features(self, prices):
        """Calculate technical analysis features for direction prediction"""
        if len(prices) < 6:
            return {}
        
        features = {}
        
        # Momentum indicators
        features['momentum_3'] = (prices[-1] - prices[-4]) / prices[-4] if len(prices) >= 4 else 0
        features['momentum_6'] = (prices[-1] - prices[-7]) / prices[-7] if len(prices) >= 7 else 0
        
        # Moving averages
        ma_3 = np.mean(prices[-3:]) if len(prices) >= 3 else prices[-1]
        ma_6 = np.mean(prices[-6:]) if len(prices) >= 6 else prices[-1]
        
        features['price_vs_ma3'] = (prices[-1] - ma_3) / ma_3
        features['price_vs_ma6'] = (prices[-1] - ma_6) / ma_6
        
        # Rate of change
        features['roc_3'] = ((prices[-1] / prices[-4]) - 1) * 100 if len(prices) >= 4 else 0
        
        # Volatility
        if len(prices) >= 6:
            volatility = np.std(prices[-6:])
            features['volatility_ratio'] = volatility / np.mean(prices[-6:])
        else:
            features['volatility_ratio'] = 0.02
        
        # Price position in recent range
        if len(prices) >= 6:
            recent_min = min(prices[-6:])
            recent_max = max(prices[-6:])
            if recent_max > recent_min:
                features['price_position'] = (prices[-1] - recent_min) / (recent_max - recent_min)
            else:
                features['price_position'] = 0.5
        else:
            features['price_position'] = 0.5
        
        # Recent trend
        if len(prices) >= 3:
            recent_change = (prices[-1] - prices[-3]) / prices[-3]
            features['recent_trend'] = recent_change
        else:
            features['recent_trend'] = 0
        
        return features
    
    def create_direction_features(self, data, category):
        """Create features for direction prediction"""
        category_data = data[data['vehicle_class'] == category].copy()
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        if len(prices) < 12:
            return None, None
        
        X = []
        y = []
        
        # Create training samples
        for i in range(6, len(prices) - 1):
            price_history = prices[:i+1]
            features = self.calculate_technical_features(price_history)
            
            # Add volume features if available
            if 'bids_received' in category_data.columns:
                volume_data = category_data['bids_received'].iloc[:i+1].values
                if len(volume_data) >= 3:
                    features['volume_trend'] = (volume_data[-1] - volume_data[-3]) / volume_data[-3]
            
            X.append(list(features.values()))
            
            # Direction: 1 if next price > current price, 0 otherwise
            direction = 1 if prices[i+1] > prices[i] else 0
            y.append(direction)
        
        return np.array(X), np.array(y)
    
    def exponential_smoothing_with_volatility(self, series, alpha=0.3):
        """Enhanced exponential smoothing considering volatility"""
        if len(series) == 0:
            return 0, 0
        
        # Calculate volatility
        if len(series) >= 6:
            volatility = np.std(series[-6:])
        else:
            volatility = np.std(series) if len(series) > 1 else series[-1] * 0.02
        
        # Adaptive alpha based on volatility
        volatility_ratio = volatility / np.mean(series[-6:]) if len(series) >= 6 else 0.02
        adaptive_alpha = alpha / (1 + volatility_ratio)
        
        # Apply exponential smoothing
        smoothed = series[0]
        for value in series[1:]:
            smoothed = adaptive_alpha * value + (1 - adaptive_alpha) * smoothed
        
        return smoothed, volatility
    
    def fit(self, data):
        """Fit models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 20:
                continue
            
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            # Store price model components
            self.models[category] = {
                'prices': prices,
                'dates': category_data['date'].values
            }
            
            # Train direction prediction model
            direction_result = self.create_direction_features(data, category)
            if direction_result is not None:
                X, y = direction_result
                if len(X) > 10:
                    try:
                        # Split for validation
                    split_point = int(len(X) * 0.8)
                    X_train, X_val = X[:split_point], X[split_point:]
                    y_train, y_val = y[:split_point], y[split_point:]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train direction classifier
                    direction_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    direction_model.fit(X_train_scaled, y_train)
                    
                    # Validate
                    val_pred = direction_model.predict(X_val_scaled)
                    direction_accuracy = accuracy_score(y_val, val_pred)
                    
                    if direction_accuracy > 0.45:  # Only use if better than random
                        self.direction_models[category] = direction_model
                        self.scalers[category] = scaler
                    
                except Exception:
                    continue
            
            # Calculate validation metrics for price prediction
            if len(prices) > 20:
                split_point = int(len(prices) * 0.8)
                train_prices = prices[:split_point]
                test_prices = prices[split_point:]
                
                # Simple validation using exponential smoothing
                predictions = []
                for i in range(len(test_prices)):
                    base_pred, _ = self.exponential_smoothing_with_volatility(train_prices)
                    predictions.append(base_pred)
                    if i < len(test_prices) - 1:
                        train_prices = np.append(train_prices, test_prices[i])
                
                # Calculate metrics
                mae = mean_absolute_error(test_prices, predictions)
                rmse = np.sqrt(mean_squared_error(test_prices, predictions))
                mape = np.mean(np.abs((test_prices - predictions) / test_prices)) * 100
                
                # Calculate direction accuracy
                actual_directions = np.diff(test_prices) > 0
                predicted_directions = np.diff(predictions) > 0
                if len(actual_directions) > 0:
                    direction_acc = np.mean(actual_directions == predicted_directions) * 100
                else:
                    direction_acc = 50
                
                self.performance_metrics[category] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2_score(test_prices, predictions),
                    'direction_accuracy': direction_acc
                }
    
    def predict(self, steps=3):
        """Generate predictions for all categories"""
        predictions = {}
        
        for category in self.categories:
            if category not in self.models:
                continue
            
            prices = self.models[category]['prices']
            if len(prices) == 0:
                continue
            
            # Get base forecast using volatility-aware exponential smoothing
            base_forecast, current_vol = self.exponential_smoothing_with_volatility(prices[-12:])
            
            # Calculate trend
            recent_prices = prices[-6:] if len(prices) >= 6 else prices
            if len(recent_prices) >= 2:
                trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
            else:
                trend = 0
            
            # Dampen trend based on volatility
            if current_vol > np.mean([np.std(prices[-12:]) if len(prices) >= 12 else current_vol]):
                trend_damping = 0.5  # High volatility -> dampen trend
            else:
                trend_damping = 0.8
            
            dampened_trend = trend * trend_damping
            
            # Generate predictions
            forecasts = []
            direction_confidences = []
            last_price = prices[-1]
            
            for step in range(1, steps + 1):
                # Base prediction
                raw_forecast = base_forecast + step * dampened_trend
                
                # Apply direction prediction if available
                if category in self.direction_models:
                    try:
                        current_features = self.calculate_technical_features(prices[-6:])
                        feature_vector = np.array([list(current_features.values())])
                        feature_vector_scaled = self.scalers[category].transform(feature_vector)
                        
                        direction_probs = self.direction_models[category].predict_proba(feature_vector_scaled)
                        if len(direction_probs) > 0 and len(direction_probs[0]) > 1:
                            prob_up = direction_probs[0][1]
                            direction_confidences.append(max(prob_up, 1 - prob_up))
                            
                            # Adjust forecast based on direction confidence
                            if prob_up > 0.6:  # Strong upward signal
                                raw_forecast = max(raw_forecast, last_price * 1.001)
                            elif prob_up < 0.4:  # Strong downward signal
                                raw_forecast = min(raw_forecast, last_price * 0.999)
                        else:
                            direction_confidences.append(0.5)
                    except Exception:
                        direction_confidences.append(0.5)
                else:
                    direction_confidences.append(0.5)
                
                # Apply volatility-based constraints
                max_change = current_vol * 1.5
                if step == 1:
                    constrained_forecast = np.clip(raw_forecast,
                                                 last_price - max_change,
                                                 last_price + max_change)
                else:
                    prev_forecast = forecasts[step - 2]
                    step_max_change = max_change * 0.8
                    constrained_forecast = np.clip(raw_forecast,
                                                 prev_forecast - step_max_change,
                                                 prev_forecast + step_max_change)
                
                forecasts.append(constrained_forecast)
                last_price = constrained_forecast
            
            # Calculate confidence intervals
            confidence_width = current_vol * 1.2
            lower_bounds = [f - confidence_width * (1 + i * 0.1) for i, f in enumerate(forecasts)]
            upper_bounds = [f + confidence_width * (1 + i * 0.1) for i, f in enumerate(forecasts)]
            
            predictions[category] = {
                'mean': forecasts,
                'lower': lower_bounds,
                'upper': upper_bounds,
                'direction_confidence': direction_confidences
            }
        
        return predictions
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a category"""
        metrics = self.performance_metrics.get(category, {})
        
        # Add direction model info if available
        if category in self.direction_models:
            metrics['has_direction_model'] = True
            metrics['model_type'] = 'enhanced_volatility_with_direction'
        else:
            metrics['has_direction_model'] = False
            metrics['model_type'] = 'volatility_aware'
        
        return metrics