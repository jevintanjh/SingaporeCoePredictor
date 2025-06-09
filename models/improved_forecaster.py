import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ImprovedCOEForecaster:
    """
    Improved COE forecaster with enhanced directional accuracy
    """
    
    def __init__(self):
        self.models = {}
        self.direction_models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def calculate_momentum_features(self, prices):
        """Calculate momentum and technical indicators"""
        if len(prices) < 3:
            return {
                'momentum_3': 0, 'momentum_6': 0, 'roc_3': 0, 'roc_6': 0,
                'ma_ratio_3': 1, 'ma_ratio_6': 1, 'volatility': 0.02,
                'trend_strength': 0, 'price_position': 0.5
            }
        
        features = {}
        
        # Momentum indicators
        if len(prices) >= 4:
            features['momentum_3'] = (prices[-1] - prices[-4]) / prices[-4]
        else:
            features['momentum_3'] = 0
            
        if len(prices) >= 7:
            features['momentum_6'] = (prices[-1] - prices[-7]) / prices[-7]
        else:
            features['momentum_6'] = 0
        
        # Rate of change
        if len(prices) >= 4:
            features['roc_3'] = ((prices[-1] / prices[-4]) - 1) * 100
        else:
            features['roc_3'] = 0
            
        if len(prices) >= 7:
            features['roc_6'] = ((prices[-1] / prices[-7]) - 1) * 100
        else:
            features['roc_6'] = 0
        
        # Moving average ratios
        if len(prices) >= 3:
            ma_3 = np.mean(prices[-3:])
            features['ma_ratio_3'] = prices[-1] / ma_3
        else:
            features['ma_ratio_3'] = 1
            
        if len(prices) >= 6:
            ma_6 = np.mean(prices[-6:])
            features['ma_ratio_6'] = prices[-1] / ma_6
        else:
            features['ma_ratio_6'] = 1
        
        # Volatility
        if len(prices) >= 6:
            features['volatility'] = np.std(prices[-6:]) / np.mean(prices[-6:])
        else:
            features['volatility'] = 0.02
        
        # Trend strength
        if len(prices) >= 6:
            x = np.arange(6)
            y = prices[-6:]
            trend_slope = np.polyfit(x, y, 1)[0]
            features['trend_strength'] = trend_slope / np.mean(y)
        else:
            features['trend_strength'] = 0
        
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
        
        return features
    
    def create_directional_dataset(self, data, category):
        """Create dataset for direction prediction"""
        category_data = data[data['vehicle_class'] == category].copy()
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        if len(prices) < 15:
            return None, None
        
        X = []
        y = []
        
        # Create training samples with lookback window
        for i in range(8, len(prices) - 1):
            price_window = prices[:i+1]
            features = self.calculate_momentum_features(price_window)
            
            # Add lagged returns
            for lag in [1, 2, 3]:
                if i >= lag:
                    lag_return = (prices[i] - prices[i-lag]) / prices[i-lag]
                    features[f'return_lag_{lag}'] = lag_return
                else:
                    features[f'return_lag_{lag}'] = 0
            
            # Add volume features if available
            if 'bids_received' in category_data.columns:
                volume_data = category_data['bids_received'].iloc[:i+1].values
                if len(volume_data) >= 3:
                    vol_change = (volume_data[-1] - volume_data[-3]) / volume_data[-3]
                    features['volume_change'] = vol_change
                else:
                    features['volume_change'] = 0
            else:
                features['volume_change'] = 0
            
            X.append(list(features.values()))
            
            # Target: 1 if price goes up next period, 0 if down
            direction = 1 if prices[i+1] > prices[i] else 0
            y.append(direction)
        
        return np.array(X), np.array(y)
    
    def fit(self, data):
        """Fit models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 20:
                continue
            
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            # Store basic model info
            self.models[category] = {
                'prices': prices,
                'dates': category_data['date'].values
            }
            
            # Train directional prediction model
            dataset_result = self.create_directional_dataset(data, category)
            if dataset_result is not None:
                X, y = dataset_result
                if X is not None and len(X) > 15:
                    try:
                        # Split data
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train direction classifier
                        clf = RandomForestClassifier(
                            n_estimators=100, 
                            max_depth=6, 
                            min_samples_split=5,
                            random_state=42
                        )
                        clf.fit(X_train_scaled, y_train)
                        
                        # Test performance
                        y_pred = clf.predict(X_test_scaled)
                        direction_accuracy = accuracy_score(y_test, y_pred)
                        
                        # Only keep model if it's better than random + margin
                        if direction_accuracy > 0.55:
                            self.direction_models[category] = clf
                            self.scalers[category] = scaler
                    
                    except Exception:
                        pass
            
            # Calculate overall performance metrics
            if len(prices) > 25:
                self.calculate_validation_metrics(prices, category)
    
    def calculate_validation_metrics(self, prices, category):
        """Calculate validation metrics using walk-forward approach"""
        try:
            # Use last 20% for validation
            split_idx = int(len(prices) * 0.8)
            train_prices = prices[:split_idx]
            test_prices = prices[split_idx:]
            
            # Simple exponential smoothing for baseline
            alpha = 0.3
            predictions = []
            current_train = train_prices.copy()
            
            for i in range(len(test_prices)):
                # Exponential smoothing prediction
                smoothed = current_train[0]
                for price in current_train[1:]:
                    smoothed = alpha * price + (1 - alpha) * smoothed
                
                # Add simple trend
                if len(current_train) >= 3:
                    trend = (current_train[-1] - current_train[-3]) / 2
                    pred = smoothed + trend * 0.5
                else:
                    pred = smoothed
                
                predictions.append(pred)
                
                # Update training set with actual value
                if i < len(test_prices) - 1:
                    current_train = np.append(current_train, test_prices[i])
            
            # Calculate metrics
            mae = mean_absolute_error(test_prices, predictions)
            rmse = np.sqrt(mean_squared_error(test_prices, predictions))
            mape = np.mean(np.abs((test_prices - predictions) / test_prices)) * 100
            r2 = r2_score(test_prices, predictions)
            
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
                'direction_accuracy': dir_acc
            }
            
        except Exception:
            pass
    
    def predict(self, steps=3):
        """Generate predictions for all categories"""
        predictions = {}
        
        for category in self.categories:
            if category not in self.models:
                continue
            
            prices = self.models[category]['prices']
            if len(prices) < 5:
                continue
            
            # Base prediction using exponential smoothing
            alpha = 0.3
            smoothed = prices[0]
            for price in prices[1:]:
                smoothed = alpha * price + (1 - alpha) * smoothed
            
            # Calculate recent trend
            if len(prices) >= 6:
                recent_trend = (prices[-1] - prices[-6]) / 5
            else:
                recent_trend = 0
            
            # Generate forecasts
            forecasts = []
            direction_confidences = []
            current_price = prices[-1]
            
            for step in range(1, steps + 1):
                # Base forecast
                base_forecast = smoothed + step * recent_trend * 0.6
                
                # Apply directional model if available
                if category in self.direction_models:
                    try:
                        # Create features for current state
                        current_features = self.calculate_momentum_features(prices[-8:])
                        
                        # Add recent returns
                        for lag in [1, 2, 3]:
                            if len(prices) > lag:
                                lag_return = (prices[-1] - prices[-1-lag]) / prices[-1-lag]
                                current_features[f'return_lag_{lag}'] = lag_return
                            else:
                                current_features[f'return_lag_{lag}'] = 0
                        
                        current_features['volume_change'] = 0  # Default
                        
                        # Get direction prediction
                        feature_vector = np.array([list(current_features.values())])
                        feature_scaled = self.scalers[category].transform(feature_vector)
                        direction_probs = self.direction_models[category].predict_proba(feature_scaled)[0]
                        
                        prob_up = direction_probs[1] if len(direction_probs) > 1 else 0.5
                        confidence = max(prob_up, 1 - prob_up)
                        direction_confidences.append(confidence)
                        
                        # Adjust forecast based on direction signal
                        if prob_up > 0.6:  # Strong up signal
                            direction_adjustment = 1.02
                        elif prob_up < 0.4:  # Strong down signal
                            direction_adjustment = 0.98
                        else:
                            direction_adjustment = 1.0
                        
                        base_forecast *= direction_adjustment
                        
                    except Exception:
                        direction_confidences.append(0.5)
                else:
                    direction_confidences.append(0.5)
                
                # Apply volatility constraints
                if len(prices) >= 6:
                    recent_volatility = np.std(prices[-6:])
                else:
                    recent_volatility = current_price * 0.02
                
                max_change = recent_volatility * 2
                
                if step == 1:
                    constrained_forecast = np.clip(
                        base_forecast,
                        current_price - max_change,
                        current_price + max_change
                    )
                else:
                    prev_forecast = forecasts[step - 2]
                    step_constraint = max_change * 0.8
                    constrained_forecast = np.clip(
                        base_forecast,
                        prev_forecast - step_constraint,
                        prev_forecast + step_constraint
                    )
                
                forecasts.append(constrained_forecast)
                current_price = constrained_forecast
            
            # Calculate confidence intervals
            if len(prices) >= 6:
                uncertainty = np.std(prices[-6:]) * 1.5
            else:
                uncertainty = prices[-1] * 0.05
            
            lower_bounds = [f - uncertainty * (1 + i * 0.1) for i, f in enumerate(forecasts)]
            upper_bounds = [f + uncertainty * (1 + i * 0.1) for i, f in enumerate(forecasts)]
            
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
        
        if category in self.direction_models:
            metrics['has_direction_model'] = True
            metrics['model_type'] = 'enhanced_directional'
        else:
            metrics['has_direction_model'] = False
            metrics['model_type'] = 'baseline_smoothing'
        
        return metrics