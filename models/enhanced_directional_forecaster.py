import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedDirectionalForecaster:
    """
    Enhanced directional forecaster with improved feature engineering and ensemble methods
    """
    
    def __init__(self):
        self.models = {}
        self.direction_models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def calculate_advanced_features(self, prices):
        """Calculate advanced technical features for better direction prediction"""
        if len(prices) < 6:
            return {
                'price_momentum_3': 0, 'price_momentum_6': 0, 'price_acceleration': 0,
                'volatility_ratio': 1, 'ma_convergence': 0, 'rsi': 50,
                'bollinger_position': 0.5, 'trend_strength': 0, 'volume_price_trend': 0,
                'momentum_divergence': 0, 'support_resistance': 0.5
            }
        
        features = {}
        prices = np.array(prices)
        
        # Multi-timeframe momentum
        if len(prices) >= 4:
            features['price_momentum_3'] = (prices[-1] - prices[-4]) / prices[-4]
        else:
            features['price_momentum_3'] = 0
            
        if len(prices) >= 7:
            features['price_momentum_6'] = (prices[-1] - prices[-7]) / prices[-7]
        else:
            features['price_momentum_6'] = features['price_momentum_3']
        
        # Price acceleration (momentum of momentum)
        if len(prices) >= 6:
            recent_momentum = (prices[-1] - prices[-3]) / prices[-3]
            older_momentum = (prices[-3] - prices[-6]) / prices[-6]
            features['price_acceleration'] = recent_momentum - older_momentum
        else:
            features['price_acceleration'] = 0
        
        # Volatility ratio (short vs long term)
        if len(prices) >= 12:
            short_vol = np.std(prices[-6:]) / np.mean(prices[-6:])
            long_vol = np.std(prices[-12:]) / np.mean(prices[-12:])
            features['volatility_ratio'] = short_vol / (long_vol + 1e-8)
        else:
            features['volatility_ratio'] = 1
        
        # Moving average convergence
        if len(prices) >= 6:
            ma_short = np.mean(prices[-3:])
            ma_long = np.mean(prices[-6:])
            features['ma_convergence'] = (ma_short - ma_long) / ma_long
        else:
            features['ma_convergence'] = 0
        
        # Enhanced RSI
        if len(prices) >= 8:
            changes = np.diff(prices[-8:])
            gains = changes[changes > 0]
            losses = -changes[changes < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.01
            
            rs = avg_gain / avg_loss
            features['rsi'] = 100 - (100 / (1 + rs))
        else:
            features['rsi'] = 50
        
        # Bollinger Band position
        if len(prices) >= 6:
            ma = np.mean(prices[-6:])
            std = np.std(prices[-6:])
            if std > 0:
                features['bollinger_position'] = (prices[-1] - ma) / (2 * std) + 0.5
                features['bollinger_position'] = np.clip(features['bollinger_position'], 0, 1)
            else:
                features['bollinger_position'] = 0.5
        else:
            features['bollinger_position'] = 0.5
        
        # Trend strength using linear regression
        if len(prices) >= 6:
            x = np.arange(6)
            y = prices[-6:]
            slope = np.polyfit(x, y, 1)[0]
            features['trend_strength'] = slope / np.mean(y)
        else:
            features['trend_strength'] = 0
        
        # Volume-Price Trend proxy (using price volatility as volume proxy)
        if len(prices) >= 6:
            price_changes = np.diff(prices[-6:])
            volatilities = np.abs(price_changes)
            # Higher volatility suggests higher "volume"
            vpt = np.sum(price_changes * volatilities) / (np.sum(volatilities) + 1e-8)
            features['volume_price_trend'] = vpt / np.mean(prices[-6:])
        else:
            features['volume_price_trend'] = 0
        
        # Momentum divergence
        if len(prices) >= 8:
            price_change = (prices[-1] - prices[-4]) / prices[-4]
            momentum_ma = np.mean([(prices[i] - prices[i-3]) / prices[i-3] for i in range(-4, 0)])
            features['momentum_divergence'] = price_change - momentum_ma
        else:
            features['momentum_divergence'] = 0
        
        # Support/Resistance level
        if len(prices) >= 12:
            recent_high = np.max(prices[-12:])
            recent_low = np.min(prices[-12:])
            if recent_high > recent_low:
                features['support_resistance'] = (prices[-1] - recent_low) / (recent_high - recent_low)
            else:
                features['support_resistance'] = 0.5
        else:
            features['support_resistance'] = 0.5
        
        return features
    
    def create_directional_training_data(self, prices, min_history=8):
        """Create enhanced training dataset for direction prediction"""
        if len(prices) < min_history + 2:
            return None, None
        
        X = []
        y = []
        
        # Create samples with varying lookback windows
        for i in range(min_history, len(prices) - 1):
            # Use expanding window for more stable features
            window_prices = prices[:i+1]
            features = self.calculate_advanced_features(window_prices)
            
            # Add price level context
            features['price_level'] = prices[i] / np.mean(prices[max(0, i-12):i+1])
            
            # Add recent volatility context
            if i >= 6:
                recent_vol = np.std(prices[max(0, i-6):i+1]) / np.mean(prices[max(0, i-6):i+1])
                features['volatility_regime'] = 1 if recent_vol > np.median([np.std(prices[max(0, j-6):j+1]) / np.mean(prices[max(0, j-6):j+1]) for j in range(6, i)]) else 0
            else:
                features['volatility_regime'] = 0
            
            X.append(list(features.values()))
            
            # Direction: 1 if next price > current price, 0 otherwise
            y.append(1 if prices[i+1] > prices[i] else 0)
        
        return np.array(X), np.array(y)
    
    def exponential_smoothing(self, series, alpha=0.3):
        """Proper exponential smoothing implementation"""
        if len(series) == 0:
            return 0
        
        smoothed = [series[0]]
        for i in range(1, len(series)):
            new_value = alpha * series[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(new_value)
        
        return smoothed[-1]
    
    def trend_forecast(self, series, periods=1):
        """Enhanced trend forecasting with momentum consideration"""
        if len(series) < 3:
            return [series[-1]] * periods if len(series) > 0 else [0] * periods
        
        # Use different trend windows based on series length
        n_points = min(8, len(series))
        y = series[-n_points:]
        x = np.arange(n_points)
        
        if n_points > 2:
            # Weighted linear regression (recent points have higher weight)
            weights = np.exp(np.linspace(0, 1, n_points))
            weights = weights / np.sum(weights)
            
            # Calculate weighted trend
            mean_x = np.average(x, weights=weights)
            mean_y = np.average(y, weights=weights)
            
            slope = np.sum(weights * (x - mean_x) * (y - mean_y)) / np.sum(weights * (x - mean_x)**2)
            intercept = mean_y - slope * mean_x
            
            # Generate forecasts with momentum damping
            forecasts = []
            for i in range(1, periods + 1):
                # Dampen trend for longer horizons
                damping = 0.8 ** (i - 1)
                forecast = intercept + slope * (n_points + i - 1) * damping
                forecasts.append(forecast)
            
            return forecasts
        else:
            return [series[-1]] * periods
    
    def fit(self, data):
        """Fit enhanced models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 20:  # Need more data for robust directional modeling
                continue
            
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            # Store basic model info
            self.models[category] = {
                'prices': prices,
                'dates': category_data['date'].values
            }
            
            # Train enhanced direction model
            self.train_enhanced_direction_model(prices, category)
            
            # Calculate validation metrics
            self.calculate_robust_validation_metrics(prices, category)
    
    def train_enhanced_direction_model(self, prices, category):
        """Train enhanced direction prediction model with ensemble methods"""
        X, y = self.create_directional_training_data(prices)
        
        if X is None or len(X) < 15:
            return
        
        try:
            # Split for validation (use more recent data for testing)
            split_idx = max(10, int(len(X) * 0.75))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Try multiple models and select the best
            models_to_try = [
                ('rf', RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=6, 
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced'
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    min_samples_split=10,
                    random_state=42
                ))
            ]
            
            best_model = None
            best_accuracy = 0
            best_scaler = None
            
            for model_name, model in models_to_try:
                # Cross-validation on training set
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)//4), scoring='accuracy')
                cv_mean = np.mean(cv_scores)
                
                # Train on full training set and test
                model.fit(X_train_scaled, y_train)
                
                if len(X_test) > 0:
                    y_pred = model.predict(X_test_scaled)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    
                    # Combined score (70% CV, 30% test)
                    combined_score = 0.7 * cv_mean + 0.3 * test_accuracy
                else:
                    combined_score = cv_mean
                
                # Only keep if significantly better than random (>55%)
                if combined_score > 0.55 and combined_score > best_accuracy:
                    best_accuracy = combined_score
                    best_model = model
                    best_scaler = scaler
            
            if best_model is not None:
                self.direction_models[category] = best_model
                self.scalers[category] = best_scaler
                
        except Exception as e:
            print(f"Direction model training failed for {category}: {e}")
    
    def calculate_robust_validation_metrics(self, prices, category):
        """Calculate validation metrics with enhanced directional accuracy"""
        if len(prices) < 12:
            return
        
        try:
            # Use walk-forward validation for more realistic assessment
            split_idx = max(8, int(len(prices) * 0.7))
            train_prices = prices[:split_idx]
            test_prices = prices[split_idx:]
            
            if len(test_prices) < 3:
                return
            
            # Generate forecasts and direction predictions
            price_predictions = []
            direction_predictions = []
            
            for i in range(len(test_prices)):
                current_train = np.concatenate([train_prices, test_prices[:i]]) if i > 0 else train_prices
                
                # Price forecast
                smoothed = self.exponential_smoothing(current_train)
                trend_forecast = self.trend_forecast(current_train, 1)[0]
                price_pred = 0.6 * smoothed + 0.4 * trend_forecast
                price_predictions.append(price_pred)
                
                # Direction prediction using enhanced model
                if category in self.direction_models and len(current_train) >= 8:
                    features = self.calculate_advanced_features(current_train)
                    
                    # Add context features
                    features['price_level'] = current_train[-1] / np.mean(current_train[-6:])
                    if len(current_train) >= 12:
                        recent_vol = np.std(current_train[-6:]) / np.mean(current_train[-6:])
                        median_vol = np.median([np.std(current_train[max(0, j-6):j+1]) / np.mean(current_train[max(0, j-6):j+1]) 
                                              for j in range(6, len(current_train), 3)])
                        features['volatility_regime'] = 1 if recent_vol > median_vol else 0
                    else:
                        features['volatility_regime'] = 0
                    
                    feature_vector = np.array([list(features.values())])
                    feature_scaled = self.scalers[category].transform(feature_vector)
                    
                    # Get direction probability
                    direction_prob = self.direction_models[category].predict_proba(feature_scaled)[0]
                    prob_up = direction_prob[1] if len(direction_prob) > 1 else 0.5
                    
                    direction_predictions.append(prob_up > 0.5)
                else:
                    # Fallback to simple trend-based direction
                    if len(current_train) >= 3:
                        recent_trend = (current_train[-1] - current_train[-3]) / current_train[-3]
                        direction_predictions.append(recent_trend > 0)
                    else:
                        direction_predictions.append(True)
            
            # Calculate metrics
            price_predictions = np.array(price_predictions)
            test_prices = np.array(test_prices)
            
            # Remove any invalid predictions
            valid_mask = np.isfinite(price_predictions) & np.isfinite(test_prices) & (test_prices > 0)
            if np.sum(valid_mask) < len(test_prices) * 0.5:
                return
            
            price_predictions = price_predictions[valid_mask]
            test_prices = test_prices[valid_mask]
            
            # Price accuracy metrics
            mae = mean_absolute_error(test_prices, price_predictions)
            rmse = np.sqrt(mean_squared_error(test_prices, price_predictions))
            mape = np.mean(np.abs((test_prices - price_predictions) / test_prices)) * 100
            r2 = np.clip(r2_score(test_prices, price_predictions), -10, 1)
            
            # Enhanced direction accuracy
            if len(test_prices) > 1:
                actual_directions = np.diff(test_prices) > 0
                
                # Use enhanced direction predictions if available
                if len(direction_predictions) == len(test_prices):
                    pred_directions = np.array(direction_predictions[1:])  # Skip first prediction
                else:
                    pred_directions = np.diff(price_predictions) > 0
                
                # Ensure same length
                min_length = min(len(actual_directions), len(pred_directions))
                actual_directions = actual_directions[:min_length]
                pred_directions = pred_directions[:min_length]
                
                if len(actual_directions) > 0:
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
                'has_direction_model': category in self.direction_models
            }
            
        except Exception as e:
            # Fallback metrics
            self.performance_metrics[category] = {
                'mae': np.std(prices),
                'rmse': np.std(prices),
                'mape': 15.0,
                'r2': 0.0,
                'direction_accuracy': 50.0,
                'n_test_points': 0,
                'error': str(e)
            }
    
    def predict(self, steps=3):
        """Generate predictions with enhanced directional accuracy"""
        predictions = {}
        
        for category in self.categories:
            if category not in self.models:
                continue
            
            prices = self.models[category]['prices']
            if len(prices) < 3:
                continue
            
            # Generate base forecasts
            base_forecasts = []
            current_prices = prices.copy()
            
            for step in range(steps):
                # Exponential smoothing component
                smoothed = self.exponential_smoothing(current_prices)
                
                # Trend component
                trend_forecast = self.trend_forecast(current_prices, 1)[0]
                
                # Combine forecasts
                base_forecast = 0.6 * smoothed + 0.4 * trend_forecast
                
                # Apply directional adjustment
                if category in self.direction_models:
                    adjusted_forecast = self.apply_enhanced_direction_adjustment(
                        current_prices, base_forecast, category, step
                    )
                else:
                    adjusted_forecast = base_forecast
                
                base_forecasts.append(adjusted_forecast)
                
                # Update prices for next iteration
                current_prices = np.append(current_prices, adjusted_forecast)
            
            # Apply volatility constraints
            constrained_forecasts = self.apply_volatility_constraints(prices, base_forecasts)
            
            # Calculate confidence intervals
            volatility = np.std(prices[-min(12, len(prices)):])
            lower_bounds = []
            upper_bounds = []
            
            for i, forecast in enumerate(constrained_forecasts):
                uncertainty = volatility * (1.1 + i * 0.2)
                lower_bounds.append(forecast - uncertainty)
                upper_bounds.append(forecast + uncertainty)
            
            predictions[category] = {
                'mean': constrained_forecasts,
                'lower': lower_bounds,
                'upper': upper_bounds
            }
        
        return predictions
    
    def apply_enhanced_direction_adjustment(self, prices, base_forecast, category, step):
        """Apply enhanced direction model adjustment"""
        try:
            features = self.calculate_advanced_features(prices)
            
            # Add context
            features['price_level'] = prices[-1] / np.mean(prices[-min(6, len(prices)):])
            if len(prices) >= 12:
                recent_vol = np.std(prices[-6:]) / np.mean(prices[-6:])
                median_vol = np.median([np.std(prices[max(0, j-6):j+1]) / np.mean(prices[max(0, j-6):j+1]) 
                                      for j in range(6, len(prices), 3)])
                features['volatility_regime'] = 1 if recent_vol > median_vol else 0
            else:
                features['volatility_regime'] = 0
            
            feature_vector = np.array([list(features.values())])
            feature_scaled = self.scalers[category].transform(feature_vector)
            
            direction_probs = self.direction_models[category].predict_proba(feature_scaled)[0]
            prob_up = direction_probs[1] if len(direction_probs) > 1 else 0.5
            
            # Enhanced adjustment based on confidence
            confidence = abs(prob_up - 0.5) * 2  # 0 to 1 scale
            
            if prob_up > 0.6:  # Strong up signal
                adjustment = 1 + confidence * 0.03 * (0.8 ** step)  # Diminishing adjustment
            elif prob_up < 0.4:  # Strong down signal
                adjustment = 1 - confidence * 0.03 * (0.8 ** step)
            else:
                adjustment = 1.0
            
            return base_forecast * adjustment
            
        except Exception:
            return base_forecast
    
    def apply_volatility_constraints(self, prices, forecasts):
        """Apply volatility-based constraints"""
        if len(prices) < 3:
            return forecasts
        
        recent_vol = np.std(prices[-min(8, len(prices)):])
        max_change = recent_vol * 1.2
        
        constrained_forecasts = []
        last_price = prices[-1]
        
        for i, forecast in enumerate(forecasts):
            if i == 0:
                constrained = np.clip(forecast, last_price - max_change, last_price + max_change)
            else:
                prev_forecast = constrained_forecasts[i-1]
                step_constraint = max_change * 0.7
                constrained = np.clip(forecast, prev_forecast - step_constraint, prev_forecast + step_constraint)
            
            constrained_forecasts.append(constrained)
        
        return constrained_forecasts
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a category"""
        metrics = self.performance_metrics.get(category, {})
        metrics['model_type'] = 'enhanced_directional'
        return metrics