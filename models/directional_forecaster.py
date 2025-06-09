import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DirectionalCOEForecaster:
    """
    Advanced directional forecasting model using proven techniques from financial literature:
    1. Ensemble classification for direction prediction
    2. Technical indicators and momentum features
    3. Regime-aware predictions
    4. Multi-timeframe analysis
    """
    
    def __init__(self):
        self.directional_models = {}
        self.price_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    def create_technical_indicators(self, prices):
        """Create technical indicators proven effective for direction prediction"""
        prices = np.array(prices)
        indicators = {}
        
        # Momentum indicators
        indicators['momentum_3'] = self.calculate_momentum(prices, 3)
        indicators['momentum_6'] = self.calculate_momentum(prices, 6)
        indicators['momentum_12'] = self.calculate_momentum(prices, 12)
        
        # Rate of change
        indicators['roc_3'] = self.calculate_roc(prices, 3)
        indicators['roc_6'] = self.calculate_roc(prices, 6)
        
        # Moving average convergence divergence (MACD-like)
        indicators['macd'] = self.calculate_macd(prices)
        
        # Relative strength index (RSI-like)
        indicators['rsi'] = self.calculate_rsi(prices)
        
        # Bollinger band position
        indicators['bb_position'] = self.calculate_bb_position(prices)
        
        # Price relative to moving averages
        indicators['price_vs_ma3'] = self.price_vs_ma(prices, 3)
        indicators['price_vs_ma6'] = self.price_vs_ma(prices, 6)
        indicators['price_vs_ma12'] = self.price_vs_ma(prices, 12)
        
        # Volatility indicators
        indicators['volatility_ratio'] = self.calculate_volatility_ratio(prices)
        
        # Support/resistance levels
        indicators['near_resistance'] = self.calculate_resistance_level(prices)
        indicators['near_support'] = self.calculate_support_level(prices)
        
        return indicators
    
    def calculate_momentum(self, prices, window):
        """Calculate price momentum"""
        if len(prices) < window + 1:
            return 0
        return (prices[-1] - prices[-window-1]) / prices[-window-1]
    
    def calculate_roc(self, prices, window):
        """Calculate rate of change"""
        if len(prices) < window + 1:
            return 0
        return ((prices[-1] / prices[-window-1]) - 1) * 100
    
    def calculate_macd(self, prices, fast=3, slow=6):
        """Calculate MACD-like indicator"""
        if len(prices) < slow:
            return 0
        
        ema_fast = self.ema(prices, fast)
        ema_slow = self.ema(prices, slow)
        return ema_fast - ema_slow
    
    def calculate_rsi(self, prices, window=6):
        """Calculate RSI-like indicator"""
        if len(prices) < window + 1:
            return 50
        
        changes = np.diff(prices[-window-1:])
        gains = changes[changes > 0]
        losses = -changes[changes < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bb_position(self, prices, window=6):
        """Calculate position within Bollinger Bands"""
        if len(prices) < window:
            return 0.5
        
        recent_prices = prices[-window:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price == 0:
            return 0.5
        
        upper_band = mean_price + 2 * std_price
        lower_band = mean_price - 2 * std_price
        
        # Position within bands (0 = lower band, 1 = upper band)
        position = (prices[-1] - lower_band) / (upper_band - lower_band)
        return np.clip(position, 0, 1)
    
    def price_vs_ma(self, prices, window):
        """Calculate price relative to moving average"""
        if len(prices) < window:
            return 0
        
        ma = np.mean(prices[-window:])
        return (prices[-1] - ma) / ma
    
    def calculate_volatility_ratio(self, prices, short_window=3, long_window=12):
        """Calculate short-term vs long-term volatility ratio"""
        if len(prices) < long_window:
            return 1
        
        short_vol = np.std(prices[-short_window:]) if len(prices) >= short_window else 0
        long_vol = np.std(prices[-long_window:])
        
        if long_vol == 0:
            return 1
        
        return short_vol / long_vol
    
    def calculate_resistance_level(self, prices, window=12):
        """Check if price is near recent resistance level"""
        if len(prices) < window:
            return 0
        
        recent_highs = []
        for i in range(len(prices) - window, len(prices) - 2):
            if i >= 1 and i < len(prices) - 1:
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    recent_highs.append(prices[i])
        
        if not recent_highs:
            return 0
        
        resistance = np.mean(recent_highs)
        distance_to_resistance = abs(prices[-1] - resistance) / resistance
        
        # Return 1 if very close to resistance (within 2%)
        return 1 if distance_to_resistance < 0.02 else 0
    
    def calculate_support_level(self, prices, window=12):
        """Check if price is near recent support level"""
        if len(prices) < window:
            return 0
        
        recent_lows = []
        for i in range(len(prices) - window, len(prices) - 2):
            if i >= 1 and i < len(prices) - 1:
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    recent_lows.append(prices[i])
        
        if not recent_lows:
            return 0
        
        support = np.mean(recent_lows)
        distance_to_support = abs(prices[-1] - support) / support
        
        # Return 1 if very close to support (within 2%)
        return 1 if distance_to_support < 0.02 else 0
    
    def ema(self, prices, window):
        """Calculate exponential moving average"""
        alpha = 2 / (window + 1)
        ema_value = prices[0]
        
        for price in prices[1:]:
            ema_value = alpha * price + (1 - alpha) * ema_value
        
        return ema_value
    
    def create_directional_features(self, data, category):
        """Create features specifically for directional prediction"""
        category_data = data[data['vehicle_class'] == category].copy()
        category_data = category_data.sort_values('date')
        
        prices = category_data['premium'].values
        features_list = []
        directions = []
        
        # Need minimum data for feature creation
        min_lookback = 15
        
        for i in range(min_lookback, len(prices) - 1):
            # Get price history up to point i
            price_history = prices[:i+1]
            
            # Create technical indicators
            indicators = self.create_technical_indicators(price_history)
            
            # Add lagged returns
            for lag in [1, 2, 3]:
                if i >= lag:
                    indicators[f'return_lag_{lag}'] = (prices[i] - prices[i-lag]) / prices[i-lag]
            
            # Add volume-related features if available
            if 'bids_received' in category_data.columns:
                recent_volume = category_data['bids_received'].iloc[:i+1].values
                if len(recent_volume) >= 3:
                    indicators['volume_trend'] = (recent_volume[-1] - recent_volume[-3]) / recent_volume[-3]
                    indicators['volume_ratio'] = recent_volume[-1] / np.mean(recent_volume[-6:]) if len(recent_volume) >= 6 else 1
            
            # Add quota utilization features
            if 'quota' in category_data.columns and 'bids_success' in category_data.columns:
                recent_quota = category_data['quota'].iloc[:i+1].values
                recent_success = category_data['bids_success'].iloc[:i+1].values
                if len(recent_quota) >= 1 and len(recent_success) >= 1:
                    indicators['quota_utilization'] = recent_success[-1] / recent_quota[-1] if recent_quota[-1] > 0 else 0
            
            features_list.append(list(indicators.values()))
            
            # Direction for next period (1 = up, 0 = down)
            next_direction = 1 if prices[i+1] > prices[i] else 0
            directions.append(next_direction)
        
        if len(features_list) == 0:
            return None
        
        # Get feature names from the last indicator calculation
        if len(features_list) > 0:
            # Recreate indicators to get feature names
            temp_indicators = self.create_technical_indicators(prices[:min_lookback+1])
            for lag in [1, 2, 3]:
                temp_indicators[f'return_lag_{lag}'] = 0
            temp_indicators['volume_trend'] = 0
            temp_indicators['volume_ratio'] = 1
            temp_indicators['quota_utilization'] = 0
            feature_names = list(temp_indicators.keys())
        else:
            feature_names = []
        
        return np.array(features_list), np.array(directions), feature_names
    
    def fit(self, data):
        """Fit directional prediction models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 20:
                continue
            
            # Create directional features
            result = self.create_directional_features(data, category)
            if result is None or result[0] is None:
                continue
            X, y, feature_names = result
            
            # Split for validation
            split_point = int(len(X) * 0.8)
            X_train, X_val = X[:split_point], X[split_point:]
            y_train, y_val = y[:split_point], y[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Create ensemble of directional classifiers
            models = {
                'rf': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
                'lr': LogisticRegression(random_state=42),
            }
            
            # Train models and select best performer
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    val_pred = model.predict(X_val_scaled)
                    score = accuracy_score(y_val, val_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                except:
                    continue
            
            if best_model is not None:
                self.directional_models[category] = best_model
                self.scalers[category] = scaler
                
                # Store feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_names, best_model.feature_importances_))
                    self.feature_importance[category] = importance_dict
                
                # Train price prediction model for magnitude
                price_model = self.fit_price_model(category_data)
                if price_model:
                    self.price_models[category] = price_model
    
    def fit_price_model(self, category_data):
        """Fit a simple price magnitude prediction model"""
        prices = category_data['premium'].values
        if len(prices) < 10:
            return None
        
        # Use exponential smoothing for price prediction
        return {'prices': prices, 'method': 'exponential_smoothing'}
    
    def predict_direction(self, data, category, steps=3):
        """Predict price direction using ensemble classifier"""
        if category not in self.directional_models:
            return None
        
        model = self.directional_models[category]
        scaler = self.scalers[category]
        
        category_data = data[data['vehicle_class'] == category].copy()
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        if len(prices) < 15:
            return None
        
        # Create features for current state
        indicators = self.create_technical_indicators(prices)
        
        # Add recent returns
        for lag in [1, 2, 3]:
            if len(prices) > lag:
                indicators[f'return_lag_{lag}'] = (prices[-1] - prices[-1-lag]) / prices[-1-lag]
        
        # Add volume and quota features if available
        if 'bids_received' in category_data.columns:
            recent_volume = category_data['bids_received'].values
            if len(recent_volume) >= 3:
                indicators['volume_trend'] = (recent_volume[-1] - recent_volume[-3]) / recent_volume[-3]
                indicators['volume_ratio'] = recent_volume[-1] / np.mean(recent_volume[-6:]) if len(recent_volume) >= 6 else 1
        
        if 'quota' in category_data.columns and 'bids_success' in category_data.columns:
            recent_quota = category_data['quota'].values
            recent_success = category_data['bids_success'].values
            if len(recent_quota) >= 1 and len(recent_success) >= 1:
                indicators['quota_utilization'] = recent_success[-1] / recent_quota[-1] if recent_quota[-1] > 0 else 0
        
        # Prepare feature vector
        feature_vector = np.array([list(indicators.values())])
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Predict direction probabilities
        direction_probs = model.predict_proba(feature_vector_scaled)[0]
        
        # Generate multi-step predictions
        predictions = []
        current_price = prices[-1]
        
        for step in range(steps):
            # Predict direction
            direction_prob_up = direction_probs[1] if len(direction_probs) > 1 else 0.5
            
            # Predict magnitude using price model
            if category in self.price_models:
                magnitude = self.predict_magnitude(category, prices, step + 1)
            else:
                # Default magnitude based on recent volatility
                recent_vol = np.std(prices[-6:]) if len(prices) >= 6 else prices[-1] * 0.02
                magnitude = recent_vol * (0.5 + np.random.random() * 0.5)  # 0.5x to 1x volatility
            
            # Combine direction and magnitude
            if direction_prob_up > 0.5:
                # Upward movement
                confidence = direction_prob_up
                price_change = magnitude * confidence
            else:
                # Downward movement  
                confidence = 1 - direction_prob_up
                price_change = -magnitude * confidence
            
            next_price = current_price + price_change
            predictions.append({
                'price': next_price,
                'direction_prob': direction_prob_up,
                'confidence': max(direction_prob_up, 1 - direction_prob_up)
            })
            
            current_price = next_price
            
            # Update direction probabilities for next step (add some decay)
            direction_probs = direction_probs * 0.9 + np.array([0.5, 0.5]) * 0.1
        
        return predictions
    
    def predict_magnitude(self, category, prices, step):
        """Predict price change magnitude"""
        price_model = self.price_models.get(category)
        if not price_model:
            return np.std(prices[-6:]) if len(prices) >= 6 else prices[-1] * 0.02
        
        # Simple exponential smoothing prediction
        alpha = 0.3
        smoothed = prices[0]
        for price in prices[1:]:
            smoothed = alpha * price + (1 - alpha) * smoothed
        
        # Predict next value
        trend = (prices[-1] - prices[-3]) / 2 if len(prices) >= 3 else 0
        predicted_price = smoothed + step * trend * 0.5
        
        # Return magnitude as absolute difference
        return abs(predicted_price - prices[-1]) / step
    
    def predict(self, data, steps=3):
        """Generate directional predictions for all categories"""
        predictions = {}
        
        for category in self.categories:
            if category not in self.directional_models:
                continue
            
            directional_preds = self.predict_direction(data, category, steps)
            if directional_preds:
                # Convert to standard format
                mean_prices = [pred['price'] for pred in directional_preds]
                confidences = [pred['confidence'] for pred in directional_preds]
                
                # Create confidence intervals based on prediction confidence
                lower_bounds = []
                upper_bounds = []
                
                for i, (price, conf) in enumerate(zip(mean_prices, confidences)):
                    # Lower confidence = wider intervals
                    uncertainty = (1 - conf) * price * 0.1  # Max 10% uncertainty
                    lower_bounds.append(price - uncertainty)
                    upper_bounds.append(price + uncertainty)
                
                predictions[category] = {
                    'mean': mean_prices,
                    'lower': lower_bounds,
                    'upper': upper_bounds,
                    'direction_confidence': [pred['confidence'] for pred in directional_preds],
                    'direction_probabilities': [pred['direction_prob'] for pred in directional_preds]
                }
        
        return predictions
    
    def get_performance_metrics(self, category):
        """Get performance metrics for directional model"""
        if category not in self.directional_models:
            return {}
        
        return {
            'model_type': 'directional_ensemble',
            'features_used': len(self.feature_importance.get(category, {})),
            'top_features': self.get_top_features(category, 5)
        }
    
    def get_top_features(self, category, n=5):
        """Get top important features for direction prediction"""
        if category not in self.feature_importance:
            return []
        
        importance = self.feature_importance[category]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]