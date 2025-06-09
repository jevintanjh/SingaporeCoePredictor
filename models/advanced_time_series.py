import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedTimeSeriesModel:
    """
    Advanced time series model using multiple proven financial forecasting techniques:
    1. ARIMA-style autoregressive components
    2. Exponential smoothing with trend and seasonality
    3. Linear trend with cyclical components
    4. Ensemble approach with volatility-aware predictions
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        
    def create_time_series_features(self, data):
        """Create specialized time series features"""
        df = data.copy()
        df = df.sort_values(['vehicle_class', 'date']).reset_index(drop=True)
        
        for category in self.categories:
            mask = df['vehicle_class'] == category
            if not mask.any():
                continue
                
            category_data = df[mask].copy()
            
            # Multi-step lagged features (proven for financial time series)
            for lag in [1, 2, 3, 6, 12]:
                df.loc[mask, f'premium_lag_{lag}'] = category_data['premium'].shift(lag)
            
            # Exponential moving averages (common in financial modeling)
            df.loc[mask, 'ema_3'] = category_data['premium'].ewm(span=3).mean()
            df.loc[mask, 'ema_6'] = category_data['premium'].ewm(span=6).mean()
            df.loc[mask, 'ema_12'] = category_data['premium'].ewm(span=12).mean()
            
            # Volatility features
            df.loc[mask, 'volatility_3'] = category_data['premium'].rolling(3).std()
            df.loc[mask, 'volatility_6'] = category_data['premium'].rolling(6).std()
            
            # Price momentum and change features
            df.loc[mask, 'price_change_1'] = category_data['premium'].pct_change(1)
            df.loc[mask, 'price_change_3'] = category_data['premium'].pct_change(3)
            df.loc[mask, 'momentum_3'] = category_data['premium'].rolling(3).apply(lambda x: (x[-1] - x[0]) / x[0])
            
            # Seasonal decomposition features
            df.loc[mask, 'seasonal_ma'] = category_data['premium'].rolling(12, center=True).mean()
            df.loc[mask, 'detrended'] = category_data['premium'] - category_data['premium'].rolling(12, center=True).mean()
        
        # Market-wide features
        df['bid_pressure'] = (df['bids_received'] - df['quota']) / df['quota']
        df['success_rate'] = df['bids_success'] / df['bids_received']
        df['market_tension'] = df['bid_pressure'] * (1 - df['success_rate'])
        
        # Cyclical time features (important for COE which has bi-monthly cycles)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['cycle_in_year'] = ((df['date'].dt.month - 1) * 2 + df['bidding_no']).astype(int)
        
        # Seasonal encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['cycle_sin'] = np.sin(2 * np.pi * df['cycle_in_year'] / 24)
        df['cycle_cos'] = np.cos(2 * np.pi * df['cycle_in_year'] / 24)
        
        return df
    
    def exponential_smoothing_predict(self, series, alpha=0.3, beta=0.1, gamma=0.1, steps=3):
        """Holt-Winters exponential smoothing for trend and seasonality"""
        series = np.array(series, dtype=float)
        if len(series) < 24:  # Need at least 2 years of data
            return self.simple_exponential_smoothing(series, alpha, steps)
        
        # Initialize components
        season_length = 12  # Monthly seasonality
        level = float(np.mean(series[:season_length]))
        trend = float((np.mean(series[season_length:2*season_length]) - np.mean(series[:season_length])) / season_length)
        seasonal = [float(x) for x in (series[:season_length] - level)]
        
        levels, trends, seasonals = [level], [trend], list(seasonal)
        
        # Fit the model
        for i in range(season_length, len(series)):
            prev_level = levels[-1]
            prev_trend = trends[-1]
            
            level = alpha * (float(series[i]) - seasonals[i % season_length]) + (1 - alpha) * (prev_level + prev_trend)
            trend = beta * (level - prev_level) + (1 - beta) * prev_trend
            seasonal_val = gamma * (float(series[i]) - level) + (1 - gamma) * seasonals[i % season_length]
            
            levels.append(level)
            trends.append(trend)
            seasonals.append(seasonal_val)
        
        # Generate forecasts
        forecasts = []
        for step in range(1, steps + 1):
            forecast = levels[-1] + step * trends[-1] + seasonals[-(season_length - step % season_length)]
            forecasts.append(float(forecast))
        
        return forecasts
    
    def simple_exponential_smoothing(self, series, alpha=0.3, steps=3):
        """Simple exponential smoothing for series with insufficient history"""
        series = np.array(series)
        if len(series) == 0:
            return [0] * steps
        
        # Initialize with first value
        smoothed = [float(series[0])]
        
        # Apply exponential smoothing
        for i in range(1, len(series)):
            smoothed.append(alpha * float(series[i]) + (1 - alpha) * smoothed[-1])
        
        # Generate forecasts
        forecasts = []
        last_smoothed = smoothed[-1]
        
        # Add slight trend based on recent changes
        if len(series) >= 3:
            recent_trend = (float(series[-1]) - float(series[-3])) / 3
        else:
            recent_trend = 0
        
        for step in range(1, steps + 1):
            forecast = last_smoothed + step * recent_trend * 0.5  # Damped trend
            forecasts.append(forecast)
        
        return forecasts
    
    def autoregressive_predict(self, series, lags=6, steps=3):
        """Simple autoregressive model"""
        if len(series) < lags + 1:
            return [series[-1]] * steps if len(series) > 0 else [0] * steps
        
        # Prepare AR data
        X, y = [], []
        for i in range(lags, len(series)):
            X.append(series[i-lags:i])
            y.append(series[i])
        
        X, y = np.array(X), np.array(y)
        
        # Simple linear regression for AR model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecasts
        forecasts = []
        last_values = list(series[-lags:])
        
        for step in range(steps):
            pred = model.predict([last_values])[0]
            forecasts.append(pred)
            last_values = last_values[1:] + [pred]
        
        return forecasts
    
    def trend_seasonal_predict(self, series, steps=3):
        """Linear trend with seasonal components"""
        if len(series) < 12:
            # Simple linear trend for short series
            x = np.arange(len(series))
            z = np.polyfit(x, series, 1)
            trend_func = np.poly1d(z)
            
            forecasts = []
            for step in range(1, steps + 1):
                forecast = trend_func(len(series) + step - 1)
                forecasts.append(forecast)
            return forecasts
        
        # Decompose into trend and seasonal
        df_temp = pd.DataFrame({'value': series, 'time': range(len(series))})
        
        # Linear trend
        z = np.polyfit(df_temp['time'], df_temp['value'], 1)
        trend = np.poly1d(z)
        
        # Seasonal component (monthly)
        detrended = series - trend(df_temp['time'])
        seasonal_period = min(12, len(series) // 2)
        
        if seasonal_period > 0:
            seasonal_avg = np.array([np.mean(detrended[i::seasonal_period]) for i in range(seasonal_period)])
        else:
            seasonal_avg = np.zeros(12)
        
        # Generate forecasts
        forecasts = []
        for step in range(1, steps + 1):
            trend_forecast = trend(len(series) + step - 1)
            seasonal_forecast = seasonal_avg[(len(series) + step - 1) % len(seasonal_avg)] if len(seasonal_avg) > 0 else 0
            forecast = trend_forecast + seasonal_forecast
            forecasts.append(forecast)
        
        return forecasts
    
    def fit(self, data):
        """Fit models for all categories"""
        df = self.create_time_series_features(data)
        
        for category in self.categories:
            category_data = df[df['vehicle_class'] == category].copy()
            if len(category_data) < 12:  # Need minimum data
                continue
            
            # Sort by date
            category_data = category_data.sort_values('date')
            series = category_data['premium'].values
            
            # Store model parameters (series for prediction)
            self.models[category] = {
                'series': series,
                'dates': category_data['date'].values,
                'features': category_data[['quota', 'bids_received', 'bid_pressure', 'success_rate']].values
            }
            
            # Calculate performance on validation set
            if len(series) > 24:
                train_size = int(len(series) * 0.8)
                train_series = series[:train_size]
                val_series = series[train_size:]
                
                # Test different methods and choose best
                methods = {
                    'exponential_smoothing': self.exponential_smoothing_predict(train_series, steps=len(val_series)),
                    'autoregressive': self.autoregressive_predict(train_series, steps=len(val_series)),
                    'trend_seasonal': self.trend_seasonal_predict(train_series, steps=len(val_series))
                }
                
                best_method = 'exponential_smoothing'
                best_mae = float('inf')
                
                for method, predictions in methods.items():
                    if len(predictions) == len(val_series):
                        mae = mean_absolute_error(val_series, predictions)
                        if mae < best_mae:
                            best_mae = mae
                            best_method = method
                
                self.models[category]['best_method'] = best_method
                
                # Calculate metrics for best method
                best_predictions = methods[best_method]
                mae = mean_absolute_error(val_series, best_predictions)
                rmse = np.sqrt(mean_squared_error(val_series, best_predictions))
                mape = np.mean(np.abs((val_series - best_predictions) / val_series)) * 100
                
                self.performance_metrics[category] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2_score(val_series, best_predictions) if len(best_predictions) == len(val_series) else 0,
                    'method': best_method
                }
            else:
                self.models[category]['best_method'] = 'exponential_smoothing'
    
    def predict(self, steps=3):
        """Generate predictions for all categories"""
        all_predictions = {}
        
        for category in self.categories:
            if category not in self.models:
                continue
            
            series = self.models[category]['series']
            best_method = self.models[category].get('best_method', 'exponential_smoothing')
            
            # Generate predictions using best method
            if best_method == 'exponential_smoothing':
                predictions = self.exponential_smoothing_predict(series, steps=steps)
            elif best_method == 'autoregressive':
                predictions = self.autoregressive_predict(series, steps=steps)
            else:  # trend_seasonal
                predictions = self.trend_seasonal_predict(series, steps=steps)
            
            # Apply volatility-based constraints
            if len(series) >= 12:
                recent_volatility = np.std(series[-12:])
            elif len(series) >= 6:
                recent_volatility = np.std(series[-6:])
            else:
                recent_volatility = float(series[-1]) * 0.1
            
            # Constrain predictions to reasonable bounds
            last_price = float(series[-1])
            vol_bound = float(recent_volatility * 2)
            pct_bound = last_price * 0.15
            max_change = vol_bound if vol_bound < pct_bound else pct_bound  # Max 15% change or 2x recent volatility
            
            constrained_predictions = []
            for i, pred in enumerate(predictions):
                if i == 0:
                    # First prediction can deviate more from last price
                    constrained = np.clip(pred, last_price - max_change, last_price + max_change)
                else:
                    # Subsequent predictions are constrained from previous prediction
                    prev_pred = constrained_predictions[i-1]
                    step_max_change = max_change * 0.5  # Smaller changes for further predictions
                    constrained = np.clip(pred, prev_pred - step_max_change, prev_pred + step_max_change)
                
                constrained_predictions.append(constrained)
            
            # Calculate confidence intervals
            confidence_width = recent_volatility * 1.96  # 95% confidence interval
            
            all_predictions[category] = {
                'mean': constrained_predictions,
                'lower': [p - confidence_width for p in constrained_predictions],
                'upper': [p + confidence_width for p in constrained_predictions],
                'method': best_method
            }
        
        return all_predictions
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a category"""
        return self.performance_metrics.get(category, {})