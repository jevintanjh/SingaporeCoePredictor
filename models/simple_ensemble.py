import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SimpleCOEModel:
    """
    Simplified ensemble model for COE price prediction
    """
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        
    def create_features(self, data):
        """Create engineered features for the model"""
        df = data.copy()
        df = df.sort_values(['vehicle_class', 'date']).reset_index(drop=True)
        
        # Create lagged features for each category
        for category in self.categories:
            mask = df['vehicle_class'] == category
            category_data = df[mask].copy()
            
            if len(category_data) > 0:
                df.loc[mask, 'premium_lag1'] = category_data['premium'].shift(1)
                df.loc[mask, 'premium_lag2'] = category_data['premium'].shift(2)
                df.loc[mask, 'premium_ma3'] = category_data['premium'].rolling(window=3, min_periods=1).mean()
        
        # Calculate ratios
        df['bid_quota_ratio'] = df['bids_received'] / df['quota']
        df['success_rate'] = df['bids_success'] / df['bids_received']
        
        # Time features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Fill missing values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def fit(self, data):
        """Train models for all categories"""
        feature_cols = ['quota', 'bids_received', 'bids_success', 'premium_lag1', 'premium_lag2', 
                       'premium_ma3', 'bid_quota_ratio', 'success_rate', 'month', 'quarter']
        
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) < 20:
                continue
                
            # Prepare features
            df = self.create_features(category_data)
            df = df.dropna()
            
            if len(df) < 10:
                continue
            
            # Split data
            split_idx = int(len(df) * 0.8)
            train_data = df[:split_idx].copy()
            val_data = df[split_idx:].copy()
            
            X_train = train_data[feature_cols].values
            y_train = train_data['premium'].values
            X_val = val_data[feature_cols].values
            y_val = val_data['premium'].values
            
            # Train ensemble models
            self.models[category] = {}
            
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
            rf_model.fit(X_train, y_train)
            self.models[category]['rf'] = rf_model
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train, y_train)
            self.models[category]['xgb'] = xgb_model
            
            # Calculate performance
            rf_pred = rf_model.predict(X_val)
            xgb_pred = xgb_model.predict(X_val)
            ensemble_pred = (rf_pred + xgb_pred) / 2
            
            mae = mean_absolute_error(y_val, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            mape = np.mean(np.abs((y_val - ensemble_pred) / y_val)) * 100
            r2 = r2_score(y_val, ensemble_pred)
            
            self.performance_metrics[category] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            }
    
    def predict(self, steps=3):
        """Generate predictions for all categories"""
        try:
            from utils.data_processor import DataProcessor
            processor = DataProcessor()
            data = processor.load_data('data/COEBiddingResultsPrices_1749430265007.csv')
            data = processor.preprocess_data(data)
        except:
            return {}
        
        all_predictions = {}
        feature_cols = ['quota', 'bids_received', 'bids_success', 'premium_lag1', 'premium_lag2', 
                       'premium_ma3', 'bid_quota_ratio', 'success_rate', 'month', 'quarter']
        
        for category in self.categories:
            if category not in self.models:
                continue
                
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) == 0:
                continue
            
            df = self.create_features(category_data)
            df = df.dropna()
            
            if len(df) == 0:
                continue
            
            predictions = []
            latest_row = df.iloc[-1:].copy()
            
            for step in range(steps):
                X_pred = latest_row[feature_cols].values
                
                # Get predictions from both models
                rf_pred = self.models[category]['rf'].predict(X_pred)[0]
                xgb_pred = self.models[category]['xgb'].predict(X_pred)[0]
                
                # Ensemble prediction
                ensemble_pred = (rf_pred + xgb_pred) / 2
                
                # Apply constraints
                base_premium = latest_row['premium'].iloc[0]
                max_change = base_premium * 0.2  # Max 20% change
                constrained_pred = np.clip(ensemble_pred, 
                                         base_premium - max_change, 
                                         base_premium + max_change)
                
                predictions.append(constrained_pred)
                
                # Update for next iteration
                new_row = latest_row.copy()
                new_row['premium'] = constrained_pred
                new_row['premium_lag1'] = latest_row['premium'].iloc[0]
                new_row['premium_lag2'] = latest_row['premium_lag1'].iloc[0] if 'premium_lag1' in latest_row.columns else latest_row['premium'].iloc[0]
                new_row['premium_ma3'] = np.mean([constrained_pred, latest_row['premium'].iloc[0], 
                                                new_row['premium_lag1'].iloc[0]])
                new_row['date'] = new_row['date'].iloc[0] + pd.DateOffset(days=15)
                
                latest_row = new_row
            
            # Calculate confidence intervals based on historical volatility
            recent_prices = df.tail(12)['premium'].values
            volatility = np.std(recent_prices)
            
            predictions = np.array(predictions)
            confidence_range = volatility * 1.96
            
            all_predictions[category] = {
                'mean': predictions.tolist(),
                'lower': (predictions - confidence_range).tolist(),
                'upper': (predictions + confidence_range).tolist()
            }
        
        return all_predictions
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a category"""
        return self.performance_metrics.get(category, {})