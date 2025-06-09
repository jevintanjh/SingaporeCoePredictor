import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class COEEnsembleModel:
    """
    Ensemble model combining LSTM, Prophet, and XGBoost for COE price prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        self.feature_cols = ['quota', 'bids_received', 'bids_success', 'premium_lag1', 'premium_lag2', 
                           'premium_ma3', 'bid_quota_ratio', 'success_rate', 'month', 'bidding_no']
        
    def create_features(self, data):
        """Create engineered features for the model"""
        df = data.copy()
        
        # Sort by date and vehicle class
        df = df.sort_values(['vehicle_class', 'date']).reset_index(drop=True)
        
        # Create lagged features
        for category in self.categories:
            mask = df['vehicle_class'] == category
            df.loc[mask, 'premium_lag1'] = df.loc[mask, 'premium'].shift(1)
            df.loc[mask, 'premium_lag2'] = df.loc[mask, 'premium'].shift(2)
            df.loc[mask, 'premium_ma3'] = df.loc[mask, 'premium'].rolling(window=3).mean()
        
        # Calculate ratios
        df['bid_quota_ratio'] = df['bids_received'] / df['quota']
        df['success_rate'] = df['bids_success'] / df['bids_received']
        
        # Time-based features
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        
        # Fill missing values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def prepare_lstm_data(self, data, lookback=6):
        """Prepare data for LSTM model"""
        # Create sequences for LSTM
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def fit_category_models(self, category_data, category):
        """Fit all models for a specific category"""
        # Prepare features
        df = self.create_features(category_data)
        df = df.dropna()
        
        if len(df) < 20:  # Minimum data requirement
            return None
        
        # Split data (use last 20% for validation)
        split_idx = int(len(df) * 0.8)
        train_data = df[:split_idx].copy()
        val_data = df[split_idx:].copy()
        
        # Initialize models dictionary for this category
        self.models[category] = {}
        self.scalers[category] = {}
        
        # Prepare training data
        X_train = train_data[self.feature_cols].values
        y_train = train_data['premium'].values
        X_val = val_data[self.feature_cols].values
        y_val = val_data['premium'].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers[category]['features'] = scaler
        
        # Scale target for LSTM
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        self.scalers[category]['target'] = target_scaler
        
        # 1. XGBoost Model
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            self.models[category]['xgboost'] = xgb_model
        except Exception as e:
            print(f"XGBoost training failed for {category}: {e}")
        
        # 2. Prophet Model
        try:
            prophet_data = train_data[['date', 'premium']].copy()
            prophet_data.columns = ['ds', 'y']
            
            prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            # Add additional regressors
            prophet_model.add_regressor('quota')
            prophet_model.add_regressor('bids_received')
            prophet_model.add_regressor('bid_quota_ratio')
            
            prophet_data['quota'] = train_data['quota'].values
            prophet_data['bids_received'] = train_data['bids_received'].values
            prophet_data['bid_quota_ratio'] = train_data['bid_quota_ratio'].values
            
            prophet_model.fit(prophet_data)
            self.models[category]['prophet'] = prophet_model
        except Exception as e:
            print(f"Prophet training failed for {category}: {e}")
        
        # 3. LSTM Model
        try:
            lookback = min(6, len(train_data) // 4)  # Adaptive lookback
            if lookback >= 3:
                # Prepare LSTM data
                lstm_features = np.column_stack([y_train_scaled, X_train_scaled])
                X_lstm, y_lstm = self.prepare_lstm_data(lstm_features, lookback)
                
                if len(X_lstm) > 0:
                    lstm_model = self.build_lstm_model((lookback, lstm_features.shape[1]))
                    lstm_model.fit(
                        X_lstm, y_lstm[:, 0],  # Predict premium (first column)
                        epochs=50,
                        batch_size=16,
                        verbose=0,
                        validation_split=0.2
                    )
                    self.models[category]['lstm'] = lstm_model
                    self.models[category]['lstm_lookback'] = lookback
        except Exception as e:
            print(f"LSTM training failed for {category}: {e}")
        
        # Calculate performance metrics
        self.calculate_performance_metrics(category, val_data)
        
        return True
    
    def fit(self, data):
        """Train ensemble models for all categories"""
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) > 0:
                print(f"Training models for {category}...")
                self.fit_category_models(category_data, category)
        
        print("Model training completed!")
    
    def predict_category(self, category, data, steps=3):
        """Predict future values for a specific category"""
        if category not in self.models:
            return None
        
        df = self.create_features(data)
        df = df.dropna()
        
        if len(df) == 0:
            return None
        
        predictions = {'xgboost': [], 'prophet': [], 'lstm': []}
        
        # Get the latest data point
        latest_data = df.iloc[-1:].copy()
        
        for step in range(steps):
            step_predictions = {}
            
            # XGBoost prediction
            if 'xgboost' in self.models[category]:
                try:
                    X_pred = latest_data[self.feature_cols].values
                    X_pred_scaled = self.scalers[category]['features'].transform(X_pred)
                    xgb_pred = self.models[category]['xgboost'].predict(X_pred_scaled)[0]
                    step_predictions['xgboost'] = xgb_pred
                except Exception as e:
                    print(f"XGBoost prediction error for {category}: {e}")
            
            # Prophet prediction
            if 'prophet' in self.models[category]:
                try:
                    future_date = latest_data['date'].iloc[0] + pd.DateOffset(days=15 * (step + 1))
                    future_df = pd.DataFrame({
                        'ds': [future_date],
                        'quota': [latest_data['quota'].iloc[0]],
                        'bids_received': [latest_data['bids_received'].iloc[0]],
                        'bid_quota_ratio': [latest_data['bid_quota_ratio'].iloc[0]]
                    })
                    
                    prophet_pred = self.models[category]['prophet'].predict(future_df)
                    step_predictions['prophet'] = prophet_pred['yhat'].iloc[0]
                except Exception as e:
                    print(f"Prophet prediction error for {category}: {e}")
            
            # LSTM prediction
            if 'lstm' in self.models[category]:
                try:
                    lookback = self.models[category]['lstm_lookback']
                    recent_data = df.tail(lookback)
                    
                    # Prepare LSTM input
                    lstm_features = []
                    for _, row in recent_data.iterrows():
                        features = [row['premium']] + [row[col] for col in self.feature_cols]
                        lstm_features.append(features)
                    
                    lstm_features = np.array(lstm_features)
                    
                    # Scale features
                    target_vals = lstm_features[:, 0].reshape(-1, 1)
                    target_scaled = self.scalers[category]['target'].transform(target_vals).flatten()
                    
                    feature_vals = lstm_features[:, 1:]
                    feature_scaled = self.scalers[category]['features'].transform(feature_vals)
                    
                    lstm_input = np.column_stack([target_scaled, feature_scaled])
                    lstm_input = lstm_input.reshape(1, lookback, -1)
                    
                    lstm_pred_scaled = self.models[category]['lstm'].predict(lstm_input, verbose=0)[0, 0]
                    lstm_pred = self.scalers[category]['target'].inverse_transform([[lstm_pred_scaled]])[0, 0]
                    step_predictions['lstm'] = lstm_pred
                except Exception as e:
                    print(f"LSTM prediction error for {category}: {e}")
            
            # Ensemble prediction (weighted average)
            if step_predictions:
                # Equal weights for available models
                ensemble_pred = np.mean(list(step_predictions.values()))
                
                # Store individual predictions
                for model_name, pred in step_predictions.items():
                    predictions[model_name].append(pred)
                
                # Update latest_data for next iteration
                new_row = latest_data.copy()
                new_row['premium'] = ensemble_pred
                new_row['premium_lag1'] = latest_data['premium'].iloc[0]
                new_row['premium_lag2'] = latest_data['premium_lag1'].iloc[0]
                new_row['date'] = new_row['date'].iloc[0] + pd.DateOffset(days=15)
                
                latest_data = new_row
            else:
                # If no predictions available, use last known value
                last_premium = latest_data['premium'].iloc[0]
                for model_name in predictions:
                    predictions[model_name].append(last_premium)
        
        return predictions
    
    def predict(self, steps=3):
        """Generate ensemble predictions for all categories"""
        all_predictions = {}
        
        # Load the most recent data for predictions
        try:
            from utils.data_processor import DataProcessor
            processor = DataProcessor()
            data = processor.load_data('data/COEBiddingResultsPrices_1749430265007.csv')
            data = processor.preprocess_data(data)
        except Exception as e:
            print(f"Error loading data for predictions: {e}")
            return {}
        
        for category in self.categories:
            category_data = data[data['vehicle_class'] == category].copy()
            if len(category_data) > 0 and category in self.models:
                predictions = self.predict_category(category, category_data, steps)
                
                if predictions and any(len(preds) > 0 for preds in predictions.values()):
                    # Calculate ensemble prediction
                    available_predictions = [preds for preds in predictions.values() if len(preds) > 0]
                    if available_predictions:
                        ensemble_mean = np.mean(available_predictions, axis=0)
                        ensemble_std = np.std(available_predictions, axis=0)
                        
                        # Calculate confidence intervals (assuming normal distribution)
                        confidence_lower = ensemble_mean - 1.96 * ensemble_std
                        confidence_upper = ensemble_mean + 1.96 * ensemble_std
                        
                        all_predictions[category] = {
                            'mean': ensemble_mean.tolist(),
                            'lower': confidence_lower.tolist(),
                            'upper': confidence_upper.tolist(),
                            'individual_models': predictions
                        }
        
        return all_predictions
    
    def calculate_performance_metrics(self, category, val_data):
        """Calculate performance metrics for a category"""
        if category not in self.models or len(val_data) == 0:
            return
        
        try:
            # Prepare validation data
            X_val = val_data[self.feature_cols].values
            y_val = val_data['premium'].values
            X_val_scaled = self.scalers[category]['features'].transform(X_val)
            
            predictions = []
            
            # Get predictions from available models
            if 'xgboost' in self.models[category]:
                xgb_preds = self.models[category]['xgboost'].predict(X_val_scaled)
                predictions.append(xgb_preds)
            
            if 'prophet' in self.models[category]:
                try:
                    prophet_data = val_data[['date', 'quota', 'bids_received', 'bid_quota_ratio']].copy()
                    prophet_data.columns = ['ds', 'quota', 'bids_received', 'bid_quota_ratio']
                    prophet_preds = self.models[category]['prophet'].predict(prophet_data)['yhat'].values
                    predictions.append(prophet_preds)
                except:
                    pass
            
            if predictions:
                # Ensemble prediction
                ensemble_preds = np.mean(predictions, axis=0)
                
                # Calculate metrics
                mae = mean_absolute_error(y_val, ensemble_preds)
                rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
                mape = np.mean(np.abs((y_val - ensemble_preds) / y_val)) * 100
                r2 = r2_score(y_val, ensemble_preds)
                
                self.performance_metrics[category] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2
                }
        except Exception as e:
            print(f"Error calculating metrics for {category}: {e}")
    
    def get_performance_metrics(self, category):
        """Get performance metrics for a specific category"""
        return self.performance_metrics.get(category, {})
    
    def get_feature_importance(self, category):
        """Get feature importance from XGBoost model"""
        if category in self.models and 'xgboost' in self.models[category]:
            importance = self.models[category]['xgboost'].feature_importances_
            feature_importance = dict(zip(self.feature_cols, importance))
            return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return []
