import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class FastModelValidation:
    """
    Optimized model validation with reduced computational complexity
    """
    
    def __init__(self):
        self.validation_results = {}
        
    def fast_walk_forward_validation(self, model_class, data, category, min_train_size=20, max_predictions=50):
        """
        Fast walk-forward validation with limited number of predictions
        """
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < min_train_size + 3:
            return None
            
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        predictions = []
        actuals = []
        prediction_dates = []
        
        # Limit number of validation points for speed
        total_points = len(prices) - min_train_size
        step_size = max(1, total_points // max_predictions)
        
        for i in range(min_train_size, len(prices) - 1, step_size):
            if len(predictions) >= max_predictions:
                break
                
            # Train on data up to point i
            train_data = category_data.iloc[:i]
            
            # Initialize and train model
            try:
                model = model_class()
                model.fit(train_data)
                
                # Make single-step prediction
                pred_result = model.predict(1)
                if category in pred_result and len(pred_result[category]['mean']) > 0:
                    pred_value = pred_result[category]['mean'][0]
                    actual_value = prices[i]
                    
                    predictions.append(pred_value)
                    actuals.append(actual_value)
                    prediction_dates.append(category_data.iloc[i]['date'])
                    
            except Exception:
                continue
        
        if len(predictions) < 3:
            return None
            
        # Calculate validation metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        r2 = r2_score(actuals, predictions)
        
        # Direction accuracy
        direction_accuracy = self.calculate_direction_accuracy(actuals, predictions)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'dates': prediction_dates,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'n_predictions': len(predictions)
        }
    
    def fast_backtest_cycles(self, model_class, data, category, n_cycles=6):
        """
        Fast backtesting with limited cycles
        """
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < n_cycles * 2:
            return None
            
        category_data = category_data.sort_values('date')
        
        # Take last n_cycles for backtesting
        backtest_data = category_data.iloc[-n_cycles:].copy()
        train_data = category_data.iloc[:-n_cycles].copy()
        
        if len(train_data) < 12:  # Need minimum training data
            return None
        
        # Train model on historical data
        try:
            model = model_class()
            model.fit(train_data)
            
            # Make predictions for the backtest period
            pred_result = model.predict(n_cycles)
            if category not in pred_result:
                return None
                
            predictions = pred_result[category]['mean']
            actuals = backtest_data['premium'].values
            
            # Ensure same length
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]
            
            if len(predictions) < 2:
                return None
            
            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            r2 = r2_score(actuals, predictions)
            
            # Direction accuracy
            direction_accuracy = self.calculate_direction_accuracy(actuals, predictions)
            
            # Simple volatility tracking
            actual_vol = np.std(actuals)
            predicted_vol = np.std(predictions)
            volatility_ratio = predicted_vol / actual_vol if actual_vol > 0 else 1
            
            return {
                'predictions': predictions,
                'actuals': actuals,
                'dates': backtest_data['date'].values[:min_len],
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'volatility_tracking': {
                    'correlation': np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else 0,
                    'ratio': volatility_ratio
                },
                'n_cycles': min_len
            }
            
        except Exception:
            return None
    
    def calculate_direction_accuracy(self, actuals, predictions):
        """
        Calculate percentage of correct up/down movement predictions
        """
        if len(actuals) < 2 or len(predictions) < 2:
            return 0
        
        actual_changes = np.diff(actuals)
        predicted_changes = np.diff(predictions)
        
        # Determine direction (1 for up, -1 for down, 0 for flat)
        actual_directions = np.sign(actual_changes)
        predicted_directions = np.sign(predicted_changes)
        
        # Calculate accuracy
        correct_directions = (actual_directions == predicted_directions).sum()
        total_directions = len(actual_directions)
        
        return (correct_directions / total_directions) * 100 if total_directions > 0 else 0
    
    def run_fast_validation(self, model_class, data, category):
        """
        Run fast validation for a single category
        """
        category_data = data[data['vehicle_class'] == category]
        if len(category_data) < 24:  # Need sufficient data
            return None
        
        # Fast walk-forward validation
        wf_result = self.fast_walk_forward_validation(model_class, data, category)
        
        # Fast backtesting
        bt_result = self.fast_backtest_cycles(model_class, data, category)
        
        return {
            'walk_forward': wf_result,
            'backtest': bt_result,
            'data_points': len(category_data)
        }
    
    def run_all_categories_fast(self, model_class, data, categories=None):
        """
        Run fast validation for all categories
        """
        if categories is None:
            categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        
        results = {}
        
        for category in categories:
            result = self.run_fast_validation(model_class, data, category)
            if result:
                results[category] = result
        
        return results
    
    def format_fast_summary(self, results):
        """
        Format validation results for display
        """
        summary = []
        
        for category, result in results.items():
            if not result:
                continue
            
            summary.append(f"\n**{category}**")
            
            # Walk-forward results
            if result['walk_forward']:
                wf = result['walk_forward']
                summary.append(f"Walk-Forward ({wf['n_predictions']} tests):")
                summary.append(f"  • MAPE: {wf['mape']:.1f}%")
                summary.append(f"  • Direction Accuracy: {wf['direction_accuracy']:.1f}%")
                summary.append(f"  • R²: {wf['r2']:.3f}")
            
            # Backtest results
            if result['backtest']:
                bt = result['backtest']
                summary.append(f"Backtesting ({bt['n_cycles']} cycles):")
                summary.append(f"  • MAPE: {bt['mape']:.1f}%")
                summary.append(f"  • Direction Accuracy: {bt['direction_accuracy']:.1f}%")
                summary.append(f"  • Volatility Ratio: {bt['volatility_tracking']['ratio']:.3f}")
        
        return "\n".join(summary)