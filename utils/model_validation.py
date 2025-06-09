import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelValidation:
    """
    Comprehensive model validation including walk-forward validation, 
    backtesting, and business-relevant metrics
    """
    
    def __init__(self):
        self.validation_results = {}
        self.backtest_results = {}
        
    def walk_forward_validation(self, model_class, data, category, min_train_size=24, steps_ahead=3):
        """
        Walk-forward validation: rolling forecasting origin
        Simulates real-world prediction setup
        """
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < min_train_size + steps_ahead:
            return None
            
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        predictions = []
        actuals = []
        prediction_dates = []
        
        # Start validation from min_train_size onwards
        for i in range(min_train_size, len(prices) - steps_ahead + 1):
            # Train on data up to point i
            train_data = category_data.iloc[:i]
            
            # Initialize and train model
            model = model_class()
            model.fit(train_data)
            
            # Make prediction for next steps_ahead periods
            try:
                pred_result = model.predict(steps_ahead)
                if category in pred_result:
                    pred_values = pred_result[category]['mean']
                    
                    # Store predictions and actuals
                    for j in range(min(steps_ahead, len(pred_values))):
                        if i + j < len(prices):
                            predictions.append(pred_values[j])
                            actuals.append(prices[i + j])
                            prediction_dates.append(category_data.iloc[i + j]['date'])
            except Exception as e:
                continue
        
        if len(predictions) == 0:
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
    
    def backtest_cycles(self, model_class, data, category, n_cycles=12):
        """
        Backtest on past n cycles: simulate "what-if" predictions
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
        
        # Train model on historical data (excluding backtest period)
        model = model_class()
        model.fit(train_data)
        
        # Make predictions for the backtest period
        try:
            pred_result = model.predict(n_cycles)
            if category not in pred_result:
                return None
                
            predictions = pred_result[category]['mean']
            actuals = backtest_data['premium'].values
            
            # Ensure same length
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]
            
            if len(predictions) == 0:
                return None
            
            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            r2 = r2_score(actuals, predictions)
            
            # Direction accuracy
            direction_accuracy = self.calculate_direction_accuracy(actuals, predictions)
            
            # Volatility tracking
            volatility_tracking = self.calculate_volatility_tracking(actuals, predictions)
            
            return {
                'predictions': predictions,
                'actuals': actuals,
                'dates': backtest_data['date'].values[:min_len],
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'volatility_tracking': volatility_tracking,
                'n_cycles': min_len
            }
            
        except Exception as e:
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
    
    def calculate_volatility_tracking(self, actuals, predictions):
        """
        Measure how well the model reflects changes in price volatility
        """
        if len(actuals) < 4 or len(predictions) < 4:
            return {'correlation': 0, 'ratio': 1}
        
        # Calculate rolling volatility (using windows of 3)
        actual_volatility = []
        predicted_volatility = []
        
        for i in range(2, len(actuals)):
            actual_vol = np.std(actuals[max(0, i-2):i+1])
            predicted_vol = np.std(predictions[max(0, i-2):i+1])
            actual_volatility.append(actual_vol)
            predicted_volatility.append(predicted_vol)
        
        if len(actual_volatility) < 2:
            return {'correlation': 0, 'ratio': 1}
        
        # Correlation between actual and predicted volatility
        correlation = np.corrcoef(actual_volatility, predicted_volatility)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # Ratio of average volatilities
        avg_actual_vol = np.mean(actual_volatility)
        avg_predicted_vol = np.mean(predicted_volatility)
        volatility_ratio = avg_predicted_vol / avg_actual_vol if avg_actual_vol > 0 else 1
        
        return {
            'correlation': correlation,
            'ratio': volatility_ratio,
            'actual_volatility': actual_volatility,
            'predicted_volatility': predicted_volatility
        }
    
    def run_comprehensive_validation(self, model_class, data, categories=None):
        """
        Run all validation tests for specified categories
        """
        if categories is None:
            categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        
        results = {}
        
        for category in categories:
            category_data = data[data['vehicle_class'] == category]
            if len(category_data) < 30:  # Need sufficient data
                continue
            
            print(f"Validating {category}...")
            
            # Walk-forward validation
            wf_result = self.walk_forward_validation(model_class, data, category)
            
            # Backtesting
            backtest_result = self.backtest_cycles(model_class, data, category)
            
            results[category] = {
                'walk_forward': wf_result,
                'backtest': backtest_result,
                'data_points': len(category_data)
            }
        
        return results
    
    def format_validation_summary(self, results):
        """
        Format validation results for display
        """
        summary = []
        
        for category, result in results.items():
            if result['walk_forward'] is None and result['backtest'] is None:
                continue
            
            summary.append(f"\n**{category}**")
            
            # Walk-forward results
            if result['walk_forward']:
                wf = result['walk_forward']
                summary.append(f"Walk-Forward Validation ({wf['n_predictions']} predictions):")
                summary.append(f"  • MAPE: {wf['mape']:.1f}%")
                summary.append(f"  • Direction Accuracy: {wf['direction_accuracy']:.1f}%")
                summary.append(f"  • R²: {wf['r2']:.3f}")
            
            # Backtest results
            if result['backtest']:
                bt = result['backtest']
                summary.append(f"Backtesting ({bt['n_cycles']} cycles):")
                summary.append(f"  • MAPE: {bt['mape']:.1f}%")
                summary.append(f"  • Direction Accuracy: {bt['direction_accuracy']:.1f}%")
                summary.append(f"  • Volatility Correlation: {bt['volatility_tracking']['correlation']:.3f}")
                summary.append(f"  • Volatility Ratio: {bt['volatility_tracking']['ratio']:.3f}")
        
        return "\n".join(summary)