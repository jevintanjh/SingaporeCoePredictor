import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return np.inf
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    
    if not mask.any():
        return np.inf
    
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def calculate_r2(y_true, y_pred):
    """Calculate R-squared (coefficient of determination)"""
    return r2_score(y_true, y_pred)

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy (percentage of correct trend predictions)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) < 2:
        return np.nan
    
    # Calculate actual and predicted directions
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    correct_directions = actual_direction == predicted_direction
    return np.mean(correct_directions) * 100

def calculate_theil_u_statistic(y_true, y_pred):
    """Calculate Theil's U statistic"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) < 2:
        return np.nan
    
    # Calculate forecast error
    forecast_error = np.sum((y_pred - y_true) ** 2)
    
    # Calculate naive forecast error (using previous period)
    naive_forecast = np.roll(y_true, 1)[1:]  # Previous values
    actual_values = y_true[1:]  # Current values
    naive_error = np.sum((naive_forecast - actual_values) ** 2)
    
    if naive_error == 0:
        return np.inf if forecast_error > 0 else 0
    
    return np.sqrt(forecast_error / naive_error)

def calculate_prediction_intervals(y_pred, residuals, confidence_level=0.95):
    """Calculate prediction intervals based on residual analysis"""
    residuals = np.array(residuals)
    y_pred = np.array(y_pred)
    
    # Calculate residual standard deviation
    residual_std = np.std(residuals)
    
    # Z-score for confidence level
    alpha = 1 - confidence_level
    z_score = 1.96  # For 95% confidence interval
    
    # Calculate intervals
    lower_bound = y_pred - z_score * residual_std
    upper_bound = y_pred + z_score * residual_std
    
    return lower_bound, upper_bound

def calculate_metrics(y_true, y_pred, return_dict=True):
    """Calculate comprehensive set of prediction metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return None
    
    metrics = {}
    
    try:
        # Basic metrics
        metrics['mae'] = calculate_mae(y_true_clean, y_pred_clean)
        metrics['rmse'] = calculate_rmse(y_true_clean, y_pred_clean)
        metrics['mape'] = calculate_mape(y_true_clean, y_pred_clean)
        metrics['smape'] = calculate_smape(y_true_clean, y_pred_clean)
        metrics['r2'] = calculate_r2(y_true_clean, y_pred_clean)
        
        # Advanced metrics
        metrics['directional_accuracy'] = calculate_directional_accuracy(y_true_clean, y_pred_clean)
        
        # Calculate residuals
        residuals = y_true_clean - y_pred_clean
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        # Theil's U statistic
        metrics['theil_u'] = calculate_theil_u_statistic(y_true_clean, y_pred_clean)
        
        # Additional statistics
        metrics['max_error'] = np.max(np.abs(residuals))
        metrics['median_error'] = np.median(np.abs(residuals))
        
        # Bias measures
        metrics['mean_bias'] = np.mean(residuals)
        metrics['median_bias'] = np.median(residuals)
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
    
    if return_dict:
        return metrics
    else:
        return (metrics['mae'], metrics['rmse'], metrics['mape'], 
                metrics['r2'], metrics['directional_accuracy'])

def format_metrics(metrics):
    """Format metrics for display"""
    if metrics is None:
        return "Metrics unavailable"
    
    formatted = {}
    
    # Format each metric with appropriate precision and units
    formatted['MAE'] = f"${metrics.get('mae', 0):,.0f}"
    formatted['RMSE'] = f"${metrics.get('rmse', 0):,.0f}"
    formatted['MAPE'] = f"{metrics.get('mape', 0):.2f}%"
    formatted['SMAPE'] = f"{metrics.get('smape', 0):.2f}%"
    formatted['R²'] = f"{metrics.get('r2', 0):.3f}"
    formatted['Directional Accuracy'] = f"{metrics.get('directional_accuracy', 0):.1f}%"
    formatted['Theil\'s U'] = f"{metrics.get('theil_u', 0):.3f}"
    formatted['Mean Bias'] = f"${metrics.get('mean_bias', 0):,.0f}"
    formatted['Max Error'] = f"${metrics.get('max_error', 0):,.0f}"
    
    return formatted

def evaluate_model_performance(metrics):
    """Evaluate model performance based on metrics"""
    if metrics is None:
        return "Unable to evaluate"
    
    mape = metrics.get('mape', float('inf'))
    r2 = metrics.get('r2', 0)
    directional_accuracy = metrics.get('directional_accuracy', 0)
    
    # Performance thresholds
    excellent_mape = 5
    good_mape = 10
    fair_mape = 20
    
    excellent_r2 = 0.8
    good_r2 = 0.6
    fair_r2 = 0.4
    
    excellent_direction = 70
    good_direction = 60
    fair_direction = 50
    
    # Evaluate MAPE
    if mape <= excellent_mape:
        mape_rating = "Excellent"
    elif mape <= good_mape:
        mape_rating = "Good"
    elif mape <= fair_mape:
        mape_rating = "Fair"
    else:
        mape_rating = "Poor"
    
    # Evaluate R²
    if r2 >= excellent_r2:
        r2_rating = "Excellent"
    elif r2 >= good_r2:
        r2_rating = "Good"
    elif r2 >= fair_r2:
        r2_rating = "Fair"
    else:
        r2_rating = "Poor"
    
    # Evaluate directional accuracy
    if directional_accuracy >= excellent_direction:
        direction_rating = "Excellent"
    elif directional_accuracy >= good_direction:
        direction_rating = "Good"
    elif directional_accuracy >= fair_direction:
        direction_rating = "Fair"
    else:
        direction_rating = "Poor"
    
    # Overall rating (weighted average)
    ratings = [mape_rating, r2_rating, direction_rating]
    rating_scores = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}
    
    avg_score = np.mean([rating_scores[rating] for rating in ratings])
    
    if avg_score >= 3.5:
        overall_rating = "Excellent"
    elif avg_score >= 2.5:
        overall_rating = "Good"
    elif avg_score >= 1.5:
        overall_rating = "Fair"
    else:
        overall_rating = "Poor"
    
    return {
        'overall': overall_rating,
        'mape': mape_rating,
        'r2': r2_rating,
        'directional_accuracy': direction_rating,
        'summary': f"Overall: {overall_rating} (MAPE: {mape_rating}, R²: {r2_rating}, Direction: {direction_rating})"
    }

def compare_model_performance(metrics_dict):
    """Compare performance across multiple models or categories"""
    if not metrics_dict:
        return None
    
    comparison = pd.DataFrame(metrics_dict).T
    
    # Add performance rankings
    ranking_cols = ['mape', 'rmse', 'mae']  # Lower is better
    for col in ranking_cols:
        if col in comparison.columns:
            comparison[f'{col}_rank'] = comparison[col].rank(ascending=True)
    
    # For R² and directional accuracy, higher is better
    improvement_cols = ['r2', 'directional_accuracy']
    for col in improvement_cols:
        if col in comparison.columns:
            comparison[f'{col}_rank'] = comparison[col].rank(ascending=False)
    
    # Calculate overall rank (average of individual ranks)
    rank_cols = [col for col in comparison.columns if col.endswith('_rank')]
    if rank_cols:
        comparison['overall_rank'] = comparison[rank_cols].mean(axis=1)
        comparison = comparison.sort_values('overall_rank')
    
    return comparison

def calculate_forecast_accuracy_by_horizon(y_true_list, y_pred_list, horizons):
    """Calculate accuracy metrics for different forecast horizons"""
    accuracy_by_horizon = {}
    
    for i, horizon in enumerate(horizons):
        if i < len(y_true_list) and i < len(y_pred_list):
            y_true = y_true_list[i]
            y_pred = y_pred_list[i]
            
            metrics = calculate_metrics(y_true, y_pred)
            if metrics:
                accuracy_by_horizon[f'horizon_{horizon}'] = metrics
    
    return accuracy_by_horizon

def statistical_significance_test(residuals1, residuals2, alpha=0.05):
    """Test statistical significance between two sets of residuals"""
    from scipy.stats import ttest_rel
    
    try:
        # Paired t-test for related samples
        statistic, p_value = ttest_rel(residuals1, residuals2)
        
        is_significant = p_value < alpha
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': alpha,
            'interpretation': 'Significantly different' if is_significant else 'Not significantly different'
        }
    except Exception as e:
        return {
            'error': str(e),
            'interpretation': 'Unable to perform test'
        }
