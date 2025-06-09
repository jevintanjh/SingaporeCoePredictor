import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelValidation:
    """
    Advanced validation techniques for financial time series models
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def bootstrap_validation(self, model_class, data, category, n_bootstraps=100, sample_ratio=0.8):
        """
        Bootstrap validation: resample data multiple times to assess model stability
        """
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < 30:
            return None
        
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        bootstrap_results = {
            'mape_scores': [],
            'direction_scores': [],
            'volatility_scores': [],
            'r2_scores': []
        }
        
        for i in range(n_bootstraps):
            # Bootstrap sample
            sample_size = int(len(category_data) * sample_ratio)
            sample_indices = np.random.choice(len(category_data), sample_size, replace=True)
            bootstrap_sample = category_data.iloc[sample_indices].sort_values('date')
            
            try:
                # Train model on bootstrap sample
                model = model_class()
                model.fit(bootstrap_sample)
                
                # Test on remaining data
                all_indices = set(range(len(category_data)))
                sample_indices_set = set(sample_indices)
                remaining_indices = list(all_indices - sample_indices_set)
                if len(remaining_indices) < 5:
                    continue
                
                test_data = category_data.iloc[remaining_indices[:5]]  # Use first 5 for testing
                test_prices = test_data['premium'].values
                
                # Make predictions
                predictions = model.predict(len(test_prices))
                if category in predictions:
                    pred_values = predictions[category]['mean'][:len(test_prices)]
                    
                    # Calculate metrics
                    mape = np.mean(np.abs((test_prices - pred_values) / test_prices)) * 100
                    bootstrap_results['mape_scores'].append(mape)
                    
                    # Direction accuracy
                    if len(test_prices) > 1:
                        actual_dirs = np.diff(test_prices) > 0
                        pred_dirs = np.diff(pred_values) > 0
                        dir_acc = np.mean(actual_dirs == pred_dirs) * 100
                        bootstrap_results['direction_scores'].append(dir_acc)
                    
                    # Volatility comparison
                    actual_vol = np.std(test_prices)
                    pred_vol = np.std(pred_values)
                    vol_ratio = pred_vol / actual_vol if actual_vol > 0 else 1
                    bootstrap_results['volatility_scores'].append(vol_ratio)
                    
                    # R² score
                    r2 = r2_score(test_prices, pred_values)
                    bootstrap_results['r2_scores'].append(r2)
                    
            except Exception:
                continue
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric, scores in bootstrap_results.items():
            if len(scores) > 10:
                confidence_intervals[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'ci_5': np.percentile(scores, 5),
                    'ci_95': np.percentile(scores, 95),
                    'n_samples': len(scores)
                }
        
        return confidence_intervals
    
    def regime_change_validation(self, model_class, data, category):
        """
        Test model performance across different market regimes (high/low volatility periods)
        """
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < 50:
            return None
        
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        # Calculate rolling volatility to identify regimes
        window = 12
        rolling_vol = []
        for i in range(window, len(prices)):
            vol = np.std(prices[i-window:i])
            rolling_vol.append(vol)
        
        # Define regimes based on volatility percentiles
        vol_75th = np.percentile(rolling_vol, 75)
        vol_25th = np.percentile(rolling_vol, 25)
        
        high_vol_periods = []
        low_vol_periods = []
        
        for i, vol in enumerate(rolling_vol):
            if vol > vol_75th:
                high_vol_periods.append(i + window)
            elif vol < vol_25th:
                low_vol_periods.append(i + window)
        
        regime_results = {}
        
        # Test performance in high volatility regime
        if len(high_vol_periods) > 10:
            regime_results['high_volatility'] = self._test_regime_performance(
                model_class, category_data, high_vol_periods, category
            )
        
        # Test performance in low volatility regime
        if len(low_vol_periods) > 10:
            regime_results['low_volatility'] = self._test_regime_performance(
                model_class, category_data, low_vol_periods, category
            )
        
        return regime_results
    
    def _test_regime_performance(self, model_class, data, period_indices, category):
        """Helper function to test model performance in specific regime"""
        try:
            # Select data from regime periods
            regime_data = data.iloc[period_indices[:20]]  # Use up to 20 periods
            
            if len(regime_data) < 10:
                return None
            
            # Split for training and testing
            split_point = int(len(regime_data) * 0.7)
            train_data = regime_data.iloc[:split_point]
            test_data = regime_data.iloc[split_point:]
            
            # Train model
            model = model_class()
            model.fit(train_data)
            
            # Test predictions
            test_prices = test_data['premium'].values
            predictions = model.predict(len(test_prices))
            
            if category in predictions:
                pred_values = predictions[category]['mean'][:len(test_prices)]
                
                # Calculate metrics
                mape = np.mean(np.abs((test_prices - pred_values) / test_prices)) * 100
                
                # Direction accuracy
                if len(test_prices) > 1:
                    actual_dirs = np.diff(test_prices) > 0
                    pred_dirs = np.diff(pred_values) > 0
                    dir_acc = np.mean(actual_dirs == pred_dirs) * 100
                else:
                    dir_acc = 50
                
                return {
                    'mape': mape,
                    'direction_accuracy': dir_acc,
                    'n_periods': len(test_prices),
                    'avg_volatility': np.std(test_prices)
                }
        
        except Exception:
            return None
    
    def rolling_origin_validation(self, model_class, data, category, min_train_size=24, step_size=3):
        """
        Rolling origin validation: incrementally expand training window
        """
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < min_train_size + 6:
            return None
        
        category_data = category_data.sort_values('date')
        
        results = {
            'training_sizes': [],
            'test_performance': [],
            'prediction_stability': []
        }
        
        for train_size in range(min_train_size, len(category_data) - 3, step_size):
            train_data = category_data.iloc[:train_size]
            test_data = category_data.iloc[train_size:train_size + 3]
            
            if len(test_data) < 3:
                break
            
            try:
                # Train model
                model = model_class()
                model.fit(train_data)
                
                # Make predictions
                test_prices = test_data['premium'].values
                predictions = model.predict(3)
                
                if category in predictions:
                    pred_values = predictions[category]['mean'][:len(test_prices)]
                    
                    # Calculate performance
                    mape = np.mean(np.abs((test_prices - pred_values) / test_prices)) * 100
                    
                    # Prediction stability (variance of predictions)
                    pred_stability = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) > 0 else 0
                    
                    results['training_sizes'].append(train_size)
                    results['test_performance'].append(mape)
                    results['prediction_stability'].append(pred_stability)
                
            except Exception:
                continue
        
        if len(results['training_sizes']) > 0:
            return {
                'performance_trend': np.polyfit(results['training_sizes'], results['test_performance'], 1)[0],
                'stability_trend': np.polyfit(results['training_sizes'], results['prediction_stability'], 1)[0],
                'final_performance': results['test_performance'][-1] if results['test_performance'] else None,
                'results': results
            }
        
        return None
    
    def stress_testing(self, model_class, data, category):
        """
        Stress test model with extreme scenarios
        """
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < 30:
            return None
        
        category_data = category_data.sort_values('date')
        prices = category_data['premium'].values
        
        stress_results = {}
        
        # Test 1: Missing data simulation
        stress_results['missing_data'] = self._test_missing_data(model_class, category_data, category)
        
        # Test 2: Extreme volatility periods
        stress_results['extreme_volatility'] = self._test_extreme_volatility(model_class, category_data, category)
        
        # Test 3: Trend breaks
        stress_results['trend_breaks'] = self._test_trend_breaks(model_class, category_data, category)
        
        return stress_results
    
    def _test_missing_data(self, model_class, data, category):
        """Test model robustness with missing data"""
        try:
            # Randomly remove 20% of data points
            n_remove = int(len(data) * 0.2)
            remove_indices = np.random.choice(len(data), n_remove, replace=False)
            
            # Create data with gaps
            incomplete_data = data.drop(data.index[remove_indices]).reset_index(drop=True)
            
            if len(incomplete_data) < 15:
                return None
            
            # Test model training with incomplete data
            model = model_class()
            model.fit(incomplete_data)
            
            # Make predictions
            predictions = model.predict(3)
            
            if category in predictions:
                pred_values = predictions[category]['mean']
                return {
                    'can_handle_missing': True,
                    'prediction_range': max(pred_values) - min(pred_values),
                    'data_completeness': len(incomplete_data) / len(data)
                }
        
        except Exception:
            return {'can_handle_missing': False}
    
    def _test_extreme_volatility(self, model_class, data, category):
        """Test model with artificially high volatility data"""
        try:
            # Add noise to create extreme volatility
            data_copy = data.copy()
            prices = data_copy['premium'].values
            
            # Add 50% random noise
            noise = np.random.normal(0, np.std(prices) * 0.5, len(prices))
            data_copy['premium'] = prices + noise
            
            # Ensure no negative prices
            data_copy['premium'] = np.maximum(data_copy['premium'], prices * 0.1)
            
            # Test model
            model = model_class()
            model.fit(data_copy)
            
            predictions = model.predict(3)
            
            if category in predictions:
                pred_values = predictions[category]['mean']
                return {
                    'handles_extreme_volatility': True,
                    'prediction_variance': np.var(pred_values),
                    'volatility_ratio': np.std(data_copy['premium']) / np.std(prices)
                }
        
        except Exception:
            return {'handles_extreme_volatility': False}
    
    def _test_trend_breaks(self, model_class, data, category):
        """Test model with artificial trend breaks"""
        try:
            # Create artificial trend break in middle of data
            data_copy = data.copy()
            mid_point = len(data_copy) // 2
            
            # Shift second half of data up by 20%
            data_copy.loc[mid_point:, 'premium'] *= 1.2
            
            # Test model
            model = model_class()
            model.fit(data_copy)
            
            predictions = model.predict(3)
            
            if category in predictions:
                pred_values = predictions[category]['mean']
                last_actual = data_copy['premium'].iloc[-1]
                
                return {
                    'handles_trend_breaks': True,
                    'prediction_continuity': abs(pred_values[0] - last_actual) / last_actual,
                    'trend_adaptation': np.mean(np.diff(pred_values))
                }
        
        except Exception:
            return {'handles_trend_breaks': False}
    
    def model_comparison_validation(self, model_classes, data, category):
        """
        Compare multiple models using various validation metrics
        """
        if len(model_classes) < 2:
            return None
        
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) < 25:
            return None
        
        category_data = category_data.sort_values('date')
        
        # Split data
        split_point = int(len(category_data) * 0.8)
        train_data = category_data.iloc[:split_point]
        test_data = category_data.iloc[split_point:]
        
        model_results = {}
        
        for i, model_class in enumerate(model_classes):
            model_name = f"Model_{i+1}_{model_class.__name__}"
            
            try:
                # Train model
                model = model_class()
                model.fit(train_data)
                
                # Test predictions
                test_prices = test_data['premium'].values
                predictions = model.predict(len(test_prices))
                
                if category in predictions:
                    pred_values = predictions[category]['mean'][:len(test_prices)]
                    
                    # Calculate comprehensive metrics
                    mape = np.mean(np.abs((test_prices - pred_values) / test_prices)) * 100
                    rmse = np.sqrt(mean_squared_error(test_prices, pred_values))
                    
                    # Direction accuracy
                    if len(test_prices) > 1:
                        actual_dirs = np.diff(test_prices) > 0
                        pred_dirs = np.diff(pred_values) > 0
                        dir_acc = np.mean(actual_dirs == pred_dirs) * 100
                    else:
                        dir_acc = 50
                    
                    # Prediction consistency
                    pred_volatility = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) > 0 else 0
                    
                    model_results[model_name] = {
                        'mape': mape,
                        'rmse': rmse,
                        'direction_accuracy': dir_acc,
                        'prediction_volatility': pred_volatility,
                        'mean_prediction': np.mean(pred_values)
                    }
            
            except Exception as e:
                model_results[model_name] = {'error': str(e)}
        
        # Rank models
        if len(model_results) > 1:
            # Calculate composite score (lower is better)
            for model_name, results in model_results.items():
                if 'error' not in results:
                    composite_score = (
                        results['mape'] * 0.4 +  # 40% weight on MAPE
                        (100 - results['direction_accuracy']) * 0.3 +  # 30% weight on direction (inverted)
                        results['rmse'] / np.mean(test_prices) * 100 * 0.3  # 30% weight on normalized RMSE
                    )
                    results['composite_score'] = composite_score
            
            # Rank by composite score
            valid_results = {k: v for k, v in model_results.items() if 'error' not in v}
            if valid_results:
                best_model = min(valid_results.keys(), key=lambda x: valid_results[x]['composite_score'])
                model_results['best_model'] = best_model
        
        return model_results
    
    def run_comprehensive_validation(self, model_class, data, category):
        """
        Run all validation tests for a comprehensive assessment
        """
        comprehensive_results = {}
        
        print(f"Running comprehensive validation for {category}...")
        
        # Bootstrap validation
        comprehensive_results['bootstrap'] = self.bootstrap_validation(model_class, data, category)
        
        # Regime change validation
        comprehensive_results['regime_analysis'] = self.regime_change_validation(model_class, data, category)
        
        # Rolling origin validation
        comprehensive_results['rolling_origin'] = self.rolling_origin_validation(model_class, data, category)
        
        # Stress testing
        comprehensive_results['stress_tests'] = self.stress_testing(model_class, data, category)
        
        return comprehensive_results
    
    def format_comprehensive_report(self, results):
        """
        Format comprehensive validation results into readable report
        """
        report = []
        
        # Bootstrap results
        if 'bootstrap' in results and results['bootstrap']:
            report.append("**Bootstrap Validation (100 samples):**")
            bootstrap = results['bootstrap']
            for metric, stats in bootstrap.items():
                if isinstance(stats, dict):
                    report.append(f"  • {metric.replace('_', ' ').title()}: {stats['mean']:.2f} ± {stats['std']:.2f}")
                    report.append(f"    95% CI: [{stats['ci_5']:.2f}, {stats['ci_95']:.2f}]")
        
        # Regime analysis
        if 'regime_analysis' in results and results['regime_analysis']:
            report.append("\n**Regime Analysis:**")
            regime = results['regime_analysis']
            for regime_type, metrics in regime.items():
                if metrics:
                    report.append(f"  • {regime_type.replace('_', ' ').title()}: MAPE {metrics['mape']:.1f}%, Direction {metrics['direction_accuracy']:.1f}%")
        
        # Rolling origin
        if 'rolling_origin' in results and results['rolling_origin']:
            report.append("\n**Rolling Origin Validation:**")
            ro = results['rolling_origin']
            if ro['performance_trend']:
                trend_desc = "improving" if ro['performance_trend'] < 0 else "degrading"
                report.append(f"  • Performance trend: {trend_desc} with more data")
                report.append(f"  • Final MAPE: {ro['final_performance']:.1f}%")
        
        # Stress tests
        if 'stress_tests' in results and results['stress_tests']:
            report.append("\n**Stress Test Results:**")
            stress = results['stress_tests']
            
            if 'missing_data' in stress and stress['missing_data']:
                md = stress['missing_data']
                status = "✓ Robust" if md.get('can_handle_missing', False) else "✗ Sensitive"
                report.append(f"  • Missing Data: {status}")
            
            if 'extreme_volatility' in stress and stress['extreme_volatility']:
                ev = stress['extreme_volatility']
                status = "✓ Stable" if ev.get('handles_extreme_volatility', False) else "✗ Unstable"
                report.append(f"  • High Volatility: {status}")
            
            if 'trend_breaks' in stress and stress['trend_breaks']:
                tb = stress['trend_breaks']
                status = "✓ Adaptive" if tb.get('handles_trend_breaks', False) else "✗ Rigid"
                report.append(f"  • Trend Breaks: {status}")
        
        return "\n".join(report)