#!/usr/bin/env python3
"""
Quick test of directional accuracy improvements
"""

import pandas as pd
import numpy as np
from models.enhanced_directional_forecaster import EnhancedDirectionalForecaster
from utils.data_processor import DataProcessor

def quick_test():
    """Quick test of direction accuracy"""
    print("Loading data...")
    processor = DataProcessor()
    data = processor.load_data('./attached_assets/COEBiddingResultsPrices_1749430265007.csv')
    data = processor.preprocess_data(data)
    
    print("Testing enhanced directional forecaster...")
    model = EnhancedDirectionalForecaster()
    
    # Test specific categories
    test_categories = ['Category A', 'Category D', 'Category E']
    
    for category in test_categories:
        category_data = data[data['vehicle_class'] == category].copy()
        if len(category_data) >= 20:
            category_data = category_data.sort_values('date')
            prices = category_data['premium'].values
            
            print(f"\n{category}: {len(prices)} data points")
            
            # Store basic model info and calculate metrics
            model.models[category] = {'prices': prices, 'dates': category_data['date'].values}
            model.train_enhanced_direction_model(prices, category)
            model.calculate_robust_validation_metrics(prices, category)
            
            metrics = model.get_performance_metrics(category)
            direction_acc = metrics.get('direction_accuracy', 0)
            has_model = metrics.get('has_direction_model', False)
            
            status = "✓ IMPROVED" if direction_acc > 50 else "✗ STILL POOR"
            model_status = "Enhanced" if has_model else "Basic"
            
            print(f"  Direction Accuracy: {direction_acc:.1f}% {status}")
            print(f"  Model Type: {model_status}")
            print(f"  MAPE: {metrics.get('mape', 0):.1f}%")
        else:
            print(f"\n{category}: Insufficient data ({len(category_data)} points)")

if __name__ == "__main__":
    quick_test()