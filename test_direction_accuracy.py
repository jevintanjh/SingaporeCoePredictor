#!/usr/bin/env python3
"""
Test script to validate directional accuracy improvements
"""

import pandas as pd
import numpy as np
from models.enhanced_directional_forecaster import EnhancedDirectionalForecaster
from utils.data_processor import DataProcessor
from utils.fast_validation import FastModelValidation

def test_direction_accuracy():
    """Test directional accuracy for all categories"""
    print("Loading data...")
    processor = DataProcessor()
    data = processor.load_data('./attached_assets/COEBiddingResultsPrices_1749430265007.csv')
    
    if data is None:
        print("Error: Could not load data")
        return
    
    # Preprocess data
    data = processor.preprocess_data(data)
    print(f"Loaded {len(data)} records")
    
    # Initialize enhanced directional forecaster
    print("\nInitializing Enhanced Directional Forecaster...")
    model = EnhancedDirectionalForecaster()
    model.fit(data)
    
    # Test validation
    print("\nRunning fast validation across all categories...")
    validator = FastModelValidation()
    all_results = validator.run_all_categories_fast(EnhancedDirectionalForecaster, data)
    
    print("\n" + "="*60)
    print("DIRECTIONAL ACCURACY RESULTS")
    print("="*60)
    
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    
    for category in categories:
        if category in all_results:
            metrics = all_results[category]
            direction_acc = metrics.get('direction_accuracy', 0)
            mape = metrics.get('mape', 0)
            r2 = metrics.get('r2', 0)
            
            status = "✓ ABOVE RANDOM" if direction_acc > 50 else "✗ BELOW RANDOM"
            
            print(f"\n{category}:")
            print(f"  Direction Accuracy: {direction_acc:.1f}% {status}")
            print(f"  MAPE: {mape:.1f}%")
            print(f"  R² Score: {r2:.3f}")
            
            if 'has_direction_model' in metrics:
                has_model = "Yes" if metrics['has_direction_model'] else "No"
                print(f"  Enhanced Direction Model: {has_model}")
        else:
            print(f"\n{category}: No validation results")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    above_random = 0
    total_tested = 0
    
    for category in categories:
        if category in all_results:
            total_tested += 1
            direction_acc = all_results[category].get('direction_accuracy', 0)
            if direction_acc > 50:
                above_random += 1
    
    print(f"Categories above 50% accuracy: {above_random}/{total_tested}")
    print(f"Improvement rate: {above_random/total_tested*100:.1f}%" if total_tested > 0 else "No results")
    
    # Specific focus on A, D, E
    problem_categories = ['Category A', 'Category D', 'Category E']
    improved_categories = []
    
    for cat in problem_categories:
        if cat in all_results:
            acc = all_results[cat].get('direction_accuracy', 0)
            if acc > 50:
                improved_categories.append(cat)
    
    print(f"\nPreviously poor categories (A, D, E) now above 50%:")
    if improved_categories:
        print(f"  Improved: {', '.join(improved_categories)}")
    else:
        print("  None improved above 50% threshold")

if __name__ == "__main__":
    test_direction_accuracy()