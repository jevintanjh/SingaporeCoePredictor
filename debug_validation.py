#!/usr/bin/env python3
"""
Debug script to identify validation issues
"""

import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from models.enhanced_directional_forecaster import EnhancedDirectionalForecaster
from utils.fast_validation import FastModelValidation

def debug_data_processing():
    """Debug data processing pipeline"""
    print("=== DATA PROCESSING DEBUG ===")
    
    # Load raw data
    processor = DataProcessor()
    print("Loading raw data...")
    raw_data = processor.load_data('./attached_assets/COEBiddingResultsPrices_1749430265007.csv')
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Raw columns: {list(raw_data.columns)}")
    print(f"Raw sample:\n{raw_data.head()}")
    
    # Process data
    print("\nProcessing data...")
    data = processor.preprocess_data(raw_data)
    print(f"Processed data shape: {data.shape}")
    print(f"Processed columns: {list(data.columns)}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Check Category A data
    print("\n=== CATEGORY A DATA ===")
    cat_a_data = data[data['vehicle_class'] == 'Category A'].copy()
    print(f"Category A records: {len(cat_a_data)}")
    if len(cat_a_data) > 0:
        cat_a_data = cat_a_data.sort_values('date')
        print(f"Date range: {cat_a_data['date'].min()} to {cat_a_data['date'].max()}")
        print(f"Premium range: ${cat_a_data['premium'].min()} to ${cat_a_data['premium'].max()}")
        print(f"Sample data:\n{cat_a_data[['date', 'premium']].head(10)}")
    
    return data, cat_a_data

def debug_model_training():
    """Debug model training"""
    print("\n=== MODEL TRAINING DEBUG ===")
    
    data, cat_a_data = debug_data_processing()
    
    if len(cat_a_data) < 20:
        print(f"ERROR: Insufficient Category A data ({len(cat_a_data)} records, need 20+)")
        return None
    
    # Test model training
    model = EnhancedDirectionalForecaster()
    prices = cat_a_data['premium'].values
    
    print(f"Training on {len(prices)} price points...")
    print(f"Price sample: {prices[:10]}")
    
    # Test individual components
    print("\n--- Testing Feature Engineering ---")
    features = model.calculate_advanced_features(prices[:10])
    print(f"Features: {features}")
    
    print("\n--- Testing Direction Training Data ---")
    X, y = model.create_directional_training_data(prices)
    if X is not None:
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Direction labels: {np.bincount(y)}")
    else:
        print("ERROR: Could not create training data")
        return None
    
    # Test model fitting
    print("\n--- Testing Model Fitting ---")
    model.fit(data)
    
    # Check if model was trained
    if 'Category A' in model.models:
        print("✓ Model fitted successfully")
        print(f"Model data shape: {len(model.models['Category A']['prices'])}")
        
        # Test validation
        print("\n--- Testing Validation ---")
        model.calculate_robust_validation_metrics(prices, 'Category A')
        metrics = model.get_performance_metrics('Category A')
        print(f"Validation metrics: {metrics}")
        
        return model
    else:
        print("ERROR: Model fitting failed")
        return None

def debug_fast_validation():
    """Debug fast validation"""
    print("\n=== FAST VALIDATION DEBUG ===")
    
    data, _ = debug_data_processing()
    model = debug_model_training()
    
    if model is None:
        print("ERROR: Cannot test validation - model training failed")
        return
    
    # Test fast validation
    validator = FastModelValidation()
    print("Running fast validation...")
    
    try:
        result = validator.run_fast_validation(EnhancedDirectionalForecaster, data, 'Category A')
        print(f"Validation result: {result}")
        
        if result:
            print("✓ Fast validation completed successfully")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print("ERROR: Fast validation returned empty result")
            
    except Exception as e:
        print(f"ERROR: Fast validation failed - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fast_validation()