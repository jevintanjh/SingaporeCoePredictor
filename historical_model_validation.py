"""
Historical Model Validation System
Evaluates model performance against actual historical COE data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_ranker import COEModelRanker
from models.fast_directional_forecaster import FastDirectionalForecaster
from models.interpretable_nbeats_v2 import InterpretableNBEATS
from models.nbeatsx import NBEATSx

def load_historical_data():
    """Load historical COE data"""
    data_paths = [
        'data/COE_Clean_2002_2025.csv',
        'attached_assets/COEBiddingResultsPrices_1749430265007.csv'
    ]
    
    for path in data_paths:
        try:
            df = pd.read_csv(path)
            print(f"Loaded data from {path}")
            break
        except Exception as e:
            print(f"Could not load {path}: {str(e)}")
            continue
    else:
        raise Exception("Could not load any data file")
    
    # Clean and standardize data
    if 'vehicle_class' not in df.columns and 'category' in df.columns:
        df['vehicle_class'] = df['category']
    
    # Convert dates
    if 'date' not in df.columns and 'month' in df.columns:
        df['date'] = pd.to_datetime(df['month'] + '-01')
        if 'bidding_no' in df.columns:
            df['date'] = df['date'] + pd.to_timedelta((df['bidding_no'] - 1) * 15, unit='D')
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    # Ensure premium column exists
    if 'premium' not in df.columns and 'price' in df.columns:
        df['premium'] = df['price']
    
    # Filter for recent data (last 2 years for validation)
    cutoff_date = datetime.now() - timedelta(days=730)
    df = df[df['date'] >= cutoff_date].copy()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Prepared {len(df)} records for validation from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    return df

def create_validation_splits(data, validation_periods=12):
    """Create validation splits for time series evaluation"""
    # Get unique dates and sort them
    unique_dates = sorted(data['date'].unique())
    
    # Take the last N periods for validation
    validation_dates = unique_dates[-validation_periods:]
    
    validation_splits = []
    
    for i, val_date in enumerate(validation_dates):
        # Training data: everything before validation date
        train_data = data[data['date'] < val_date].copy()
        
        # Validation data: the specific date
        val_data = data[data['date'] == val_date].copy()
        
        if len(train_data) > 50 and len(val_data) > 0:  # Ensure sufficient training data
            validation_splits.append({
                'split_id': i + 1,
                'train_data': train_data,
                'val_data': val_data,
                'val_date': val_date
            })
    
    print(f"Created {len(validation_splits)} validation splits")
    return validation_splits

def evaluate_model_on_split(model_class, model_name, split):
    """Evaluate a single model on a validation split"""
    try:
        print(f"Evaluating {model_name} on split {split['split_id']} (date: {split['val_date'].strftime('%Y-%m-%d')})")
        
        # Initialize and train model
        model = model_class()
        model.fit(split['train_data'])
        
        # Generate predictions
        try:
            predictions = model.predict(steps=1)
        except Exception as pred_error:
            print(f"Prediction error for {model_name}: {str(pred_error)}")
            # Try alternative prediction approach
            try:
                predictions = model.predict(1)
            except:
                return []
        
        # Extract actual results from validation data
        actual_results = {}
        for _, row in split['val_data'].iterrows():
            category = row['vehicle_class']
            actual_price = row['premium']
            actual_results[category] = actual_price
        
        # Extract predicted results
        predicted_results = {}
        for category in actual_results.keys():
            if category in predictions:
                # Take first prediction (1-step ahead)
                if isinstance(predictions[category], list):
                    predicted_results[category] = predictions[category][0]
                else:
                    predicted_results[category] = predictions[category]
        
        # Calculate performance metrics
        evaluations = []
        for category in actual_results.keys():
            if category in predicted_results:
                actual_price = actual_results[category]
                predicted_price = predicted_results[category]
                
                # Price accuracy
                price_error = abs(actual_price - predicted_price)
                price_accuracy = max(0, 1 - (price_error / actual_price))
                
                # Combined score (simplified - no directional for single point)
                combined_score = price_accuracy
                
                evaluations.append({
                    'model_name': model_name,
                    'category': category,
                    'split_id': split['split_id'],
                    'val_date': split['val_date'],
                    'actual_price': actual_price,
                    'predicted_price': predicted_price,
                    'price_error': price_error,
                    'price_accuracy': price_accuracy,
                    'combined_score': combined_score
                })
        
        return evaluations
        
    except Exception as e:
        print(f"Error evaluating {model_name} on split {split['split_id']}: {str(e)}")
        return []

def run_historical_validation():
    """Run complete historical validation"""
    print("Starting historical model validation...")
    
    # Load data
    data = load_historical_data()
    
    # Create validation splits
    validation_splits = create_validation_splits(data, validation_periods=8)
    
    if not validation_splits:
        print("No validation splits available")
        return
    
    # Initialize models
    models = {
        'Fast Directional Forecaster': FastDirectionalForecaster,
        'Interpretable N-BEATS': InterpretableNBEATS,
        'N-BEATSx': NBEATSx
    }
    
    # Collect all evaluations
    all_evaluations = []
    
    # Evaluate each model on each split
    for model_name, model_class in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        model_evaluations = []
        
        for split in validation_splits:
            split_evaluations = evaluate_model_on_split(model_class, model_name, split)
            model_evaluations.extend(split_evaluations)
            all_evaluations.extend(split_evaluations)
        
        # Calculate model summary
        if model_evaluations:
            df_eval = pd.DataFrame(model_evaluations)
            overall_score = df_eval['combined_score'].mean()
            print(f"{model_name} Overall Score: {overall_score:.3f}")
            print(f"Evaluations: {len(model_evaluations)}")
            
            # Category-wise performance
            for category in df_eval['category'].unique():
                cat_data = df_eval[df_eval['category'] == category]
                cat_score = cat_data['combined_score'].mean()
                print(f"  {category}: {cat_score:.3f} ({len(cat_data)} evaluations)")
    
    # Generate comprehensive results
    if all_evaluations:
        print(f"\n{'='*60}")
        print("HISTORICAL VALIDATION RESULTS")
        print(f"{'='*60}")
        
        df_all = pd.DataFrame(all_evaluations)
        
        # Overall model rankings
        model_scores = df_all.groupby('model_name')['combined_score'].agg(['mean', 'count', 'std']).round(3)
        model_scores = model_scores.sort_values('mean', ascending=False)
        model_scores['rank'] = range(1, len(model_scores) + 1)
        
        print("\nModel Rankings (Historical Performance):")
        print(model_scores)
        
        # Update model ranker with historical results
        print(f"\nUpdating model ranker with {len(all_evaluations)} historical evaluations...")
        ranker = COEModelRanker()
        
        # Group evaluations by validation date and update ranker
        for val_date in df_all['val_date'].unique():
            date_evaluations = df_all[df_all['val_date'] == val_date]
            
            # Convert to format expected by model ranker
            actual_results = {}
            predictions_by_model = {model: {} for model in models.keys()}
            
            for _, row in date_evaluations.iterrows():
                category = row['category']
                actual_results[category] = row['actual_price']
                predictions_by_model[row['model_name']][category] = row['predicted_price']
            
            # Log predictions
            prediction_date = val_date - timedelta(days=1)
            exercise_date = val_date.strftime('%Y-%m-%d')
            
            for model_name, predictions in predictions_by_model.items():
                for category, predicted_price in predictions.items():
                    ranker.log_prediction(
                        model_name=model_name,
                        category=category,
                        predicted_price=predicted_price,
                        prediction_date=prediction_date,
                        exercise_date=exercise_date
                    )
            
            # Evaluate predictions
            ranker.evaluate_predictions(actual_results)
        
        print("Historical validation complete!")
        print("Model rankings have been updated with historical performance data.")
        
        # Display final rankings
        rankings = ranker.get_current_rankings()
        if rankings:
            print(f"\nFinal Model Rankings:")
            for rank_data in rankings:
                print(f"{rank_data['rank']}. {rank_data['model_name']} - Score: {rank_data['average_score']:.3f} ({rank_data['total_evaluations']} evaluations)")
        
        return True
    
    else:
        print("No evaluations completed")
        return False

if __name__ == "__main__":
    run_historical_validation()