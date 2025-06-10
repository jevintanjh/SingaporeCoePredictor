"""
Update model rankings with comprehensive historical validation
This script runs historical validation and updates the ranking system
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_ranker import COEModelRanker
from models.fast_directional_forecaster import FastDirectionalForecaster
from models.interpretable_nbeats_v2 import InterpretableNBEATS
from models.nbeatsx import NBEATSx

def load_and_prepare_data():
    """Load and prepare historical COE data for validation"""
    try:
        # Load the clean historical data
        df = pd.read_csv('data/COE_Clean_2002_2025.csv')
        
        # Create date column from month and bidding_no
        df['date'] = pd.to_datetime(df['month'] + '-01')
        # Adjust date based on bidding number (1st or 2nd bidding of the month)
        df['date'] = df['date'] + pd.to_timedelta((df['bidding_no'] - 1) * 15, unit='D')
        
        # Filter for recent data (last 18 months for comprehensive validation)
        cutoff_date = datetime.now() - timedelta(days=545)
        df = df[df['date'] >= cutoff_date].copy()
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} historical records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_time_series_splits(data, min_training_days=180):
    """Create time series validation splits"""
    unique_dates = sorted(data['date'].unique())
    
    # Create monthly validation points
    validation_dates = []
    start_date = unique_dates[0] + timedelta(days=min_training_days)
    
    current_date = start_date
    while current_date <= unique_dates[-1]:
        # Find closest actual date
        closest_date = min(unique_dates, key=lambda x: abs((x - current_date).days))
        if closest_date not in validation_dates:
            validation_dates.append(closest_date)
        current_date += timedelta(days=30)  # Monthly intervals
    
    splits = []
    for val_date in validation_dates:
        train_data = data[data['date'] < val_date].copy()
        val_data = data[data['date'] == val_date].copy()
        
        if len(train_data) >= 50 and len(val_data) > 0:
            splits.append({
                'train_data': train_data,
                'val_data': val_data,
                'val_date': val_date
            })
    
    return splits

def evaluate_model_comprehensively(model_class, model_name, splits):
    """Evaluate model across all splits with comprehensive metrics"""
    print(f"\nEvaluating {model_name} across {len(splits)} time periods...")
    
    all_results = []
    successful_predictions = 0
    
    for i, split in enumerate(splits):
        try:
            # Initialize and train model
            model = model_class()
            model.fit(split['train_data'])
            
            # Generate predictions
            predictions = model.predict(steps=1)
            
            # Process validation data
            val_results = {}
            for _, row in split['val_data'].iterrows():
                category = row['vehicle_class']
                actual_price = row['premium']
                val_results[category] = actual_price
            
            # Extract predictions
            pred_results = {}
            for category in val_results.keys():
                if category in predictions:
                    if isinstance(predictions[category], list):
                        pred_results[category] = predictions[category][0]
                    else:
                        pred_results[category] = predictions[category]
            
            # Calculate metrics for each category
            for category in val_results.keys():
                if category in pred_results:
                    actual = val_results[category]
                    predicted = pred_results[category]
                    
                    # Price accuracy
                    error = abs(actual - predicted)
                    accuracy = max(0, 1 - (error / actual))
                    
                    # Store result
                    all_results.append({
                        'model': model_name,
                        'date': split['val_date'],
                        'category': category,
                        'actual': actual,
                        'predicted': predicted,
                        'error': error,
                        'accuracy': accuracy
                    })
                    
                    successful_predictions += 1
        
        except Exception as e:
            print(f"  Error on split {i+1}: {str(e)}")
            continue
    
    # Calculate summary statistics
    if all_results:
        results_df = pd.DataFrame(all_results)
        overall_accuracy = results_df['accuracy'].mean()
        
        print(f"  {model_name} Results:")
        print(f"    Overall Accuracy: {overall_accuracy:.3f}")
        print(f"    Successful Predictions: {successful_predictions}")
        print(f"    Evaluation Periods: {len(splits)}")
        
        # Category-wise performance
        for category in results_df['category'].unique():
            cat_acc = results_df[results_df['category'] == category]['accuracy'].mean()
            cat_count = len(results_df[results_df['category'] == category])
            print(f"    {category}: {cat_acc:.3f} ({cat_count} predictions)")
    
    return all_results

def update_model_rankings_with_historical_data():
    """Main function to update rankings with historical validation"""
    print("Starting comprehensive historical model validation...")
    
    # Load data
    data = load_and_prepare_data()
    if data is None:
        return False
    
    # Create validation splits
    splits = create_time_series_splits(data)
    if not splits:
        print("No validation splits created")
        return False
    
    print(f"Created {len(splits)} validation splits for comprehensive evaluation")
    
    # Initialize models
    models = {
        'Fast Directional Forecaster': FastDirectionalForecaster,
        'Interpretable N-BEATS': InterpretableNBEATS,
        'N-BEATSx': NBEATSx
    }
    
    # Evaluate all models
    all_model_results = {}
    
    for model_name, model_class in models.items():
        model_results = evaluate_model_comprehensively(model_class, model_name, splits)
        all_model_results[model_name] = model_results
    
    # Update model ranker with historical results
    print(f"\nUpdating model ranker with historical validation results...")
    ranker = COEModelRanker()
    
    # Process results and update ranker
    total_evaluations = 0
    
    for model_name, results in all_model_results.items():
        if not results:
            continue
            
        # Group by date for proper evaluation
        results_df = pd.DataFrame(results)
        
        for date in results_df['date'].unique():
            date_results = results_df[results_df['date'] == date]
            
            # Prepare actual results
            actual_results = {}
            predicted_results = {}
            
            for _, row in date_results.iterrows():
                category = row['category']
                actual_results[category] = row['actual']
                predicted_results[category] = row['predicted']
            
            # Log predictions
            prediction_date = date - timedelta(days=1)
            exercise_date = date.strftime('%Y-%m-%d')
            
            for category, pred_price in predicted_results.items():
                ranker.log_prediction(
                    model_name=model_name,
                    category=category,
                    predicted_price=pred_price,
                    prediction_date=prediction_date,
                    exercise_date=exercise_date
                )
            
            total_evaluations += 1
        
        # Evaluate all predictions for this model
        for date in results_df['date'].unique():
            date_results = results_df[results_df['date'] == date]
            actual_results = {row['category']: row['actual'] for _, row in date_results.iterrows()}
            ranker.evaluate_predictions(actual_results)
    
    # Display final rankings
    print(f"\n{'='*60}")
    print("HISTORICAL VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Evaluations: {total_evaluations}")
    
    rankings = ranker.get_current_rankings()
    if rankings:
        print(f"\nFinal Model Rankings (Based on Historical Performance):")
        for rank_data in rankings:
            print(f"{rank_data['rank']}. {rank_data['model_name']}")
            print(f"   Score: {rank_data['average_score']:.3f}")
            print(f"   Evaluations: {rank_data['total_evaluations']}")
            print(f"   Performance: {rank_data.get('performance_rating', 'N/A')}")
            print()
    
    print("Model rankings updated with comprehensive historical validation data.")
    print("The system is now ready to automatically update rankings when new COE results arrive.")
    
    return True

if __name__ == "__main__":
    success = update_model_rankings_with_historical_data()
    if success:
        print("\n✓ Historical validation and ranking update completed successfully")
    else:
        print("\n✗ Historical validation failed")