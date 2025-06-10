"""
Recent Cycles Model Validation
Evaluates model performance against the last 6 COE cycles for current accuracy assessment
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

def load_recent_coe_data():
    """Load the most recent 6 cycles of COE data"""
    try:
        # Load the clean historical data
        df = pd.read_csv('data/COE_Clean_2002_2025.csv')
        
        # Create date column from month and bidding_no
        df['date'] = pd.to_datetime(df['month'] + '-01')
        df['date'] = df['date'] + pd.to_timedelta((df['bidding_no'] - 1) * 15, unit='D')
        
        # Sort by date to get most recent
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        # Get unique exercise dates and take the last 8 (to use 6 for validation, 2 for context)
        unique_dates = sorted(df['date'].unique(), reverse=True)
        recent_dates = unique_dates[:8]
        
        # Filter for recent cycles
        recent_data = df[df['date'].isin(recent_dates)].copy()
        recent_data = recent_data.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded last 6 COE cycles from {recent_data['date'].min().strftime('%Y-%m-%d')} to {recent_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total records: {len(recent_data)}")
        
        return recent_data
        
    except Exception as e:
        print(f"Error loading recent data: {str(e)}")
        return None

def create_recent_validation_splits(data):
    """Create validation splits for the last 6 cycles"""
    unique_dates = sorted(data['date'].unique())
    
    if len(unique_dates) < 6:
        print(f"Only {len(unique_dates)} cycles available, need at least 6")
        return []
    
    # Take the last 6 dates for validation
    validation_dates = unique_dates[-6:]
    
    splits = []
    
    for i, val_date in enumerate(validation_dates):
        # Use all data before validation date as training
        train_data = data[data['date'] < val_date].copy()
        val_data = data[data['date'] == val_date].copy()
        
        if len(train_data) >= 10 and len(val_data) > 0:  # Minimum training data
            splits.append({
                'cycle': i + 1,
                'train_data': train_data,
                'val_data': val_data,
                'val_date': val_date,
                'exercise_name': f"Cycle {len(validation_dates) - i} ({val_date.strftime('%Y-%m-%d')})"
            })
    
    print(f"Created {len(splits)} validation splits for recent cycles")
    return splits

def evaluate_model_on_recent_cycle(model_class, model_name, split):
    """Evaluate a single model on a recent cycle"""
    try:
        print(f"  Evaluating {split['exercise_name']}...")
        
        # Initialize and train model
        model = model_class()
        model.fit(split['train_data'])
        
        # Generate predictions
        try:
            predictions = model.predict(steps=1)
        except Exception as pred_error:
            print(f"    Prediction error: {str(pred_error)}")
            try:
                predictions = model.predict(1)
            except:
                return None
        
        # Extract actual results
        actual_results = {}
        for _, row in split['val_data'].iterrows():
            category = row['vehicle_class']
            actual_price = row['premium']
            actual_results[category] = actual_price
        
        # Extract predicted results
        predicted_results = {}
        for category in actual_results.keys():
            if category in predictions:
                if isinstance(predictions[category], list):
                    predicted_results[category] = predictions[category][0]
                else:
                    predicted_results[category] = predictions[category]
        
        # Calculate performance metrics for each category
        cycle_results = []
        total_score = 0
        valid_predictions = 0
        
        for category in actual_results.keys():
            if category in predicted_results:
                actual_price = actual_results[category]
                predicted_price = predicted_results[category]
                
                # Price accuracy
                price_error = abs(actual_price - predicted_price)
                price_accuracy = max(0, 1 - (price_error / actual_price))
                
                # Directional accuracy (if we have previous price for comparison)
                directional_accuracy = 0.5  # Neutral if no previous data
                
                # Try to get previous cycle price for directional comparison
                previous_data = split['train_data']
                if len(previous_data) > 0:
                    category_history = previous_data[previous_data['vehicle_class'] == category]
                    if len(category_history) > 0:
                        previous_price = category_history.iloc[-1]['premium']
                        actual_direction = 1 if actual_price > previous_price else 0
                        predicted_direction = 1 if predicted_price > previous_price else 0
                        directional_accuracy = 1.0 if actual_direction == predicted_direction else 0.0
                
                # Combined score (70% price accuracy + 30% directional accuracy)
                combined_score = (0.7 * price_accuracy) + (0.3 * directional_accuracy)
                
                cycle_results.append({
                    'category': category,
                    'actual_price': actual_price,
                    'predicted_price': predicted_price,
                    'price_error': price_error,
                    'price_accuracy': price_accuracy,
                    'directional_accuracy': directional_accuracy,
                    'combined_score': combined_score
                })
                
                total_score += combined_score
                valid_predictions += 1
        
        if valid_predictions > 0:
            average_score = total_score / valid_predictions
            print(f"    Average Score: {average_score:.3f} ({valid_predictions} categories)")
            
            return {
                'model_name': model_name,
                'cycle': split['cycle'],
                'exercise_name': split['exercise_name'],
                'val_date': split['val_date'],
                'average_score': average_score,
                'valid_predictions': valid_predictions,
                'category_results': cycle_results,
                'actual_results': actual_results,
                'predicted_results': predicted_results
            }
        
        return None
        
    except Exception as e:
        print(f"    Error evaluating {model_name} on {split['exercise_name']}: {str(e)}")
        return None

def run_recent_cycles_validation():
    """Run validation on the last 6 COE cycles"""
    print("Starting recent cycles model validation (Last 6 COE Exercises)...")
    
    # Load recent data
    data = load_recent_coe_data()
    if data is None:
        return False
    
    # Create validation splits
    splits = create_recent_validation_splits(data)
    if not splits:
        return False
    
    # Initialize models
    models = {
        'Fast Directional Forecaster': FastDirectionalForecaster,
        'Interpretable N-BEATS': InterpretableNBEATS,
        'N-BEATSx': NBEATSx
    }
    
    # Store all results
    all_results = []
    model_summaries = {}
    
    # Evaluate each model
    for model_name, model_class in models.items():
        print(f"\nEvaluating {model_name} on last 6 cycles:")
        
        model_results = []
        total_score = 0
        successful_cycles = 0
        
        for split in splits:
            result = evaluate_model_on_recent_cycle(model_class, model_name, split)
            if result:
                model_results.append(result)
                all_results.append(result)
                total_score += result['average_score']
                successful_cycles += 1
        
        if successful_cycles > 0:
            overall_score = total_score / successful_cycles
            model_summaries[model_name] = {
                'overall_score': overall_score,
                'successful_cycles': successful_cycles,
                'total_cycles': len(splits),
                'results': model_results
            }
            print(f"  Overall Recent Performance: {overall_score:.3f} ({successful_cycles}/{len(splits)} cycles)")
        else:
            print(f"  No successful evaluations for {model_name}")
    
    # Update model ranker with recent results
    print(f"\nUpdating model ranker with recent cycles validation...")
    ranker = COEModelRanker()
    
    # Clear existing demo data and add recent validation results
    for result in all_results:
        # Log predictions
        prediction_date = result['val_date'] - timedelta(days=1)
        exercise_date = result['val_date'].strftime('%Y-%m-%d')
        
        for category, predicted_price in result['predicted_results'].items():
            ranker.log_prediction(
                model_name=result['model_name'],
                category=category,
                predicted_price=predicted_price,
                prediction_date=prediction_date,
                exercise_date=exercise_date
            )
        
        # Evaluate predictions
        ranker.evaluate_predictions(result['actual_results'])
    
    # Display results summary
    print(f"\n{'='*60}")
    print("RECENT CYCLES VALIDATION RESULTS (LAST 6 COE EXERCISES)")
    print(f"{'='*60}")
    
    # Sort models by recent performance
    sorted_models = sorted(model_summaries.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    print(f"\nModel Rankings (Based on Last 6 Cycles Performance):")
    for rank, (model_name, summary) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}")
        print(f"   Recent Score: {summary['overall_score']:.3f}")
        print(f"   Successful Predictions: {summary['successful_cycles']}/{summary['total_cycles']} cycles")
        print(f"   Performance Rating: {'Excellent' if summary['overall_score'] >= 0.95 else 'Good' if summary['overall_score'] >= 0.90 else 'Fair' if summary['overall_score'] >= 0.80 else 'Poor'}")
        print()
    
    # Display cycle-by-cycle performance
    print("Cycle-by-Cycle Performance Summary:")
    for split in splits:
        print(f"\n{split['exercise_name']}:")
        cycle_results = [r for r in all_results if r['val_date'] == split['val_date']]
        for result in sorted(cycle_results, key=lambda x: x['average_score'], reverse=True):
            print(f"  {result['model_name']}: {result['average_score']:.3f}")
    
    # Get updated rankings from model ranker
    final_rankings = ranker.get_current_rankings()
    if final_rankings:
        print(f"\nFinal Updated Model Rankings:")
        for rank_data in final_rankings:
            print(f"{rank_data['rank']}. {rank_data['model_name']} - Score: {rank_data['average_score']:.3f} ({rank_data['total_evaluations']} evaluations)")
    
    print("\nModel rankings updated with recent cycles performance data.")
    print("Rankings now reflect current model accuracy based on the most recent 6 COE exercises.")
    
    return True

if __name__ == "__main__":
    success = run_recent_cycles_validation()
    if success:
        print("\n✓ Recent cycles validation completed successfully")
    else:
        print("\n✗ Recent cycles validation failed")