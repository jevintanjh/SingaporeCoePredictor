"""
Demonstration script for the model ranking system
Creates sample predictions and evaluations to show how the ranking system works
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.model_ranker import COEModelRanker

def create_demo_rankings():
    """
    Create demo model rankings to demonstrate the system
    """
    print("Creating demonstration of model ranking system...")
    
    # Initialize model ranker
    ranker = COEModelRanker()
    
    # Simulate some historical predictions and their evaluations
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    models = ['Fast Directional Forecaster', 'Interpretable N-BEATS', 'N-BEATSx']
    
    # Create sample evaluation scenarios
    evaluation_rounds = [
        {
            'date': '2025-05-15',
            'actual_results': {
                'Category A': 95000,
                'Category B': 105000,
                'Category C': 88000,
                'Category D': 45000,
                'Category E': 92000
            },
            'predictions': {
                'Fast Directional Forecaster': {
                    'Category A': 94500,  # Close prediction
                    'Category B': 108000,  # Slightly off
                    'Category C': 87500,   # Very close
                    'Category D': 46000,   # Good
                    'Category E': 89000    # Reasonable
                },
                'Interpretable N-BEATS': {
                    'Category A': 96000,   # Good
                    'Category B': 102000,  # Very close
                    'Category C': 90000,   # Off by more
                    'Category D': 44500,   # Close
                    'Category E': 93500    # Close
                },
                'N-BEATSx': {
                    'Category A': 98000,   # Further off
                    'Category B': 106000,  # Good
                    'Category C': 86000,   # Reasonable
                    'Category D': 47000,   # Off
                    'Category E': 94000    # Good
                }
            }
        },
        {
            'date': '2025-05-29',
            'actual_results': {
                'Category A': 97000,
                'Category B': 110000,
                'Category C': 85000,
                'Category D': 48000,
                'Category E': 95000
            },
            'predictions': {
                'Fast Directional Forecaster': {
                    'Category A': 96800,   # Excellent
                    'Category B': 112000,  # Good
                    'Category C': 84500,   # Excellent
                    'Category D': 48500,   # Good
                    'Category E': 94200    # Very good
                },
                'Interpretable N-BEATS': {
                    'Category A': 95000,   # Off
                    'Category B': 108000,  # Reasonable
                    'Category C': 87000,   # Off
                    'Category D': 47500,   # Good
                    'Category E': 96000    # Good
                },
                'N-BEATSx': {
                    'Category A': 99000,   # Off
                    'Category B': 111000,  # Very good
                    'Category C': 83000,   # Reasonable
                    'Category D': 49000,   # Good
                    'Category E': 93000    # Reasonable
                }
            }
        }
    ]
    
    # Process each evaluation round
    for round_data in evaluation_rounds:
        print(f"\nProcessing evaluation round for {round_data['date']}...")
        
        # First, log predictions (simulating they were made earlier)
        prediction_date = datetime.strptime(round_data['date'], '%Y-%m-%d') - timedelta(days=2)
        exercise_date = round_data['date']
        
        for model_name, predictions in round_data['predictions'].items():
            for category, predicted_price in predictions.items():
                ranker.log_prediction(
                    model_name=model_name,
                    category=category,
                    predicted_price=predicted_price,
                    prediction_date=prediction_date,
                    exercise_date=exercise_date
                )
        
        # Then evaluate against actual results
        ranker.evaluate_predictions(round_data['actual_results'])
    
    # Display final rankings
    print("\n" + "="*60)
    print("FINAL MODEL RANKINGS")
    print("="*60)
    
    rankings = ranker.get_current_rankings()
    
    if rankings:
        for rank_data in rankings:
            print(f"{rank_data['rank']}. {rank_data['model_name']}")
            print(f"   Overall Score: {rank_data['average_score']:.3f}")
            print(f"   Total Evaluations: {rank_data['total_evaluations']}")
            print(f"   Last Updated: {rank_data['last_updated']}")
            print()
    
    # Show performance summary
    performance_summary = ranker.get_model_performance_summary()
    if performance_summary and 'models' in performance_summary:
        print("PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        for model, metrics in performance_summary['models'].items():
            print(f"\n{model}:")
            print(f"  Overall Score: {metrics['overall_score']:.3f}")
            print(f"  Recent Score: {metrics['recent_score']:.3f}")
            print(f"  Total Evaluations: {metrics['total_evaluations']}")
            print(f"  Trend: {metrics['trend']}")
    
    print(f"\nDemo rankings created successfully!")
    print("The ranking system is now ready and will automatically update when actual COE results are available.")
    
    return True

if __name__ == "__main__":
    create_demo_rankings()