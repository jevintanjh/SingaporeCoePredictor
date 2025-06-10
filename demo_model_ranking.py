"""
Demonstration script for the model ranking system
Creates sample predictions and evaluations to show how the ranking system works
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    
    # Create sample evaluation scenarios for 6 cycles
    evaluation_rounds = [
        {
            'date': '2025-04-17',
            'actual_results': {
                'Category A': 92000,
                'Category B': 102000,
                'Category C': 86000,
                'Category D': 42000,
                'Category E': 89000
            },
            'predictions': {
                'Fast Directional Forecaster': {
                    'Category A': 91500,
                    'Category B': 103500,
                    'Category C': 85800,
                    'Category D': 42500,
                    'Category E': 88500
                },
                'Interpretable N-BEATS': {
                    'Category A': 90000,
                    'Category B': 101000,
                    'Category C': 87500,
                    'Category D': 41800,
                    'Category E': 90000
                },
                'N-BEATSx': {
                    'Category A': 94000,
                    'Category B': 105000,
                    'Category C': 84000,
                    'Category D': 43500,
                    'Category E': 87500
                }
            }
        },
        {
            'date': '2025-05-01',
            'actual_results': {
                'Category A': 94000,
                'Category B': 104000,
                'Category C': 87500,
                'Category D': 44000,
                'Category E': 91000
            },
            'predictions': {
                'Fast Directional Forecaster': {
                    'Category A': 93800,
                    'Category B': 104200,
                    'Category C': 87200,
                    'Category D': 44300,
                    'Category E': 90800
                },
                'Interpretable N-BEATS': {
                    'Category A': 92500,
                    'Category B': 102500,
                    'Category C': 89000,
                    'Category D': 43500,
                    'Category E': 92000
                },
                'N-BEATSx': {
                    'Category A': 95500,
                    'Category B': 106000,
                    'Category C': 86000,
                    'Category D': 45000,
                    'Category E': 89500
                }
            }
        },
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
                    'Category A': 94500,
                    'Category B': 105500,
                    'Category C': 87800,
                    'Category D': 45200,
                    'Category E': 91700
                },
                'Interpretable N-BEATS': {
                    'Category A': 96000,
                    'Category B': 103000,
                    'Category C': 89500,
                    'Category D': 44200,
                    'Category E': 93000
                },
                'N-BEATSx': {
                    'Category A': 97000,
                    'Category B': 106500,
                    'Category C': 86500,
                    'Category D': 46000,
                    'Category E': 90000
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
                    'Category A': 96800,
                    'Category B': 110300,
                    'Category C': 84800,
                    'Category D': 48200,
                    'Category E': 94700
                },
                'Interpretable N-BEATS': {
                    'Category A': 95500,
                    'Category B': 108000,
                    'Category C': 86500,
                    'Category D': 47200,
                    'Category E': 96000
                },
                'N-BEATSx': {
                    'Category A': 98500,
                    'Category B': 111500,
                    'Category C': 83500,
                    'Category D': 49000,
                    'Category E': 93500
                }
            }
        },
        {
            'date': '2025-06-05',
            'actual_results': {
                'Category A': 99000,
                'Category B': 112000,
                'Category C': 82000,
                'Category D': 49500,
                'Category E': 97000
            },
            'predictions': {
                'Fast Directional Forecaster': {
                    'Category A': 98700,
                    'Category B': 111800,
                    'Category C': 82300,
                    'Category D': 49300,
                    'Category E': 96800
                },
                'Interpretable N-BEATS': {
                    'Category A': 97000,
                    'Category B': 110500,
                    'Category C': 83500,
                    'Category D': 48800,
                    'Category E': 98000
                },
                'N-BEATSx': {
                    'Category A': 100500,
                    'Category B': 113500,
                    'Category C': 80500,
                    'Category D': 50500,
                    'Category E': 95500
                }
            }
        },
        {
            'date': '2025-06-10',
            'actual_results': {
                'Category A': 101000,
                'Category B': 115000,
                'Category C': 80000,
                'Category D': 51000,
                'Category E': 99000
            },
            'predictions': {
                'Fast Directional Forecaster': {
                    'Category A': 100800,
                    'Category B': 114700,
                    'Category C': 80200,
                    'Category D': 50800,
                    'Category E': 98700
                },
                'Interpretable N-BEATS': {
                    'Category A': 99500,
                    'Category B': 113000,
                    'Category C': 81500,
                    'Category D': 50200,
                    'Category E': 100000
                },
                'N-BEATSx': {
                    'Category A': 102500,
                    'Category B': 116500,
                    'Category C': 78500,
                    'Category D': 52000,
                    'Category E': 97500
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