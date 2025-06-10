import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class COEModelRanker:
    """
    Ranks COE prediction models based on accuracy and directional performance
    """
    
    def __init__(self):
        self.rankings_file = 'data/model_rankings.json'
        self.predictions_log = 'data/predictions_log.json'
        self.model_names = [
            'Fast Directional Forecaster',
            'Interpretable N-BEATS',
            'N-BEATSx'
        ]
        
    def log_prediction(self, model_name, category, predicted_price, prediction_date, exercise_date):
        """Log a prediction for later evaluation"""
        try:
            # Load existing predictions log
            predictions = self.load_predictions_log()
            
            prediction_id = f"{model_name}_{category}_{exercise_date}"
            
            predictions[prediction_id] = {
                'model_name': model_name,
                'category': category,
                'predicted_price': float(predicted_price),
                'prediction_date': prediction_date.isoformat(),
                'exercise_date': exercise_date,
                'actual_price': None,
                'evaluated': False
            }
            
            # Save updated predictions
            self.save_predictions_log(predictions)
            logger.info(f"Logged prediction: {model_name} - {category} - ${predicted_price:,.0f}")
            
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
    
    def evaluate_predictions(self, actual_results):
        """
        Evaluate predictions against actual results and update model rankings
        
        Args:
            actual_results: dict with format {category: actual_price}
        """
        try:
            predictions = self.load_predictions_log()
            current_rankings = self.load_rankings()
            
            # Find predictions that match the actual results
            evaluated_predictions = []
            
            for pred_id, prediction in predictions.items():
                if prediction['evaluated']:
                    continue
                    
                category = prediction['category']
                if category in actual_results:
                    actual_price = actual_results[category]
                    predicted_price = prediction['predicted_price']
                    
                    # Calculate accuracy metrics
                    price_error = abs(actual_price - predicted_price)
                    price_accuracy = max(0, 1 - (price_error / actual_price))
                    
                    # Calculate directional accuracy
                    # Get previous price for direction calculation
                    prev_price = self.get_previous_price(category, prediction['exercise_date'])
                    
                    if prev_price is not None:
                        actual_direction = 1 if actual_price > prev_price else -1 if actual_price < prev_price else 0
                        predicted_direction = 1 if predicted_price > prev_price else -1 if predicted_price < prev_price else 0
                        direction_accuracy = 1.0 if actual_direction == predicted_direction else 0.0
                    else:
                        direction_accuracy = 0.5  # Neutral if no previous price available
                    
                    # Combined score (70% price accuracy, 30% direction accuracy)
                    combined_score = (0.7 * price_accuracy) + (0.3 * direction_accuracy)
                    
                    evaluation = {
                        'model_name': prediction['model_name'],
                        'category': category,
                        'predicted_price': predicted_price,
                        'actual_price': actual_price,
                        'price_error': price_error,
                        'price_accuracy': price_accuracy,
                        'direction_accuracy': direction_accuracy,
                        'combined_score': combined_score,
                        'exercise_date': prediction['exercise_date'],
                        'evaluation_date': datetime.now().isoformat()
                    }
                    
                    evaluated_predictions.append(evaluation)
                    
                    # Mark prediction as evaluated
                    predictions[pred_id]['actual_price'] = actual_price
                    predictions[pred_id]['evaluated'] = True
                    
                    logger.info(f"Evaluated {prediction['model_name']} - {category}: "
                              f"Predicted ${predicted_price:,.0f}, Actual ${actual_price:,.0f}, "
                              f"Score: {combined_score:.3f}")
            
            # Update model rankings if we have evaluations
            if evaluated_predictions:
                self.update_rankings(evaluated_predictions, current_rankings)
                self.save_predictions_log(predictions)
                
                logger.info(f"Updated model rankings based on {len(evaluated_predictions)} evaluations")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {str(e)}")
            return False
    
    def update_rankings(self, evaluations, current_rankings):
        """Update model rankings based on new evaluations"""
        try:
            # Group evaluations by model
            model_scores = {}
            
            for eval_data in evaluations:
                model = eval_data['model_name']
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(eval_data['combined_score'])
            
            # Calculate average scores for this evaluation round
            round_averages = {}
            for model, scores in model_scores.items():
                round_averages[model] = np.mean(scores)
            
            # Update cumulative rankings
            timestamp = datetime.now().isoformat()
            
            for model in self.model_names:
                if model not in current_rankings:
                    current_rankings[model] = {
                        'total_evaluations': 0,
                        'cumulative_score': 0.0,
                        'average_score': 0.0,
                        'rank': 0,
                        'last_updated': timestamp,
                        'evaluation_history': []
                    }
                
                if model in round_averages:
                    score = round_averages[model]
                    model_data = current_rankings[model]
                    
                    # Update cumulative statistics
                    model_data['total_evaluations'] += 1
                    model_data['cumulative_score'] += score
                    model_data['average_score'] = model_data['cumulative_score'] / model_data['total_evaluations']
                    model_data['last_updated'] = timestamp
                    
                    # Add to evaluation history
                    model_data['evaluation_history'].append({
                        'date': timestamp,
                        'score': score,
                        'evaluations_count': len(model_scores[model])
                    })
                    
                    # Keep only last 20 evaluations in history
                    if len(model_data['evaluation_history']) > 20:
                        model_data['evaluation_history'] = model_data['evaluation_history'][-20:]
            
            # Calculate ranks based on average scores
            models_with_scores = [(model, data['average_score']) 
                                for model, data in current_rankings.items() 
                                if data['total_evaluations'] > 0]
            
            models_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (model, score) in enumerate(models_with_scores, 1):
                current_rankings[model]['rank'] = rank
            
            # Save updated rankings
            self.save_rankings(current_rankings)
            
        except Exception as e:
            logger.error(f"Error updating rankings: {str(e)}")
    
    def get_previous_price(self, category, exercise_date):
        """Get the previous COE price for direction calculation"""
        try:
            # Load main dataset to find previous price
            data_paths = [
                'data/COE_Clean_2002_2025.csv',
                'attached_assets/COEBiddingResultsPrices_1749430265007.csv'
            ]
            
            for path in data_paths:
                try:
                    df = pd.read_csv(path)
                    break
                except:
                    continue
            else:
                return None
            
            # Convert exercise_date to datetime for comparison
            target_date = pd.to_datetime(exercise_date)
            
            # Filter for the category and dates before the exercise date
            if 'vehicle_class' in df.columns:
                category_data = df[df['vehicle_class'] == category].copy()
            else:
                return None
            
            # Convert date column
            if 'month' in df.columns and 'bidding_no' in df.columns:
                category_data['date'] = pd.to_datetime(category_data['month'] + '-01') + \
                                      pd.to_timedelta((category_data['bidding_no'] - 1) * 15, unit='D')
            elif 'date' in df.columns:
                category_data['date'] = pd.to_datetime(category_data['date'])
            else:
                return None
            
            # Find the most recent price before the exercise date
            previous_data = category_data[category_data['date'] < target_date]
            
            if len(previous_data) > 0:
                latest_record = previous_data.loc[previous_data['date'].idxmax()]
                return float(latest_record['premium'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting previous price: {str(e)}")
            return None
    
    def get_current_rankings(self):
        """Get current model rankings"""
        try:
            rankings = self.load_rankings()
            
            # Sort by rank
            ranked_models = []
            for model, data in rankings.items():
                if data['total_evaluations'] > 0:
                    ranked_models.append({
                        'model_name': model,
                        'rank': data['rank'],
                        'average_score': data['average_score'],
                        'total_evaluations': data['total_evaluations'],
                        'last_updated': data['last_updated']
                    })
            
            ranked_models.sort(key=lambda x: x['rank'])
            return ranked_models
            
        except Exception as e:
            logger.error(f"Error getting rankings: {str(e)}")
            return []
    
    def load_predictions_log(self):
        """Load predictions log from file"""
        try:
            if os.path.exists(self.predictions_log):
                with open(self.predictions_log, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading predictions log: {str(e)}")
            return {}
    
    def save_predictions_log(self, predictions):
        """Save predictions log to file"""
        try:
            os.makedirs(os.path.dirname(self.predictions_log), exist_ok=True)
            with open(self.predictions_log, 'w') as f:
                json.dump(predictions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving predictions log: {str(e)}")
    
    def load_rankings(self):
        """Load model rankings from file"""
        try:
            if os.path.exists(self.rankings_file):
                with open(self.rankings_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading rankings: {str(e)}")
            return {}
    
    def save_rankings(self, rankings):
        """Save model rankings to file"""
        try:
            os.makedirs(os.path.dirname(self.rankings_file), exist_ok=True)
            with open(self.rankings_file, 'w') as f:
                json.dump(rankings, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving rankings: {str(e)}")
    
    def get_model_performance_summary(self):
        """Get comprehensive performance summary for all models"""
        try:
            rankings = self.load_rankings()
            
            summary = {
                'last_updated': datetime.now().isoformat(),
                'models': {}
            }
            
            for model, data in rankings.items():
                if data['total_evaluations'] > 0:
                    # Calculate recent performance (last 5 evaluations)
                    recent_scores = [eval_data['score'] for eval_data in data['evaluation_history'][-5:]]
                    recent_avg = np.mean(recent_scores) if recent_scores else 0.0
                    
                    # Get last 6 cycles' scores
                    last_6_scores = [eval_data['score'] for eval_data in data['evaluation_history'][-6:]]
                    last_6_dates = [eval_data['date'] for eval_data in data['evaluation_history'][-6:]]
                    
                    # Calculate trend metrics
                    trend = 'stable'
                    if recent_avg > data['average_score']:
                        trend = 'improving'
                    elif recent_avg < data['average_score']:
                        trend = 'declining'
                    
                    # Calculate performance consistency (standard deviation of last 6 scores)
                    consistency_score = 0.0
                    if len(last_6_scores) > 1:
                        consistency_score = 1.0 - min(1.0, np.std(last_6_scores) / np.mean(last_6_scores))
                    
                    summary['models'][model] = {
                        'rank': data['rank'],
                        'overall_score': data['average_score'],
                        'recent_score': recent_avg,
                        'total_evaluations': data['total_evaluations'],
                        'trend': trend,
                        'last_6_scores': last_6_scores,
                        'last_6_dates': last_6_dates,
                        'consistency_score': consistency_score,
                        'best_score': max(last_6_scores) if last_6_scores else 0.0,
                        'worst_score': min(last_6_scores) if last_6_scores else 0.0
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}