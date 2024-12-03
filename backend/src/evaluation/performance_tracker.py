from typing import Dict, List
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

class ModelPerformanceTracker:
    def __init__(self):
        self.metrics_history = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'recommendation_accuracy': [],
            'user_satisfaction': []
        }
        self.timestamp = []
        
    def track_recommendation_performance(self, 
                                       predicted_jobs: List[Dict], 
                                       actual_selections: List[str],
                                       user_feedback: Dict) -> Dict:
        """Track recommendation performance metrics"""
        # Calculate metrics
        precision = len(set(predicted_jobs) & set(actual_selections)) / len(predicted_jobs) if predicted_jobs else 0
        recall = len(set(predicted_jobs) & set(actual_selections)) / len(actual_selections) if actual_selections else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update history
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        self.metrics_history['f1_score'].append(f1)
        self.metrics_history['user_satisfaction'].append(user_feedback.get('satisfaction_score', 0))
        self.timestamp.append(datetime.now())
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'user_satisfaction': user_feedback.get('satisfaction_score', 0)
        }
        
    def plot_learning_curves(self, save_path: str = None):
        """Generate and optionally save learning curves"""
        plt.figure(figsize=(12, 8))
        
        # Plot metrics
        plt.subplot(2, 1, 1)
        for metric in ['precision', 'recall', 'f1_score']:
            plt.plot(self.timestamp, self.metrics_history[metric], label=metric.replace('_', ' ').title())
        plt.title('Model Performance Metrics Over Time')
        plt.legend()
        
        # Plot user satisfaction
        plt.subplot(2, 1, 2)
        plt.plot(self.timestamp, self.metrics_history['user_satisfaction'], label='User Satisfaction')
        plt.title('User Satisfaction Score Over Time')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()