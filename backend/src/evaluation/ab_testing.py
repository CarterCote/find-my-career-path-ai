from typing import Dict, List
import random
from ..utils.job_search import JobSearchService
from .performance_tracker import ModelPerformanceTracker

class RecommendationABTesting:
    def __init__(self, settings, db):
        self.models = {
            'baseline': JobSearchService(settings, db, use_enhanced=False),
            'enhanced': JobSearchService(settings, db, use_enhanced=True)
        }
        self.performance_trackers = {
            'baseline': ModelPerformanceTracker(),
            'enhanced': ModelPerformanceTracker()
        }
        
    def run_ab_test(self, user_profiles: List[Dict], test_duration_days: int = 30):
        """Run A/B test comparing different recommendation approaches"""
        results = {model_name: [] for model_name in self.models.keys()}
        
        for profile in user_profiles:
            # Randomly assign user to test group
            model_name = random.choice(list(self.models.keys()))
            model = self.models[model_name]
            
            # Get recommendations
            recommendations = model.get_initial_recommendations(profile)
            
            # Track performance
            performance = self.performance_trackers[model_name].track_recommendation_performance(
                recommendations,
                profile.get('selected_jobs', []),
                profile.get('feedback', {})
            )
            
            results[model_name].append(performance)
            
        return self.analyze_test_results(results)
        
    def analyze_test_results(self, results: Dict) -> Dict:
        """Analyze A/B test results"""
        analysis = {}
        for model_name, performances in results.items():
            analysis[model_name] = {
                'avg_precision': np.mean([p['precision'] for p in performances]),
                'avg_recall': np.mean([p['recall'] for p in performances]),
                'avg_f1': np.mean([p['f1_score'] for p in performances]),
                'avg_satisfaction': np.mean([p['user_satisfaction'] for p in performances])
            }
        return analysis