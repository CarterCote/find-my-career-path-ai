from typing import Dict, List
import numpy as np
from ..models import JobPosting, UserProfile

class RecommendationMetrics:
    @staticmethod
    def calculate_skill_alignment(persona_skills: List[str], profile_skills: List[str]) -> float:
        """Calculate skill alignment score"""
        if not persona_skills or not profile_skills:
            return 0.0
        
        persona_set = set(s.lower() for s in persona_skills)
        profile_set = set(s.lower() for s in profile_skills)
        
        return len(persona_set.intersection(profile_set)) / len(persona_set)

    @staticmethod
    def calculate_values_alignment(persona_values: List[str], profile_values: List[str]) -> float:
        """Calculate values alignment score"""
        if not persona_values or not profile_values:
            return 0.0
            
        persona_set = set(v.lower() for v in persona_values)
        profile_set = set(v.lower() for v in profile_values)
        
        return len(persona_set.intersection(profile_set)) / len(persona_set)

    @staticmethod
    def aggregate_scores(evaluations: List[Dict]) -> Dict:
        """Aggregate scores across multiple evaluations"""
        scores = {
            'skills_alignment': [],
            'values_compatibility': [],
            'culture_fit': [],
            'growth_potential': []
        }
        
        for eval in evaluations:
            scores['skills_alignment'].append(eval['skills_alignment'])
            scores['values_compatibility'].append(eval['values_compatibility'])
            scores['culture_fit'].append(eval['culture_fit'])
            scores['growth_potential'].append(eval['growth_potential'])
            
        return {k: np.mean(v) for k, v in scores.items()}