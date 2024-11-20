from typing import Dict, List
import json
from ..models import JobPosting, UserProfile
from ..prompts.evaluation_prompts import (
    TEST_PROFILE_GENERATION,
    TEST_QUERY_GENERATION
)

class TestCaseGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate_test_profiles(self, job_posting: JobPosting) -> List[Dict]:
        """Generate synthetic user profiles that would be good matches for a job"""
        prompt = TEST_PROFILE_GENERATION.format(
            title=job_posting.title,
            description=job_posting.description,
            company=job_posting.company_name
        )
        
        response = self.llm.chat([
            {"role": "system", "content": "You are an expert at creating realistic candidate profiles."},
            {"role": "user", "content": prompt}
        ])
        
        return json.loads(response.content)

    def generate_test_queries(self, profile: Dict) -> List[str]:
        """Generate realistic search queries a user might make"""
        prompt = TEST_QUERY_GENERATION.format(
            profile_json=json.dumps(profile, indent=2)
        )
        
        response = self.llm.chat([
            {"role": "system", "content": "You are an expert at creating realistic job search queries."},
            {"role": "user", "content": prompt}
        ])
        
        return json.loads(response.content)

    def create_evaluation_dataset(self, job_postings: List[JobPosting]) -> Dict:
        """Create comprehensive evaluation dataset"""
        dataset = {
            'profile_job_pairs': [],
            'query_answer_pairs': []
        }
        
        for job in job_postings:
            # Generate matching profiles
            profiles = self.generate_test_profiles(job)
            
            # Create profile-job pairs
            for profile in profiles:
                dataset['profile_job_pairs'].append({
                    'profile': profile,
                    'job': {
                        'job_id': job.job_id,
                        'title': job.title,
                        'company': job.company_name,
                        'description': job.description
                    },
                    'expected_match_score': 0.8  # High score since profile was generated to match
                })
                
                # Generate queries for this profile
                queries = self.generate_test_queries(profile)
                
                # Add query-answer pairs
                for query in queries:
                    dataset['query_answer_pairs'].append({
                        'profile': profile,
                        'query': query,
                        'expected_jobs': [job.job_id],  # This job should be in results
                        'relevance_score': 0.8  # High relevance since query was generated for this job
                    })
        
        return dataset

def run_evaluation(test_cases: Dict, search_service: 'JobSearchService') -> Dict:
    """Run evaluation using test cases"""
    metrics = {
        'profile_match_accuracy': [],
        'query_relevance': [],
        'ranking_accuracy': []
    }
    
    # Evaluate profile-job matching
    for pair in test_cases['profile_job_pairs']:
        results = search_service.evaluate_recommendations(
            recommendations=[pair['job']],
            user_profile=pair['profile']
        )
        
        if results['evaluations']:
            actual_score = results['evaluations'][0]['persona_score']
            expected_score = pair['expected_match_score']
            metrics['profile_match_accuracy'].append(
                1 - abs(actual_score - expected_score)
            )
    
    # Evaluate query-based search
    for qa_pair in test_cases['query_answer_pairs']:
        results = search_service.search(
            query=qa_pair['query'],
            profile_data=qa_pair['profile']
        )
        
        # Check if expected jobs are in results
        found_jobs = [job['job_id'] for job in results.get('jobs', [])]
        relevant_count = len(set(found_jobs) & set(qa_pair['expected_jobs']))
        
        if found_jobs:
            metrics['query_relevance'].append(
                relevant_count / len(found_jobs)
            )
            
            # Check ranking (expected jobs should be ranked higher)
            correct_rank = 0
            for idx, job_id in enumerate(found_jobs):
                if job_id in qa_pair['expected_jobs']:
                    correct_rank = 1.0 / (idx + 1)  # Higher score for higher ranks
                    break
            metrics['ranking_accuracy'].append(correct_rank)
    
    return {
        'profile_match_accuracy': sum(metrics['profile_match_accuracy']) / len(metrics['profile_match_accuracy']) if metrics['profile_match_accuracy'] else 0,
        'query_relevance': sum(metrics['query_relevance']) / len(metrics['query_relevance']) if metrics['query_relevance'] else 0,
        'ranking_accuracy': sum(metrics['ranking_accuracy']) / len(metrics['ranking_accuracy']) if metrics['ranking_accuracy'] else 0
    }