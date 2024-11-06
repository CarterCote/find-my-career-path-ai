from typing import Dict, List
import json

def extract_job_features(structured_description: Dict) -> Dict:
    """Extract searchable features from structured job descriptions"""
    features = {
        'required_skills': set(),
        'experience_years': 0,
        'education_level': None,
        'work_environment': None,
        'benefits': set()
    }
    
    try:
        desc = json.loads(structured_description) if isinstance(structured_description, str) else structured_description
        
        # Extract skills
        if 'Required skills' in desc:
            features['required_skills'].update(desc['Required skills'])
            
        # Extract years of experience
        if 'Years of experience' in desc:
            # Convert text to number (e.g., "3-5 years" -> 3)
            exp = desc['Years of experience'].lower()
            features['experience_years'] = parse_years_experience(exp)
            
        # Extract other features...
        
    except Exception as e:
        print(f"Error extracting features: {e}")
    
    return features

def parse_years_experience(exp_text: str) -> int:
    """Convert experience text to minimum years required"""
    # Add parsing logic here
    pass