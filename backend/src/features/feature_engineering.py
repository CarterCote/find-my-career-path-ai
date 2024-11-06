from typing import Dict, List, Set, Optional
import re
import json

def parse_years_experience(exp_text: str) -> int:
    """Convert experience text to minimum years required"""
    if not exp_text:
        return 0
        
    # Convert text patterns to minimum years
    exp_text = exp_text.lower()
    
    # Extract numbers from text
    numbers = re.findall(r'\d+', exp_text)
    if not numbers:
        if 'entry' in exp_text or 'junior' in exp_text:
            return 0
        if 'mid' in exp_text or 'intermediate' in exp_text:
            return 3
        if 'senior' in exp_text or 'experienced' in exp_text:
            return 5
        return 0
        
    # Get the minimum number mentioned
    return int(min(numbers))

def parse_education_level(edu_text: str) -> str:
    """Normalize education requirements"""
    if not edu_text:
        return "Not specified"
        
    edu_text = edu_text.lower()
    
    if any(term in edu_text for term in ['phd', 'doctorate']):
        return "PhD"
    if any(term in edu_text for term in ['master', 'ms', 'msc']):
        return "Master's"
    if any(term in edu_text for term in ['bachelor', 'bs', 'ba']):
        return "Bachelor's"
    if any(term in edu_text for term in ['associate', 'aa']):
        return "Associate's"
    
    return "Not specified"

def extract_job_features(structured_description: Dict) -> Dict:
    """Extract searchable features from structured job descriptions"""
    features = {
        'required_skills': set(),
        'preferred_skills': set(),
        'experience_years': 0,
        'education_level': "Not specified",
        'work_environment': "Not specified",
        'benefits': set(),
        'certifications': set(),
        'job_type': "Not specified",
        'responsibilities': set()
    }
    
    try:
        desc = json.loads(structured_description) if isinstance(structured_description, str) else structured_description
        
        # Extract skills
        if 'Required skills' in desc:
            features['required_skills'].update(
                skill.lower().strip() for skill in desc['Required skills']
            )
        
        # Extract years of experience
        if 'Years of experience' in desc:
            features['experience_years'] = parse_years_experience(desc['Years of experience'])
            
        # Extract education
        if 'Education requirements' in desc:
            features['education_level'] = parse_education_level(desc['Education requirements'])
            
        # Extract work environment
        if 'Work environment' in desc:
            features['work_environment'] = desc['Work environment'].lower().strip()
            
        # Extract benefits
        if 'Benefits mentioned' in desc:
            features['benefits'].update(
                benefit.lower().strip() for benefit in desc['Benefits mentioned']
            )
            
        # Extract certifications
        if 'Required certifications' in desc:
            features['certifications'].update(
                cert.lower().strip() for cert in desc['Required certifications']
            )
            
        # Extract responsibilities
        if 'Key responsibilities' in desc:
            features['responsibilities'].update(
                resp.lower().strip() for resp in desc['Key responsibilities']
            )
            
    except Exception as e:
        print(f"Error extracting features: {e}")
    
    return features