import pandas as pd
import json
import os
from typing import List, Dict

def create_evaluation_cases() -> List[Dict]:
    """Create specific test cases for selected careers."""
    
    evaluation_cases = [
        {
            "case_name": "Software Engineer",
            "profile": {
                "skills": [
                    "programming",
                    "solve complex problems",
                    "technical expertise",
                    "think critically",
                    "analyze"
                ],
                "work_culture": [
                    "innovation",
                    "challenging",
                    "professional development",
                    "flexibility"
                ],
                "core_values": [
                    "intellectual challenge",
                    "learning",
                    "excellence"
                ]
            },
            "queries": [
                "I want a job where I can code and solve complex technical problems",
                "Looking for software development positions with opportunities to learn and grow",
                "Tech job with focus on programming and innovation"
            ],
            "expected_titles": [
                "Software Engineer",
                "Software Developer",
                "Full Stack Developer",
                "Backend Engineer"
            ]
        },
        {
            "case_name": "Business Analyst",
            "profile": {
                "skills": [
                    "interpret data",
                    "analyze",
                    "communicate verbally",
                    "think critically",
                    "project management"
                ],
                "work_culture": [
                    "collaboration",
                    "professional development",
                    "engaging work"
                ],
                "core_values": [
                    "impact",
                    "excellence",
                    "learning"
                ]
            },
            "queries": [
                "Looking for a role analyzing business data and processes",
                "Want to work with data and communicate insights to stakeholders",
                "Business analysis position with project management responsibilities"
            ],
            "expected_titles": [
                "Business Analyst",
                "Business Systems Analyst",
                "Process Analyst",
                "Business Intelligence Analyst"
            ]
        },
        {
            "case_name": "Architect",
            "profile": {
                "skills": [
                    "design/create",
                    "technical expertise",
                    "build",
                    "solve complex problems",
                    "project management"
                ],
                "work_culture": [
                    "creativity",
                    "professional development",
                    "collaboration"
                ],
                "core_values": [
                    "excellence",
                    "creativity",
                    "environmental responsibility"
                ]
            },
            "queries": [
                "Looking for architectural design and project management role",
                "Want to design buildings and manage construction projects",
                "Architecture position focusing on sustainable design"
            ],
            "expected_titles": [
                "Architect",
                "Project Architect",
                "Design Architect",
                "Architectural Designer"
            ]
        },
        {
            "case_name": "Mechanical Engineer",
            "profile": {
                "skills": [
                    "technical expertise",
                    "solve complex problems",
                    "design/create",
                    "analyze",
                    "build"
                ],
                "work_culture": [
                    "innovation",
                    "challenging",
                    "collaboration"
                ],
                "core_values": [
                    "excellence",
                    "intellectual challenge",
                    "impact"
                ]
            },
            "queries": [
                "Seeking mechanical engineering position with focus on design",
                "Looking for role in mechanical systems and product development",
                "Engineering job working on complex mechanical problems"
            ],
            "expected_titles": [
                "Mechanical Engineer",
                "Design Engineer",
                "Product Engineer",
                "Manufacturing Engineer"
            ]
        },
        {
            "case_name": "Registered Nurse",
            "profile": {
                "skills": [
                    "customer/client focus",
                    "empathy",
                    "work in teams",
                    "technical expertise",
                    "make decisions"
                ],
                "work_culture": [
                    "fast-paced",
                    "supportive environment",
                    "service focus"
                ],
                "core_values": [
                    "helping",
                    "compassion",
                    "safety"
                ]
            },
            "queries": [
                "Looking for nursing position in patient care",
                "Want to work as a registered nurse in healthcare",
                "Nursing role focused on patient care and teamwork"
            ],
            "expected_titles": [
                "Registered Nurse",
                "RN",
                "Staff Nurse",
                "Clinical Nurse"
            ]
        },
        {
            "case_name": "Web Designer",
            "profile": {
                "skills": [
                    "design/create",
                    "creativity",
                    "technical expertise",
                    "customer/client focus",
                    "build"
                ],
                "work_culture": [
                    "innovation",
                    "creativity",
                    "flexibility"
                ],
                "core_values": [
                    "creativity",
                    "excellence",
                    "authenticity"
                ]
            },
            "queries": [
                "Looking for web design position with creative focus",
                "Want to design and build websites",
                "Creative web design role with client interaction"
            ],
            "expected_titles": [
                "Web Designer",
                "UI Designer",
                "Digital Designer",
                "Web Developer"
            ]
        },
        {
            "case_name": "Emergency Medical Technician",
            "profile": {
                "skills": [
                    "make decisions",
                    "work in teams",
                    "technical expertise",
                    "empathy",
                    "customer/client focus"
                ],
                "work_culture": [
                    "fast-paced",
                    "challenging",
                    "service focus"
                ],
                "core_values": [
                    "helping",
                    "safety",
                    "community"
                ]
            },
            "queries": [
                "Looking for EMT position in emergency medical services",
                "Want to work in emergency medical response",
                "EMT role helping people in emergency situations"
            ],
            "expected_titles": [
                "EMT",
                "Emergency Medical Technician",
                "Paramedic",
                "Emergency Medical Services"
            ]
        },
        {
            "case_name": "Retail Store Manager",
            "profile": {
                "skills": [
                    "lead",
                    "manage and develop others",
                    "customer/client focus",
                    "organize",
                    "make decisions"
                ],
                "work_culture": [
                    "fast-paced",
                    "service focus",
                    "mentoring"
                ],
                "core_values": [
                    "success",
                    "relationships",
                    "achievement"
                ]
            },
            "queries": [
                "Looking for retail management position",
                "Want to manage a retail store and lead teams",
                "Retail leadership role with focus on customer service"
            ],
            "expected_titles": [
                "Store Manager",
                "Retail Manager",
                "Store Leader",
                "Retail Store Manager"
            ]
        },
        {
            "case_name": "Financial Analyst",
            "profile": {
                "skills": [
                    "analyze",
                    "interpret data",
                    "quantitative",
                    "think critically",
                    "manage finances"
                ],
                "work_culture": [
                    "professional development",
                    "challenging",
                    "stability"
                ],
                "core_values": [
                    "excellence",
                    "integrity",
                    "intellectual challenge"
                ]
            },
            "queries": [
                "Seeking financial analysis position",
                "Looking for role analyzing financial data and markets",
                "Financial analyst position with focus on data analysis"
            ],
            "expected_titles": [
                "Financial Analyst",
                "Investment Analyst",
                "Finance Analyst",
                "Business Financial Analyst"
            ]
        },
        {
            "case_name": "Medical Assistant",
            "profile": {
                "skills": [
                    "customer/client focus",
                    "empathy",
                    "organize",
                    "work in teams",
                    "technical expertise"
                ],
                "work_culture": [
                    "service focus",
                    "supportive environment",
                    "collaboration"
                ],
                "core_values": [
                    "helping",
                    "compassion",
                    "excellence"
                ]
            },
            "queries": [
                "Looking for medical assistant position in healthcare",
                "Want to work as a medical assistant helping patients",
                "Healthcare support role with patient care focus"
            ],
            "expected_titles": [
                "Medical Assistant",
                "Clinical Assistant",
                "Healthcare Assistant",
                "Clinical Medical Assistant"
            ]
        }
    ]
    
    return evaluation_cases

def find_matching_job_ids(df: pd.DataFrame, job_titles: List[str]) -> List[str]:
    """Find job IDs for specific job titles in the dataset."""
    pattern = '|'.join(job_titles)
    matching_jobs = df[df['title'].str.contains(pattern, case=False, na=False)]
    return matching_jobs['job_id'].tolist()

def create_evaluation_dataset(input_csv: str, output_dir: str):
    """Create evaluation dataset with specific test cases."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load job data
    df = pd.read_csv(input_csv)
    
    # Get test cases
    test_cases = create_evaluation_cases()
    
    # Process each test case
    evaluation_data = []
    for case in test_cases:
        # Find actual job IDs for expected matches
        expected_job_ids = find_matching_job_ids(df, case['expected_titles'])
        
        if expected_job_ids:
            evaluation_data.append({
                "case_name": case['case_name'],
                "profile": case['profile'],
                "queries": case['queries'],
                "expected_job_ids": expected_job_ids
            })
        else:
            print(f"Warning: No matching jobs found for case '{case['case_name']}'")
    
    # Save evaluation dataset
    output_file = os.path.join(output_dir, 'evaluation_dataset.json')
    with open(output_file, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
    print(f"Created evaluation dataset at {output_file}")
    
    return evaluation_data

if __name__ == "__main__":
    input_csv = 'backend/data/checkpoints/enriched_data_checkpoint_testing.csv'
    output_dir = 'backend/data/evaluation'
    
    evaluation_dataset = create_evaluation_dataset(input_csv, output_dir)
    print(f"Generated {len(evaluation_dataset)} evaluation cases") 