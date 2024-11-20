import requests
import json

def display_evaluations(evaluations_data):
    """Helper function to display evaluation results consistently"""
    if not evaluations_data:
        return
        
    print("\nEvaluation Results:")
    print("-" * 50)
    
    evaluations = evaluations_data.get('evaluations', [])
    metrics = evaluations_data.get('metrics', {})
    
    if evaluations:
        print("\nOverall Metrics:")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.2f}/10")
        
        print("\nDetailed Job Evaluations:")
        for eval in evaluations:
            print(f"\nJob: {eval['title']}")
            print(f"System Score: {eval['system_score']:.2f}")
            print(f"Persona Score: {eval['persona_score']:.2f}")
            
            detailed = eval['detailed_evaluation']
            print("\nDetailed Scores:")
            print(f"Skills Alignment: {detailed['skills_alignment']}/10")
            print(f"Values Compatibility: {detailed['values_compatibility']}/10")
            print(f"Culture Fit: {detailed['culture_fit']}/10")
            print(f"Growth Potential: {detailed['growth_potential']}/10")
            
            print("\nReasoning:")
            for category, reason in detailed['reasoning'].items():
                print(f"{category.title()}: {reason}")
            
            if detailed.get('skill_gaps'):
                print("\nSkill Gaps:")
                for gap in detailed['skill_gaps']:
                    print(f"- {gap}")
                    
            if detailed.get('culture_fit_details'):
                print("\nCulture Fit Details:")
                for detail in detailed['culture_fit_details']:
                    print(f"- {detail}")
    else:
        print("No evaluation results available")

def chat():
    session_id = "test-session"
    
    # Your profile JSON
    profile_data = {
        "session_id": session_id,
        "core_values": [
            "integrity", "personal development", "impact", "creativity",
            "excellence", "intellectual challenge", "learning", "autonomy",
            "relationships", "trust"
        ],
        "work_culture": [
            "inspiring", "growth potential", "innovation", "professional development",
            "flexibility", "supportive environment", "mentoring", "inclusive workplace",
            "life balance", "recognition"
        ],
        "skills": [
            "programming", "solve complex problems", "think critically",
            "technical expertise", "strategic planning", "project management",
            "analyze", "research", "build relationships", "communicate verbally"
        ],
        "additional_interests": ""
    }
    
    # First, create a user profile with preferences
    print("Creating user profile with preferences...")
    profile_response = requests.post(
        'http://127.0.0.1:8000/users/preferences',
        json=profile_data
    )
    
    print(f"\nProfile creation status code: {profile_response.status_code}")
    print(f"Profile creation response: {profile_response.text}")
    
    if profile_response.status_code != 200:
        print(f"Error creating profile: {profile_response.text}")
        return
    
    print("\nProfile created successfully!")
    
    # Start Q&A process
    print("\nGenerating questions based on your profile...")
    initial_message = "Based on my profile, what specific questions should I answer to help refine my job search?"
    
    print(f"\nDebug - About to make POST request to /chat with:")
    print(f"message: {initial_message}")
    print(f"session_id: {session_id}")
    
    questions_response = requests.post(
        'http://127.0.0.1:8000/chat',
        params={
            "message": initial_message,
            "session_id": session_id
        }
    )
    
    print(f"\nDebug - Raw response from /chat:")
    print(f"Status code: {questions_response.status_code}")
    print(f"Headers: {dict(questions_response.headers)}")
    print(f"Content: {questions_response.content}")
    
    if questions_response.status_code != 200:
        print("\nError starting Q&A:", questions_response.status_code)
        return
    
    # Handle Q&A session
    qa_complete = False
    while not qa_complete:
        try:
            result = questions_response.json()
            print("\nAssistant:", result.get('response', 'No response found'))
            
            # Check if Q&A is complete
            if result.get('qa_complete'):
                print("\nQ&A session complete! Getting job recommendations...")
                qa_complete = True
                break
            
            # Get user's answer
            answer = input("\nYour answer: ")
            
            # Send answer
            questions_response = requests.post(
                'http://127.0.0.1:8000/chat',
                params={
                    "message": answer,
                    "session_id": session_id
                }
            )
            
            if questions_response.status_code != 200:
                print("\nError:", questions_response.status_code)
                break
                
        except Exception as e:
            print(f"\nError in Q&A session: {str(e)}")
            break
    
    # Get recommendations after Q&A complete
    if qa_complete:
        print("\nFetching job recommendations...")
        recommendations_response = requests.get(
            f'http://127.0.0.1:8000/users/recommendations/{session_id}'
        )
        
        if recommendations_response.status_code == 200:
            recommendations = recommendations_response.json()
            print("\nJob Recommendations:")
            print("-" * 50)
            
            for job in recommendations.get('recommendations', []):
                print(f"\nJob: {job['title']}")
                print(f"Company: {job['company_name']}")
                print(f"Match Score: {job['match_score']:.2f}")
                print(f"Matching Skills: {', '.join(job['matching_skills'])}")
                print(f"Matching Culture: {', '.join(job['matching_culture'])}")
                if job.get('location'):
                    print(f"Location: {job['location']}")
                print("-" * 30)
            
            # Handle evaluations if present
            if 'evaluation' in recommendations:
                display_evaluations(recommendations['evaluation'])
        else:
            print("\nError getting recommendations:", recommendations_response.status_code)
    
    # Continue with regular chat
    print("\nCareer Path AI Assistant (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        query = input("\nYou: ")
        if query.lower() == 'quit':
            break
            
        response = requests.post(
            'http://127.0.0.1:8000/chat',
            params={
                "message": query,
                "session_id": session_id
            }
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\nAssistant:", result.get('response', 'No response found'))
                
                # Display evaluations if present in chat response
                if 'evaluation' in result:
                    display_evaluations(result['evaluation'])
                    
            except json.JSONDecodeError:
                print("\nError: Could not parse JSON response")
        else:
            print("\nError:", response.status_code, response.text)

if __name__ == "__main__":
    chat()