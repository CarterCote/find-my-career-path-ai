import requests
import json

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
        "additional_interests": "Interested in tech companies with strong mentorship programs"
    }
    
    # First, create a user profile with preferences
    print("Creating user profile with preferences...")
    profile_response = requests.post(
        'http://127.0.0.1:8000/users/preferences',
        json=profile_data
    )
    
    # Debug output
    print(f"\nProfile creation status code: {profile_response.status_code}")
    print(f"Profile creation response: {profile_response.text}")
    
    if profile_response.status_code != 200:
        print(f"Error creating profile: {profile_response.text}")
        return
        
    print("\nProfile created successfully!")
    print(f"Profile details: {profile_response.text}")
    
    # Generate initial questions based on profile
    print("\nGenerating questions based on your profile...")
    initial_message = "Based on my profile, what specific questions should I answer to help refine my job search?"
    questions_response = requests.post(
        'http://127.0.0.1:8000/chat',
        params={
            "message": initial_message,
            "session_id": session_id
        }
    )
    
    # Debug output
    print(f"\nQuestions generation status code: {questions_response.status_code}")
    print(f"Questions generation response: {questions_response.text}")
    
    if questions_response.status_code == 200:
        try:
            result = questions_response.json()
            print("\nAssistant:", result.get('response', 'No response found'))
        except json.JSONDecodeError:
            print("\nError: Could not parse JSON response")
    else:
        print("\nError generating questions:", questions_response.status_code, questions_response.text)
    
    print("\nCareer Path AI Assistant (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        # Get user input
        query = input("\nYou: ")
        if query.lower() == 'quit':
            break
            
        # Send request to API
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
            except json.JSONDecodeError:
                print("\nError: Could not parse JSON response")
        else:
            print("\nError:", response.status_code, response.text)

if __name__ == "__main__":
    chat()