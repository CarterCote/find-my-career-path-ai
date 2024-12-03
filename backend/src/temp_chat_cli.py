import requests
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List

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

def plot_performance_data(performance_data: Dict):
    """Plot performance metrics from the API response"""
    try:
        metrics_history = performance_data.get('metrics_history', {})
        timestamps = [datetime.fromisoformat(ts) for ts in performance_data.get('timestamp', [])]
        
        if not metrics_history or not timestamps:
            print("No performance data available to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot metrics
        plt.subplot(2, 1, 1)
        for metric, values in metrics_history.items():
            if metric != 'user_satisfaction':
                plt.plot(timestamps, values, label=metric.replace('_', ' ').title())
        plt.title('Model Performance Metrics Over Time')
        plt.legend()
        
        # Plot user satisfaction
        plt.subplot(2, 1, 2)
        if 'user_satisfaction' in metrics_history:
            plt.plot(timestamps, metrics_history['user_satisfaction'], label='User Satisfaction')
        plt.title('User Satisfaction Score Over Time')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('experiments/learning_curves.png')
        plt.close()
        
        print("\nPerformance visualization saved to experiments/learning_curves.png")
    except Exception as e:
        print(f"Error plotting performance data: {str(e)}")

def chat():
    print("\nCareer Path AI Assistant")
    print("-" * 50)
    print("1. Create recommendations from a profile")
    print("2. Update recommendations from a profile")
    choice = input("\nEnter your choice (1 or 2): ")

    if choice == "2":
        # Get existing profiles
        response = requests.get('http://127.0.0.1:8000/users/profiles')
        if response.status_code == 200:
            profiles = response.json()
            if not profiles:
                print("\nNo existing profiles found.")
                return
                
            print("\nExisting Profiles:")
            for profile in profiles:
                print(f"ID: {profile['id']} - Session: {profile['user_session_id']}")
            
            profile_id = input("\nEnter profile ID to access: ")
            
            # Get recommendations for existing profile
            rec_response = requests.get(f'http://127.0.0.1:8000/users/profile/{profile_id}/recommendations')
            if rec_response.status_code == 200:
                recommendations = rec_response.json()
                print("\nExisting Job Recommendations:")
                print("-" * 50)
                
                # Enhanced recommendation display
                for job in recommendations:
                    print(f"\nJob Title: {job['title']}")
                    print(f"Company: {job['company_name']}")
                    print(f"Match Score: {job['match_score']:.2f}")
                    print(f"Type: {job['recommendation_type']}")
                    print(f"Job ID: {job['job_id']}")
                    print(f"Created: {job['created_at']}")
                    print("-" * 30)
                
                # Ask if user wants to start a new chat session
                new_chat = input("\nWould you like to start a new chat session? (y/n): ")
                if new_chat.lower() == 'y':
                    # Generate a new chat session ID for this user
                    chat_session_id = f"chat-{random.randint(1000, 9999999)}"
                    print(f"\nStarting new chat session: {chat_session_id}")
                    # Continue with chat...
                else:
                    return
            else:
                print("\nError retrieving recommendations:", rec_response.status_code)
            return
            
    else:  # Choice 1: Create recommendations from a profile
        # Get existing profiles to let user select one
        print("\nFetching existing user profiles...")
        response = requests.get('http://127.0.0.1:8000/users/profiles')
        if response.status_code == 200:
            profiles = response.json()
            if not profiles:
                print("\nNo existing profiles found.")
                return
                
            print("\nExisting Profiles:")
            for profile in profiles:
                print(f"ID: {profile['id']} - Session: {profile['user_session_id']}")
            
            user_id = input("\nEnter user ID to use their preferences: ")
            
            # Fetch user's preferences
            user_response = requests.get(f'http://127.0.0.1:8000/users/{user_id}')
            if user_response.status_code != 200:
                print(f"\nError fetching user data: {user_response.status_code}")
                return
                
            user_data = user_response.json()
            chat_session_id = f"session-{random.randint(1000, 9999999)}"
            
            print("\nUsing the following preferences:")
            print(f"Core Values: {', '.join(user_data['core_values'])}")
            print(f"Work Culture: {', '.join(user_data['work_culture'])}")
            print(f"Skills: {', '.join(user_data['skills'])}")
            
            # Start Q&A process
            print("\nGenerating questions based on your profile...")
            initial_message = "Based on my profile, what specific questions should I answer to help refine my job search?"
            
            print(f"\nDebug - About to make POST request to /chat with:")
            print(f"message: {initial_message}")
            print(f"chat_session_id: {chat_session_id}")
            
            questions_response = requests.post(
                'http://127.0.0.1:8000/chat',
                params={
                    "message": initial_message,
                    "chat_session_id": chat_session_id
                }
            )
            
            if questions_response.status_code != 200:
                print("\nError starting Q&A:", questions_response.status_code)
                return
            
            # Handle Q&A session
            qa_complete = False
            while not qa_complete:
                try:
                    result = questions_response.json()
                    print("\nAssistant:", result.get('response', 'No response found'))
                    
                    if result.get('qa_complete'):
                        print("\nQ&A session complete! Getting job recommendations...")
                        qa_complete = True
                        break
                    
                    answer = input("\nYour answer: ")
                    
                    questions_response = requests.post(
                        'http://127.0.0.1:8000/chat',
                        params={
                            "message": answer,
                            "chat_session_id": chat_session_id
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
                    f'http://127.0.0.1:8000/users/recommendations/{chat_session_id}'
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
                    
                    if 'evaluation' in recommendations:
                        display_evaluations(recommendations['evaluation'])
                    
                    if 'performance_data' in recommendations:
                        plot_performance_data(recommendations['performance_data'])
                    
                    while True:
                        choice = input("\nWhat would you like to do?\n1. Start new profile\n2. Exit\nYour choice (1 or 2): ")
                        if choice == "1":
                            chat()
                            break
                        elif choice == "2":
                            print("\nThank you for using Career Path AI Assistant!")
                            return
                        else:
                            print("\nInvalid choice. Please enter 1 or 2.")
                else:
                    print("\nError getting recommendations:", recommendations_response.status_code)

if __name__ == "__main__":
    chat()